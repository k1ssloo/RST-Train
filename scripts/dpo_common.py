#!/usr/bin/env python3
"""Shared pieces for the DPO fallback path: fingerprints, masked logprobs, losses.

One implementation, imported by both `18_dpo_ref_logprobs.py` (which runs it under
`no_grad` to freeze the reference) and `19_train_dpo.py` (which runs the identical
code with gradients on the policy). That sameness is not tidiness -- it is what
makes the step-0 calibration check meaningful: at initialization the policy IS the
reference, so the two logprob sums must agree to within float noise. If the two
scripts computed logprobs differently, that check would be measuring the
difference between two implementations instead of the difference between two
models, and a silently wrong mask or dtype would sail through.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FINGERPRINT_HEAD_BYTES = 1 << 20  # 1 MiB per weight file


def checkpoint_fingerprint(model_path: str | Path) -> str:
    """Content-addressed identity for a local HF checkpoint directory.

    Reference logprobs are only valid for the exact weights that produced them. A
    path string is not identity (checkpoints get overwritten in place, and
    `out-hf-full` is a moving target across runs), so we hash the config plus the
    size and leading bytes of every weight shard. Reading 1 MiB per shard is fast
    and still catches a re-export, a different conversion, or a resumed-then-saved
    checkpoint; it is a mismatch detector, not a cryptographic commitment.
    """
    root = Path(model_path)
    digest = hashlib.sha256()
    config = root / "config.json"
    if config.is_file():
        digest.update(b"config\0")
        digest.update(config.read_bytes())
    weights = sorted(
        [p for p in root.iterdir() if p.suffix in (".safetensors", ".bin")],
        key=lambda p: p.name,
    )
    for path in weights:
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("utf-8"))
        with path.open("rb") as handle:
            digest.update(handle.read(FINGERPRINT_HEAD_BYTES))
    if not weights:
        digest.update(b"no-weight-files")
    return digest.hexdigest()


AUTO_CLASSES = ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel")


def load_model(model_path: str | Path, *, dtype, device_map: str | None = None):
    """Load whatever kind of checkpoint this is, and say which auto class worked.

    Same probe as `06b_eval_offline.py`: the Qwen3.5 checkpoints declare
    `Qwen3_5ForConditionalGeneration`, so `AutoModelForCausalLM` alone is not enough,
    and a hard-coded class would break on the next model in the registry.
    """
    import transformers

    errors: list[str] = []
    for name in AUTO_CLASSES:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
            if device_map:
                kwargs["device_map"] = device_map
            model = cls.from_pretrained(str(model_path), **kwargs)
        except Exception as exc:  # noqa: BLE001 - probing which class fits
            errors.append(f"{name}: {type(exc).__name__}: {exc}"[:300])
            continue
        return model, name
    raise SystemExit("could not load the checkpoint with any auto class:\n  "
                     + "\n  ".join(errors))


def decoder_and_head(model) -> tuple[Any, Any]:
    """The two halves needed to score tokens without materializing full logits."""
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else None
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if decoder is None or head is None:
        raise SystemExit(
            "this model exposes no get_decoder()/get_output_embeddings(). The DPO path "
            "needs both so the LM head can be applied in slices; a single "
            "model(input_ids) call would materialize seq x vocab logits (~10 GB for one "
            "32k row at vocab 248,320) and OOM before computing anything."
        )
    return decoder, head


def supervised_positions(mask, device):
    """Indices of tokens that are TARGETS, i.e. produced by the policy.

    `loss_mask[i] == 1` means token i is supervised (see the dataset manifest: the
    mask is aligned 1:1 with input_ids, no offset). Token i is predicted from
    hidden state i-1, so position 0 can never be a target.
    """
    import torch

    flags = torch.tensor(mask, dtype=torch.bool, device=device)
    flags[0] = False
    return torch.nonzero(flags, as_tuple=False).flatten()


def _score_impl(model, ids: list[int], mask: list[int], *, chunk: int, grad: bool):
    """Body of masked_logprob_sum; see it for what this computes.

    Always entered with every parameter a plain tensor, either because the run is
    single-process or because the caller routed this through the root module's
    FSDP forward hook.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    decoder, head = decoder_and_head(model)
    param = next(model.parameters())
    device = getattr(param, "to_local", lambda: param)().device

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    targets = torch.tensor(ids, dtype=torch.long, device=device)
    positions = supervised_positions(mask, device)
    n = int(positions.numel())
    if n == 0:
        raise ValueError("no supervised positions: this row should have been dropped")

    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        hidden = decoder(input_ids=input_ids, use_cache=False).last_hidden_state[0]

        def block_nll(states, gold):
            # fp32 for the softmax: bf16 logsumexp over a 248,320-way vocab loses
            # enough precision to move a whole-sequence logprob sum by nats, and DPO
            # compares two such sums against a frozen reference.
            logits = head(states).float()
            return torch.nn.functional.cross_entropy(logits, gold, reduction="sum")

        # fp64 accumulator. The per-chunk cross-entropy is fp32, but the running total
        # reaches ~1e3 nats where an fp32 ulp is 6e-5, and DPO's `logits` is the
        # difference of two such totals that are equal at step 0 -- textbook
        # catastrophic cancellation. Summing in fp64 keeps the absolute error near
        # 1e-13 for the price of one scalar add per chunk.
        total = torch.zeros((), device=device, dtype=torch.float64)
        for start in range(0, n, chunk):
            block = positions[start : start + chunk]
            states = hidden[block - 1]
            gold = targets[block]
            if grad:
                part = checkpoint(block_nll, states, gold, use_reentrant=False)
            else:
                part = block_nll(states, gold)
            total = total + part.double()
        logp = -total
    return logp, n


_SCORER_ATTR = "_rst_fsdp_score"


def _scorer(model):
    """Return a callable that scores one row with every parameter unsharded.

    FSDP2 all-gathers a module's parameters only around that module's own
    forward. This path calls the decoder and the head separately, so the head can
    be applied in slices instead of materialising seq x vocab logits, which means
    the root module's hook never fires: the parameters the root owns (the input
    embedding, the final norm, the head) stay DTensors while every decoder
    layer's own hook has already all-gathered its parameters into plain tensors.
    One operation seeing both kinds is a hard error, and it is not fixable from
    the input side. Passing a plain input fails in the embedding:

        RuntimeError: aten.embedding.default got mixed torch.Tensor and DTensor

    and promoting the input to a DTensor only moves the same failure one module
    down, into the first layer's norm, because that layer's weights are plain:

        RuntimeError: aten.mul.Tensor got mixed torch.Tensor and DTensor

    `register_fsdp_forward_method` is the supported way to give a method other
    than forward the root's unshard, reshard and backward hooks, so the whole
    scoring pass runs with every operand plain. Registering the pass as a whole,
    rather than the decoder and the head separately, also all-gathers the root
    group once per row instead of once per logit chunk.
    """
    import types

    existing = getattr(model, _SCORER_ATTR, None)
    if existing is not None:
        return existing

    from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method

    setattr(model, _SCORER_ATTR, types.MethodType(_score_impl, model))
    if isinstance(model, FSDPModule):
        register_fsdp_forward_method(model, _SCORER_ATTR)
    return getattr(model, _SCORER_ATTR)


def masked_logprob_sum(model, ids: list[int], mask: list[int], *, chunk: int,
                       grad: bool = False):
    """Sum log p(token | prefix) over supervised positions only.

    Returns ``(logp_sum, n_supervised)`` where ``logp_sum`` is a 0-dim tensor. With
    ``grad=True`` each chunk is wrapped in activation checkpointing, so the peak
    logit footprint stays at one chunk instead of the whole sequence: without that,
    holding every chunk's logits for backward costs exactly as much as never having
    chunked at all, which is the trap this function exists to avoid.
    """
    return _scorer(model)(ids, mask, chunk=chunk, grad=grad)


def as_f64(value):
    """Promote a logprob to fp64 before any comparison against a reference.

    Reference logprobs come back from parquet as Python floats (fp64) while the
    policy's come back as tensors. Subtracting them in the tensor's dtype would round
    the fp64 side down to fp32 and make the step-0 identity approximate, which shows
    up as random signs on margins that should be exactly zero. Differentiable, so the
    training path can use it too.
    """
    import torch

    if torch.is_tensor(value):
        return value.double()
    return float(value)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, *,
             beta: float, variant: str = "sigmoid", label_smoothing: float = 0.0):
    """DPO objective on already-summed (or already-normalized) logprobs.

    ``logits`` here is the implicit reward margin divided by beta: how much more the
    policy prefers the winner than the reference does. Positive means the update is
    working in the intended direction.
    """
    import torch
    import torch.nn.functional as F

    policy_chosen, policy_rejected = as_f64(policy_chosen), as_f64(policy_rejected)
    ref_chosen, ref_rejected = as_f64(ref_chosen), as_f64(ref_rejected)
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    if variant == "ipo":
        # IPO replaces the log-sigmoid with a squared error toward 1/(2*beta),
        # which removes DPO's incentive to keep pushing an already-won pair further.
        loss = (logits - 1.0 / (2.0 * beta)) ** 2
    elif variant == "sigmoid":
        loss = (
            -F.logsigmoid(beta * logits) * (1.0 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    else:
        raise ValueError(f"unknown DPO variant {variant!r}")
    with torch.no_grad():
        metrics = {
            "reward_chosen": float(beta * (policy_chosen - ref_chosen)),
            "reward_rejected": float(beta * (policy_rejected - ref_rejected)),
            "reward_margin": float(beta * logits),
            "logits": float(logits),
        }
    return loss, metrics


def read_ref_logps(path: str | Path) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Load reference logprobs from a file or a directory of shard files.

    Returns ``(by_pair_id, manifests)``. Shards are read together so a sharded
    reference pass (one process per GPU) needs no merge step.
    """
    import pandas as pd

    root = Path(path)
    files = sorted(root.glob("ref_logps*.parquet")) if root.is_dir() else [root]
    if not files:
        raise SystemExit(f"no ref_logps*.parquet under {root}")
    table: dict[str, dict[str, float]] = {}
    manifests: list[dict] = []
    for file in files:
        frame = pd.read_parquet(file)
        for row in frame.itertuples():
            table[row.pair_id] = {
                "chosen_ref_logp": float(row.chosen_ref_logp),
                "rejected_ref_logp": float(row.rejected_ref_logp),
                "chosen_ref_n": int(row.chosen_ref_n),
                "rejected_ref_n": int(row.rejected_ref_n),
            }
        sidecar = file.with_name(file.stem + "_manifest.json")
        if sidecar.is_file():
            manifests.append(json.loads(sidecar.read_text(encoding="utf-8")))
    return table, manifests
