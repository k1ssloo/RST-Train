"""`19_train_dpo.shard_model` must shard the modules the DPO forward actually CALLS.

THE BUG THIS LOCKS OUT
----------------------
`dpo_common.masked_logprob_sum` never calls `model(...)`. It calls

    model.get_decoder()(input_ids=...)          # hidden states
    model.get_output_embeddings()(states)        # LM head, one --logit-chunk slice at a time

on purpose: one `model(input_ids)` would materialize seq x 248,320 logits and OOM before
computing anything. FSDP2, though, registers its unshard hook on whichever module is handed
to `fully_shard`. `shard_model` used to do

    for layer in decoder.layers: fully_shard(layer)
    fully_shard(model)                           # <-- the root

so the root group -- `embed_tokens`, the final norm, `lm_head`, the ViT -- was sharded and
then never unsharded, because `model.__call__` never runs. Measured on one H100 at
world_size 1 (fully_shard still wraps params as DTensor there, so the layout is testable
without a second GPU):

    RuntimeError: aten.embedding.default got mixed torch.Tensor and DTensor, need to
    convert all torch.Tensor to DTensor before calling distributed operators!

It fires in gate 3's calibration forward, before a single gradient, i.e. on every
multi-rank DPO run at once. It survived review because `shard_model` returns early at
world_size == 1 and the only DPO run so far was single-GPU -- so the test below asserts the
LAYOUT rather than the runtime, which is what makes it run on a laptop.

WHY NOT JUST RUN IT
-------------------
There is a real end-to-end check (`fully_shard` + the actual `masked_logprob_sum` +
`clip_grad_norm_` over DTensor params), but it needs CUDA and a process group. This file
records the invariant so a future edit that "simplifies" the sharding back to the root is
caught by `python tests/run_tests.py` on any machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import load_script, need  # noqa: E402

TRAIN = load_script("19_train_dpo")


class _Module:
    """Enough of nn.Module for shard_model: children by attribute, params by name.

    `parameters()` RECURSES into `.layers`, because nn.Module.parameters() does. Getting
    that wrong here would make the test pass for the wrong reason: shard_model decides
    what is "outside decoder+lm_head" from `decoder.parameters()`, so a non-recursing stub
    reports every decoder layer as outside and freezes it, and the assertions below would
    then be describing the stub rather than the code.
    """

    def __init__(self, name: str, params: dict[str, int]):
        self._name = name
        self._params = {key: _Param(size) for key, size in params.items()}
        self.layers: list[_Module] = []

    def parameters(self):
        own = list(self._params.values())
        for child in self.layers:
            own += child.parameters()
        return own

    def named_parameters(self):
        out = [(f"{self._name}.{k}", v) for k, v in self._params.items()]
        for child in self.layers:
            out += child.named_parameters()
        return out


class _Param:
    def __init__(self, numel: int):
        self._numel = numel
        self.requires_grad = True

    def numel(self) -> int:
        return self._numel

    def requires_grad_(self, flag: bool):
        self.requires_grad = flag
        return self


class _Model:
    """A VLM-shaped stand-in: decoder + lm_head + a ViT that is a sibling of both."""

    def __init__(self, *, with_vision: bool = True):
        self.layers = [_Module(f"layer{i}", {"weight": 100}) for i in range(3)]
        self.decoder = _Module("language_model", {"embed_tokens.weight": 10, "norm.weight": 1})
        self.decoder.layers = self.layers
        self.lm_head = _Module("lm_head", {"weight": 20})
        self.visual = _Module("visual", {"weight": 5}) if with_vision else None

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.lm_head

    def named_parameters(self):
        # decoder.named_parameters() already covers the layers, as nn.Module's would.
        out = list(self.decoder.named_parameters())
        out += self.lm_head.named_parameters()
        if self.visual is not None:
            out += self.visual.named_parameters()
        return out


def _run(model, monkeypatched_calls: list):
    """Call the real shard_model with fully_shard replaced by a recorder."""
    torch = need("torch")
    fsdp = need("torch.distributed.fsdp")

    real = fsdp.fully_shard

    def recorder(module, **kwargs):
        monkeypatched_calls.append((module, kwargs))
        return module

    fsdp.fully_shard = recorder
    try:
        return TRAIN.shard_model(model, param_dtype=torch.bfloat16, world_size=2)
    finally:
        fsdp.fully_shard = real


def test_the_two_modules_the_forward_calls_are_both_fsdp_entry_points():
    model = _Model()
    calls: list = []
    _run(model, calls)
    sharded = [module for module, _ in calls]

    assert model.get_decoder() in sharded, (
        "decoder is not an FSDP entry point, so masked_logprob_sum's "
        "decoder(input_ids=...) call will not unshard embed_tokens and dies on "
        "aten.embedding with mixed Tensor/DTensor"
    )
    assert model.get_output_embeddings() in sharded, (
        "lm_head is not an FSDP entry point, so applying it by hand hits a sharded DTensor"
    )
    for layer in model.layers:
        assert layer in sharded, "a decoder layer was left unsharded"


def test_the_root_module_is_not_sharded():
    # The regression itself. Sharding the root is what put embed_tokens/lm_head into a
    # group whose hook never fires.
    model = _Model()
    calls: list = []
    _run(model, calls)
    assert model not in [module for module, _ in calls], (
        "shard_model sharded the root model again. FSDP2 unshards a group from the hook on "
        "the module passed to fully_shard, and the DPO forward never calls model(...), so "
        "the root group's params stay sharded DTensors for the whole run."
    )


def test_the_lm_head_does_not_reshard_between_chunks():
    # masked_logprob_sum applies the head once per --logit-chunk slice. With the default
    # reshard_after_forward that is one all-gather of a [vocab, hidden] matrix per chunk:
    # at 248,320 x 5120 bf16 and a 3k-token side, ~2.4 GiB moved six times per forward.
    model = _Model()
    calls: list = []
    _run(model, calls)
    head_kwargs = [kwargs for module, kwargs in calls if module is model.get_output_embeddings()]
    assert head_kwargs, "lm_head was never sharded"
    assert head_kwargs[0].get("reshard_after_forward") is False, (
        "lm_head must be sharded with reshard_after_forward=False; it is re-entered once "
        "per logit chunk"
    )


def test_params_outside_the_sharded_modules_are_frozen_rather_than_stranded():
    # A sharded-but-never-unsharded param is a crash; an unsharded-and-trainable one is a
    # silent 4 B/param of fp32 master per GPU plus AdamW weight decay on a zero gradient.
    # Freezing is the only option that is neither.
    model = _Model(with_vision=True)
    _run(model, [])
    named = dict(model.named_parameters())
    assert named["visual.weight"].requires_grad is False, (
        "the ViT is outside decoder+lm_head, so DPO can never produce a gradient for it; "
        "leaving requires_grad=True lets AdamW decay it every step"
    )
    # The other half, and the one that would actually ruin a run: everything reachable from
    # the DPO forward must still be trainable. `decoder.parameters()` recurses, so the
    # layers are inside -- a freeze that reached them would train nothing but report a
    # falling loss from the head alone.
    still_trained = ["lm_head.weight", "language_model.embed_tokens.weight",
                     "language_model.norm.weight", "layer0.weight", "layer2.weight"]
    frozen_by_mistake = [k for k in still_trained if not named[k].requires_grad]
    assert not frozen_by_mistake, f"freezing spilled onto trained params: {frozen_by_mistake}"


def test_nothing_is_reported_as_outside_when_the_model_has_no_vision_tower():
    # A text-only checkpoint has no siblings of decoder/lm_head, so the "outside" list must
    # be empty. If it is not, the id()-based membership test in shard_model is not seeing
    # through whatever container the params live in, and it would freeze real weights.
    model = _Model(with_vision=False)
    _run(model, [])
    frozen = [k for k, p in model.named_parameters() if not p.requires_grad]
    assert not frozen, f"a text-only model had params frozen: {frozen}"


def test_a_model_without_an_output_embedding_is_refused_not_silently_wrong():
    model = _Model()
    model.get_output_embeddings = lambda: None
    try:
        _run(model, [])
    except SystemExit as exc:
        assert "get_output_embeddings" in str(exc)
        return
    raise AssertionError("a model whose LM head cannot be located was accepted; the head "
                         "would then be applied to a sharded DTensor mid-run")


def test_single_process_still_skips_sharding_entirely():
    torch = need("torch")
    model = _Model()
    assert TRAIN.shard_model(model, param_dtype=torch.bfloat16, world_size=1) is model
    assert dict(model.named_parameters())["visual.weight"].requires_grad is True, (
        "world_size=1 must not freeze anything: there is no FSDP, and the ViT's state is "
        "whatever the checkpoint had"
    )


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
