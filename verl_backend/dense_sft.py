"""BUG-29: bounded Llama/Phi SFT heads and one full sequence per micro-batch.

The implementation is verl's Torch chunked head, not a new loss. This module
checks the final launch configuration and the installed backend's numerical
contract on tiny CPU models before any production weights/GPU workers load.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHUNKED_PROFILES = {"llama3": "llama", "phi3": "phi3"}


def supports_chunked_head(model_path: str | Path) -> bool:
    config = json.loads((Path(model_path) / "config.json").read_text())
    # verl can shard an untied lm_head separately. dense_common reads .weight
    # directly, bypassing that module's FSDP all-gather hooks. Keep the native
    # head for those checkpoints until that path has its own tested adapter.
    return (config.get("model_type") in CHUNKED_PROFILES.values()
            and config.get("tie_word_embeddings") is True)


def validate_config(config, profile: str, *, chunked_head: bool | None = None) -> None:
    """Check resolved Hydra values, including caller overrides, without torch."""
    if profile == "qwen3_5":
        return
    if chunked_head is None:
        chunked_head = profile in CHUNKED_PROFILES
    required = {
        "data.pad_mode": "no_padding",
        "data.use_dynamic_bsz": False,
        "data.micro_batch_size_per_gpu": 1,
        "model.use_remove_padding": False,
        "model.use_liger": False,
        "model.enable_gradient_checkpointing": True,
        "engine.ulysses_sequence_parallel_size": 1,
        "model.use_fused_kernels": chunked_head,
    }
    if chunked_head:
        required["model.fused_kernel_options.impl_backend"] = "torch"
    problems = []
    for dotted, expected in required.items():
        current = config
        for key in dotted.split("."):
            current = current.get(key) if hasattr(current, "get") else None
        if current != expected:
            problems.append(f"{dotted}={current!r}; required {expected!r}")
    if problems:
        raise ValueError("BUG-29: incompatible dense SFT overrides:\n  " + "\n  ".join(problems))


def check_backend(model_type: str) -> dict:
    """Exercise actual verl loading/dispatch, layouts, masking and gradients on CPU.

    This does not construct distributed FSDP or claim production GPU acceptance.
    The long tiny row crosses the upstream head's default 512-token chunk boundary.
    """
    if model_type not in CHUNKED_PROFILES.values():
        raise ValueError(f"no validated chunked SFT profile for {model_type!r}")
    import torch
    import transformers as hf
    from verl.models.transformers.monkey_patch import patch_forward_with_backends
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import SFTTensorCollator
    from verl.utils.experimental import torch_functional as chunked
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from verl.workers.engine.utils import prepare_micro_batches
    from verl.workers.utils.losses import sft_loss

    torch.set_num_threads(1)
    torch.manual_seed(29)
    kwargs = dict(vocab_size=97, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=1024, bos_token_id=1, eos_token_id=2,
                  pad_token_id=2, tie_word_embeddings=True, attention_dropout=0.0)
    if model_type == "phi3":
        kwargs.update(original_max_position_embeddings=1024, resid_pdrop=0.0, embd_pdrop=0.0)
    model_config = hf.AutoConfig.for_model(model_type, **kwargs)
    model_config._attn_implementation = "eager"
    model = hf.AutoModelForCausalLM.from_config(model_config).eval()
    samples = []
    for length in (529, 13):
        ids = torch.randint(3, kwargs["vocab_size"], (length,))
        ids[0], ids[-1] = 1, 2
        mask = torch.ones(length, dtype=torch.long)
        mask[:3], mask[5:8] = 0, 0  # prompt/observation and a masked assistant span
        samples.append(dict(input_ids=ids, loss_mask=mask, position_ids=torch.arange(length)))
    ids = torch.nn.utils.rnn.pad_sequence([s["input_ids"] for s in samples],
                                         batch_first=True, padding_value=2)
    masks = torch.nn.utils.rnn.pad_sequence([s["loss_mask"] for s in samples], batch_first=True)
    attention = torch.arange(ids.shape[1])[None, :] < torch.tensor([529, 13])[:, None]
    labels = ids.clone()
    labels[masks == 0] = -100
    baseline = model(input_ids=ids, attention_mask=attention, labels=labels, use_cache=False).loss
    baseline.backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    batch = tu.get_tensordict(SFTTensorCollator("no_padding")(samples), non_tensor_dict={
        "use_dynamic_bsz": False, "micro_batch_size_per_gpu": 1,
        # Deliberately below the longest row: ignored for static micro-batches.
        "max_token_len_per_gpu": 8, "use_remove_padding": False,
        "use_fused_kernels": True, "pad_mode": "no_padding", "pad_token_id": 2,
        "temperature": 1.0, "dp_size": 1, "batch_num_tokens": masks.sum().item(),
    })
    micro_batches, _ = prepare_micro_batches(batch)
    assert len(micro_batches) == 2 and all(len(b) == 1 for b in micro_batches)
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.pad_to_length = False
    engine.use_ulysses_sp = False
    native_forward = type(model).forward
    # Optional flash-attn CE only accepts CUDA tensors. Use the real helper's
    # PyTorch fallback for this CPU gate and restore the choice afterwards.
    flash_ce = chunked._FLASH_ATTN_CROSS_ENTROPY_AVAILABLE
    chunked._FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = False
    total = 0.0
    shapes = []
    try:
        patch_forward_with_backends(model, use_fused_kernels=True, fused_kernels_backend="torch")
        for micro in micro_batches:
            inputs, args = engine.prepare_model_inputs(micro)
            shapes.append(list(inputs["input_ids"].shape))
            output = model(**inputs, use_cache=False)
            assert output.logits is None, "chunked path materialized full logits"
            prepared = engine.prepare_model_outputs(output, args, micro, logits_processor_func=None)
            loss, _ = sft_loss(None, prepared, micro)
            total += loss.item()
            loss.backward()
    finally:
        type(model).forward = native_forward
        chunked._FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = flash_ce
    torch.testing.assert_close(torch.tensor(total), baseline.detach(), rtol=2e-5, atol=2e-5)
    max_grad_error = 0.0
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name], rtol=2e-4, atol=2e-5, msg=name)
            max_grad_error = max(max_grad_error, (parameter.grad - expected[name]).abs().max().item())
    return {"model_type": model_type, "native_loss": baseline.item(), "chunked_loss": total,
            "max_grad_abs_error": max_grad_error, "model_input_shapes": shapes,
            "full_logits_materialized": False, "scope": "CPU numerical/layout check; no FSDP/GPU run"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="read config only; no full weights loaded")
    args = parser.parse_args()
    if not supports_chunked_head(args.model):
        parser.error("the verified dense head requires Llama/Phi with tied embeddings")
    model_type = json.loads((args.model / "config.json").read_text())["model_type"]
    print("[rst-dense-sft] " + json.dumps(check_backend(model_type), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
