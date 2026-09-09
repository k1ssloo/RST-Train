"""BUG-29: opt-in, small single-rank CUDA/FSDP2 memory and gradient regression.

RST_RUN_CUDA_TESTS=1 python tests/run_tests.py test_dense_sft_cuda
Run in a separate process with an allocated GPU. This is not a full-model test.
"""

from __future__ import annotations

import gc
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import need, skip


def test_tiny_cuda_fsdp2_chunked_head_matches_native_and_reduces_memory():
    if os.environ.get("RST_RUN_CUDA_TESTS") != "1":
        skip("set RST_RUN_CUDA_TESTS=1 on an allocated GPU for the bounded CUDA/FSDP2 probe")
    torch = need("torch")
    hf = need("transformers")
    need("verl")
    if not torch.cuda.is_available():
        skip("CUDA unavailable")
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from verl.models.transformers.monkey_patch import patch_forward_with_backends

    if torch.cuda.mem_get_info()[0] < 4 * (1 << 30):
        skip("probe requires at least 4 GiB free; no workloads are stopped")
    # Bound this process even if a regression accidentally creates huge logits.
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 3 * (1 << 30) / total_memory))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    reports = []
    with tempfile.TemporaryDirectory() as directory:
        torch.distributed.init_process_group(
            "nccl", init_method=(Path(directory) / "rendezvous").as_uri(), rank=0, world_size=1,
            device_id=torch.device("cuda:0"),
        )
        try:
            for model_type, param_dtype in itertools.product(
                ("llama", "phi3"), (torch.float32, torch.bfloat16)
            ):
                runs = []
                for chunked in (False, True):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.manual_seed(29)
                    kwargs = dict(vocab_size=8192, hidden_size=64, intermediate_size=128,
                                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                  max_position_embeddings=4096, bos_token_id=1, eos_token_id=2,
                                  pad_token_id=2, tie_word_embeddings=True, attention_dropout=0.0)
                    if model_type == "phi3":
                        kwargs.update(original_max_position_embeddings=4096,
                                      resid_pdrop=0.0, embd_pdrop=0.0)
                    config = hf.AutoConfig.for_model(model_type, **kwargs)
                    config._attn_implementation = "sdpa"
                    model = hf.AutoModelForCausalLM.from_config(config).cuda().train()
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    model_class = type(model)
                    native_forward = model_class.forward
                    if chunked:
                        patch_forward_with_backends(model, use_fused_kernels=True, fused_kernels_backend="torch")
                    policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
                    for layer in model.model.layers:
                        fully_shard(layer, mp_policy=policy)
                    fully_shard(model, mp_policy=policy)
                    ids = torch.randint(3, 8192, (1, 2049), device="cuda")
                    ids[0, 0], ids[0, -1] = 1, 2
                    mask = torch.ones_like(ids)
                    mask[:, :5], mask[:, 100:200] = 0, 0
                    labels = ids.clone()
                    labels[mask == 0] = -100
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    try:
                        if chunked:
                            output = model(input_ids=ids, use_cache=False, return_dict=True)
                            assert output.logits is None
                            loss = -(output.log_probs * torch.roll(mask, -1, -1)).sum() / mask.sum()
                        else:
                            output = model(input_ids=ids, labels=labels, use_cache=False)
                            loss = output.loss
                        loss.backward()
                        torch.cuda.synchronize()
                        peak = torch.cuda.max_memory_allocated()
                        grads = {name: p.grad.full_tensor().cpu().clone()
                                 for name, p in model.named_parameters() if p.grad is not None}
                        assert grads and all(torch.isfinite(g).all() for g in grads.values())
                        runs.append({"loss": loss.item(), "peak_allocated_bytes": peak, "grads": grads})
                        torch.optim.SGD(model.parameters(), lr=1e-3).step()
                    finally:
                        model_class.forward = native_forward
                    del loss, output, model, ids, labels, mask, layer
                native, fused = runs
                assert native["grads"].keys() == fused["grads"].keys()
                max_error = 0.0
                rtol = 2e-4 if param_dtype == torch.float32 else 0.02
                for name in native["grads"]:
                    torch.testing.assert_close(fused["grads"][name], native["grads"][name],
                                               rtol=rtol, atol=2e-5, msg=name)
                    max_error = max(max_error, (fused["grads"][name] - native["grads"][name]).abs().max().item())
                assert abs(fused["loss"] - native["loss"]) < 2e-5
                assert fused["peak_allocated_bytes"] < native["peak_allocated_bytes"], runs
                reports.append({"model_type": model_type, "native_loss": native["loss"],
                                "parameter_dtype": str(param_dtype),
                                "chunked_loss": fused["loss"], "max_grad_abs_error": max_error,
                                "native_peak_bytes": native["peak_allocated_bytes"],
                                "chunked_peak_bytes": fused["peak_allocated_bytes"]})
        finally:
            torch.distributed.destroy_process_group()
    report = {"scope": "tiny 2-layer fp32/bf16, tied embeddings, gradient checkpointing, single-rank FSDP2",
              "seq_len": 2049, "vocab_size": 8192, "gpu": torch.cuda.get_device_name(0), "models": reports}
    print(json.dumps(report, sort_keys=True))
    if os.environ.get("RST_DENSE_GPU_REPORT"):
        Path(os.environ["RST_DENSE_GPU_REPORT"]).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
