"""BUG-30: actual OLMo3 SFT/DPO FSDP2 hooks preserve FP32 rotary tensors."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import load_repo_module, load_script, need, skip


def test_olmo_bf16_fsdp2_rotary_inputs_and_gradients():
    if os.environ.get("RST_RUN_CUDA_TESTS") != "1":
        skip("set RST_RUN_CUDA_TESTS=1 for the bounded OLMo3 CUDA/FSDP2 regression")
    torch = need("torch")
    hf = need("transformers")
    need("verl")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 3 * (1 << 30):
        skip("requires CUDA and 3 GiB free; no workloads are stopped")
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from verl.utils import fsdp_utils
    from verl.workers.engine.fsdp import transformer_impl

    backend = load_repo_module("verl_backend.model_precision")
    backend.apply()
    train, dpo = load_script("19_train_dpo"), load_script("dpo_common")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((1 << 30) / total)

    def model():
        torch.manual_seed(30)
        config = hf.Olmo3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                               max_position_embeddings=128, tie_word_embeddings=False,
                               bos_token_id=1, eos_token_id=2, pad_token_id=0,
                               layer_types=["sliding_attention", "full_attention"],
                               sliding_window=16, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        instance = hf.Olmo3ForCausalLM(config).cuda().train()
        instance.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return instance

    ids = [1] + [3 + (i * 7) % 57 for i in range(59)] + [2]
    mask = [0] * 7 + [1] * 20 + [0] * 8 + [1] * 26
    x = torch.tensor([ids], device="cuda")
    labels = x.clone()
    labels[torch.tensor([mask], device="cuda") == 0] = -100
    with tempfile.TemporaryDirectory() as directory:
        torch.distributed.init_process_group(
            "nccl", init_method=(Path(directory) / "rendezvous").as_uri(), rank=0, world_size=1,
            device_id=torch.device("cuda:0"),
        )
        try:
            baseline = model()
            # FSDP casts parameters, not floating-point rotary buffers.
            for parameter in baseline.parameters():
                parameter.data = parameter.data.to(torch.bfloat16)
            expected_loss = baseline(input_ids=x, labels=labels, use_cache=False).loss
            expected_loss.backward()
            expected = {n: p.grad.float().cpu().clone() for n, p in baseline.named_parameters()
                        if p.grad is not None}
            value = expected_loss.item()
            del baseline, expected_loss
            gc.collect()
            for route in ("sft", "dpo"):
                candidate = model()
                if route == "sft":
                    # Exercise the engine's actual imported binding, not a mock.
                    transformer_impl.apply_fsdp2(candidate, {"mp_policy": MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16, reduce_dtype=torch.float32)}, {})
                else:
                    # Force the grouping branch; the actual process group/mesh
                    # remain size ONE. No multi-rank claim is made by this test.
                    candidate = train.shard_model(candidate, param_dtype=torch.bfloat16, world_size=2)
                observed = []

                def observe(module, args, kwargs):
                    observed.extend(t.dtype for t in kwargs["position_embeddings"])

                handles = [layer.self_attn.register_forward_pre_hook(observe, with_kwargs=True)
                           for layer in candidate.get_decoder().layers]
                if route == "sft":
                    loss = candidate(input_ids=x, labels=labels, use_cache=False).loss
                else:
                    logp, count = dpo.masked_logprob_sum(candidate, ids, mask, chunk=13, grad=True)
                    loss = -logp / count
                loss.backward()
                assert observed and set(observed) == {torch.float32}, (route, observed)
                assert abs(loss.item() - value) < 2e-4
                for name, parameter in candidate.named_parameters():
                    if name in expected:
                        torch.testing.assert_close(parameter.grad.full_tensor().float().cpu(), expected[name],
                                                   rtol=0.03, atol=3e-4, msg=f"{route}: {name}")
                for handle in handles:
                    handle.remove()
                del candidate, loss
                if route == "dpo":
                    del logp
                gc.collect()
                torch.cuda.empty_cache()
            assert getattr(fsdp_utils.apply_fsdp2, "_rst_rope_precision", False)
        finally:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
