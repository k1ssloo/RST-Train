"""BUG-29: bounded full-context SFT must preserve loss, masks and gradients."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import ROOT, load_repo_module, need

DENSE = load_repo_module("verl_backend.dense_sft")


def config():
    return {
        "data": {"pad_mode": "no_padding", "use_dynamic_bsz": False,
                 "micro_batch_size_per_gpu": 1, "max_length": 32768, "train_batch_size": 128},
        "model": {"use_remove_padding": False, "use_liger": False,
                  "enable_gradient_checkpointing": True, "use_fused_kernels": True,
                  "fused_kernel_options": {"impl_backend": "torch"}},
        "engine": {"ulysses_sequence_parallel_size": 1},
    }


def test_final_config_rejects_stale_overrides_that_reintroduce_oom_or_invalid_layout():
    good = config()
    for profile in ("llama3", "phi3"):
        DENSE.validate_config(good, profile)
        for section, key, bad in (
            ("model", "use_fused_kernels", False),
            ("model", "use_remove_padding", True),
            ("model", "enable_gradient_checkpointing", False),
            ("data", "use_dynamic_bsz", True),
            ("data", "micro_batch_size_per_gpu", 4),
            ("data", "pad_mode", "padding"),
            ("engine", "ulysses_sequence_parallel_size", 2),
            ("model", "fused_kernel_options", {"impl_backend": "triton"}),
        ):
            cfg = copy.deepcopy(good)
            cfg[section][key] = bad
            try:
                DENSE.validate_config(cfg, profile)
            except ValueError as exc:
                assert f"{section}.{key}" in str(exc), str(exc)
            else:
                raise AssertionError(f"accepted stale {section}.{key}={bad}")


def test_other_models_keep_their_validated_head_and_qwen_layout_is_unchanged():
    other = config()
    other["model"]["use_fused_kernels"] = False
    for profile in ("gemma4", "smollm3", "olmo3"):
        DENSE.validate_config(other, profile)
    DENSE.validate_config(other, "llama3", chunked_head=False)
    DENSE.validate_config({}, "qwen3_5")


def test_untied_checkpoint_keeps_native_head_for_fsdp_module_hooks():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        for model_type in ("llama", "phi3", "gemma4", "qwen3_5"):
            for tied in (True, False):
                path.write_text(json.dumps({"model_type": model_type, "tie_word_embeddings": tied}))
                assert DENSE.supports_chunked_head(directory) == (tied and model_type in ("llama", "phi3"))


def test_launcher_selects_single_sequence_microbatches_and_rejects_old_fused_env():
    source = (ROOT / "scripts/30_run_sft_verl.sh").read_text()
    block = source[source.index("SFT_LAYOUT_ARGS=("):source.index("# ---- multi-node rendezvous gate")]
    for profile in ("llama3", "phi3", "gemma4", "qwen3_5"):
        env = dict(os.environ, SFT_GENERIC=str(int(profile != "qwen3_5")),
                   LOSS_MASK_TYPE=profile, MODEL_KEY=profile)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        env.pop("FUSED_KERNELS", None)
        command = block + '\nprintf "%s\\n" "${SFT_LAYOUT_ARGS[@]}" "${SFT_BATCH_ARGS[@]}" "${FUSED_KERNELS:-qwen-default}"\n'
        with tempfile.TemporaryDirectory() as directory:
            env["MODEL_PATH"] = directory
            model_type = {"llama3": "llama", "phi3": "phi3"}.get(profile, profile)
            (Path(directory) / "config.json").write_text(json.dumps({"model_type": model_type,
                                                                    "tie_word_embeddings": True}))
            result = subprocess.run(["bash", "-eu", "-c", command], env=env, cwd=ROOT,
                                    text=True, capture_output=True)
            if profile in ("llama3", "phi3"):
                env["FUSED_KERNELS"] = "0"
                bad = subprocess.run(["bash", "-eu", "-c", block], env=env, cwd=ROOT,
                                     text=True, capture_output=True)
                assert bad.returncode != 0 and "FUSED_KERNELS=1" in bad.stderr
        assert result.returncode == 0, result.stderr
        options = result.stdout.splitlines()
        assert "data.pad_mode=no_padding" in options
        if profile == "qwen3_5":
            assert "data.use_dynamic_bsz=True" in options
        else:
            assert "model.use_remove_padding=False" in options
            assert "data.micro_batch_size_per_gpu=1" in options
            assert "data.use_dynamic_bsz=False" in options
            assert options[-1] == ("1" if profile in ("llama3", "phi3") else "0")


def test_real_verl_llama_layout_loss_and_gradients():
    need("verl")
    report = DENSE.check_backend("llama")
    assert report["model_input_shapes"] == [[1, 529], [1, 13]]
    assert not report["full_logits_materialized"]


def test_real_verl_phi_layout_loss_and_gradients():
    need("verl")
    report = DENSE.check_backend("phi3")
    assert report["model_input_shapes"] == [[1, 529], [1, 13]]
    assert not report["full_logits_materialized"]


def test_chunked_head_does_not_retain_sequence_by_vocabulary_activations():
    need("verl")
    torch = need("torch")
    backend = need("verl.utils.experimental.torch_functional")
    torch.set_num_threads(1)
    torch.manual_seed(29)
    hidden = torch.randn(1031, 32, requires_grad=True)
    weight = torch.randn(257, 32, requires_grad=True)
    labels = torch.randint(0, 257, (1031,))
    scales = torch.rand(1031)
    expected = (torch.log_softmax(hidden @ weight.T, -1)
                .gather(-1, labels[:, None]).squeeze(-1) * scales).sum()
    expected.backward()
    hidden_grad, weight_grad = hidden.grad.clone(), weight.grad.clone()
    hidden.grad, weight.grad = None, None
    saved = []

    def record(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with patch.object(backend, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False):
        with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
            log_probs, _ = backend.FusedLinearForPPO(chunk_size=127)(hidden, weight, labels)
            actual = (log_probs * scales).sum()
        actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hidden.grad, hidden_grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(weight.grad, weight_grad, rtol=2e-4, atol=2e-5)
    assert (1031, 257) not in saved, saved
    assert all(shape[-1] != 257 for shape in saved if len(shape) > 1), saved


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
