"""BUG-30: FSDP must preserve OLMo3's FP32 rotary inputs."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import load_repo_module, load_script, need

COMMON = load_repo_module("rst_common.model_precision")
BACKEND = load_repo_module("verl_backend.model_precision")


def test_only_olmo_input_cast_changes_and_other_precision_fields_are_retained():
    torch = need("torch")
    from torch.distributed.fsdp import MixedPrecisionPolicy

    original = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                    output_dtype=torch.bfloat16)
    for family in ("smollm3", "gemma4", "llama", "qwen3_5", "phi3"):
        model = SimpleNamespace(config=SimpleNamespace(model_type=family))
        assert COMMON.preserve_rope_precision(model, original) is original
    model = SimpleNamespace(config=SimpleNamespace(model_type="olmo3"))
    corrected = COMMON.preserve_rope_precision(model, original)
    assert original.cast_forward_inputs and not corrected.cast_forward_inputs
    assert corrected.param_dtype == original.param_dtype
    assert corrected.reduce_dtype == original.reduce_dtype
    assert corrected.output_dtype == original.output_dtype
    assert COMMON.preserve_rope_precision(model, corrected) is corrected


def test_verl_wrapper_preserves_caller_kwargs_and_wraps_only_once():
    torch = need("torch")
    from torch.distributed.fsdp import MixedPrecisionPolicy

    calls = []

    def original(model, kwargs, config):
        calls.append((model, kwargs, config))
        return "wrapped"

    wrapper = BACKEND.wrap_apply_fsdp2(original)
    assert BACKEND.wrap_apply_fsdp2(wrapper) is wrapper
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    kwargs = {"mp_policy": policy, "mesh": object(), "reshard_after_forward": True}
    model = SimpleNamespace(config=SimpleNamespace(model_type="olmo3"))
    config = {"wrap_policy": {}}
    assert wrapper(model, kwargs, config) == "wrapped"
    _, passed, received = calls[0]
    assert kwargs["mp_policy"] is policy and policy.cast_forward_inputs
    assert not passed["mp_policy"].cast_forward_inputs and received is config
    assert passed["mesh"] is kwargs["mesh"] and passed["reshard_after_forward"]


def test_dpo_applies_olmo_precision_to_every_shard_group():
    torch = need("torch")
    from test_multimodel_training import tiny_model

    train = load_script("19_train_dpo")
    model = tiny_model("Olmo3")
    calls = []
    with patch("torch.distributed.fsdp.fully_shard", side_effect=lambda m, **kw: calls.append(kw)):
        train.shard_model(model, param_dtype=torch.bfloat16, world_size=2)
    assert len(calls) >= 4
    assert all(not call["mp_policy"].cast_forward_inputs for call in calls)


def test_real_verl_installs_policy_at_both_import_bindings():
    need("verl")
    from verl.utils import fsdp_utils
    from verl.workers.engine.fsdp import transformer_impl

    assert BACKEND.apply()
    first = transformer_impl.apply_fsdp2
    assert first is fsdp_utils.apply_fsdp2
    assert getattr(fsdp_utils.apply_fsdp2, "_rst_rope_precision", False)
    assert getattr(first, "_rst_rope_precision", False)
    assert BACKEND.apply() and transformer_impl.apply_fsdp2 is first


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
