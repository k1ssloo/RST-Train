"""Install BUG-30's OLMo3 policy before verl constructs its FSDP2 modules."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from rst_common.model_precision import preserve_rope_precision


def wrap_apply_fsdp2(original: Callable) -> Callable:
    if getattr(original, "_rst_rope_precision", False):
        return original

    @wraps(original)
    def wrapped(model: Any, fsdp_kwargs: dict[str, Any], config: Any) -> Any:
        policy = fsdp_kwargs["mp_policy"]
        corrected = preserve_rope_precision(model, policy)
        if corrected is not policy:
            fsdp_kwargs = {**fsdp_kwargs, "mp_policy": corrected}
            print("[rst-fsdp2] OLMo3: preserve FP32 rotary inputs; parameter/reduction dtypes retained",
                  flush=True)
        return original(model, fsdp_kwargs, config)

    wrapped._rst_rope_precision = True
    return wrapped


def apply() -> bool:
    try:
        from verl.utils import fsdp_utils
        from verl.workers.engine.fsdp import transformer_impl
    except ImportError:
        return False  # CPU data tools can be used without verl installed.
    # The engine imported this function by name; update both bindings.
    original = fsdp_utils.apply_fsdp2
    fsdp_utils.apply_fsdp2 = wrap_apply_fsdp2(original)
    if transformer_impl.apply_fsdp2 is original:
        transformer_impl.apply_fsdp2 = fsdp_utils.apply_fsdp2
    else:
        transformer_impl.apply_fsdp2 = wrap_apply_fsdp2(transformer_impl.apply_fsdp2)
    return True
