"""Architecture-specific FSDP2 precision contracts (BUG-30)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.distributed.fsdp import MixedPrecisionPolicy


def preserve_rope_precision(model: Any, policy: MixedPrecisionPolicy) -> MixedPrecisionPolicy:
    """OLMo3 deliberately passes FP32 cos/sin into BF16 decoder layers.

    FSDP2's default input cast otherwise rounds those tensors before rotary
    attention, changing both forward values and gradients. Text hidden states
    already have the parameter dtype; preserve the model's auxiliary tensors.
    Parameter/reduction/output dtypes and every other model's policy are retained.
    """
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "olmo3" and policy.cast_forward_inputs:
        return replace(policy, cast_forward_inputs=False)
    return policy
