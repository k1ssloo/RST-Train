"""The paper's published numbers, in one place.

Both `scripts/06_eval.py` (which stamps them into results.json for provenance)
and `scripts/14_make_report.py` (which compares against them) used to carry their
own copy of this table. Two copies of a reference value is one copy too many:
whichever one somebody corrects, the other silently becomes the number a
conclusion gets drawn from.

Source: *Recursive Synthesis for Long-Horizon Terminal Tasks*, Tables 3-4,
pass rate %. LHTB is kept for reference only -- its verifiers are withheld
upstream, so this harness cannot score it (see `06_eval.UNSCORABLE`).
"""

from __future__ import annotations

PAPER = {
    "base":   {"tb2": 41.20, "tb-hard": 22.67, "lhtb": 18.10},
    "sft_r1": {"tb2": 42.32, "tb-hard": 23.00, "lhtb": 21.32},
    "sft_r3": {"tb2": 47.94, "tb-hard": 28.33, "lhtb": 22.44},
    "rl":     {"tb2": 49.44, "tb-hard": 32.00, "lhtb": 22.07},
}

# The released SFT checkpoint should land near here. A big miss indicts the
# harness, not the checkpoint -- see check_eval() in 14_make_report.py.
REF_TARGET = {"tb2": 47.94, "tb-hard": 28.33}

# The paper only published numbers for Qwen3.5-27B and 122B-A10B. For any other
# model in configs/models.json there is NO reference point, so "regression vs
# base" and "does the reference checkpoint reproduce the paper" do not apply --
# asserting them anyway would manufacture a finding out of nothing.
PAPER_MODELS = {"qwen3.5-27b"}


def has_paper_reference(model_key: str | None) -> bool:
    return (model_key or "qwen3.5-27b") in PAPER_MODELS
