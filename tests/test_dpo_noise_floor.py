"""Whether a finished DPO run says its measured preference was arithmetic.

Both runs that reached the end reported an improvement and no warnings:

    4B  holdout_reward_accuracy 0.5 -> 0.5938   holdout_reward_margin 6e-05
    9B  holdout_reward_accuracy 0.5 -> 0.5391   holdout_reward_margin 1e-05
        holdout_ties 64 -> 1     clip_active_fraction 0.0     warnings []

The accuracy is a sign test on that margin (`rank_score` in `19_train_dpo.py` credits a
pair above `TIE_EPS = 1e-6`), while the reference logprobs it is differenced against were
scored in bf16 -- good to ~1e-3 nats. At beta 0.1 a reward margin under 1e-4 is therefore
below the arithmetic floor of the quantity it comes from, and three decades below the
1e-2..1 the trainer's own comment calls trained. The accuracy is the number that gets
quoted; the magnitude that discounts it is a different field of the same file.

So the warning is the deliverable, and these tests pin it to the observed numbers: it
fires on both runs, stays quiet for a margin that is genuinely trained, and scales with
beta rather than hard-coding a threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dpo_common import REF_LOGP_NOISE_NATS, noise_floor_warning  # noqa: E402

# The 9B run, verbatim from dpo/dpo_training_summary.json on the Hub.
NINE_B_BEFORE = {"holdout_loss": 0.69315, "holdout_pairs": 64,
                 "holdout_reward_accuracy": 0.5, "holdout_reward_margin": -0.0,
                 "holdout_ties": 64}
NINE_B_AFTER = {"holdout_loss": 0.69314, "holdout_pairs": 64,
                "holdout_reward_accuracy": 0.5391, "holdout_reward_margin": 1e-05,
                "holdout_ties": 1}
NINE_B_LAST = {"step": 75, "loss": 0.69311, "accuracy": 0.6875, "grad_norm": 0.0766,
               "reward_chosen": -6e-05, "reward_margin": 7e-05,
               "reward_rejected": -0.00013}


def test_the_9b_run_is_reported_as_a_training_no_op():
    out = noise_floor_warning(beta=0.1, baseline_eval=NINE_B_BEFORE,
                              final_eval=NINE_B_AFTER, last_step=NINE_B_LAST)
    assert out is not None
    assert "NOISE FLOOR" in out, out
    # Both numbers, next to each other, because the pair is the argument.
    assert "0.5391" in out, out
    assert "1e-05" in out, out
    # The tie collapse is the same fact from the other side, so quote it.
    assert "64 -> 1" in out, out
    # And what to do with the checkpoint, which is the operator's actual question.
    assert "near-indistinguishable" in out, out
    assert "--beta" in out and "--lr" in out, out


def test_the_4b_run_fires_too_at_six_times_the_margin():
    # 6e-05 is six times the 9B's and still under beta x 1e-3, which is the point: the
    # floor is not a hair's breadth away from what was observed.
    out = noise_floor_warning(
        beta=0.1,
        baseline_eval={"holdout_reward_accuracy": 0.5, "holdout_reward_margin": 0.0},
        final_eval={"holdout_reward_accuracy": 0.5938, "holdout_reward_margin": 6e-05},
    )
    assert out is not None and "0.5938" in out, out


def test_a_genuinely_trained_margin_says_nothing():
    # 1e-2 is the low end of what 19_train_dpo.py's TIE_EPS comment calls trained.
    assert noise_floor_warning(
        beta=0.1,
        baseline_eval={"holdout_reward_accuracy": 0.5, "holdout_reward_margin": 0.0},
        final_eval={"holdout_reward_accuracy": 0.71, "holdout_reward_margin": 0.012},
    ) is None
    # A negative margin of real size is a different problem (the objective ran backwards)
    # and must not be dressed up as noise.
    assert noise_floor_warning(
        beta=0.1, baseline_eval=None,
        final_eval={"holdout_reward_accuracy": 0.31, "holdout_reward_margin": -0.05},
    ) is None


def test_the_floor_scales_with_beta_because_the_reward_does():
    # reward = beta * (logp_pi - logp_ref), so the same logprob difference is a smaller
    # reward at a smaller beta. A fixed threshold would misjudge every beta but one.
    margin = 5e-05
    assert noise_floor_warning(
        beta=0.1, baseline_eval=None,
        final_eval={"holdout_reward_accuracy": 0.55, "holdout_reward_margin": margin},
    ) is not None, "5e-05 < 0.1 x 1e-3, so it is noise at beta 0.1"
    assert noise_floor_warning(
        beta=0.01, baseline_eval=None,
        final_eval={"holdout_reward_accuracy": 0.55, "holdout_reward_margin": margin},
    ) is None, "5e-05 > 0.01 x 1e-3, so the same margin is real signal at beta 0.01"


def test_the_floor_is_the_bf16_logprob_noise_and_not_a_round_number():
    # Named and reused, so the reason it is 1e-3 stays attached to the value.
    assert REF_LOGP_NOISE_NATS == 1e-3
    out = noise_floor_warning(
        beta=0.1, baseline_eval=None,
        final_eval={"holdout_reward_accuracy": 0.55, "holdout_reward_margin": 1e-05},
    )
    assert "bf16" in out, out
    assert "1e-04" in out, out  # beta x 1e-3, formatted


def test_a_run_with_no_holdout_eval_is_not_accused_of_anything():
    # --eval-pairs 0 produces no holdout dict at all. Absence of a measurement is not a
    # measurement of zero, and warning here would be a false positive on every such run.
    assert noise_floor_warning(beta=0.1, baseline_eval=None, final_eval=None) is None
    assert noise_floor_warning(
        beta=0.1, baseline_eval=None, final_eval={"holdout_pairs": 0}
    ) is None


def test_the_trainer_appends_this_to_the_summary_warnings():
    src = (Path(__file__).resolve().parent.parent / "scripts" / "19_train_dpo.py") \
        .read_text(encoding="utf-8")
    # In runtime_warnings, which is both printed and written to
    # dpo_training_summary.json -- an empty `warnings` list was how the 9B run passed.
    assert "noise_floor_warning" in src, src
    assert "runtime_warnings.append(at_noise)" in src, src
    appended = src.index("runtime_warnings.append(at_noise)")
    assert appended < src.index('"warnings": runtime_warnings'), (
        "the warning has to be appended before the summary dict is built, or it is "
        "printed but not archived"
    )
