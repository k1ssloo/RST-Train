"""`scripts/10c_build_swegym_taskset.py` -- the SWE-Gym RL pool.

WHY THIS IS A TASKSET BUILDER AND NOT AN SFT CONVERTER
    `SWE-Gym/SWE-Gym` is SWE-bench format -- `instance_id`, `problem_statement`,
    `patch`, `test_patch`, `FAIL_TO_PASS` -- with no `messages` or `conversations`
    column at all. Fourth dataset handed to this repo that looks like SFT data and is
    not, so the builder asserts the absence rather than assuming it.

    Its trajectories live in `SWE-Gym/OpenHands-SFT-Trajectories`, which is exactly the
    491 `resolved` rollouts out of `OpenHands-Sampled-Trajectories`' 6,055. They are
    not converted, because they are in the OpenHands/CodeAct action space and
    rewriting `str_replace_editor` as terminus-2 `{analysis, plan, commands}` would
    mean inventing the analysis and plan text.

WHAT THE TESTS ARE FOR
    Two things that would be invisible if wrong.

    **The tier arithmetic.** 88.0 % of this pool is `hard` (0-10 % pass rate), i.e.
    zero-gradient at full sandbox cost. If the banding is off by an epsilon at a
    boundary, a pool advertised as `sweet` silently contains all-fail groups and the
    only symptom is a GRPO run that costs a lot and learns nothing.

    **The two exclusions.** All 2,438 instances ship `patch`, the gold solution, and
    1,528 ship `hints_text` that frequently names the fix. Both are kept out of the
    output. A leak here is unrecoverable once published, and it invalidates every
    evaluation anyone runs on the pool afterwards -- silently, and in the direction
    that looks like success.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, load_script, need  # noqa: E402

builder = load_script("10c_build_swegym_taskset")


# ------------------------------------------------------------------ as_list


def test_a_multi_element_repeated_field_is_coerced_without_truthiness():
    """The bug this helper exists for.

    `value or []` raises `ValueError: truth value of an array ... is ambiguous` for a
    numpy array of length > 1. It passed on `FAIL_TO_PASS`, which is usually one
    element, and failed on `PASS_TO_PASS`, which usually is not -- so the naive version
    worked on the first rows and died partway through the pool.
    """
    # `need()` and not a bare import: without numpy this has to say "skipped" out loud,
    # or an environment gap is reported as a code failure.
    np = need("numpy")
    assert builder.as_list(np.array(["a", "b", "c"])) == ["a", "b", "c"]
    assert builder.as_list(np.array(["only"])) == ["only"]
    assert builder.as_list(np.array([], dtype=object)) == []


def test_none_and_empty_become_the_empty_list():
    assert builder.as_list(None) == []
    assert builder.as_list([]) == []


def test_elements_are_stringified():
    assert builder.as_list([1, 2]) == ["1", "2"]


# ------------------------------------------------------------------ tiers


def test_the_bands_match_the_repos_own_rl_taskset_builder():
    # Divergent bands between the two pools would make "sweet" mean two different
    # things in one project.
    other = load_script("10_build_rl_taskset")
    assert builder.TIERS == other.TIERS


def test_every_representable_pass_rate_lands_in_exactly_one_tier():
    for step in range(0, 101):
        rate = step / 100
        matches = [name for name, low, high in builder.TIERS if low <= rate < high]
        assert len(matches) == 1, f"pass rate {rate} matched {matches}"


def test_the_band_boundaries_go_the_way_the_comments_claim():
    assert builder.tier_of(0.0) == "hard"        # all-fail: zero gradient
    assert builder.tier_of(0.09) == "hard"
    assert builder.tier_of(0.10) == "sweet"      # 0.10 is IN sweet, not hard
    assert builder.tier_of(0.5) == "sweet"
    assert builder.tier_of(0.89) == "sweet"
    assert builder.tier_of(0.90) == "easy"       # 0.90 is IN easy, not sweet
    assert builder.tier_of(1.0) == "easy"        # all-pass must not fall off the end


def test_a_pass_rate_outside_zero_to_one_is_an_error_not_a_silent_tier():
    for bad in (-0.01, 1.01, 2.0):
        try:
            builder.tier_of(bad)
        except AssertionError:
            continue
        raise AssertionError(f"pass rate {bad} was silently assigned a tier")


# ------------------------------------------------------------------ exclusions


def test_the_gold_patch_is_never_written_to_the_output():
    source = (ROOT / "scripts" / "10c_build_swegym_taskset.py").read_text(encoding="utf-8")
    record = source[source.index('records.append('):source.index('stats["selected"]')]
    assert "row.patch" not in record, "the reference solution reached the task record"
    assert '"gold_patch_excluded": True' in record


def test_hints_text_is_recorded_as_a_flag_and_never_as_text():
    source = (ROOT / "scripts" / "10c_build_swegym_taskset.py").read_text(encoding="utf-8")
    record = source[source.index('records.append('):source.index('stats["selected"]')]
    # The boolean is fine and useful; the text is what must not travel.
    assert '"hints_text_available_upstream": bool(' in record
    assert '"hints_text": ' not in record


def test_the_prompt_is_the_problem_statement_alone():
    # Anything else in `prompt` changes the task the model is asked to solve, and would
    # make the measured pass rates -- all from no-hint runs -- describe a different task.
    source = (ROOT / "scripts" / "10c_build_swegym_taskset.py").read_text(encoding="utf-8")
    assert '"prompt": problem,' in source


def test_the_verifier_fields_are_kept_because_the_pool_is_useless_without_them():
    source = (ROOT / "scripts" / "10c_build_swegym_taskset.py").read_text(encoding="utf-8")
    for field in ('"test_patch"', '"FAIL_TO_PASS"', '"PASS_TO_PASS"'):
        assert field in source, field
    # ...which is why the uploader must not default to public.
    uploader = (ROOT / "scripts" / "13g_upload_swegym_hf.py").read_text(encoding="utf-8")
    assert "private=not args.public" in uploader


# ------------------------------------------------------------------ gates / wiring


def test_a_task_with_no_failing_test_is_refused():
    # With nothing to flip, the reward is undefined: a do-nothing rollout scores the
    # same as a correct one, so the group has no variance by construction.
    source = (ROOT / "scripts" / "10c_build_swegym_taskset.py").read_text(encoding="utf-8")
    assert "drop_no_fail_to_pass" in source


def test_an_instance_with_no_measured_rollouts_is_refused_not_defaulted():
    """The whole value-add of this pool is that its tiers are measured.

    Defaulting an uncovered instance to any pass rate would put an unscreened task in a
    tier that claims to be screened. All 2,438 happen to be covered, so this gate fires
    zero times today -- which is exactly when it is cheapest to get right.
    """
    source = (ROOT / "scripts" / "10c_build_swegym_taskset.py").read_text(encoding="utf-8")
    assert "drop_no_rollout_coverage" in source


def test_the_absence_of_responses_is_asserted_not_assumed():
    source = (ROOT / "scripts" / "10c_build_swegym_taskset.py").read_text(encoding="utf-8")
    assert "may be " in source and "check before discarding the responses" in source


def test_the_pass_rates_are_labelled_with_the_policy_that_produced_them():
    # A pass rate is not a property of a task alone. gpt-4o-2024-08-06's 7.4 % mean
    # transfers to a stronger policy as an ordering, not as an absolute, and a report
    # that quotes it as intrinsic difficulty is wrong.
    source = (ROOT / "scripts" / "10c_build_swegym_taskset.py").read_text(encoding="utf-8")
    assert "gpt-4o-2024-08-06" in source
    assert "rollout_run_ids" in source


def test_the_output_matches_the_schema_grpo_actually_consumes():
    source = (ROOT / "scripts" / "10c_build_swegym_taskset.py").read_text(encoding="utf-8")
    for key in ('"prompt":', '"label":', '"metadata":', "rl_tasks.jsonl"):
        assert key in source, key


def test_the_reason_the_trajectories_were_not_converted_is_recorded():
    # So "why is there no SWE-Gym SFT set?" is answerable from the artifact.
    source = (ROOT / "scripts" / "10c_build_swegym_taskset.py").read_text(encoding="utf-8")
    for evidence in ("sft_trajectories_exist_but_were_not_converted", "str_replace_editor",
                     "OpenHands-SFT-Trajectories"):
        assert evidence in source, evidence


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
