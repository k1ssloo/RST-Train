"""`scripts/03g_curate_sft.py` -- the soft gate after the verifier.

Pinned on synthetic episodes in both dialects. The first real run on 30,135 rows
mis-banded two things this file now guards: TMax's `<tool_call>` turns were all
"unparseable" because the curator had its own JSON-only parser (it now shares the
offline eval's), and every clean RST success was "claimed completion then continued"
because Terminus-2 asks the agent to repeat its completion claim -- two consecutive
claims are the normal ending, not a flip-flop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import load_script  # noqa: E402

curate = load_script("03g_curate_sft")


def action(cmds, complete=False, plan="do the thing"):
    obj = {"analysis": "a", "plan": plan,
           "commands": [{"keystrokes": c + "\n", "duration": 0.1} for c in cmds]}
    if complete:
        obj["task_complete"] = True
    return {"role": "assistant", "content": json.dumps(obj, indent=2)}


def obs(text="New Terminal Output:\nroot@x:/app# ok\n"):
    return {"role": "user", "content": text}


def episode(turns):
    """turns: list of (assistant message, observation text or None)."""
    messages = [{"role": "user", "content": "TASK: make it work"}]
    for assistant, observation in turns:
        messages.append(assistant)
        if observation is not None:
            messages.append(obs(observation))
    return messages


CLEAN = episode([(action(["ls"]), "New Terminal Output:\nfile\n"),
                 (action(["make"]), "New Terminal Output:\nbuilt\n"),
                 (action([], complete=True), "Are you sure? repeat task_complete"),
                 (action([], complete=True), None)])


def test_a_clean_success_with_the_repeated_completion_claim_is_clear_keep():
    f = curate.trajectory_features(CLEAN)
    assert f["task_complete_count"] == 2 and f["completion_retractions"] == 0
    assert f["final_complete"] is True and f["unparseable_turns"] == 0
    assert curate.band_of(f, None) == ("clear_keep", [])


def test_claiming_done_and_carrying_on_is_a_retraction():
    flip = episode([(action(["ls"]), "x"), (action([], complete=True), "not done: tests fail"),
                    (action(["fix"]), "ok"), (action([], complete=True), "sure?"),
                    (action([], complete=True), None)])
    f = curate.trajectory_features(flip)
    assert f["completion_retractions"] == 1
    band, reasons = curate.band_of(f, None)
    assert band == "borderline" and "completion_claimed_then_continued" in reasons
    twice = episode([(action([], complete=True), "no"), (action(["a"]), "x"),
                     (action([], complete=True), "no"), (action(["b"]), "x"),
                     (action([], complete=True), None)])
    band, reasons = curate.band_of(curate.trajectory_features(twice), None)
    assert band == "clear_drop" and "repeated_false_completion" in reasons


def test_thrashing_is_clear_drop_and_mild_repetition_is_borderline():
    thrash = episode([(action(["make"]), "error: no rule")] * 6 + [(action([], complete=True), None)])
    f = curate.trajectory_features(thrash)
    assert f["repeat_ratio"] >= 0.5 and f["max_consecutive_repeat"] >= 4
    band, reasons = curate.band_of(f, None)
    assert band == "clear_drop" and {"repeat_ratio", "consecutive_repeat"} <= set(reasons)
    mild = episode([(action(["ls"]), "a"), (action(["ls"]), "a"), (action(["make"]), "b"),
                    (action(["cat x"]), "c"), (action([], complete=True), None)])
    band, reasons = curate.band_of(curate.trajectory_features(mild), None)
    assert band == "borderline" and reasons == ["repeat_ratio"]


def test_error_laden_observations_are_seen_and_the_ratio_needs_enough_of_them():
    noisy = episode([(action([f"cmd{i}"]), "Traceback (most recent call last):\nboom") for i in range(6)]
                    + [(action([], complete=True), None)])
    f = curate.trajectory_features(noisy)
    assert f["error_obs_ratio"] > 0.6 and f["n_obs"] == 6
    assert curate.band_of(f, None)[0] == "clear_drop"
    short = episode([(action(["a"]), "error: x"), (action([], complete=True), None)])
    assert curate.band_of(curate.trajectory_features(short), None)[0] == "borderline", (
        "one error in two observations is a flag, not a verdict")


def test_the_tool_call_dialect_parses_through_the_shared_probe():
    xml = ("<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n"
           "</function>\n</tool_call>")
    done = ("<tool_call>\n<function=bash>\n<parameter=command>\n"
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n</parameter>\n</function>\n</tool_call>")
    messages = [{"role": "user", "content": "task"},
                {"role": "assistant", "content": xml}, obs("fine"),
                {"role": "assistant", "content": done}]
    f = curate.trajectory_features(messages)
    assert f["unparseable_turns"] == 0 and f["n_commands"] == 2
    assert f["final_complete"] is True
    assert curate.band_of(f, None)[0] == "clear_keep"


def test_a_turn_that_parses_in_neither_dialect_is_clear_drop():
    messages = [{"role": "user", "content": "task"},
                {"role": "assistant", "content": "I would rather not."}]
    f = curate.trajectory_features(messages)
    assert f["unparseable_turns"] == 1
    assert curate.band_of(f, None) == ("clear_drop", ["unparseable_turn"])


def test_far_longer_than_its_siblings_is_borderline_only_with_enough_siblings():
    # With population std one outlier among n siblings is bounded by sqrt(n-1), so
    # the 2.0 threshold needs six or more siblings to be reachable at all -- RST's
    # cap10 groups have ~8-10, which is the population this feature was sized for.
    rows = [{"trajectory_id": f"t{i}", "task_group_id": "g",
             "features": {"n_turns": n}} for i, n in enumerate([4, 5, 4, 5, 4, 5, 4, 5, 4, 30])]
    z = curate.turn_z_scores(rows, min_group=3)
    assert z["t9"] > 2.0 and abs(z["t0"]) < 1.0
    assert curate.band_of(curate.trajectory_features(CLEAN), z["t9"])[0] == "borderline"
    lonely = curate.turn_z_scores(rows[:2], min_group=3)
    assert lonely["t0"] is None, "two siblings cannot define an outlier"


def test_condense_keeps_the_ends_and_the_request_names_the_schema():
    long = episode([(action([f"step{i}"], plan=f"plan {i}"), f"out {i}") for i in range(20)]
                   + [(action([], complete=True), None)])
    text = curate.condense(long)
    assert "TASK:" in text and "T1:" in text and "T21:" in text
    assert "turns elided" in text and "T10:" not in text
    assert "[claims task complete]" in text
    row = {"trajectory_id": "t", "messages": long, "reasons": ["repeat_ratio"],
           "features": curate.trajectory_features(long)}
    request = curate.judge_request(row)
    assert request.required_keys == curate.JUDGE_KEYS
    assert "heuristic_reasons=repeat_ratio" in request.user
    assert '"keep": true|false' in request.user


def test_judge_keep_accepts_booleans_and_their_string_spellings():
    assert curate.judge_says_keep({"keep": True}) is True
    assert curate.judge_says_keep({"keep": "false"}) is False
    assert curate.judge_says_keep({"keep": "TRUE"}) is True
    assert curate.judge_says_keep({}) is False


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
