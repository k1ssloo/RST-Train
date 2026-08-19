"""`normalize_assistant` — the one canonical form for an agent turn.

Every dataset in this repo goes through this function: the RST builder, the DPO
builder's format control, `06b_eval_offline.py` (which loads it by path precisely
so there is only one implementation), and the OpenThoughts converter. So a change
in what it accepts changes what every one of them contains.

Two of the cases below are regressions from real data, both found by running
15,209 upstream trajectories through it:

  * agents emit **literal newlines inside JSON string values**. That is invalid
    JSON by spec and `json.loads` rejects it as an "Invalid control character",
    but the intent is unambiguous and re-dumping escapes it correctly. 769 turns
    still fail for other reasons; this repair recovered ~420.
  * `find . -exec ls {} \\;` inside a prose preamble **balances as the empty
    object**. The old code took the first balanced object, found `{}`, saw no
    required keys, and reported `missing_required_keys` for a turn whose real
    response was two lines further down. That masked 180 good turns.

The third case is the boundary: unescaped inner quotes (`directories "1" and
"2"`) are NOT repaired. Where the string ends is a guess, and a guess here writes
wrong supervision into training data. It stays a drop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import load_script  # noqa: E402

builder = load_script("03_build_sft_data")
normalize = builder.normalize_assistant

GOOD = {"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\n", "duration": 0.1}]}


def canonical_of(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def test_a_clean_turn_is_accepted_and_reported_as_not_rewritten():
    canonical, rewritten, reason = normalize(canonical_of(GOOD))
    assert reason == "ok"
    assert not rewritten, "an already-canonical turn must not be counted as repaired"
    assert json.loads(canonical) == GOOD


def test_the_canonical_form_is_a_fixed_point():
    # Otherwise `rewritten_turn_fraction` counts the same turn forever and the
    # warning-preamble repair fires on turns nothing was done to.
    once, _, _ = normalize(canonical_of(GOOD))
    twice, rewritten, reason = normalize(once)
    assert reason == "ok" and twice == once and not rewritten


def test_a_literal_newline_inside_a_string_is_repaired():
    raw = '{"analysis": "step one\nstep two", "plan": "p", "commands": []}'
    # Establish that this really is invalid JSON, so the test cannot pass vacuously.
    try:
        json.loads(raw)
        raise AssertionError("input is valid strict JSON; the regression is not covered")
    except json.JSONDecodeError:
        pass
    canonical, rewritten, reason = normalize(raw)
    assert reason == "ok", f"literal newline in a string was not repaired ({reason})"
    assert rewritten, "a repaired turn must be reported as rewritten"
    assert json.loads(canonical)["analysis"] == "step one\nstep two"


def test_the_repaired_output_is_valid_strict_json():
    # The point of the repair: downstream (verl, the eval parser, terminus-2) sees
    # a string it can parse without knowing anything about this leniency.
    canonical, _, reason = normalize('{"analysis": "a\nb", "plan": "p", "commands": []}')
    assert reason == "ok"
    json.loads(canonical, strict=True)


def test_an_empty_object_from_a_shell_command_does_not_mask_the_response():
    raw = ("Let me check the files:\n"
           "I ran find . -type f -exec ls -la {} \\; to list them.\n"
           + canonical_of(GOOD))
    assert builder._balanced_json_object(raw) == "{}", (
        "this test asserts nothing unless the first balanced object really is the "
        "find -exec placeholder"
    )
    canonical, rewritten, reason = normalize(raw)
    assert reason == "ok", f"the real response was masked by an earlier {{}} ({reason})"
    assert rewritten, "extracting from a prose preamble is a rewrite"
    assert json.loads(canonical) == GOOD


def test_prose_around_the_response_is_dropped_not_kept():
    # terminus-2's own parser complains "Extra text detected before JSON object",
    # so the training target must be the JSON alone.
    canonical, _, reason = normalize("Thinking out loud.\n" + canonical_of(GOOD) + "\nDone.")
    assert reason == "ok"
    assert "Thinking out loud" not in canonical and "Done." not in canonical


def test_a_fenced_block_is_accepted():
    canonical, rewritten, reason = normalize("```json\n" + json.dumps(GOOD) + "\n```")
    assert reason == "ok" and rewritten
    assert json.loads(canonical) == GOOD


def test_a_genuinely_keyless_object_is_still_refused():
    canonical, _, reason = normalize('{"thought": "no required keys here"}')
    assert canonical is None and reason == "missing_required_keys"


def test_commands_must_be_a_list_not_a_sentence():
    raw = '{"analysis": "a", "plan": "p", "commands": "no commands to run"}'
    canonical, _, reason = normalize(raw)
    assert canonical is None and reason == "commands_not_list", (
        "a string `commands` would be replayed as zero actions, silently teaching "
        "the model that a sentence is a valid action batch"
    )


def test_an_unescaped_inner_quote_is_refused_rather_than_guessed():
    raw = '{"analysis": "made directories "1" and "2" here", "plan": "p", "commands": []}'
    canonical, _, reason = normalize(raw)
    assert canonical is None, (
        "the normalizer guessed where a broken string ended; repairing this means "
        "inventing content, and it lands in training data"
    )
    assert reason in ("unparseable", "missing_required_keys")


def test_a_truncated_object_is_unparseable_and_does_not_crash():
    canonical, _, reason = normalize('{"analysis": "cut off mid-way, no closing brace')
    assert canonical is None and reason == "unparseable"


def test_no_json_at_all_is_unparseable():
    canonical, _, reason = normalize("I think the task is done, no JSON here.")
    assert canonical is None and reason == "unparseable"


def test_empty_and_whitespace_input_are_refused():
    for raw in ("", "   \n\t "):
        canonical, _, reason = normalize(raw)
        assert canonical is None and reason == "unparseable"


def test_the_first_valid_response_wins_when_several_objects_are_present():
    # A retry preamble can contain an earlier, keyless object; the response is the
    # first one that actually satisfies the contract, scanning left to right.
    other = {"analysis": "second", "plan": "p", "commands": []}
    raw = '{"note": "keyless"}\n' + canonical_of(GOOD) + "\n" + canonical_of(other)
    canonical, _, reason = normalize(raw)
    assert reason == "ok"
    assert json.loads(canonical)["analysis"] == "a"


def test_balanced_objects_are_found_left_to_right_and_not_nested():
    found = builder._balanced_json_objects('{"a": {"b": 1}} tail {"c": 2}')
    assert found == ['{"a": {"b": 1}}', '{"c": 2}'], (
        f"nested objects must not be separate candidates, got {found}"
    )


def test_a_brace_inside_a_string_does_not_end_the_object():
    obj = {"analysis": "wrote } to the file", "plan": "p", "commands": []}
    canonical, _, reason = normalize(json.dumps(obj))
    assert reason == "ok"
    assert json.loads(canonical)["analysis"] == "wrote } to the file"


def test_command_signature_ignores_prose_and_tracks_keystrokes():
    a = [{"role": "assistant", "content": canonical_of(GOOD)}]
    b = [{"role": "assistant", "content": "preamble\n" + canonical_of(GOOD)}]
    # b's content is not canonical, so normalize it the way the builders do first.
    b_norm = [{"role": "assistant", "content": normalize(b[0]["content"])[0]}]
    assert builder.command_signature(a) == builder.command_signature(b_norm)
    other = dict(GOOD, commands=[{"keystrokes": "rm -rf /\n", "duration": 0.1}])
    assert builder.command_signature(a) != builder.command_signature(
        [{"role": "assistant", "content": canonical_of(other)}]
    )


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
