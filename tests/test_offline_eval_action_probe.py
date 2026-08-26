"""The action probe, and the two ways it was silently measuring nothing.

Both defects had the same shape: a metric that reported a clean-looking value (or an
honest `None`) while comparing the wrong thing, on data the repo trains on every day.

**1. One dialect only.** `action_of` went through `normalize_assistant`, which requires
the RST/OpenThoughts/Nemotron JSON contract `{analysis, plan, commands}`. Half the
corpora here are not that: TMax -- and any tool-calling corpus -- emits

    <tool_call>
    <function=bash>
    <parameter=command>
    ls -la /app
    </parameter>
    </function>
    </tool_call>

Measured on the 4B TMax checkpoint before the fix: **80 of 80 probed turns counted
`reference_unparseable`**, every action rate came back `None`, and `note` said "no turns
were probed". The teacher-forced loss was real; the action probe measured nothing, and
only the skip counter said so.

**2. A key that exists in no dataset.** `compare_actions` read
`reference.get("is_task_complete")`. Every dataset here writes **`task_complete`**
(verified: 398/398 assistant turns in the Nemotron holdout). So both sides of the
comparison were `None`, `None == None` was true, and `task_complete_agreement_rate`
reported **1.0 for every parsed pair regardless of what the model predicted** -- the one
failure mode worse than a missing metric, because it looks like a passing one.

The trailing-newline normalization in `keystrokes_of` is the third piece: the JSON
dialect writes `"ls -la\\n"` and the tool-calling dialect writes `ls -la` for the
identical action, so comparing raw would score every cross-dialect pair as a mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import load_script  # noqa: E402

probe = load_script("06b_eval_offline")

TOOL_CALL = (
    "<think>\nreasoning\n</think>\n\nTHOUGHT: look around.\n\n"
    "<tool_call>\n<function=bash>\n<parameter=command>\nls -la /app\n"
    "</parameter>\n</function>\n</tool_call>"
)
JSON_ACTION = (
    '{"analysis":"a","plan":"p",'
    '"commands":[{"keystrokes":"ls -la /app\\n","duration":0.1}],"task_complete":false}'
)
DONE_CALL = (
    "<tool_call>\n<function=bash>\n<parameter=command>\n"
    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n</parameter>\n</function>\n</tool_call>"
)


# ---------------------------------------------------------------- dialects


def test_the_tool_calling_dialect_parses():
    """The regression: 80 of 80 TMax turns were unparseable before this."""
    action = probe.action_of(TOOL_CALL)
    assert action is not None, "a tool-calling turn was counted reference_unparseable"
    assert action["_dialect"] == "tool_call"
    assert probe.keystrokes_of(action) == ["ls -la /app"]


def test_the_json_contract_still_parses_and_is_preferred():
    action = probe.action_of(JSON_ACTION)
    assert action is not None and action["_dialect"] == "json"
    assert action["analysis"] == "a", "the JSON body was not returned intact"


def test_the_same_action_in_either_dialect_compares_as_identical():
    """What the trailing-newline strip is for.

    `"ls -la /app\\n"` (JSON keystrokes) and `ls -la /app` (tool parameter) are the same
    command. Raw, every cross-dialect pair would score commands_exact=False.
    """
    tool, js = probe.action_of(TOOL_CALL), probe.action_of(JSON_ACTION)
    assert probe.keystrokes_of(tool) == probe.keystrokes_of(js)
    assert probe.compare_actions(tool, js)["commands_exact"] is True
    assert probe.compare_actions(js, tool)["commands_exact"] is True


def test_prose_with_no_action_is_still_none():
    # The probe must keep being able to say "the model emitted no action at all".
    assert probe.action_of("I think I should look at the files, but here is no command.") is None
    assert probe.action_of("") is None


def test_several_tool_calls_in_one_turn_all_land():
    keys = probe.keystrokes_of(probe.action_of(TOOL_CALL + "\n" + DONE_CALL))
    assert len(keys) == 2
    assert keys[0] == "ls -la /app"


def test_a_single_differently_named_parameter_is_not_dropped():
    # `command` is the bash tool's parameter name. A one-parameter tool with another
    # name is still an action, and silently dropping it would understate the parse rate.
    action = probe.action_of(
        "<tool_call>\n<function=run>\n<parameter=script>\npwd\n</parameter>\n"
        "</function>\n</tool_call>")
    assert probe.keystrokes_of(action) == ["pwd"]


def test_a_tool_call_with_no_usable_parameter_is_not_a_phantom_action():
    action = probe.action_of("<tool_call>\n<function=bash>\n</function>\n</tool_call>")
    assert action is None


# ---------------------------------------------------------------- task_complete


def test_the_completion_key_the_data_actually_uses_is_read():
    """The silent one.

    `is_task_complete` appears in no dataset here; `task_complete` appears in all of
    them. Reading the wrong key made both sides None and the agreement trivially true.
    """
    assert probe.task_complete_of({"task_complete": True}) is True
    assert probe.task_complete_of({"task_complete": False}) is False
    # tolerated in case some producer emits it, but it is not what the data uses
    assert probe.task_complete_of({"is_task_complete": True}) is True
    assert probe.task_complete_of({}) is False
    assert probe.task_complete_of(None) is False


def test_the_tool_calling_dialect_signals_completion_by_the_sentinel_command():
    # There is no structured field in that dialect -- the agent says it is done by
    # running the sentinel, so that is what "the model thinks it is done" means.
    assert probe.task_complete_of(probe.action_of(DONE_CALL)) is True
    assert probe.task_complete_of(probe.action_of(TOOL_CALL)) is False


def test_disagreement_about_completion_is_now_actually_detected():
    """Before the fix this returned True for every parsed pair."""
    not_done, done = probe.action_of(TOOL_CALL), probe.action_of(DONE_CALL)
    assert probe.compare_actions(not_done, done)["task_complete_agreement"] is False
    assert probe.compare_actions(not_done, not_done)["task_complete_agreement"] is True
    assert probe.compare_actions(done, done)["task_complete_agreement"] is True


def test_an_unparseable_prediction_never_counts_as_agreement():
    # `predicted is None` must short-circuit, or a model that emitted nothing would be
    # credited with agreeing about completion.
    assert probe.compare_actions(probe.action_of(TOOL_CALL), None) == {
        "parsed": False, "commands_exact": False, "first_keystrokes_exact": False,
        "command_count_match": False, "task_complete_agreement": False,
    }


# ---------------------------------------------------------------- comparison shape


def test_analysis_and_plan_are_not_compared():
    # Two different words for the same plan are not a disagreement; only the executable
    # part is comparable. Pinned so a future edit does not start scoring prose.
    other = (
        '{"analysis":"COMPLETELY DIFFERENT PROSE","plan":"ALSO DIFFERENT",'
        '"commands":[{"keystrokes":"ls -la /app\\n","duration":0.1}],"task_complete":false}'
    )
    verdict = probe.compare_actions(probe.action_of(JSON_ACTION), probe.action_of(other))
    assert verdict["commands_exact"] is True


def test_a_different_command_is_not_scored_exact():
    other = probe.action_of(
        "<tool_call>\n<function=bash>\n<parameter=command>\nrm -rf /\n"
        "</parameter>\n</function>\n</tool_call>")
    verdict = probe.compare_actions(probe.action_of(TOOL_CALL), other)
    assert verdict["commands_exact"] is False
    assert verdict["first_keystrokes_exact"] is False
    assert verdict["command_count_match"] is True   # one command each


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
