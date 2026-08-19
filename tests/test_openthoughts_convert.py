"""`03d_build_openthoughts_sft.py::reconstruct` — the upstream-to-ours mapping.

The upstream shape is so close to ours that the dangerous failures are the quiet
ones: a `system` turn that shifts the rendered prefix, a trailing user turn with
no assistant answer after it, or a stale "Previous response had warnings:"
preamble left in front of an observation whose complaint was repaired away.

Nothing here needs the tokenizer, the 110 MB parquet, or the network.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import load_script  # noqa: E402

convert = load_script("03d_build_openthoughts_sft")
builder = convert.load_builder()

GOOD = {"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\n", "duration": 0.1}]}
CANON = json.dumps(GOOD, indent=2, ensure_ascii=False)
OBSERVATION = "Current terminal state:\nNew Terminal Output:\nroot@host:/# ls\nfile\n"


def turn(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def run(conversation):
    stats: Counter = Counter()
    return convert.reconstruct(conversation, builder, stats), stats


def test_a_minimal_conversation_maps_straight_through():
    record, stats = run([turn("user", "solve this"), turn("assistant", CANON)])
    assert record is not None, dict(stats)
    assert [m["role"] for m in record["messages"]] == ["user", "assistant"]
    assert record["messages"][0]["content"] == "solve this"
    assert json.loads(record["messages"][1]["content"]) == GOOD
    assert record["n_assistant_turns"] == 1
    assert record["n_rewritten_turns"] == 0


def test_a_multi_turn_conversation_keeps_the_alternation():
    record, _ = run([turn("user", "go"), turn("assistant", CANON),
                     turn("user", OBSERVATION), turn("assistant", CANON)])
    assert record is not None
    assert [m["role"] for m in record["messages"]] == \
        ["user", "assistant", "user", "assistant"]
    assert record["n_assistant_turns"] == 2


def test_a_system_turn_is_refused_rather_than_folded_in():
    record, stats = run([turn("system", "you are an agent"), turn("user", "go"),
                         turn("assistant", CANON)])
    assert record is None, "a system turn changes the rendered prefix and the mask"
    assert stats["drop_unexpected_role"] == 1


def test_a_conversation_that_does_not_start_with_a_user_turn_is_refused():
    record, stats = run([turn("assistant", CANON)])
    assert record is None and stats["drop_no_user_prompt"] == 1


def test_a_trailing_user_turn_is_refused():
    # There is no target after it, so it would train the model on nothing while
    # inflating the sequence with a full terminal dump.
    record, stats = run([turn("user", "go"), turn("assistant", CANON),
                         turn("user", OBSERVATION)])
    assert record is None and stats["drop_not_assistant_terminated"] == 1


def test_one_bad_assistant_turn_drops_the_whole_trajectory():
    record, stats = run([turn("user", "go"), turn("assistant", CANON),
                         turn("user", OBSERVATION), turn("assistant", "no json here")])
    assert record is None, "a mid-conversation failure must not be silently skipped"
    assert stats["drop_unparseable"] == 1


def test_the_warning_preamble_is_stripped_after_a_rewritten_turn():
    scolded = ("Previous response had warnings:\n- Extra text detected before JSON "
               "object\n\n" + OBSERVATION)
    record, stats = run([turn("user", "go"),
                         turn("assistant", "Let me think.\n" + CANON),  # rewritten
                         turn("user", scolded),
                         turn("assistant", CANON)])
    assert record is not None
    assert record["n_rewritten_turns"] == 1
    assert stats["repaired_warning_preamble"] == 1
    assert not record["messages"][2]["content"].startswith("Previous response had"), (
        "the model is being trained to expect a scolding for output this dataset "
        "now shows as correct"
    )
    assert record["messages"][2]["content"].startswith("Current terminal state:")


def test_the_warning_preamble_is_kept_when_the_previous_turn_was_clean():
    # Then the complaint is about something real that survived into the data.
    scolded = "Previous response had warnings:\n- Command timed out\n\n" + OBSERVATION
    record, stats = run([turn("user", "go"), turn("assistant", CANON),
                         turn("user", scolded), turn("assistant", CANON)])
    assert record is not None
    assert stats["repaired_warning_preamble"] == 0
    assert record["messages"][2]["content"].startswith("Previous response had")


def test_an_empty_conversation_is_refused():
    record, stats = run([])
    assert record is None and stats["drop_empty_conversation"] == 1


def test_a_nonstring_content_is_refused_rather_than_coerced():
    record, stats = run([turn("user", "go"), {"role": "assistant", "content": None}])
    assert record is None and stats["drop_nonstring_message"] == 1


def test_identical_trajectories_hash_identically_and_differing_ones_do_not():
    a, _ = run([turn("user", "go"), turn("assistant", CANON)])
    b, _ = run([turn("user", "go"), turn("assistant", "prose\n" + CANON)])
    assert a is not None and b is not None
    # Same canonical content despite different raw input -> same hash, so the
    # dedup pass cannot be fooled by a prose preamble.
    assert a["content_hash"] == b["content_hash"]
    c, _ = run([turn("user", "different task"), turn("assistant", CANON)])
    assert c is not None and c["content_hash"] != a["content_hash"]


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
