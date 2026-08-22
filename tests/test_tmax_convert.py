"""`scripts/03e_build_tmax_sft.py` -- the TMax conversion, and its four repairs.

WHY THESE TESTS AND NOT OTHERS
    The converter's one real claim is that its `{role, content}` pre-render is
    byte-identical to the native tool-calling render, and that claim is already
    asserted per row at build time against a live tokenizer -- 5,795/5,795 pass, so
    re-asserting it here would test the tokenizer, not the code.

    What is worth pinning is the part with judgement in it: which upstream defects
    get repaired and which get refused. Every threshold below came out of a
    measurement over all 82,203 assistant turns, and each one is a place where a
    plausible-looking "fix" would quietly write a guess into supervision:

        528 turns  a `</think>` inside content while `reasoning_content` is ALSO
                   set. 483 are one THOUGHT then the tag with nothing after it
                   (repairable); 45 are two to six THOUGHT paragraphs each with
                   their own tag (not repairable -- which one did it mean?).

        269 turns  `reasoning_content` ending in tool-call closing scaffolding.
                   Strippable when prose precedes it, refusable when the reasoning
                   IS the scaffolding.

          4 turns  `reasoning_content` that is the bare literal `<|im_start|>`.
                   One token under this tokenizer. Must never be trained on.

        409 turns  malformed turns followed by a "Format error:" scolding, spliced
                   out so the surviving trajectory has exactly one user turn --
                   which is what keeps the chat template from discarding the CoT of
                   every turn before the last one (71.3 % of assistant tokens).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, load_script  # noqa: E402

convert = load_script("03e_build_tmax_sft")


def _assistant(content: str, reasoning: str | None = None, command: str | None = None):
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if command is not None:
        message["tool_calls"] = [
            {"type": "function",
             "function": {"name": "bash", "arguments": {"command": command}}}
        ]
    return message


# ------------------------------------------------------------ the format-error splice


def test_the_scolding_and_the_turn_that_caused_it_are_both_removed():
    # Keeping the malformed turn trains the model to omit the tool call; keeping the
    # scolding trains it to expect to be told off for output it was just shown.
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Please solve this task:\n\ndo it"},
        _assistant("THOUGHT: a", command="ls"),
        {"role": "tool", "content": "out"},
        _assistant("THOUGHT: forgot the tool call"),          # malformed
        {"role": "user", "content": "Format error: Your last response did not..."},
        _assistant("THOUGHT: b", command="pwd"),
        {"role": "tool", "content": "out2"},
    ]
    spliced, removed = convert.splice_format_errors(messages)
    assert removed == 1
    assert [m["role"] for m in spliced] == [
        "system", "user", "assistant", "tool", "assistant", "tool"]
    assert "forgot" not in "".join(m["content"] for m in spliced)
    assert not any(m["content"].startswith("Format error:") for m in spliced)


def test_the_splice_leaves_exactly_one_user_turn():
    """The property the CoT depends on.

    The template emits reasoning only for assistant turns after the LAST user turn
    (chat_template line 99). A surviving mid-conversation user turn therefore
    silently deletes the reasoning of everything before it.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant("bad"),
        {"role": "user", "content": "Format error: one"},
        _assistant("bad again"),
        {"role": "user", "content": "Format error: two"},
        _assistant("THOUGHT: good", command="ls"),
    ]
    spliced, removed = convert.splice_format_errors(messages)
    assert removed == 2
    assert sum(1 for m in spliced if m["role"] == "user") == 1


def test_the_opening_task_description_is_never_treated_as_a_scolding():
    # It is a `user` turn too. Guarded by position, so a task whose text happens to
    # begin with "Format error:" cannot decapitate its own trajectory.
    messages = [
        {"role": "user", "content": "Format error: is the string to grep for"},
        _assistant("THOUGHT: a", command="ls"),
    ]
    spliced, removed = convert.splice_format_errors(messages)
    assert removed == 0
    assert spliced == messages


# ------------------------------------------------------------ the stray </think>


def test_a_lone_trailing_close_tag_is_dropped():
    # 483 turns. Rendered as-is the turn closes a think block the template already
    # closed, teaching two `</think>` per turn.
    content = "THOUGHT: I will look at the files.\n</think>\n\n"
    repaired, changed = convert.repair_stray_think(content)
    assert changed is True
    assert repaired == "THOUGHT: I will look at the files.\n"
    assert "</think>" not in repaired


def test_repeated_thought_paragraphs_are_refused_rather_than_picked_from():
    # 45 turns. An upstream sampling loop emitted the thought several times; which
    # one is canonical is unknowable, so this must not be repaired.
    content = ("THOUGHT: first attempt.\n</think>\n\n"
               "THOUGHT: second attempt.\n</think>\n\n")
    repaired, changed = convert.repair_stray_think(content)
    assert repaired is None, "a guess was made about which THOUGHT was meant"


def test_a_close_tag_with_text_after_it_is_refused():
    # Dropping the tag here would silently splice two paragraphs into one.
    repaired, _ = convert.repair_stray_think("THOUGHT: a\n</think>\nand then more text")
    assert repaired is None


def test_clean_content_is_returned_unchanged_and_unflagged():
    content = "THOUGHT: nothing wrong here."
    repaired, changed = convert.repair_stray_think(content)
    assert (repaired, changed) == (content, False)


# ------------------------------------------------------------ reasoning sanitation


def test_trailing_tool_scaffolding_is_stripped_and_the_prose_survives():
    # The call itself is intact in `tool_calls`, so the closing tags are a duplicate,
    # not a guess. Recovers the CoT on 34 turns instead of dropping the row.
    reasoning = ("Let me verify the PNG is valid by checking its header.\n"
                 "</parameter>\n</function>\n</tool_call>")
    cleaned, changed = convert.sanitize_reasoning(reasoning)
    assert changed is True
    assert cleaned == "Let me verify the PNG is valid by checking its header."
    assert not any(lit in cleaned for lit in convert.FORBIDDEN_IN_SOURCE)


def test_reasoning_that_is_nothing_but_a_malformed_call_is_refused():
    # Stripping the trailing run leaves `<parameter=command>...`, which is markup, so
    # there is no prose to recover and nothing to do but refuse.
    reasoning = ("<parameter=command>\nsed -i 's/a/b/' main.go\n"
                 "</parameter>\n</function>\n</tool_call>")
    cleaned, _ = convert.sanitize_reasoning(reasoning)
    assert cleaned is None


def test_a_bare_control_token_in_reasoning_is_refused():
    """The one that actually costs something.

    `<|im_start|>` is a single token here. A model trained to emit it can forge a
    turn boundary mid-answer and inject its own system or tool turn, which ends a
    harbor rollout in a way no reward signal attributes to the right cause. No
    strip can rescue this, so the row goes.
    """
    assert convert.sanitize_reasoning("<|im_start|>\n")[0] is None
    assert convert.sanitize_reasoning("thinking <|im_end|> more")[0] is None


def test_ordinary_reasoning_passes_through_untouched():
    reasoning = "First I will read the file, then patch the off-by-one."
    assert convert.sanitize_reasoning(reasoning) == (reasoning, False)


def test_the_forbidden_list_covers_every_special_token_the_template_emits():
    # If a new control token is added to the template but not here, it becomes
    # trainable text. Cheap to assert, and the failure mode is invisible otherwise.
    for literal in ("<|im_start|>", "<|im_end|>", "<think>", "</think>",
                    "<tool_call>", "</tool_call>", "<tool_response>"):
        assert literal in convert.FORBIDDEN_IN_SOURCE, literal


# ------------------------------------------------------------ the pre-render


def test_a_tool_observation_becomes_a_tool_response_wrapped_user_turn():
    """Why this does not cost us the reasoning.

    The template's scan for the last real query SKIPS a user turn that starts with
    `<tool_response>` and ends with `</tool_response>` (chat_template lines 69-74),
    so `last_query_index` still lands on the task description and every assistant
    turn stays after it. The wrapper is load-bearing: without it, each observation
    would count as a new query and strip the CoT from everything before it.
    """
    rendered = convert.pre_render(
        [{"role": "tool", "content": "total 16\ndrwxr-xr-x  2 root root"}], "sys")
    assert rendered[0]["role"] == "user"
    assert rendered[0]["content"].startswith("<tool_response>")
    assert rendered[0]["content"].endswith("</tool_response>")


def test_assistant_content_is_trimmed_before_it_is_used():
    """The bug that cost 5,792 of 5,795 rows on the first run.

    TMax content is wrapped in newlines -- "\\n\\nTHOUGHT: ...\\n\\n" -- and the
    template trims every message's content (chat_template line 81). Left raw, the
    trailing newlines survive into our render but not the template's, and the two
    disagree by four characters immediately before every tool call.
    """
    rendered = convert.pre_render(
        [_assistant("\n\nTHOUGHT: do the thing.\n\n", reasoning="because", command="ls")],
        "sys")
    content = rendered[0]["content"]
    assert content == ("<think>\nbecause\n</think>\n\nTHOUGHT: do the thing."
                       "\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n"
                       "</parameter>\n</function>\n</tool_call>")


def test_a_tool_call_after_empty_content_gets_no_blank_line():
    # The template makes the leading "\n\n" conditional on `content|trim`
    # (lines 109-114). 7,486 turns have an empty content field, so this branch is
    # not an edge case, and a wrong newline here is a byte-level render mismatch.
    rendered = convert.pre_render([_assistant("", reasoning="r", command="pwd")], "sys")
    assert rendered[0]["content"] == (
        "<think>\nr\n</think>\n\n<tool_call>\n<function=bash>\n"
        "<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>")


def test_the_pre_render_emits_only_roles_the_existing_pipeline_reads():
    messages = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "task"},
        _assistant("THOUGHT: a", reasoning="r", command="ls"),
        {"role": "tool", "content": "out"},
    ]
    rendered = convert.pre_render(messages, "baked")
    assert {m["role"] for m in rendered} <= {"system", "user", "assistant"}
    # `{role, content}` and nothing else, or the parquet schema stops matching the
    # one 15_export_pretokenized.py and the verl dataset already consume.
    assert all(set(m) == {"role", "content"} for m in rendered)
    assert rendered[0]["content"] == "baked", "the tool schema was not baked in"


def test_multiple_arguments_render_in_order():
    calls = [{"type": "function",
              "function": {"name": "bash", "arguments": {"a": "1", "b": "2"}}}]
    block = convert.render_tool_calls(calls, "")
    assert block.index("<parameter=a>") < block.index("<parameter=b>")


def test_a_structured_argument_is_json_encoded_like_the_template_does():
    # chat_template line 121 tojson's a mapping or a non-string sequence.
    calls = [{"type": "function",
              "function": {"name": "bash", "arguments": {"argv": ["ls", "-la"]}}}]
    assert '["ls", "-la"]' in convert.render_tool_calls(calls, "")


# ------------------------------------------------------------ wiring


def test_the_converter_refuses_a_second_tool_schema():
    # Baking one system prompt is only legitimate while upstream ships exactly one
    # tool schema. If a second appears, one baked prompt would describe the wrong
    # tools for some rows -- silently, since the render would still be self-consistent.
    source = (ROOT / "scripts" / "03e_build_tmax_sft.py").read_text(encoding="utf-8")
    assert "distinct `tools` payloads" in source


def test_the_dpo_verdict_is_recorded_with_its_numbers():
    # So that "why is there no DPO stage for TMax" is answerable from the artifact
    # rather than from memory.
    source = (ROOT / "scripts" / "03e_build_tmax_sft.py").read_text(encoding="utf-8")
    for evidence in ("pairable_tasks", "291", "2020", "444"):
        assert evidence in source, evidence


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
