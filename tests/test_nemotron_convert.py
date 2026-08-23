"""`scripts/03f_build_nemotron_sft.py` -- the Nemotron conversion, and its three repairs.

WHY THESE TESTS AND NOT OTHERS
    The conversion itself is mostly `03d`'s: upstream is already this repo's
    terminus-2 contract, so the JSON goes through the shared `normalize_assistant`
    and there is nothing new to pin there.

    What is new, and what these tests are about, is that every assistant turn
    carries a real `<think>` block -- 47.1 % of assistant tokens -- and that
    reasoning interacts with the Qwen3.5 template in a way that is easy to get
    wrong in both directions. Each threshold below came out of a measurement over
    a multi-file sample of the real corpus:

        17.4 %   of assistant turns have an EMPTY body: the think block ran long,
                 swallowed the start of the JSON, and `</think>` landed after it.
                 The next turn is always a parse-error scolding carrying no
                 observation (761/761), so both are spliced out.

        118/118  of the normalization failures that survive the splice are the
                 LAST turn with body exactly `null`. So this file truncates where
                 `03d` refuses to -- the same test, opposite answer, because there
                 the failures clustered on the FIRST turn.

        1 pair   of `<think>`/`</think>` is required. The template recovers
                 reasoning with `split('</think>')[0]` and content with
                 `split('</think>')[-1]`, so a second tag makes those two splits
                 disagree about where the reasoning ended, silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, load_script  # noqa: E402

convert = load_script("03f_build_nemotron_sft")

ACTION = '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n", "duration": 0.1}]}'


def _assistant(think: str | None, body: str = ACTION) -> dict[str, str]:
    content = body if think is None else f"<think>\n{think}\n</think>\n\n{body}"
    return {"role": "assistant", "content": content}


def _observation(text: str = "ls -la\ntotal 0") -> dict[str, str]:
    return {"role": "user", "content": f"New Terminal Output:\n{text}"}


# ------------------------------------------------------------------ split_think


def test_a_single_balanced_pair_splits_into_reasoning_and_action():
    think, body, reason = convert.split_think(f"<think>\nwhy\n</think>\n\n{ACTION}")
    assert reason == "ok"
    assert think.strip() == "why"
    assert body.strip() == ACTION


def test_a_turn_with_no_think_block_is_allowed_through_unchanged():
    # 12 of ~6,600 turns in the sample have none. The template renders an empty
    # think block for them, which is a shape the RST and OpenThoughts data are
    # entirely made of, so there is nothing to refuse.
    think, body, reason = convert.split_think(ACTION)
    assert (think, reason) == ("", "ok")
    assert body == ACTION


def test_a_second_close_tag_is_refused_rather_than_split_on_the_wrong_one():
    think, _, reason = convert.split_think(f"<think>\na\n</think>\nb\n</think>\n\n{ACTION}")
    assert think is None and reason == "bad_think_shape"


def test_tags_in_the_wrong_order_are_refused():
    think, _, reason = convert.split_think(f"</think>\nstray\n<think>\n\n{ACTION}")
    assert think is None and reason == "bad_think_shape"


def test_an_unclosed_think_block_is_refused():
    think, _, reason = convert.split_think(f"<think>\nnever closed\n\n{ACTION}")
    assert think is None and reason == "bad_think_shape"


# ------------------------------------------------------------ the retry splice


def test_a_parse_error_scolding_and_the_turn_that_caused_it_are_both_removed():
    conversation = [
        {"role": "user", "content": "harness prompt"},
        _assistant("first thought"),
        _observation(),
        _assistant("thought that ate the JSON", body=""),          # malformed
        {"role": "user", "content": convert.PARSE_ERROR_PREFIX
                                    + "\nERROR: No valid JSON found in response"},
        _assistant("the retry"),
    ]
    spliced, removed = convert.splice_retries(conversation)
    assert removed == 1
    assert [m["role"] for m in spliced] == ["user", "assistant", "user", "assistant"]
    assert "ate the JSON" not in "".join(m["content"] for m in spliced)


def test_the_splice_preserves_strict_user_assistant_alternation():
    """What the loss mask depends on.

    The scolding carries no observation, so removing the pair reconnects the previous
    observation directly to the retry. Removing only one of the two would leave two
    adjacent turns of the same role and shift every assistant span after it.
    """
    conversation = [{"role": "user", "content": "prompt"}]
    for i in range(3):
        conversation += [
            _assistant(f"bad {i}", body=""),
            {"role": "user", "content": convert.PARSE_ERROR_PREFIX + "\nERROR: ..."},
        ]
    conversation += [_assistant("good")]
    spliced, removed = convert.splice_retries(conversation)
    assert removed == 3
    roles = [m["role"] for m in spliced]
    assert roles == ["user", "assistant"]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


def test_the_harness_prompt_is_never_treated_as_a_scolding():
    conversation = [{"role": "user", "content": convert.PARSE_ERROR_PREFIX + " is the text"},
                    _assistant("t")]
    spliced, removed = convert.splice_retries(conversation)
    assert removed == 0 and spliced == conversation


def test_a_real_observation_is_not_spliced():
    conversation = [{"role": "user", "content": "prompt"}, _assistant("t"),
                    _observation(), _assistant("u")]
    spliced, removed = convert.splice_retries(conversation)
    assert removed == 0 and len(spliced) == 4


# ------------------------------------------------------------ the dead tail


def test_a_trailing_null_action_is_truncated_with_the_observation_it_answered():
    # The harness's marker for an episode that ended without an action. Everything
    # before it is valid supervision, so dropping the trajectory would be the lossy
    # choice -- unlike in 03d, where failures clustered on the first turn.
    conversation = [{"role": "user", "content": "prompt"}, _assistant("t"),
                    _observation(), _assistant("dead", body="null")]
    out, removed = convert.truncate_dead_tail(conversation)
    assert removed == 1
    assert [m["role"] for m in out] == ["user", "assistant"]


def test_a_well_formed_trajectory_is_left_completely_alone():
    conversation = [{"role": "user", "content": "prompt"}, _assistant("t"),
                    _observation(), _assistant("u")]
    out, removed = convert.truncate_dead_tail(conversation)
    assert removed == 0 and out == conversation


def test_an_observation_left_dangling_by_the_splice_is_dropped():
    # After splice_retries removes a trailing scolding and its malformed turn, the
    # observation before them has no answer. It is context carrying no target.
    conversation = [{"role": "user", "content": "prompt"}, _assistant("t"), _observation()]
    out, removed = convert.truncate_dead_tail(conversation)
    assert removed == 0
    assert [m["role"] for m in out] == ["user", "assistant"]


# ------------------------------------------------------------ reconstruct


def _stats():
    from collections import Counter
    return Counter()


def test_a_clean_trajectory_reconstructs_with_reasoning_kept_in_every_turn():
    builder = load_script("03_build_sft_data")
    stats = _stats()
    conversation = [{"role": "user", "content": "harness prompt"},
                    _assistant("thought one"), _observation(), _assistant("thought two")]
    messages = convert.reconstruct(conversation, builder, stats)
    assert messages is not None
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert len(assistants) == 2
    # Kept, not stripped. The template drops the non-final ones at render time; the
    # stored data keeps them so nothing is destroyed in the published artifact.
    assert all(m["content"].startswith("<think>\n") for m in assistants)
    assert all("</think>\n\n{" in m["content"] for m in assistants)


def test_the_json_body_is_canonicalized_by_the_shared_normalizer():
    """Not reimplemented here, on purpose.

    `normalize_assistant` is the definition of this repo's canonical assistant form.
    A second copy would mean two canonical forms in one training mixture, which is
    the thing that makes RST, OpenThoughts and Nemotron mixable row-for-row.
    """
    import json

    builder = load_script("03_build_sft_data")
    stats = _stats()
    messy = '{"analysis":"a","plan":"p","commands":[{"keystrokes":"ls\\n","duration":0.1}]}'
    messages = convert.reconstruct(
        [{"role": "user", "content": "p"}, _assistant("t", body=messy)], builder, stats)
    assert messages is not None
    body = messages[-1]["content"].split("</think>")[1]
    assert json.loads(body)["analysis"] == "a"
    assert body.startswith("\n\n{\n  "), "not re-dumped with the canonical indent"


def test_a_control_token_in_an_assistant_turn_is_refused():
    # One token under this tokenizer, so this is not markup in text -- it would train
    # the model to emit a turn boundary it can then forge.
    builder = load_script("03_build_sft_data")
    stats = _stats()
    out = convert.reconstruct(
        [{"role": "user", "content": "p"}, _assistant("<|im_start|> smuggled")],
        builder, stats)
    assert out is None and stats["drop_control_token"] == 1


def test_think_markup_inside_the_json_body_is_refused():
    # `content.split('</think>')[-1]` would take the wrong half.
    builder = load_script("03_build_sft_data")
    stats = _stats()
    body = '{"analysis": "a </think> b", "plan": "p", "commands": []}'
    out = convert.reconstruct(
        [{"role": "user", "content": "p"}, _assistant("t", body=body)], builder, stats)
    assert out is None


def test_a_trajectory_not_ending_on_an_assistant_turn_is_refused():
    builder = load_script("03_build_sft_data")
    stats = _stats()
    out = convert.reconstruct(
        [{"role": "user", "content": "p"}, _assistant("t"), _observation(), _observation()],
        builder, stats)
    assert out is None


def test_an_unexpected_role_is_refused_before_anything_else():
    # A `system` turn would shift the rendered prefix and therefore the loss mask.
    builder = load_script("03_build_sft_data")
    stats = _stats()
    out = convert.reconstruct(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "p"},
         _assistant("t")], builder, stats)
    assert out is None and stats["drop_unexpected_role"] == 1


# ------------------------------------------------------------ signatures / wiring


def test_the_command_signature_reads_through_the_think_prefix():
    """The repo's own signature cannot be reused here.

    `03_build_sft_data.command_signature` does `json.loads(message["content"])`, which
    fails on content that opens with a think block -- silently, because it catches
    JSONDecodeError and moves on, so every trajectory would get the signature of an
    empty command list and dedup would collapse the corpus to one row per task.
    """
    messages = [{"role": "user", "content": "p"}, _assistant("t")]
    other = [{"role": "user", "content": "p"},
             _assistant("t", body='{"analysis":"a","plan":"p","commands":'
                                  '[{"keystrokes":"pwd\\n","duration":0.1}]}')]
    first = convert.command_signature(messages)
    assert first != convert.command_signature(other), "the keystrokes were not read"
    assert first == convert.command_signature(list(messages)), "not stable"


def test_large_source_files_are_split_across_workers():
    # math.parquet is 162,692 of the corpus's 366,154 rows in ONE row group, so
    # without splitting one worker does 44 % of the job while the rest idle.
    source = (ROOT / "scripts" / "03f_build_nemotron_sft.py").read_text(encoding="utf-8")
    assert "rows_per_part" in source
    assert "ordinal % parts != part" in source


def test_source_rows_are_counted_once_per_file_not_once_per_part():
    # Every part opens the same file, so `metadata.num_rows` is the whole file's count
    # in each of them. Summed blindly, a 14-part file would claim 14x its rows and the
    # manifest's drop rate would be meaningless.
    source = (ROOT / "scripts" / "03f_build_nemotron_sft.py").read_text(encoding="utf-8")
    assert "if part == 0 else 0" in source


def test_the_reasoning_decision_is_recorded_in_the_manifest():
    # "Why is only the last turn's reasoning supervised?" must be answerable from the
    # artifact, not from memory.
    source = (ROOT / "scripts" / "03f_build_nemotron_sft.py").read_text(encoding="utf-8")
    for evidence in ("kept_in_messages", "supervised_turns", "last_query_index"):
        assert evidence in source, evidence


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
