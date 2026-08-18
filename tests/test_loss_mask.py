"""The loss mask is the one thing in this pipeline that fails silently.

There are two ports of slime's `gen_multi_turn_loss_mask_qwen3_5` in this repo, on
purpose:

  scripts/15_export_pretokenized.py::qwen3_5_mask       the PRODUCER (also imported
                                                        by 17_build_dpo_data.py)
  scripts/03b_validate_sft_data.py::qwen3_5_loss_mask   the independent AUDITOR

Collapsing them into one function would make the auditor audit itself. Keeping
them apart is only safe if something checks that they still agree -- which is what
this file is. `test_two_ports_agree_*` is the lock: any edit to one port that
changes behaviour fails here until the other is edited too.

The rest of the file pins the *semantics* both ports must have, so that "they
agree" cannot be satisfied by two identically-broken copies: nothing before the
first assistant header is trained, no header token is ever trained, the prompt's
`<think>\\n` opener is not a target while the content after it is, and
`step_loss_mask=0` turns are excluded.

These tests run on a synthetic tokenizer (see FakeQwenTokenizer), so they need
neither transformers nor the 27 B tokenizer. Real-tokenizer verification is a
separate, heavier thing: `scripts/03b_validate_sft_data.py --sample 300`, whose
result is quoted in README.md.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import load_script, need  # noqa: E402

# Every special token is atomic in the real Qwen tokenizer, and `<` ends a plain
# run so that "content<|im_end|>" cannot be swallowed into one token.
TOKEN_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|<think>|</think>|\s+|[^\s<]+|<")
THINK_OPENER = "<think>\n\n</think>\n\n"


class FakeQwenTokenizer:
    """A stand-in with the two behaviours the mask ports actually depend on.

    1. `apply_chat_template` renders `<|im_start|>{role}\\n{content}<|im_end|>\\n`
       per message and injects `<think>\\n\\n</think>\\n\\n` at the start of the
       FINAL assistant turn only -- the Qwen3.5 quirk the whole mask design is
       about, and the reason per-turn tokenize-and-concat does not reproduce the
       sequence.
    2. Tokenizing the rendered string returns character offsets, with special
       tokens atomic.

    Ids are arbitrary but stable within an instance; no test asserts their values,
    only that the two ports assign the same mask to the same offsets.
    """

    is_fast = True

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def render(self, messages: list[dict]) -> str:
        last_assistant = max(
            (i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1
        )
        parts = []
        for index, message in enumerate(messages):
            content = message["content"]
            if index == last_assistant:
                content = THINK_OPENER + content
            parts.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>\n")
        return "".join(parts)

    def _split(self, text: str) -> list[tuple[str, int, int]]:
        return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]

    def _id(self, token: str) -> int:
        return self._vocab.setdefault(token, 1000 + len(self._vocab))

    def apply_chat_template(self, messages, tokenize=False, return_dict=False,
                            add_generation_prompt=False):
        rendered = self.render(messages)
        if not tokenize:
            return rendered
        return [self._id(tok) for tok, _, _ in self._split(rendered)]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        pieces = self._split(text)
        out: dict = {"input_ids": [self._id(tok) for tok, _, _ in pieces]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(start, end) for _, start, end in pieces]
        return out


class CharTokenizer(FakeQwenTokenizer):
    """One token per character. Used only by the equivalence lock.

    Both ports build a *character* mask and then mark a token trained if any of its
    characters is. With realistic multi-character tokens that last step hides
    char-level disagreements: change `THINK_PREFIX` in one port from `"<think>\\n"`
    to `"<think>"` and the boundary moves by exactly one newline, which sits inside
    the same `"\\n\\n"` token and so produces an identical mask. Tokenizing per
    character removes that blind spot, so the lock fails on any divergence at all.
    """

    def _split(self, text: str) -> list[tuple[str, int, int]]:
        return [(char, index, index + 1) for index, char in enumerate(text)]


class DriftingTokenizer(FakeQwenTokenizer):
    """`apply_chat_template(tokenize=True)` disagrees with render-then-tokenize.

    This is the failure both ports are supposed to refuse rather than mask around:
    if the two paths disagree, the character offsets do not describe the sequence
    the trainer will see, so the mask is meaningless.
    """

    def apply_chat_template(self, messages, tokenize=False, return_dict=False,
                            add_generation_prompt=False):
        result = super().apply_chat_template(
            messages, tokenize=tokenize, return_dict=return_dict,
            add_generation_prompt=add_generation_prompt)
        return [*result, 999_999] if tokenize else result


def ports():
    """(producer, auditor) mask functions, normalized to (ids, mask)."""
    need("transformers")   # 03b imports it at module level
    producer = load_script("15_export_pretokenized").qwen3_5_mask
    auditor = load_script("03b_validate_sft_data").qwen3_5_loss_mask
    return producer, lambda tok, msgs: auditor(tok, msgs)[:2]


def trained_spans(tokenizer, messages, mask) -> list[tuple[int, int]]:
    rendered = tokenizer.render(messages)
    offsets = tokenizer(rendered, return_offsets_mapping=True)["offset_mapping"]
    return [span for span, flag in zip(offsets, mask) if flag]


def trained_tokens(tokenizer, messages, mask) -> list[str]:
    rendered = tokenizer.render(messages)
    return [rendered[s:e] for s, e in trained_spans(tokenizer, messages, mask)]


CONVERSATIONS = {
    "single_turn": [
        {"role": "user", "content": "USERONE list the files"},
        {"role": "assistant", "content": "ASSISTONE ls -la"},
    ],
    "with_system": [
        {"role": "system", "content": "SYSONE you are a terminal agent"},
        {"role": "user", "content": "USERONE list the files"},
        {"role": "assistant", "content": "ASSISTONE ls -la"},
    ],
    "multi_turn": [
        {"role": "system", "content": "SYSONE you are a terminal agent"},
        {"role": "user", "content": "USERONE find the config"},
        {"role": "assistant", "content": "ASSISTONE grep -r config ."},
        {"role": "user", "content": "USERTWO here is the output\nline2"},
        {"role": "assistant", "content": "ASSISTTWO cat config.yaml"},
        {"role": "user", "content": "USERTHREE ok"},
        {"role": "assistant", "content": "ASSISTTHREE done\n\nsummary: ok"},
    ],
    "masked_middle_turn": [
        {"role": "user", "content": "USERONE find the config"},
        {"role": "assistant", "content": "ASSISTONE grep -r config .", "step_loss_mask": 0},
        {"role": "user", "content": "USERTWO here is the output"},
        {"role": "assistant", "content": "ASSISTTWO cat config.yaml"},
    ],
    "content_mentions_markup": [
        {"role": "user", "content": "USERONE the log says <think> and <|im_end|>-ish"},
        {"role": "assistant", "content": "ASSISTONE noted, <think> is just text here"},
    ],
    "empty_assistant_content": [
        {"role": "user", "content": "USERONE anything?"},
        {"role": "assistant", "content": ""},
    ],
}


def test_two_ports_agree_on_every_conversation():
    """The C2 lock: producer and auditor must be behaviourally identical.

    Run against both tokenizations: the realistic one (atomic specials) and the
    per-character one, which is what makes the lock sensitive to a boundary moving
    by a single character.
    """
    producer, auditor = ports()
    for kind in (FakeQwenTokenizer, CharTokenizer):
        for name, messages in CONVERSATIONS.items():
            p_ids, p_mask = producer(kind(), messages)
            a_ids, a_mask = auditor(kind(), messages)
            where = f"{name}/{kind.__name__}"
            assert p_ids == a_ids, f"{where}: token ids differ between the two ports"
            assert p_mask == a_mask, f"{where}: LOSS MASKS DIFFER between the two ports"
            assert len(p_mask) == len(p_ids), f"{where}: mask/ids length mismatch"


def test_two_ports_agree_that_a_template_contract_break_is_fatal():
    producer, auditor = ports()
    messages = CONVERSATIONS["single_turn"]
    for label, func in (("producer", producer), ("auditor", auditor)):
        try:
            func(DriftingTokenizer(), messages)
        except ValueError:
            continue
        raise AssertionError(f"{label} accepted a chat-template contract mismatch")


def test_nothing_before_the_first_assistant_header_is_trained():
    producer, _ = ports()
    for name, messages in CONVERSATIONS.items():
        tokenizer = FakeQwenTokenizer()
        rendered = tokenizer.render(messages)
        _, mask = producer(tokenizer, messages)
        first_header = rendered.index("<|im_start|>assistant\n")
        for start, end in trained_spans(tokenizer, messages, mask):
            assert start >= first_header, f"{name}: prompt char {start} is a training target"


def test_no_role_header_is_ever_trained():
    """`<|im_start|>` and the role word are prompt scaffolding, never targets."""
    producer, _ = ports()
    for name, messages in CONVERSATIONS.items():
        tokenizer = FakeQwenTokenizer()
        _, mask = producer(tokenizer, messages)
        text = "".join(trained_tokens(tokenizer, messages, mask))
        assert "<|im_start|>" not in text, f"{name}: a role header is being trained on"
        for role in ("user", "system"):
            assert f"{role}\n" not in text, f"{name}: a {role} header leaked into the target"


def test_user_content_never_leaks_into_the_target():
    producer, _ = ports()
    for name, messages in CONVERSATIONS.items():
        tokenizer = FakeQwenTokenizer()
        _, mask = producer(tokenizer, messages)
        text = "".join(trained_tokens(tokenizer, messages, mask))
        for marker in ("USERONE", "USERTWO", "USERTHREE", "SYSONE"):
            assert marker not in text, f"{name}: {marker} (a non-assistant turn) is trained"


def test_think_opener_is_prompt_and_the_rest_of_the_turn_is_target():
    """The final turn opens with `<think>\\n`, which the model is asked to continue.

    So `<think>` itself must not be a target, while `</think>`, the content and the
    closing `<|im_end|>` must be -- that is the exact contract 03b measured against
    the real tokenizer (32.6 % trained).
    """
    producer, _ = ports()
    tokenizer = FakeQwenTokenizer()
    messages = CONVERSATIONS["multi_turn"]
    _, mask = producer(tokenizer, messages)
    tokens = trained_tokens(tokenizer, messages, mask)
    assert "<think>" not in tokens, "the prompt's <think> opener is being trained on"
    assert "</think>" in tokens, "the closing </think> should be part of the target"
    assert "<|im_end|>" in tokens, "the turn terminator must be trained or the model "\
                                   "never learns to stop"
    assert "ASSISTTHREE" in "".join(tokens)


def test_step_loss_mask_zero_excludes_exactly_that_turn():
    producer, auditor = ports()
    tokenizer = FakeQwenTokenizer()
    messages = CONVERSATIONS["masked_middle_turn"]
    _, mask = producer(tokenizer, messages)
    text = "".join(trained_tokens(tokenizer, messages, mask))
    assert "ASSISTONE" not in text, "step_loss_mask=0 turn was trained on anyway"
    assert "ASSISTTWO" in text, "the unmasked turn was dropped as well"
    # and the auditor must reach the same verdict, not just the same total
    _, other = auditor(FakeQwenTokenizer(), messages)
    assert other == mask


def test_every_assistant_turn_is_trained_when_nothing_is_masked():
    producer, _ = ports()
    tokenizer = FakeQwenTokenizer()
    messages = CONVERSATIONS["multi_turn"]
    _, mask = producer(tokenizer, messages)
    text = "".join(trained_tokens(tokenizer, messages, mask))
    for marker in ("ASSISTONE", "ASSISTTWO", "ASSISTTHREE"):
        assert marker in text, f"{marker} is an assistant turn and must be a target"
    assert 0 < sum(mask) < len(mask), "a mask that is all-0 or all-1 is a bug, not a mask"


if __name__ == "__main__":  # allow `python tests/test_loss_mask.py`
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
