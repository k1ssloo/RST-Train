#!/usr/bin/env python3
"""Validate the built SFT parquet against slime's qwen3_5 loss-mask semantics.

Ports ``slime/utils/mask_utils.py::gen_multi_turn_loss_mask_qwen3_5`` verbatim so
we can confirm, before touching the cluster, that (a) the contract assertion
passes, (b) the trained-token span is exactly the assistant content, and (c) the
trained-token budget is what the LR/step schedule assumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ASSISTANT_HEADER = "<|im_start|>assistant\n"
THINK_PREFIX = "<think>\n"
END_MARKER = "<|im_end|>"


def qwen3_5_loss_mask(tokenizer, messages: list[dict]) -> tuple[list[int], list[int], str]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, return_dict=False)
    tokenized = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]

    expected = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    if token_ids != expected:
        raise ValueError("contract mismatch")

    char_mask = [0] * len(rendered)
    cursor = 0
    for message in messages:
        if message["role"] != "assistant":
            continue
        header = rendered.find(ASSISTANT_HEADER, cursor)
        if header < 0:
            raise ValueError("assistant header not found")
        content_start = header + len(ASSISTANT_HEADER)
        end = rendered.find(END_MARKER, content_start)
        if end < 0:
            raise ValueError("im_end not found")
        span_end = end + len(END_MARKER)
        if span_end < len(rendered) and rendered[span_end] == "\n":
            span_end += 1
        cursor = span_end
        if message.get("step_loss_mask", 1) != 1:
            continue
        start = content_start
        if rendered[content_start : content_start + len(THINK_PREFIX)] == THINK_PREFIX:
            start += len(THINK_PREFIX)
        for pos in range(start, span_end):
            char_mask[pos] = 1

    prefix = [0]
    for value in char_mask:
        prefix.append(prefix[-1] + value)
    loss_mask = [
        0 if end <= start else (1 if prefix[end] - prefix[start] > 0 else 0)
        for start, end in offsets
    ]
    return token_ids, loss_mask, rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--show", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    frame = pd.read_parquet(args.parquet)
    print(f"rows={len(frame)} columns={list(frame.columns)}")

    subset = frame.head(args.sample)
    total_tokens = 0
    trained_tokens = 0
    failures = 0
    leaked = 0

    for row in subset.itertuples():
        messages = [dict(m) for m in row.messages]
        try:
            ids, mask, rendered = qwen3_5_loss_mask(tokenizer, messages)
        except ValueError as exc:
            failures += 1
            print(f"  FAIL {row.trajectory_id}: {exc}")
            continue
        total_tokens += len(ids)
        trained_tokens += sum(mask)

        # Every trained character must live inside an assistant span.
        trained_text = "".join(
            rendered[s:e]
            for (s, e), m in zip(tokenizer(rendered, add_special_tokens=False,
                                           return_offsets_mapping=True)["offset_mapping"], mask)
            if m
        )
        if "<|im_start|>user" in trained_text:
            leaked += 1

    print(f"\n=== mask audit over {len(subset)} examples ===")
    print(f"  contract failures : {failures}")
    print(f"  user-turn leakage : {leaked}")
    print(f"  total tokens      : {total_tokens:,}")
    print(f"  trained tokens    : {trained_tokens:,}  ({trained_tokens/max(1,total_tokens):.2%})")

    scale = len(frame) / max(1, len(subset))
    print(f"  projected trained tokens for full split: {int(trained_tokens*scale):,}")

    lengths = frame.n_tokens.to_numpy()
    print(f"\n=== length distribution (full split) ===")
    for q in (0.5, 0.9, 0.95, 0.99):
        print(f"  p{int(q*100)}: {np.quantile(lengths, q):,.0f}")
    print(f"  max: {lengths.max():,}   total: {lengths.sum():,}")

    for row in frame.head(args.show).itertuples():
        messages = [dict(m) for m in row.messages]
        ids, mask, rendered = qwen3_5_loss_mask(tokenizer, messages)
        print("\n" + "=" * 78)
        print(f"EXAMPLE {row.trajectory_id}  tokens={len(ids)} trained={sum(mask)} "
              f"turns={row.n_assistant_turns} group={row.task_group_id}")
        print("=" * 78)
        offs = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        # Show the boundary region around the first two assistant turns.
        marks = []
        for (s, e), m in zip(offs, mask):
            marks.append((rendered[s:e], m))
        head = "".join(f"\033[92m{t}\033[0m" if m else t for t, m in marks[:180])
        print("--- first 180 tokens (green = trained) ---")
        print(head)
        print("\n--- LAST assistant turn (green = trained) ---")
        print("".join(f"\033[92m{t}\033[0m" if m else t for t, m in marks[-120:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
