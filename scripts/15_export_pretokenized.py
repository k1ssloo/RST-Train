#!/usr/bin/env python3
"""Export pre-tokenized SFT data: `input_ids` + `loss_mask`, backend-agnostic.

    python scripts/15_export_pretokenized.py \
        --parquet   $BASE_FOLDER/sft-v1-cap10/rst_sft_train.parquet \
        --tokenizer $BASE_FOLDER/Qwen3.5-27B \
        --out       $BASE_FOLDER/sft-v1-cap10/pretokenized_train.parquet

WHY THIS EXISTS
---------------
The loss mask is the one thing in this pipeline that is easy to get wrong and
impossible to notice afterwards: a wrong mask trains on terminal output and the
harness prompt, the loss still goes down, and the model just comes out worse. We
verified slime's `qwen3_5` mask token-by-token (0 contract failures, 0 user-turn
leakage, 32.6 % of tokens trained). Baking that verified mask into the data makes
it portable: any trainer that accepts `input_ids` + a per-token mask reproduces
the exact same training target, with no second implementation to get wrong.

This matters concretely for verl. Its `sft_trainer_engine.yaml` warns:

    MultiTurnSFTDataset apply_chat_template to each turn separately and concat
    `input_ids` as a whole sequence, which may not equal to apply_chat_template
    to whole messages at once. For example, Qwen Thinking series models add
    <think></think> tags to last turn ...

That is exactly the Qwen3.5 behaviour we measured: the template injects
`<think>\\n\\n</think>\\n\\n` before the FINAL assistant turn only, so per-turn
tokenize-and-concat does not equal the single whole-conversation render. verl's
escape hatch is `ignore_input_ids_mismatch: True`, which silences the check
rather than fixing the sequence. Feeding pre-tokenized data through verl's
`data.custom_cls` avoids the whole problem.

The mask implementation here is a straight port of
`slime/utils/mask_utils.py::gen_multi_turn_loss_mask_qwen3_5`, and the script
re-asserts slime's own contract (render-then-tokenize == tokenize-directly) on
every row, so a divergence is a hard error rather than a silent difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ASSISTANT_HEADER = "<|im_start|>assistant\n"
THINK_PREFIX = "<think>\n"
END_MARKER = "<|im_end|>"


def qwen3_5_mask(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Port of slime's gen_multi_turn_loss_mask_qwen3_5. Raises on contract failure."""
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, return_dict=False)
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    token_ids, offsets = enc["input_ids"], enc["offset_mapping"]

    expected = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    if token_ids != expected:
        raise ValueError("chat-template contract mismatch (render vs direct tokenize)")

    char_mask = [0] * len(rendered)
    cursor = 0
    for message in messages:
        if message["role"] != "assistant":
            continue
        header = rendered.find(ASSISTANT_HEADER, cursor)
        if header < 0:
            raise ValueError("assistant header not found in rendered text")
        content_start = header + len(ASSISTANT_HEADER)
        end = rendered.find(END_MARKER, content_start)
        if end < 0:
            raise ValueError("<|im_end|> not found for an assistant message")
        span_end = end + len(END_MARKER)
        if span_end < len(rendered) and rendered[span_end] == "\n":
            span_end += 1
        cursor = span_end
        if message.get("step_loss_mask", 1) != 1:
            continue
        start = content_start
        # The template opens the final assistant turn with "<think>\n", which is
        # part of the PROMPT (the model is asked to continue from it), so it is not
        # a training target. Everything after -- including "\n</think>\n\n", the
        # content, and <|im_end|> -- is.
        if rendered[content_start : content_start + len(THINK_PREFIX)] == THINK_PREFIX:
            start += len(THINK_PREFIX)
        for pos in range(start, span_end):
            char_mask[pos] = 1

    prefix = [0]
    for value in char_mask:
        prefix.append(prefix[-1] + value)
    loss_mask = [
        0 if e <= s else (1 if prefix[e] - prefix[s] > 0 else 0) for s, e in offsets
    ]
    return token_ids, loss_mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, required=True, help="messages parquet from 03_build_sft_data.py")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--strict", action="store_true",
                    help="abort on the first bad row instead of dropping it")
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    if not tokenizer.is_fast:
        sys.exit("a fast tokenizer is required (offset mapping)")
    frame = pd.read_parquet(args.parquet)
    print(f"[in] {args.parquet}  rows={len(frame)}")

    ids_col: list[list[int]] = []
    mask_col: list[list[int]] = []
    keep_idx: list[int] = []
    dropped = {"contract": 0, "too_long": 0, "no_trained_tokens": 0,
               "supervised_first_token": 0, "error": 0}
    trained_total = 0
    token_total = 0

    for i, row in enumerate(frame.itertuples()):
        messages = [dict(m) for m in row.messages]
        try:
            ids, mask = qwen3_5_mask(tokenizer, messages)
        except ValueError as exc:
            dropped["contract" if "contract" in str(exc) else "error"] += 1
            if args.strict:
                sys.exit(f"row {i}: {exc}")
            continue
        if len(ids) != len(mask):
            dropped["error"] += 1
            continue
        if len(ids) > args.max_seq_len:
            dropped["too_long"] += 1
            continue
        if sum(mask) == 0:
            # Nothing to learn from; would contribute a zero gradient and, with
            # per-token loss normalization, silently skew the average.
            dropped["no_trained_tokens"] += 1
            continue
        if mask[0] != 0:
            # Downstream packing depends on this. verl's `sft_loss` aligns the mask to the
            # log-probs with a cyclic `torch.roll(..., -1)` over the whole packed
            # micro-batch, so a supervised token 0 leaks supervision onto the PREVIOUS
            # document's last hidden state. Every row built from the Qwen3.5 template opens
            # on `<|im_start|>` and cannot hit this; if it ever does, the mask is wrong in a
            # way no loss curve would show, so drop the row rather than ship it.
            dropped["supervised_first_token"] += 1
            if args.strict:
                sys.exit(f"row {i}: loss_mask[0] == 1, but the first token of a conversation "
                         f"is never a training target. The mask is misaligned.")
            continue
        ids_col.append(ids)
        mask_col.append(mask)
        keep_idx.append(i)
        trained_total += sum(mask)
        token_total += len(ids)

    kept = frame.iloc[keep_idx].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "input_ids": ids_col,
            "loss_mask": mask_col,
            "n_tokens": [len(x) for x in ids_col],
            "n_trained_tokens": [sum(m) for m in mask_col],
            "trajectory_id": kept.get("trajectory_id"),
            "task_group_id": kept.get("task_group_id"),
            "model_name": kept.get("model_name"),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    manifest = {
        "source_parquet": str(args.parquet),
        "tokenizer": str(args.tokenizer),
        "rows_in": int(len(frame)),
        "rows_out": int(len(out)),
        "dropped": dropped,
        "total_tokens": int(token_total),
        "trained_tokens": int(trained_total),
        "trained_fraction": round(trained_total / max(1, token_total), 4),
        "max_seq_len": args.max_seq_len,
        "mask_source": "slime/utils/mask_utils.py::gen_multi_turn_loss_mask_qwen3_5 (ported)",
        "schema": {
            "input_ids": "list[int] — the exact tokens, whole-conversation render",
            "loss_mask": "list[int] — 1 = train on this token, 0 = context only",
        },
        "note": "loss_mask is aligned 1:1 with input_ids: mask[i] refers to token i, "
                "with no offset applied. To get `labels`, set "
                "labels[i] = input_ids[i] where loss_mask[i]==1 else -100, and do NOT "
                "shift — every HuggingFace CausalLM (and Liger's fused CE) shifts "
                "internally, so shifting here misaligns the supervision by one token. "
                "Shift only if you hand-write the cross-entropy against unshifted "
                "logits, in which case shift logits and labels together as usual.",
    }
    (args.out.parent / (args.out.stem + "_manifest.json")).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    if dropped["contract"]:
        print(f"\nWARNING: {dropped['contract']} rows failed the chat-template contract. "
              f"That should be 0 for data built by 03_build_sft_data.py — investigate "
              f"before training.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
