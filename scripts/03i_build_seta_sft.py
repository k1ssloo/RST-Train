#!/usr/bin/env python3
"""Convert CAMEL-AI SETA's Kimi thinking trajectories to Qwen3.5 SFT messages.

    python scripts/03i_build_seta_sft.py \
        --source data/seta-sft/source-thinking/train.parquet \
        --source-info data/seta-sft/source-thinking/source.json \
        --tokenizer data/Qwen3.5-27B-tokenizer --out-dir data/seta-sft/sft-v1

Reconstruct raw_conv_json, never reuse the upstream Qwen3-8B ids or mask. Keep
only reward=1, normally completed conversations with valid tool arguments and
paired observations. Preserve reasoning and the six native SETA tools; this is
not a conversion to Terminus-2's analysis/plan/commands protocol. The published
snapshots omit tool schemas, so none are invented. See SETA.md and BUG-21.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sft_common import contract_length_gate, dedup_records, group_disjoint_split, token_stats
from siblings import load_script

native_tools = load_script("03e_build_tmax_sft")
exporter = load_script("15_export_pretokenized")

SOURCE_DATASET = "camel-ai/seta-sft-kimi-k2.5-thinking"
SOURCE_COLUMNS = ["task_id", "trial_uid", "reward", "model", "raw_conv_json"]
FORBIDDEN = native_tools.FORBIDDEN_IN_SOURCE + ("</function>", "</parameter>")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class Rejected(ValueError):
    """One source row cannot be converted without guessing or losing content."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_json(value: str) -> Any:
    return json.loads(value, object_pairs_hook=_object, parse_constant=_nonfinite)


def check_text(value: Any) -> None:
    """Source text, including tool arguments, must not forge template boundaries."""
    if isinstance(value, str):
        if any(marker in value for marker in FORBIDDEN):
            raise Rejected("control_markup")
    elif isinstance(value, dict):
        for key, item in value.items():
            check_text(key)
            check_text(item)
    elif isinstance(value, list):
        for item in value:
            check_text(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise Rejected("invalid_arguments")


def reconstruct(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict]]:
    """Recover one complete conversation and validate its ordered tool exchanges.

    Reject duplicate responses, malformed JSON, or truncated final calls rather
    than deleting turns or repairing code. No command from this data is executed.
    """
    try:
        request = raw["request"]
        choices = raw["response"]["choices"]
        if len(choices) != 1:
            raise Rejected("ambiguous_response")
        if choices[0].get("finish_reason") != "stop":
            raise Rejected("incomplete_finish")
        history = request["messages"]
        final = choices[0]["message"]
        if not isinstance(history, list) or not history:
            raise Rejected("invalid_messages")
        if (final.get("role") != "assistant" or final.get("tool_calls")
                or not isinstance(final.get("content"), str) or not final["content"].strip()):
            raise Rejected("incomplete_final_message")
        tools = request.get("tools") or []
        if not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools):
            raise Rejected("invalid_tools_schema")
        check_text(tools)
        messages = history + [final]
        if ([m["role"] for m in messages[:2]] != ["system", "user"]
                or sum(m["role"] == "system" for m in messages) != 1
                or sum(m["role"] == "user" for m in messages) != 1):
            raise Rejected("unexpected_user_or_system_turn")

        out: list[dict[str, Any]] = []
        pending: list[str] = []
        seen_calls: set[str] = set()
        previous = ""
        for message in messages:
            role = message["role"]
            content = message.get("content")
            if role not in {"system", "user", "assistant", "tool"}:
                raise Rejected("unexpected_role")
            if not isinstance(content, str):
                raise Rejected("nontext_content")
            if any(message.get(key) for key in ("audio", "refusal", "function_call")):
                raise Rejected("unsupported_message_field")
            clean: dict[str, Any] = {"role": role, "content": content}
            if role == "assistant":
                if pending:
                    raise Rejected("missing_tool_response")
                if previous not in {"user", "tool"}:
                    raise Rejected("unexpected_assistant_turn")
                reasoning = message.get("reasoning_content") or ""
                if not isinstance(reasoning, str):
                    raise Rejected("nontext_reasoning")
                clean["reasoning_content"] = reasoning
                calls = message.get("tool_calls") or []
                if not isinstance(calls, list):
                    raise Rejected("invalid_tool_calls")
                if not content.strip() and not calls:
                    raise Rejected("empty_assistant")
                normalized = []
                for call in calls:
                    call_id = call["id"]
                    function = call["function"]
                    name = function["name"]
                    if (call.get("type") != "function" or not isinstance(call_id, str)
                            or not call_id or not isinstance(name, str) or not NAME.fullmatch(name)):
                        raise Rejected("invalid_tool_call")
                    if call_id in seen_calls:
                        raise Rejected("duplicate_call_id")
                    arguments = function["arguments"]
                    if isinstance(arguments, str):
                        try:
                            arguments = strict_json(arguments)
                        except ValueError as exc:
                            raise Rejected("invalid_arguments") from exc
                    if not isinstance(arguments, dict) or any(
                        not isinstance(k, str) or not NAME.fullmatch(k) for k in arguments
                    ):
                        raise Rejected("invalid_arguments")
                    normalized.append({"id": call_id, "type": "function", "function": {
                        "name": name, "arguments": arguments,
                    }})
                    pending.append(call_id)
                    seen_calls.add(call_id)
                if normalized:
                    clean["tool_calls"] = normalized
            elif role == "tool":
                call_id = message.get("tool_call_id")
                if not pending or call_id != pending[0]:
                    raise Rejected("unpaired_or_out_of_order_tool_response")
                pending.pop(0)
                clean["tool_call_id"] = call_id
            elif message.get("tool_calls"):
                raise Rejected("unexpected_tool_calls")
            check_text(clean)
            out.append(clean)
            previous = role
        if pending:
            raise Rejected("missing_tool_response")
        return out, tools
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise Rejected("invalid_record_structure") from exc


def action_signature(messages: list[dict[str, Any]]) -> str:
    """Include every tool and its arguments, including file writes and waits."""
    actions = [call["function"] for message in messages
               for call in message.get("tool_calls", [])]
    return digest_json(actions)


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_provenance(source: Path, info_path: Path | None) -> dict[str, Any]:
    info = json.loads(info_path.read_text()) if info_path else {}
    actual = file_sha256(source)
    if info.get("sha256") and info["sha256"] != actual:
        raise ValueError("source SHA-256 does not match --source-info")
    if "size_bytes" in info and info["size_bytes"] != source.stat().st_size:
        raise ValueError("source size does not match --source-info")
    if info.get("dataset", SOURCE_DATASET) != SOURCE_DATASET:
        raise ValueError("this builder expects the SETA thinking dataset")
    return {"dataset": SOURCE_DATASET, "revision": info.get("revision"),
            "filename": info.get("filename", source.name), "sha256": actual,
            "size_bytes": source.stat().st_size, "license": "apache-2.0"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-info", type=Path, help="source revision, SHA-256 and size JSON")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--holdout", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1228)
    args = parser.parse_args()
    if args.max_seq_len <= 0 or args.holdout < 0:
        parser.error("--max-seq-len must be positive; --holdout must be nonnegative")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("--out-dir must be empty; use a new version directory")

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    provenance = source_provenance(args.source, args.source_info)
    rows = pq.read_table(args.source, columns=SOURCE_COLUMNS).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    if not tokenizer.is_fast:
        parser.error("a fast tokenizer is required for the loss mask")
    print(f"[source] rows={len(rows)} sha256={provenance['sha256']}", flush=True)

    stats: Counter = Counter()
    finishes: Counter = Counter()
    rewards: Counter = Counter()
    task_counts: Counter = Counter()
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if index % 100 == 0:
            print(f"[progress] {index}/{len(rows)} built={stats['built']}", flush=True)
        task_counts[row["task_id"]] += 1
        rewards[str(row["reward"])] += 1
        try:
            try:
                raw = strict_json(row["raw_conv_json"])
            except (ValueError, TypeError) as exc:
                raise Rejected("invalid_raw_json") from exc
            try:
                finishes[str(raw["response"]["choices"][0].get("finish_reason"))] += 1
                stats["source_rows_without_tool_schema"] += int(not raw["request"].get("tools"))
            except (KeyError, TypeError, AttributeError, IndexError) as exc:
                raise Rejected("invalid_record_structure") from exc
            if row["reward"] != 1.0:
                raise Rejected("reward_not_one")
            native, tools = reconstruct(raw)
            baked = native_tools.bake_system(tokenizer, native, tools)
            messages = native_tools.pre_render(native, baked)
            want = tokenizer.apply_chat_template(native, tools=tools, tokenize=False,
                                                  return_dict=False)
            got = tokenizer.apply_chat_template(messages, tokenize=False, return_dict=False)
            if got != want:
                raise Rejected("prerender_mismatch")
            n_tokens, reason = contract_length_gate(tokenizer, messages, args.max_seq_len)
            if reason:
                raise Rejected(reason)
            try:
                ids, mask = exporter.qwen3_5_mask(tokenizer, messages)
            except ValueError as exc:
                raise Rejected("mask_failed") from exc
            if not mask or len(ids) != len(mask) or len(ids) != n_tokens or mask[0] != 0:
                raise Rejected("invalid_mask")
            if not sum(mask):
                raise Rejected("no_trained_tokens")
        except Rejected as exc:
            reason = str(exc)
            stats[f"drop_{reason}"] += 1
            rejected.append({"trajectory_id": row["trial_uid"],
                             "task_group_id": row["task_id"], "reason": reason})
            continue
        records.append({
            "messages": messages, "trajectory_id": row["trial_uid"],
            "task_group_id": row["task_id"], "model_name": row["model"],
            "reward": float(row["reward"]), "n_tokens": n_tokens,
            "n_trained_tokens": sum(mask),
            "n_assistant_turns": sum(m["role"] == "assistant" for m in native),
            "tool_names": sorted({c["function"]["name"] for m in native
                                  for c in m.get("tool_calls", [])}),
            "tool_schema_present": bool(tools),
            "command_signature": action_signature(native),
            "content_hash": digest_json(messages), "prompt_hash": digest_json(native[1]),
        })
        stats["built"] += 1

    kept, cross_task = dedup_records(records, stats)
    if not kept:
        raise SystemExit(f"no usable rows: {dict(stats)}")
    train, held, held_tasks, target = group_disjoint_split(
        kept, holdout=args.holdout, seed=args.seed, max_fraction=0.2,
    )
    if ({r["task_group_id"] for r in train} & held_tasks
            or {r["prompt_hash"] for r in train} & {r["prompt_hash"] for r in held}):
        raise SystemExit("task or identical prompt overlaps train and holdout")

    schema = pa.schema([
        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("trajectory_id", pa.string()), ("task_group_id", pa.string()),
        ("model_name", pa.string()), ("reward", pa.float64()),
        ("n_tokens", pa.int64()), ("n_trained_tokens", pa.int64()),
        ("n_assistant_turns", pa.int64()), ("tool_names", pa.list_(pa.string())),
        ("tool_schema_present", pa.bool_()),
    ])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, split in (("train", train), ("holdout", held)):
        path = args.out_dir / f"seta_sft_{name}.parquet"
        pq.write_table(pa.Table.from_pylist(split, schema=schema), path)
        outputs[name] = {"path": str(path), "rows": len(split), "sha256": file_sha256(path)}
        print(f"[write] {path} rows={len(split)}", flush=True)
    (args.out_dir / "rejected_rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected), encoding="utf-8",
    )
    manifest = {
        "source": provenance, "source_rows": len(rows), "source_tasks": len(task_counts),
        "source_tasks_with_multiple_rollouts": sum(n > 1 for n in task_counts.values()),
        "source_reward_counts": dict(rewards), "source_finish_reason_counts": dict(finishes),
        "stats": dict(stats), "after_dedup": len(kept),
        "train_examples": len(train), "holdout_examples": len(held),
        "holdout_tasks": len(held_tasks), "holdout_requested": args.holdout,
        "holdout_target": target, "seed": args.seed,
        "split": "whole task_id groups; identical cross-split prompts also forbidden",
        "command_signatures_shared_across_tasks": cross_task,
        "max_seq_len": args.max_seq_len, "tokenizer": str(args.tokenizer),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "tokenizer_files_sha256": {p.name: file_sha256(p) for p in
                                   sorted(args.tokenizer.glob("*.json"))},
        "tokens": token_stats([r["n_tokens"] for r in kept],
                              [r["n_trained_tokens"] for r in kept]),
        "tool_names": sorted({name for r in kept for name in r["tool_names"]}),
        "tool_schema_policy": "preserve request.tools if present; never infer missing schemas",
        "protocol": "native SETA tool calls rendered with Qwen3.5; not Terminus-2 JSON",
        "gates": ["reward == 1", "finish_reason == stop", "strict JSON tool arguments",
                  "one ordered observation per call", "no source control markup",
                  "native render == flattened render", "template tokenization contract",
                  "length <= max_seq_len without truncation", "canonical assistant-only mask"],
        "repairs": {}, "outputs": outputs,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({"stats": dict(stats), "tokens": manifest["tokens"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
