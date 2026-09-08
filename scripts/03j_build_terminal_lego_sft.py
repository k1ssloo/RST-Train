#!/usr/bin/env python3
"""Download released Terminal-Lego trajectories and build reward-filtered SFT.

    python scripts/03j_build_terminal_lego_sft.py --release deepseek-15k --download \
        --tokenizer data/Qwen3.5-27B-tokenizer \
        --out-dir data/terminal-lego-trajectories/deepseek-sft-v1

    python scripts/03j_build_terminal_lego_sft.py --release opus-8k \
        --download --download-only

The 8k release lacks per-trajectory rewards: oracle_passed_task identifies an
environment, not a successful model rollout. Never synthesize reward=1. The
DeepSeek release reuses task directory numbers across different task batches;
group by the actual instruction, not the basename or the separate task pool.
Shared RST normalizer, dedup, template, length and split gates apply. BUG-23.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sft_common import contract_length_gate, dedup_records, group_disjoint_split, token_stats
from siblings import load_script

conversation_builder = load_script("03d_build_openthoughts_sft")
rst_builder = conversation_builder.load_builder()
FORBIDDEN = load_script("03e_build_tmax_sft").FORBIDDEN_IN_SOURCE

SOURCES = {
    "deepseek-15k": {
        "dataset": "Lego-X/Terminal-Lego-Traj-Deepseek-V3-2-15k",
        "revision": "922c2fcb019854911fc05eb1a208d2685ac6e006",
        "filename": "terminal-lego-deepseek-v3-2-15k.json",
        "sha256": "af729a6e8cb3c78a5bb8eb315bbc88b8a197fab8ea8082bc69d466bbd7b77b5b",
        "size_bytes": 540497982,
        "model_name": "deepseek-v3.2",
    },
    "opus-8k": {
        "dataset": "Lego-X/Terminal-Lego-Traj-8k",
        "revision": "02f33fb252b15308f309e7d10472bb781eeb1ecf",
        "filename": "terminal-lego-opus-4-6-8k.json",
        "sha256": "925da53ae7686e22320c4ac9e506b43ad1c8a55979f2f8bf500283a686a24c10",
        "size_bytes": 168615262,
        "model_name": "claude-opus-4.6",
    },
}
TASK_START = "Task Description:\n"
TERMINAL_START = "\nCurrent terminal state:\n"


class Rejected(ValueError):
    """A trajectory cannot pass a data gate without guessing or truncation."""


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_source(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing source {path}; use --download")
    if path.stat().st_size != spec["size_bytes"] or file_sha256(path) != spec["sha256"]:
        raise ValueError("source SHA-256/size does not match the pinned release")
    return {**spec, "path": str(path.resolve()), "model_name_source": "release filename",
            "license": None}


def download_source(path: Path, spec: dict[str, Any]) -> None:
    """Check existing bytes; publish a new download only after size/hash checks."""
    if path.exists():
        check_source(path, spec)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://huggingface.co/datasets/{spec['dataset']}/resolve/"
           f"{spec['revision']}/{spec['filename']}")
    temporary = path.with_suffix(path.suffix + ".part")
    total = 0
    next_report = 64 * 1024 * 1024
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
            total += len(chunk)
            if total >= next_report:
                print(f"[download] {total / 1e6:.1f}/{spec['size_bytes'] / 1e6:.1f} MB",
                      flush=True)
                next_report += 64 * 1024 * 1024
    check_source(temporary, spec)
    temporary.rename(path)


def task_prompt(prompt: str) -> str:
    """Remove only the fixed harness prefix and initial terminal screen for grouping.

    Training messages keep the complete original prompt. The terminal's container
    ID must not separate repeated demonstrations of the same instruction.
    """
    if TASK_START not in prompt:
        raise Rejected("missing_task_prompt_boundary")
    body = prompt.split(TASK_START, 1)[1]
    if TERMINAL_START not in body:
        raise Rejected("missing_task_prompt_boundary")
    body = body.rsplit(TERMINAL_START, 1)[0].strip()
    if not body:
        raise Rejected("empty_task_prompt")
    return body


def validate_action(action: dict[str, Any]) -> None:
    if (not isinstance(action.get("analysis"), str)
            or not isinstance(action.get("plan"), str)
            or not isinstance(action.get("commands"), list)):
        raise Rejected("invalid_action_schema")
    if "task_complete" in action and not isinstance(action["task_complete"], bool):
        raise Rejected("invalid_completion_flag")
    for command in action["commands"]:
        if not isinstance(command, dict) or not isinstance(command.get("keystrokes"), str):
            raise Rejected("invalid_command_schema")
        duration = command.get("duration", 1.0)
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration < 0):
            raise Rejected("invalid_command_duration")


def convert_row(row: dict[str, Any], index: int, release: str,
                spec: dict[str, Any], stats: Counter) -> dict[str, Any]:
    if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
        raise Rejected("invalid_metadata")
    metadata = row["metadata"]
    if "reward" not in metadata:
        raise Rejected("missing_trajectory_reward")
    reward = metadata["reward"]
    if (isinstance(reward, bool) or not isinstance(reward, (int, float))
            or not math.isfinite(reward) or reward not in (0, 1)):
        raise Rejected("invalid_reward")
    if reward != 1:
        raise Rejected("reward_not_one")
    if any(not isinstance(metadata.get(k), str) or not metadata[k].strip()
           for k in ("task_path", "task_name", "source")):
        raise Rejected("missing_trial_provenance")

    conversation = row.get("conversations")
    if not isinstance(conversation, list) or len(conversation) < 2:
        raise Rejected("empty_conversation")
    if len(conversation) % 2:
        raise Rejected("incomplete_conversation")
    mapped = []
    for position, turn in enumerate(conversation):
        role = "human" if position % 2 == 0 else "gpt"
        if not isinstance(turn, dict) or turn.get("from") != role:
            raise Rejected("nonalternating_roles")
        text = turn.get("value")
        if not isinstance(text, str) or not text.strip():
            raise Rejected("empty_or_nontext_turn")
        if any(marker in text for marker in FORBIDDEN):
            raise Rejected("source_control_markup")
        mapped.append({"role": "user" if role == "human" else "assistant", "content": text})

    body = task_prompt(mapped[0]["content"])
    prompt_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    local_stats: Counter = Counter()
    record = conversation_builder.reconstruct(mapped, rst_builder, local_stats)
    if record is None:
        reason = next(k.removeprefix("drop_") for k in local_stats if k.startswith("drop_"))
        raise Rejected(reason)
    for message in record["messages"]:
        if message["role"] == "assistant":
            validate_action(json.loads(message["content"]))
    if json.loads(record["messages"][-1]["content"]).get("task_complete") is not True:
        raise Rejected("incomplete_final_action")
    stats.update(local_stats)
    return {
        **record, "trajectory_id": f"terminal_lego_{release}_{index:05d}",
        "task_group_id": f"terminal_lego_prompt_{prompt_hash}", "prompt_hash": prompt_hash,
        "model_name": spec["model_name"], "reward": float(reward), "source_row": index,
        "source_trial_id": metadata["task_name"], "source_task_path": metadata["task_path"],
        "source_run": metadata["source"],
    }


def build(args: argparse.Namespace, *, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec or SOURCES[args.release]
    if args.out_dir.exists() and (not args.out_dir.is_dir() or any(args.out_dir.iterdir())):
        raise ValueError("out-dir must be new or empty")
    if args.max_seq_len <= 0 or args.holdout < 0:
        raise ValueError("max-seq-len must be positive and holdout nonnegative")
    if args.download:
        download_source(args.source, spec)
    provenance = check_source(args.source, spec)
    rows = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("source must be a JSON array of trajectories")

    stats: Counter = Counter()
    rewards: Counter = Counter()
    records = []
    rejected = []
    for index, row in enumerate(rows):
        metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
        rewards[str(metadata.get("reward", "missing")) if isinstance(metadata, dict)
                else "missing"] += 1
        try:
            records.append(convert_row(row, index, args.release, spec, stats))
        except Rejected as exc:
            reason = str(exc)
            stats[f"drop_{reason}"] += 1
            rejected.append({"source_row": index, "reason": reason})
    del rows
    print(f"[source] {sum(rewards.values())} rows; reward={dict(rewards)}; "
          f"reconstructed={len(records)}", flush=True)
    kept, cross_task = dedup_records(records, stats)
    kept_ids = {r["trajectory_id"] for r in kept}
    rejected.extend({"source_row": r["source_row"], "reason": "duplicate"}
                    for r in records if r["trajectory_id"] not in kept_ids)
    if not kept:
        raise ValueError(f"no reward-validated complete trajectories: {dict(stats)}")

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer is required for assistant-only loss masks")
    final = []
    for index, record in enumerate(kept, 1):
        n_tokens, reason = contract_length_gate(tokenizer, record["messages"], args.max_seq_len)
        if reason:
            stats[f"drop_{reason}"] += 1
            rejected.append({"source_row": record["source_row"], "reason": reason})
        else:
            record["n_tokens"] = n_tokens
            final.append(record)
        if index % 500 == 0 or index == len(kept):
            print(f"[tokenize] {index}/{len(kept)} kept={len(final)}", flush=True)
    if not final:
        raise ValueError("no trajectories passed the tokenizer and length gates")

    train, held, held_groups, target = group_disjoint_split(
        final, holdout=args.holdout, seed=args.seed, max_fraction=0.2,
    )
    if ({r["source_task_path"] for r in train} & {r["source_task_path"] for r in held}
            or {r["prompt_hash"] for r in train} & {r["prompt_hash"] for r in held}):
        raise ValueError("source task or identical instruction overlaps train and holdout")
    schema = pa.schema([
        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("trajectory_id", pa.string()), ("task_group_id", pa.string()),
        ("model_name", pa.string()), ("reward", pa.float64()),
        ("n_tokens", pa.int64()), ("n_assistant_turns", pa.int64()),
        ("n_rewritten_turns", pa.int64()), ("source_row", pa.int64()),
        ("source_trial_id", pa.string()), ("source_task_path", pa.string()),
        ("source_run", pa.string()), ("prompt_hash", pa.string()),
    ])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, split in (("train", train), ("holdout", held)):
        path = args.out_dir / f"terminal_lego_sft_{name}.parquet"
        pq.write_table(pa.Table.from_pylist(split, schema=schema), path)
        outputs[name] = {"path": str(path), "rows": len(split), "sha256": file_sha256(path)}
        print(f"[write] {path} rows={len(split)}", flush=True)
    (args.out_dir / "rejected_rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in sorted(rejected, key=lambda r: r["source_row"])),
        encoding="utf-8",
    )
    manifest = {
        "source": provenance, "source_rows": sum(rewards.values()),
        "source_reward_counts": dict(rewards), "reconstructed": len(records),
        "after_dedup": len(kept), "final_examples": len(final),
        "train_examples": len(train), "holdout_examples": len(held),
        "groups_covered": len({r["task_group_id"] for r in final}),
        "holdout_groups": sorted(held_groups), "holdout_requested": args.holdout,
        "holdout_target": target, "seed": args.seed, "stats": dict(stats),
        "split_policy": "SHA-256 of instruction between Task Description and terminal screen; "
                        "source task paths also checked for cross-split overlap",
        "task_pool_id_mapping_verified": False, "benchmark_overlap_checked": False,
        "command_signatures_shared_across_tasks": cross_task,
        "max_seq_len": args.max_seq_len, "tokenizer": str(args.tokenizer),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "tokenizer_files_sha256": {p.name: file_sha256(p) for p in
                                   sorted(args.tokenizer.glob("*.json"))},
        "token_stats": token_stats([r["n_tokens"] for r in final]),
        "n_rewritten_turns": sum(r["n_rewritten_turns"] for r in final),
        "protocol": "Terminus-2 analysis/plan/commands JSON; shared RST canonicalizer",
        "reward_policy": "source metadata.reward == 1; not inferred from oracle or task_complete",
        "validation_scope": "source integrity and SFT structure; upstream rewards not replayed locally",
        "local_model_rollouts": 0, "outputs": outputs,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"stats": dict(stats), "token_stats": manifest["token_stats"]}, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=sorted(SOURCES), default="deepseek-15k")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--holdout", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1228)
    args = parser.parse_args()
    spec = SOURCES[args.release]
    if args.source is None:
        args.source = Path("data/terminal-lego-trajectories") / f"source-{args.release}" / spec["filename"]
    if args.download_only:
        if args.download:
            download_source(args.source, spec)
        provenance = check_source(args.source, spec)
        info = args.source.parent / "source.json"
        if not info.exists():
            info.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print(f"[verified] {args.source} sha256={provenance['sha256']}")
    else:
        if args.out_dir is None or args.tokenizer is None:
            parser.error("SFT conversion requires --tokenizer and --out-dir")
        build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
