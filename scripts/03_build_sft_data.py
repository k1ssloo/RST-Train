#!/usr/bin/env python3
"""Build a slime-ready SFT parquet from the RST trajectory release.

Pipeline
--------
1. Metadata gate on ``metadata/trajectories.parquet``:
   ``status == completed``, ``has_trajectory``, ``not has_exception``,
   ``reward == 1.0``, ``task_present_in_task_dataset``.
2. Deterministic, diversity-aware group-capped sampling (<= --per-group per
   ``task_group_id``, round-robin across ``model_name`` sources).
3. Per-trajectory reconstruction of an OpenAI-style ``messages`` list from the
   ATIF-v1.7 record, with assistant-output JSON normalization.
4. Warning-feedback repair: when an assistant turn is renormalized, the
   "Previous response had warnings:" preamble is stripped from the *following*
   observation so the conversation stays self-consistent.
5. Exact and command-signature dedup.
6. Tokenization against the real Qwen3.5 tokenizer to measure sequence length
   and to assert slime's ``--loss-mask-type qwen3_5`` contract.
7. Write ``messages`` parquet (+ train/holdout split) and a manifest.

Every drop is counted and reported; nothing is silently discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- normalization

_FENCE_BLOCK = re.compile(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", re.DOTALL)
_WARNING_PREAMBLE = re.compile(
    r"\APrevious response had warnings:\n.*?\n\n(?=New Terminal Output:|Current terminal state:)",
    re.DOTALL,
)
_REQUIRED_KEYS = ("analysis", "plan", "commands")


def _balanced_json_object(text: str) -> str | None:
    """Return the first brace-balanced JSON object in ``text``, honoring strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def normalize_assistant(raw: str) -> tuple[str | None, bool, str]:
    """Return ``(canonical_json, was_rewritten, reason)`` for one agent output."""
    stripped = raw.strip()
    candidates: list[tuple[str, bool]] = []
    if stripped:
        candidates.append((stripped, False))
    fenced = _FENCE_BLOCK.search(raw)
    if fenced:
        candidates.append((fenced.group(1), True))
    balanced = _balanced_json_object(raw)
    if balanced:
        candidates.append((balanced, True))

    for text, rewritten in candidates:
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        if not all(key in obj for key in _REQUIRED_KEYS):
            return None, False, "missing_required_keys"
        if not isinstance(obj.get("commands"), list):
            return None, False, "commands_not_list"
        canonical = json.dumps(obj, indent=2, ensure_ascii=False)
        # Rewritten if we had to extract, or if canonical form differs from raw.
        return canonical, rewritten or canonical != stripped, "ok"
    return None, False, "unparseable"


def observation_text(step: dict[str, Any]) -> str | None:
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return None
    results = observation.get("results")
    if not isinstance(results, list) or not results:
        return None
    parts = [
        str(item.get("content", ""))
        for item in results
        if isinstance(item, dict) and item.get("content")
    ]
    return "\n".join(parts) if parts else None


def command_signature(messages: list[dict[str, str]]) -> str:
    """Stable signature of the executed command sequence (near-dup detection)."""
    keystrokes: list[str] = []
    for message in messages:
        if message["role"] != "assistant":
            continue
        try:
            obj = json.loads(message["content"])
        except (json.JSONDecodeError, ValueError):
            continue
        for command in obj.get("commands") or []:
            if isinstance(command, dict):
                keystrokes.append(str(command.get("keystrokes", "")))
    return hashlib.sha256("\x00".join(keystrokes).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ per-shard

def build_from_shard(job: tuple[str, str, dict[str, str]]) -> tuple[list[dict], Counter]:
    """Reconstruct conversations for the selected members of one tar shard."""
    shard_path, root, wanted = job  # wanted: member_prefix -> trajectory_id
    stats: Counter = Counter()
    out: list[dict] = []
    prefixes = {f"{p.rstrip('/')}/trajectory.json": tid for p, tid in wanted.items()}

    with tarfile.open(Path(root) / shard_path) as tar:
        for member in tar:
            tid = prefixes.get(member.name)
            if tid is None:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                stats["drop_unreadable"] += 1
                continue
            try:
                record = json.load(handle)
            except (json.JSONDecodeError, ValueError):
                stats["drop_bad_json"] += 1
                continue

            steps = record.get("steps") or []
            if not steps or steps[0].get("source") != "user":
                stats["drop_no_user_prompt"] += 1
                continue

            prompt = steps[0].get("message")
            if not isinstance(prompt, str) or not prompt.strip():
                stats["drop_empty_prompt"] += 1
                continue

            messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
            agent_steps = [s for s in steps[1:] if s.get("source") == "agent"]
            if not agent_steps:
                stats["drop_no_agent_steps"] += 1
                continue

            rewritten_turns = 0
            failed = False
            pending_observation: str | None = None
            pending_rewritten = False

            for step in agent_steps:
                raw = step.get("message")
                if not isinstance(raw, str):
                    failed = True
                    stats["drop_nonstring_message"] += 1
                    break
                canonical, rewritten, reason = normalize_assistant(raw)
                if canonical is None:
                    failed = True
                    stats[f"drop_{reason}"] += 1
                    break

                # Flush the previous observation as a user turn, repairing its
                # warning preamble if we just rewrote the turn it complains about.
                if pending_observation is not None:
                    text = pending_observation
                    if pending_rewritten:
                        repaired = _WARNING_PREAMBLE.sub("", text)
                        if repaired != text:
                            stats["repaired_warning_preamble"] += 1
                        text = repaired
                    messages.append({"role": "user", "content": text})

                messages.append({"role": "assistant", "content": canonical})
                rewritten_turns += int(rewritten)
                pending_observation = observation_text(step)
                pending_rewritten = rewritten

            if failed:
                continue
            # Trailing observation intentionally dropped: no assistant turn follows it.
            if messages[-1]["role"] != "assistant":
                stats["drop_not_assistant_terminated"] += 1
                continue
            if len(messages) < 2:
                stats["drop_too_short"] += 1
                continue

            out.append(
                {
                    "trajectory_id": tid,
                    "messages": messages,
                    "n_assistant_turns": sum(m["role"] == "assistant" for m in messages),
                    "n_rewritten_turns": rewritten_turns,
                    "command_signature": command_signature(messages),
                    "content_hash": hashlib.sha256(
                        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
                    ).hexdigest(),
                }
            )
            stats["built"] += 1
    return out, stats


# ------------------------------------------------------------------- selection

def select_trajectories(frame, per_group: int, seed: int):
    """Deterministic group cap that round-robins across model sources."""
    import pandas as pd  # local import keeps worker processes lean

    frame = frame.copy()
    # Deterministic per-row ordering key independent of file order.
    frame["_key"] = [
        hashlib.sha256(f"{seed}:{tid}".encode("utf-8")).hexdigest()
        for tid in frame["trajectory_id"]
    ]
    picked_indices: list[Any] = []
    for _, group in frame.groupby("task_group_id", sort=True):
        buckets: dict[str, list] = defaultdict(list)
        for row in group.sort_values("_key").itertuples():
            buckets[row.model_name].append(row.Index)
        # round-robin over model sources (sorted for determinism)
        order = sorted(buckets)
        taken: list[Any] = []
        position = 0
        while len(taken) < per_group:
            progressed = False
            for name in order:
                if position < len(buckets[name]) and len(taken) < per_group:
                    taken.append(buckets[name][position])
                    progressed = True
            if not progressed:
                break
            position += 1
        picked_indices.extend(taken)
    return frame.loc[picked_indices].drop(columns=["_key"])


# ------------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traj-root", type=Path, required=True, help="dir containing data/*.tar and metadata/")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Qwen3.5 tokenizer dir")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--holdout", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1228)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--models", type=str, default="", help="comma-separated model_name allowlist")
    args = parser.parse_args()

    import pandas as pd

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(args.traj_root / "metadata" / "trajectories.parquet")
    total = len(meta)

    gate = (
        (meta.status == "completed")
        & meta.has_trajectory
        & (~meta.has_exception)
        & meta.reward.notna()
        & (meta.reward >= 1.0)
        & meta.task_present_in_task_dataset
    )
    eligible = meta[gate]
    if args.models:
        allow = {m.strip() for m in args.models.split(",") if m.strip()}
        eligible = eligible[eligible.model_name.isin(allow)]

    selected = select_trajectories(eligible, args.per_group, args.seed)
    print(f"[gate] total={total} eligible={len(eligible)} "
          f"groups={eligible.task_group_id.nunique()} selected={len(selected)}", flush=True)

    jobs: dict[str, dict[str, str]] = defaultdict(dict)
    for row in selected.itertuples():
        jobs[row.shard][row.member_prefix] = row.trajectory_id

    stats: Counter = Counter()
    built: list[dict] = []
    payload = [(shard, str(args.traj_root), wanted) for shard, wanted in sorted(jobs.items())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for records, shard_stats in pool.map(build_from_shard, payload):
            built.extend(records)
            stats.update(shard_stats)
    print(f"[build] reconstructed={len(built)} stats={dict(stats)}", flush=True)

    # ---- dedup -------------------------------------------------------------
    by_id = {r["trajectory_id"]: r for r in built}
    group_of = dict(zip(selected.trajectory_id, selected.task_group_id))
    model_of = dict(zip(selected.trajectory_id, selected.model_name))

    seen_content: set[str] = set()
    seen_cmd: set[tuple[str, str]] = set()
    kept: list[dict] = []
    for tid in sorted(by_id):
        record = by_id[tid]
        if record["content_hash"] in seen_content:
            stats["dedup_exact"] += 1
            continue
        cmd_key = (group_of[tid], record["command_signature"])
        if cmd_key in seen_cmd:
            stats["dedup_command_signature"] += 1
            continue
        seen_content.add(record["content_hash"])
        seen_cmd.add(cmd_key)
        record["task_group_id"] = group_of[tid]
        record["model_name"] = model_of[tid]
        kept.append(record)
    print(f"[dedup] kept={len(kept)} exact_dropped={stats['dedup_exact']} "
          f"cmd_dropped={stats['dedup_command_signature']}", flush=True)

    # ---- tokenize + verify slime contract ----------------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    lengths: list[int] = []
    final: list[dict] = []
    for record in kept:
        messages = record["messages"]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, return_dict=False)
        encoding = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoding["input_ids"]
        expected = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
        if ids != expected:
            stats["drop_slime_contract_mismatch"] += 1
            continue
        if len(ids) > args.max_seq_len:
            stats["drop_too_long"] += 1
            continue
        record["n_tokens"] = len(ids)
        lengths.append(len(ids))
        final.append(record)
    print(f"[tokenize] kept={len(final)} contract_mismatch={stats['drop_slime_contract_mismatch']} "
          f"too_long={stats['drop_too_long']}", flush=True)

    # ---- split + write -----------------------------------------------------
    final.sort(key=lambda r: r["trajectory_id"])
    rng = __import__("random").Random(args.seed)
    rng.shuffle(final)
    holdout = final[: args.holdout]
    train = final[args.holdout :]

    def write(rows: list[dict], name: str) -> Path:
        frame = pd.DataFrame(
            {
                "messages": [r["messages"] for r in rows],
                "trajectory_id": [r["trajectory_id"] for r in rows],
                "task_group_id": [r["task_group_id"] for r in rows],
                "model_name": [r["model_name"] for r in rows],
                "n_tokens": [r["n_tokens"] for r in rows],
                "n_assistant_turns": [r["n_assistant_turns"] for r in rows],
                "n_rewritten_turns": [r["n_rewritten_turns"] for r in rows],
            }
        )
        path = args.out_dir / name
        frame.to_parquet(path, index=False)
        return path

    train_path = write(train, "rst_sft_train.parquet")
    holdout_path = write(holdout, "rst_sft_holdout.parquet")

    import numpy as np

    array = np.array(lengths) if lengths else np.array([0])
    manifest = {
        "source_dataset": "Zhongzhi1228/Recursive-Task-Synthesis-Trajectories",
        "trajectories_total": int(total),
        "eligible_after_gate": int(len(eligible)),
        "eligible_groups": int(eligible.task_group_id.nunique()),
        "per_group_cap": args.per_group,
        "selected": int(len(selected)),
        "reconstructed": int(len(built)),
        "after_dedup": int(len(kept)),
        "final_examples": int(len(final)),
        "train_examples": int(len(train)),
        "holdout_examples": int(len(holdout)),
        "max_seq_len": args.max_seq_len,
        "token_stats": {
            "mean": float(array.mean()),
            "p50": float(np.quantile(array, 0.50)),
            "p90": float(np.quantile(array, 0.90)),
            "p99": float(np.quantile(array, 0.99)),
            "max": int(array.max()),
            "total_tokens": int(array.sum()),
        },
        "groups_covered": len({r["task_group_id"] for r in final}),
        "model_mix": dict(Counter(r["model_name"] for r in final)),
        "turns": {
            "mean": float(np.mean([r["n_assistant_turns"] for r in final])) if final else 0.0,
            "max": int(max((r["n_assistant_turns"] for r in final), default=0)),
        },
        "rewritten_turn_fraction": (
            float(sum(r["n_rewritten_turns"] for r in final))
            / max(1, sum(r["n_assistant_turns"] for r in final))
        ),
        "drop_counters": {k: int(v) for k, v in sorted(stats.items())},
        "seed": args.seed,
        "train_parquet": str(train_path),
        "holdout_parquet": str(holdout_path),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
