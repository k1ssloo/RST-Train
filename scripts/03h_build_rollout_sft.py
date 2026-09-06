#!/usr/bin/env python3
"""Turn the repo's OWN rollouts into SFT data -- the step that closes the loop.

    # Harbor job dirs left by an eval run made with --export-trajectories
    python scripts/03h_build_rollout_sft.py \
        --jobs-dir $BASE_FOLDER/eval/mine/jobs \
        --tokenizer $BASE_FOLDER/Qwen3.5-27B --out-dir $BASE_FOLDER/rollout-sft-v1

    # TerminalEvo's golden export (each episode embeds its ATIF trajectory)
    python scripts/03h_build_rollout_sft.py \
        --golden-episodes ../TerminalEvo/outputs/golden/golden_episodes.jsonl \
        --tokenizer $BASE_FOLDER/Qwen3.5-27B --out-dir $BASE_FOLDER/evo-sft-v1

WHAT WAS MISSING
    Every dataset in this repo so far came from someone else's policy: the RST release,
    OpenThoughts, TMax, Nemotron. The repo can serve a checkpoint, drive it through
    Harbor, score it with the task's verifier -- and then delete the job directory.
    Nothing turned a verifier-passed rollout of *our* model back into training data,
    so "recursive self-improvement" ended at the eval table. This script is the
    missing arrow: Harbor job dir (or TerminalEvo golden episode) -> the canonical
    `messages` parquet every other 03* builder writes, through the SAME
    `reconstruct_trajectory` / `normalize_assistant` / dedup / template gate.

THE ONE THING THE ROLLOUT MUST HAVE BEEN RUN WITH
    Terminus-2's default ATIF stores its own rendering of each turn --
    "Analysis: ...\\nPlan: ..." plus `tool_calls` -- and drops the model's completion.
    That file cannot become a training target without inventing text. Runs meant for
    this script pass `trajectory_config={"raw_content":true,"linear_history":true}`
    (`06_eval.py --export-trajectories`, `RST_EXPORT_TRAJECTORIES=1` for RL; see
    `rst_common.harbor.export_agent_kwargs` and BUG.md BUG-19). A rendered trajectory
    is therefore REFUSED by default and counted as `drop_rendered_trajectory`.
    `--allow-rendered` re-synthesizes the canonical JSON from the rendering and the
    tool calls, marks every such turn in `n_resynthesized_turns`, and is for salvaging
    a run that cannot be repeated -- not for routine use.

REWARD IS THE GATE
    The verifier's reward is read with `rst_common.harbor.read_reward`, so an
    infrastructure failure is "unmeasured" and never "reward 0". Only trials with
    reward >= --min-reward (default 1.0) become SFT rows. EVERY reconstructed trial,
    failures included, is also written to `rollouts.jsonl` with its reward: that is
    the on-policy success/failure pool a future DPO pass can pair on, which the
    off-policy release cannot provide.

CONTEXT COMPACTION
    A long episode may hold Terminus-2 summarization boundaries. With linear_history
    Harbor splits them into `trajectory.json`, `trajectory.cont-1.json`, ...; each is
    exactly the history the model saw and becomes its own row. Without it, the one
    file stitches segments together. `--on-compaction split` (default) cuts it at
    every boundary the way Harbor would have; `stitch` mirrors what the release
    builder does with such files; `drop` refuses them.

OUTPUT
    <out-dir>/rollout_sft_train.parquet    the 03_build_sft_data.py schema + reward,
    <out-dir>/rollout_sft_holdout.parquet  source_trial, segment_index, n_resynthesized_turns
    <out-dir>/rollouts.jsonl               every reconstructed trial with its reward
    <out-dir>/manifest.json                every counter
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rst_common.harbor import HARNESS_INFRA, Outcome, read_reward  # noqa: E402
from sft_common import (  # noqa: E402
    contract_length_gate,
    dedup_records,
    group_disjoint_split,
    token_stats,
)
from siblings import load_script  # noqa: E402

RENDERED = re.compile(r"\A(?:Analysis: (?P<analysis>.*?))?(?:\n?Plan: (?P<plan>.*))?\Z", re.DOTALL)
CONTINUATION = re.compile(r"\Atrajectory(?:\.cont-(\d+))?\.json\Z")


# ------------------------------------------------------------------ discovery

def trial_dirs(jobs_dir: Path) -> list[Path]:
    """Every Harbor trial directory under `jobs_dir`: has result.json and agent/."""
    out = []
    for result in sorted(jobs_dir.rglob("result.json")):
        trial = result.parent
        if (trial / "agent").is_dir() and (trial / "agent" / "trajectory.json").is_file():
            out.append(trial)
    return out


def trajectory_files(agent_dir: Path) -> list[Path]:
    """`trajectory.json` first, then the linear-history continuations in order."""
    found: list[tuple[int, Path]] = []
    for path in agent_dir.glob("trajectory*.json"):
        match = CONTINUATION.match(path.name)
        if match:
            found.append((int(match.group(1) or 0), path))
    return [p for _, p in sorted(found)]


# --------------------------------------------------------------- one trajectory

def is_rendered(record: dict[str, Any]) -> bool:
    """True when Terminus-2 wrote its own rendering instead of the raw completion."""
    return any(s.get("source") == "agent" and s.get("tool_calls") is not None
               for s in record.get("steps") or [])


def resynthesize_step(step: dict[str, Any]) -> tuple[str, bool]:
    """Rebuild the canonical action JSON from a rendered step. Returns (text, changed).

    Only used under --allow-rendered. The analysis and plan come back verbatim from
    the rendering; commands from `bash_command` tool calls; `task_complete` from the
    `mark_task_complete` call. Anything else the model wrote is gone, which is why
    this is opt-in and counted.
    """
    tool_calls = step.get("tool_calls")
    if tool_calls is None:
        return str(step.get("message", "")), False
    match = RENDERED.match(str(step.get("message", "")))
    analysis = (match.group("analysis") or "").strip() if match else ""
    plan = (match.group("plan") or "").strip() if match else ""
    commands = []
    complete = False
    for call in tool_calls:
        name = call.get("function_name")
        arguments = call.get("arguments") or {}
        if name == "bash_command":
            command = {"keystrokes": str(arguments.get("keystrokes", ""))}
            if "duration" in arguments:
                command["duration"] = arguments["duration"]
            commands.append(command)
        elif name == "mark_task_complete":
            complete = True
    obj: dict[str, Any] = {"analysis": analysis, "plan": plan, "commands": commands}
    if complete:
        obj["task_complete"] = True
    return json.dumps(obj, indent=2, ensure_ascii=False), True


def split_on_compaction(steps: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Cut at every context-compaction marker; each segment starts at a user step."""
    segments: list[list[dict[str, Any]]] = [[]]
    for step in steps:
        extra = step.get("extra") or {}
        if step.get("source") == "system" and isinstance(extra.get("context_management"), dict):
            segments.append([])
            continue
        if step.get("source") == "system":
            continue
        segments[-1].append(step)
    return [seg for seg in segments if seg]


def has_compaction(steps: list[dict[str, Any]]) -> bool:
    return any(s.get("source") == "system"
               and isinstance((s.get("extra") or {}).get("context_management"), dict)
               for s in steps)


def build_segments(record: dict[str, Any], *, on_compaction: str, allow_rendered: bool,
                   builder: Any, stats: Counter) -> list[dict[str, Any]]:
    """One ATIF record -> zero or more canonical records (one per linear segment)."""
    steps = list(record.get("steps") or [])
    rendered = is_rendered(record)
    resynthesized = 0
    if rendered:
        if not allow_rendered:
            stats["drop_rendered_trajectory"] += 1
            return []
        new_steps = []
        for step in steps:
            if step.get("source") == "agent":
                text, changed = resynthesize_step(step)
                step = {**step, "message": text}
                resynthesized += int(changed)
            new_steps.append(step)
        steps = new_steps

    if has_compaction(steps):
        stats["trajectories_with_compaction"] += 1
        if on_compaction == "drop":
            stats["drop_compaction"] += 1
            return []
        pieces = split_on_compaction(steps) if on_compaction == "split" else [steps]
    else:
        pieces = [steps]

    out = []
    for index, piece in enumerate(pieces):
        built = builder.reconstruct_trajectory({"steps": piece}, stats)
        if built is None:
            continue
        built["segment_index"] = index
        built["n_resynthesized_turns"] = resynthesized if index == 0 else 0
        out.append(built)
    return out


# ------------------------------------------------------------------- sources

def from_jobs_dir(jobs_dir: Path, *, model_name: str | None, stats: Counter,
                  **kwargs: Any) -> list[dict[str, Any]]:
    records = []
    for trial in trial_dirs(jobs_dir):
        outcome: Outcome = read_reward(trial)
        if outcome.kind == HARNESS_INFRA:
            stats["trials_infra_unmeasured"] += 1
            continue
        try:
            result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stats["trials_infra_unmeasured"] += 1
            continue
        task = str(result.get("task_name") or trial.parent.name)
        trial_name = str(result.get("trial_name") or trial.name)
        agent_model = ((result.get("agent_info") or {}).get("model_info") or {}).get("name")
        files = trajectory_files(trial / "agent")
        stats["trials_read"] += 1
        stats["trajectory_files_read"] += len(files)
        for file_index, path in enumerate(files):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stats["drop_bad_json"] += 1
                continue
            for built in build_segments(record, builder=kwargs["builder"], stats=stats,
                                        on_compaction=kwargs["on_compaction"],
                                        allow_rendered=kwargs["allow_rendered"]):
                segment = file_index * 1000 + built["segment_index"]
                built.update({
                    "trajectory_id": f"{trial_name}#{segment}" if (len(files) > 1 or built["segment_index"]) else trial_name,
                    "task_group_id": task,
                    "model_name": model_name or agent_model or "unknown",
                    "reward": float(outcome.reward or 0.0),
                    "budget_failure": outcome.budget_reason,
                    "source_trial": str(trial),
                    "segment_index": segment,
                })
                records.append(built)
    return records


def from_golden_episodes(path: Path, *, model_name: str | None, stats: Counter,
                         **kwargs: Any) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = json.loads(line)
            rollout = episode.get("rollout") or {}
            record = rollout.get("trajectory")
            if not isinstance(record, dict):
                stats["drop_episode_without_trajectory"] += 1
                continue
            stats["trials_read"] += 1
            stats["trajectory_files_read"] += 1
            reward = rollout.get("reward")
            task = str((episode.get("task") or {}).get("task_id") or "unknown")
            agent_model = ((rollout.get("agent_info") or {}).get("model_info") or {}).get("name")
            trial_name = str(rollout.get("trial_name") or episode.get("episode_id"))
            for built in build_segments(record, builder=kwargs["builder"], stats=stats,
                                        on_compaction=kwargs["on_compaction"],
                                        allow_rendered=kwargs["allow_rendered"]):
                built.update({
                    "trajectory_id": (f"{trial_name}#{built['segment_index']}"
                                      if built["segment_index"] else trial_name),
                    "task_group_id": task,
                    "model_name": model_name or agent_model or "unknown",
                    "reward": float(reward) if isinstance(reward, (int, float)) else 0.0,
                    "budget_failure": None,
                    "source_trial": f"{path}:{trial_name}",
                })
                records.append(built)
    return records


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-dir", type=Path, action="append", default=[],
                    help="a Harbor jobs tree (06_eval.py --out/jobs, RST_JOBS_ROOT); repeatable")
    ap.add_argument("--golden-episodes", type=Path, action="append", default=[],
                    help="TerminalEvo golden_episodes.jsonl; repeatable")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-reward", type=float, default=1.0,
                    help="trials below this are kept in rollouts.jsonl but not in the SFT parquet")
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--holdout", type=int, default=0,
                    help="rows to hold out, whole tasks at a time (default 0: rollouts are train)")
    ap.add_argument("--on-compaction", choices=("split", "stitch", "drop"), default="split")
    ap.add_argument("--allow-rendered", action="store_true",
                    help="re-synthesize the action JSON for trajectories written WITHOUT "
                         "raw_content. Lossy; for salvage only. See the module docstring.")
    ap.add_argument("--model-name", default=None, help="override the policy name recorded per row")
    ap.add_argument("--seed", type=int, default=1228)
    args = ap.parse_args()
    if not args.jobs_dir and not args.golden_episodes:
        sys.exit("give at least one --jobs-dir or --golden-episodes")

    import pandas as pd

    builder = load_script("03_build_sft_data")
    stats: Counter = Counter()
    records: list[dict[str, Any]] = []
    common = dict(builder=builder, stats=stats, on_compaction=args.on_compaction,
                  allow_rendered=args.allow_rendered, model_name=args.model_name)
    for jobs_dir in args.jobs_dir:
        if not jobs_dir.is_dir():
            sys.exit(f"not a directory: {jobs_dir}")
        before = stats["trials_read"], stats["trials_infra_unmeasured"]
        found = from_jobs_dir(jobs_dir, **common)
        print(f"[jobs] {jobs_dir}: trials={stats['trials_read'] - before[0]} "
              f"infra_unmeasured={stats['trials_infra_unmeasured'] - before[1]} "
              f"reconstructed={len(found)}", flush=True)
        records.extend(found)
    for path in args.golden_episodes:
        if not path.is_file():
            sys.exit(f"missing input: {path}")
        before_n = stats["trials_read"]
        found = from_golden_episodes(path, **common)
        print(f"[golden] {path}: episodes={stats['trials_read'] - before_n} "
              f"reconstructed={len(found)}", flush=True)
        records.extend(found)
    if stats["drop_rendered_trajectory"]:
        print(f"[note] {stats['drop_rendered_trajectory']} trajectories were written by Terminus-2 "
              f"WITHOUT raw_content (they hold the harness's 'Analysis:/Plan:' rendering, not "
              f"the model's output) and were refused. Rerun the rollout with "
              f"06_eval.py --export-trajectories / RST_EXPORT_TRAJECTORIES=1, or pass "
              f"--allow-rendered to salvage them lossily.", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda r: r["trajectory_id"]):
            handle.write(json.dumps({k: v for k, v in record.items()}, ensure_ascii=False) + "\n")
    rewards = Counter("success" if r["reward"] >= 1.0 else "failure" for r in records)
    print(f"[rollouts] {len(records)} reconstructed: {dict(rewards)} -> rollouts.jsonl", flush=True)

    successes = [r for r in records if r["reward"] >= args.min_reward]
    stats["below_min_reward"] = len(records) - len(successes)
    if not successes:
        (args.out_dir / "manifest.json").write_text(json.dumps({
            "reconstructed": len(records), "sft_rows": 0, "min_reward": args.min_reward,
            "counters": dict(sorted(stats.items())),
            "note": "no trial reached --min-reward; rollouts.jsonl still holds every "
                    "reconstructed trajectory with its reward",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("[sft] nothing reached --min-reward; manifest written, no parquet", flush=True)
        return 0

    kept, cross_task = dedup_records(successes, stats)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    final: list[dict[str, Any]] = []
    lengths: list[int] = []
    for record in kept:
        n_tokens, reason = contract_length_gate(tokenizer, record["messages"], args.max_seq_len)
        if reason:
            stats[f"drop_{reason}"] += 1
            continue
        record["n_tokens"] = n_tokens
        lengths.append(n_tokens)
        final.append(record)
    final.sort(key=lambda r: r["trajectory_id"])
    train, holdout, holdout_groups, _target = group_disjoint_split(
        final, holdout=args.holdout, seed=args.seed)
    holdout.sort(key=lambda r: r["trajectory_id"])

    columns = ["messages", "trajectory_id", "task_group_id", "model_name", "n_tokens",
               "n_assistant_turns", "n_rewritten_turns", "reward", "source_trial",
               "segment_index", "n_resynthesized_turns"]
    paths = {}
    for name, rows in (("train", train), ("holdout", holdout)):
        path = args.out_dir / f"rollout_sft_{name}.parquet"
        pd.DataFrame([{k: r[k] for k in columns} for r in rows], columns=columns).to_parquet(
            path, index=False)
        paths[name] = str(path)
        print(f"[write] {path} rows={len(rows)}", flush=True)

    manifest = {
        "sources": {"jobs_dirs": [str(p) for p in args.jobs_dir],
                    "golden_episodes": [str(p) for p in args.golden_episodes]},
        "trials_read": stats["trials_read"],
        "trials_infra_unmeasured": stats["trials_infra_unmeasured"],
        "reconstructed": len(records),
        "reconstructed_by_reward": dict(rewards),
        "min_reward": args.min_reward,
        "sft_candidates": len(successes),
        "after_dedup": len(kept),
        "final_examples": len(final),
        "train_examples": len(train),
        "holdout_examples": len(holdout),
        "holdout_groups": sorted(holdout_groups),
        "groups_covered": len({r["task_group_id"] for r in final}),
        "command_signatures_shared_across_tasks": cross_task,
        "on_compaction": args.on_compaction,
        "allow_rendered": args.allow_rendered,
        "resynthesized_turns": sum(r["n_resynthesized_turns"] for r in final),
        "model_mix": dict(Counter(r["model_name"] for r in final)),
        "max_seq_len": args.max_seq_len,
        "tokenizer": str(args.tokenizer),
        "token_stats": token_stats(lengths),
        "turns": {"mean": (sum(r["n_assistant_turns"] for r in final) / len(final)) if final else 0.0,
                  "max": max((r["n_assistant_turns"] for r in final), default=0)},
        "rewritten_turn_fraction": (sum(r["n_rewritten_turns"] for r in final)
                                    / max(1, sum(r["n_assistant_turns"] for r in final))),
        "counters": dict(sorted(stats.items())),
        "seed": args.seed,
        "train_parquet": paths["train"],
        "holdout_parquet": paths["holdout"],
        "rollouts_jsonl": str(args.out_dir / "rollouts.jsonl"),
        "schema_note": "same columns as 03_build_sft_data.py plus reward/source_trial/"
                       "segment_index/n_resynthesized_turns; 15_export_pretokenized.py "
                       "and 03g_curate_sft.py consume it unchanged",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("reconstructed_by_reward", "final_examples",
                                                "train_examples", "counters")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
