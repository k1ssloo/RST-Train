#!/usr/bin/env python3
"""Select and materialize the GRPO task pool from the RST task release.

Why selection matters more than it looks
----------------------------------------
GRPO's advantage is computed *within* a group of `n_samples_per_prompt` rollouts
on the same prompt. If every rollout in a group gets the same reward, the
advantage is identically zero and the group contributes **nothing** to the
gradient while still paying the full sandbox cost. The paper's reward curve sat
at 0.11 -> 0.14, i.e. the overwhelming majority of its groups were all-fail.

We can do better, because the trajectory release tells us the empirical pass
rate of each task group. Measured over 231,092 clean trajectories:

    all-fail (0%)   897 groups     <- zero advantage, pure cost
    0-10%           252
    10-35%          469            }
    35-65%          394            }  the useful band
    65-90%          144            }
    >90%             90            <- near-zero advantage, little to learn
    (and 9,764 of the 12,010 groups have no trajectory data at all)

So we sort tasks into tiers and let the launcher spend its sandbox budget on the
band that actually produces gradient signal.

Outputs
-------
  <out>/rl_tasks.jsonl      slime prompt-data: {prompt, label, metadata{...}}
  <out>/tasks/<task_id>/    materialized 6-file RST task dirs (Harbor input)
  <out>/manifest.json       tier counts, image list, selection provenance
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

TIERS = (
    ("sweet", 0.10, 0.90),   # primary GRPO pool: reliable within-group variance
    ("hard", 0.00, 0.10),    # exploration: only worth it once the policy improves
    ("easy", 0.90, 1.01),    # near-saturated: keep a trickle to avoid regression
)
TRACKED = (
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "solution/solve.sh",
    "tests/test.sh",
    "tests/test_state.py",
)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_image(dockerfile: str) -> str:
    match = re.search(r"^\s*FROM\s+(\S+)", dockerfile or "", re.M | re.I)
    return match.group(1) if match else "?"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-root", type=Path, required=True, help="dir with data/*.tar + metadata/")
    parser.add_argument("--traj-root", type=Path, required=True, help="dir with metadata/trajectories.parquet")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tier", default="sweet", choices=[t[0] for t in TIERS] + ["all", "any-variance"])
    parser.add_argument("--max-tasks", type=int, default=0, help="0 = no cap")
    parser.add_argument("--min-trials", type=int, default=4)
    parser.add_argument("--per-group", type=int, default=0, help="cap tasks per group; 0 = keep all variants")
    parser.add_argument("--materialize", action="store_true", help="extract task dirs from the tars")
    args = parser.parse_args()

    import pandas as pd

    tasks = pd.read_parquet(args.tasks_root / "metadata" / "tasks.parquet")
    traj = pd.read_parquet(args.traj_root / "metadata" / "trajectories.parquet")
    tasks = tasks[tasks.validation_status == "passed"]

    clean = traj[
        (traj.status == "completed") & traj.has_trajectory & (~traj.has_exception) & traj.reward.notna()
    ]
    stats = clean.groupby("task_group_id").agg(
        n_trials=("reward", "size"),
        pass_rate=("reward", lambda s: float((s >= 1.0).mean())),
    )

    tasks = tasks.merge(stats, left_on="task_group_id", right_index=True, how="left")
    tasks["n_trials"] = tasks.n_trials.fillna(0).astype(int)

    def tier_of(row) -> str:
        if row.n_trials < args.min_trials:
            return "unknown"
        for name, lo, hi in TIERS:
            if lo <= row.pass_rate < hi:
                return name
        return "unknown"

    tasks["tier"] = tasks.apply(tier_of, axis=1)
    tier_counts = Counter(tasks.tier)
    print(f"[tiers] tasks={len(tasks):,} " + " ".join(f"{k}={v}" for k, v in sorted(tier_counts.items())))

    if args.tier == "all":
        chosen = tasks
    elif args.tier == "any-variance":
        chosen = tasks[(tasks.n_trials >= args.min_trials) & (tasks.pass_rate > 0) & (tasks.pass_rate < 1)]
    else:
        chosen = tasks[tasks.tier == args.tier]

    if args.per_group:
        chosen = chosen.sort_values("task_id").groupby("task_group_id").head(args.per_group)
    chosen = chosen.sort_values("task_id")
    if args.max_tasks:
        chosen = chosen.head(args.max_tasks)
    print(f"[select] tier={args.tier} -> {len(chosen):,} tasks / {chosen.task_group_id.nunique():,} groups")

    args.out.mkdir(parents=True, exist_ok=True)
    task_root = args.out / "tasks"

    # ---- materialize task dirs from the tars -------------------------------
    materialized = 0
    if args.materialize:
        task_root.mkdir(parents=True, exist_ok=True)
        wanted: dict[str, dict[str, str]] = defaultdict(dict)
        for row in chosen.itertuples():
            wanted[row.shard][row.member_prefix.rstrip("/")] = row.task_id
        for shard, members in sorted(wanted.items()):
            prefix_to_id = {p + "/": tid for p, tid in members.items()}
            with tarfile.open(args.tasks_root / shard) as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    for prefix, tid in prefix_to_id.items():
                        if not member.name.startswith(prefix):
                            continue
                        rel = member.name[len(prefix) :]
                        # path-safety: never write outside the task dir
                        if rel.startswith("/") or ".." in Path(rel).parts:
                            raise SystemExit(f"unsafe member path: {member.name}")
                        dest = task_root / tid / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        handle = tar.extractfile(member)
                        if handle is not None:
                            dest.write_bytes(handle.read())
                        break
            print(f"  materialized from {shard}", flush=True)
        # verify the six-file RST contract
        incomplete = []
        for row in chosen.itertuples():
            missing = [f for f in TRACKED if not (task_root / row.task_id / f).is_file()]
            if missing:
                incomplete.append((row.task_id, missing))
            else:
                materialized += 1
        if incomplete:
            print(f"[warn] {len(incomplete)} tasks missing tracked files, e.g. {incomplete[:3]}")
        print(f"[materialize] complete task dirs: {materialized:,}")

        # ---- verifier-leak guard -------------------------------------------
        # `environment/` is the Docker build context, so anything in it becomes
        # visible to the agent. If the private verifier ends up there, the task's
        # reward is hackable and must not enter the RL pool.
        #
        # NOTE: ~46 tasks legitimately contain `environment/tests/` holding the
        # *project's own* fixtures (a PHP suite, Ansible playbooks, JSON files) --
        # that is workspace content the agent is meant to work on, NOT the RST
        # verifier. So do not flag on the directory name; flag on the verifier's
        # filenames or on byte-identical content. Measured: 0 leaks in 5,140 tasks.
        leaked: list[tuple[str, str]] = []
        for row in chosen.itertuples():
            task_dir = task_root / row.task_id
            env_dir = task_dir / "environment"
            if not env_dir.is_dir():
                continue
            private = {
                _sha256(task_dir / "tests" / name)
                for name in ("test.sh", "test_state.py")
                if (task_dir / "tests" / name).is_file()
            }
            for path in env_dir.rglob("*"):
                if not path.is_file():
                    continue
                if path.name in ("test_state.py", "test.sh") or _sha256(path) in private:
                    leaked.append((row.task_id, str(path.relative_to(task_dir))))
                    break
        if leaked:
            print(f"[LEAK] {len(leaked)} tasks expose the private verifier in the build "
                  f"context and are EXCLUDED, e.g. {leaked[:3]}")
            excluded = {tid for tid, _ in leaked}
            chosen = chosen[~chosen.task_id.isin(excluded)]
            (args.out / "excluded_verifier_leak.json").write_text(
                json.dumps(leaked, indent=2) + "\n", encoding="utf-8"
            )
        else:
            print("[leak-guard] 0 tasks expose the private verifier to the agent image")

    # ---- slime prompt-data -------------------------------------------------
    jsonl = args.out / "rl_tasks.jsonl"
    images = Counter()
    with jsonl.open("w", encoding="utf-8") as fh:
        for row in chosen.itertuples():
            image = base_image(row.dockerfile)
            images[image] += 1
            record = {
                # `prompt` is required by slime's data loader but the real prompt is
                # built by Terminus-2 inside Harbor from instruction.md. We keep the
                # instruction here so the row is self-describing and debuggable.
                "prompt": row.instruction,
                "label": row.task_id,
                "metadata": {
                    "task_id": row.task_id,
                    "task_group_id": row.task_group_id,
                    "task_dir": str((task_root / row.task_id).resolve()),
                    "task_content_sha256": row.task_content_sha256,
                    "base_image": image,
                    "tier": row.tier,
                    "empirical_pass_rate": (None if row.n_trials == 0 else float(row.pass_rate)),
                    "n_reference_trials": int(row.n_trials),
                },
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "tasks_selected": int(len(chosen)),
        "groups_selected": int(chosen.task_group_id.nunique()),
        "tier": args.tier,
        "min_trials": args.min_trials,
        "tier_counts_all_tasks": dict(sorted(tier_counts.items())),
        "materialized_task_dirs": materialized,
        "distinct_base_images": len(images),
        "base_images": dict(images.most_common()),
        "pass_rate_summary": {
            "mean": float(chosen.pass_rate.mean()) if len(chosen) else None,
            "min": float(chosen.pass_rate.min()) if len(chosen) else None,
            "max": float(chosen.pass_rate.max()) if len(chosen) else None,
        },
        "prompt_data": str(jsonl),
        "task_root": str(task_root),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
