#!/usr/bin/env python3
"""Convert `SWE-Gym/SWE-Gym` into a tiered RL task pool.

    python scripts/10c_build_swegym_taskset.py \
        --pool data/swegym/source-pool.parquet \
        --rollouts data/swegym/rollouts/*.parquet \
        --out data/swegym

WHY THIS IS NOT AN SFT CONVERTER
--------------------------------
`SWE-Gym/SWE-Gym` is SWE-bench format: `instance_id`, `problem_statement`, `patch`,
`test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `repo`, `base_commit`. There is no
`messages` or `conversations` column at all, so there is nothing to supervise on. It
is the fourth dataset handed to this repo that looks like SFT data and is not, after
`allenai/TMax-15K`, `allenai/tmax-15k-open-instruct` and
`allenai/open-instruct-termigen`.

The trajectories that go with it are in `SWE-Gym/OpenHands-SFT-Trajectories` (491
rows) -- which is exactly the `resolved == True` subset of
`SWE-Gym/OpenHands-Sampled-Trajectories` (6,055 rollouts, 491 resolved). They are
**not** converted here, and that is a deliberate call rather than an omission: they
are in the OpenHands/CodeAct action space,

    <function=str_replace_editor>
    <parameter=command>view</parameter>
    <parameter=path>/workspace/...</parameter>
    </function>

which has no terminus-2 equivalent. Rewriting `str_replace_editor` as
`{"analysis", "plan", "commands": [{"keystrokes", "duration"}]}` would mean inventing
the analysis and plan text and re-expressing structured edits as shell heredocs --
fabricating supervision, which is the same reason `03e_build_tmax_sft.py` refused to
synthesize a `plan` key. Converted faithfully instead, they would be a 491-row corpus
(4 steps/epoch at GBS 128) in an action space this repo's Harbor/terminus-2 eval
cannot drive, distilled from `gpt-4o-2024-08-06`. Say so before spending GPU on it.

WHAT THIS DOES ADD OVER UPSTREAM
--------------------------------
Tiers, measured. `10_build_rl_taskset.py` exists mostly because GRPO's advantage is
computed *within* a group of rollouts on one prompt: if every rollout scores the same,
the advantage is identically zero and the group contributes nothing to the gradient
while paying the full sandbox cost. Screening for that needs per-instance pass rates,
and `Termigen-RL-Taskset` had to ship without them.

Here they exist. `OpenHands-Sampled-Trajectories` carries a `resolved` boolean per
rollout and covers **all 2,438** SWE-Gym instances, so every task in this pool gets a
real pass rate off 6,055 measured rollouts:

    hard   (0-10 %)   2,144      88.0 %   <- zero-gradient, pure cost
    sweet  (10-90 %)    187       7.7 %   <- the only usable GRPO band
    easy   (>=90 %)     107       4.4 %

That 88 % is the number to plan around. A naive GRPO run over this pool spends most
of its sandbox budget on groups that cannot produce a gradient.

The rates come from a `gpt-4o-2024-08-06` policy, so they measure difficulty *for that
policy*. They transfer as an ordering, not as absolutes -- a stronger policy shifts
everything up. Recorded with the run_ids they came from so this stays checkable.

TWO THINGS DELIBERATELY LEFT OUT OF THE OUTPUT
----------------------------------------------
**The gold patch.** All 2,438 instances ship `patch`, the reference solution. Nothing
needs it at rollout time, and a ready-made copy alongside the prompts makes "did the
model see the answer?" impossible to argue. Excluded entirely; it is one `hf download`
away from anyone who legitimately needs it.

**`hints_text`.** Non-empty on 1,528 of 2,438 instances (63 %), and it is maintainer
discussion that frequently names the fix. Every SWE-Gym rollout run in
`OpenHands-Sampled-Trajectories` is a `no-hint` run, so including it in the prompt
would also make the measured pass rates describe a different task than the one being
served. Recorded as a boolean, never as text.

`test_patch`, `FAIL_TO_PASS` and `PASS_TO_PASS` *are* kept: they are what the verifier
needs to score a rollout, and without them the pool is not runnable. That is why the
uploader defaults to a private repo.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_POOL = "SWE-Gym/SWE-Gym"
SOURCE_ROLLOUTS = "SWE-Gym/OpenHands-Sampled-Trajectories"
SOURCE_LICENSE = "mit"

# Same bands and the same reasoning as scripts/10_build_rl_taskset.py.
TIERS = (
    ("sweet", 0.10, 0.90),   # primary GRPO pool: reliable within-group variance
    ("hard", 0.00, 0.10),    # exploration: only worth it once the policy improves
    ("easy", 0.90, 1.01),    # near-saturated: keep a trickle to avoid regression
)

REQUIRED_POOL_COLUMNS = {
    "instance_id", "problem_statement", "patch", "test_patch",
    "repo", "base_commit", "version", "FAIL_TO_PASS", "PASS_TO_PASS", "hints_text",
}


def as_list(value: object) -> list[str]:
    """Coerce a repeated parquet field to `list[str]`.

    `value or []` is wrong here: pyarrow hands these back as numpy arrays, and
    `bool(array)` raises for length > 1. It silently worked on `FAIL_TO_PASS`, which is
    usually a single element, and blew up on `PASS_TO_PASS`, which usually is not.
    """
    if value is None:
        return []
    return [str(item) for item in value]


def tier_of(pass_rate: float) -> str:
    for name, low, high in TIERS:
        if low <= pass_rate < high:
            return name
    raise AssertionError(f"pass rate {pass_rate} fell outside every tier")


def load_rollout_outcomes(patterns: list[str]) -> tuple[dict[str, list[int]], Counter]:
    """Per-instance `[resolved, total]` from the sampled-rollout parquets.

    Read column-projected through pyarrow rather than `pd.read_parquet`, because the
    `messages` and `tools` columns are nested and large -- 287 MiB of them -- and this
    needs two scalars per row.
    """
    import pyarrow.parquet as pq

    outcomes: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    runs: Counter = Counter()
    files = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not files:
        sys.exit(f"no rollout parquet matched {patterns}")
    for path in files:
        reader = pq.ParquetFile(path)
        columns = set(reader.schema_arrow.names)
        missing = {"instance_id", "resolved"} - columns
        if missing:
            sys.exit(f"{path} is missing {sorted(missing)}; upstream schema changed")
        want = ["instance_id", "resolved"] + (["run_id"] if "run_id" in columns else [])
        for batch in reader.iter_batches(batch_size=512, columns=want):
            for row in batch.to_pylist():
                entry = outcomes[row["instance_id"]]
                entry[0] += int(bool(row["resolved"]))
                entry[1] += 1
                if "run_id" in row:
                    runs[row["run_id"]] += 1
    print(f"[rollouts] {len(files)} file(s), {sum(v[1] for v in outcomes.values()):,} rollouts "
          f"over {len(outcomes):,} instances", flush=True)
    return dict(outcomes), runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, required=True, help=f"{SOURCE_POOL} train parquet")
    ap.add_argument("--rollouts", nargs="+", required=True,
                    help=f"{SOURCE_ROLLOUTS} parquet(s); globs allowed")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tier", default="all",
                    help="comma-separated subset of sweet,hard,easy, or 'all'")
    args = ap.parse_args()

    import pandas as pd

    if not args.pool.is_file():
        sys.exit(f"missing input: {args.pool}")
    pool = pd.read_parquet(args.pool)
    missing = REQUIRED_POOL_COLUMNS - set(pool.columns)
    if missing:
        sys.exit(f"{args.pool} is missing {sorted(missing)}; upstream schema changed")

    # The claim the docstring rests on, asserted rather than trusted: if upstream ever
    # adds trajectories, this must stop calling itself a task-pool-only converter.
    for column in ("messages", "conversations"):
        if column in pool.columns:
            sys.exit(f"{args.pool} now has a `{column}` column. This dataset may be "
                     f"SFT-able -- check before discarding the responses.")
    print(f"[pool] {len(pool):,} instances from {SOURCE_POOL}, no response column "
          f"(a task pool, as expected)", flush=True)

    outcomes, runs = load_rollout_outcomes(args.rollouts)
    wanted = ({name for name, _, _ in TIERS} if args.tier == "all"
              else {t.strip() for t in args.tier.split(",")})
    unknown = wanted - {name for name, _, _ in TIERS}
    if unknown:
        sys.exit(f"unknown tier(s) {sorted(unknown)}")

    stats: Counter = Counter()
    tier_counts: Counter = Counter()
    repos: Counter = Counter()
    records: list[dict[str, object]] = []
    verifiers: list[dict[str, object]] = []
    rates: list[float] = []

    for row in pool.itertuples(index=False):
        instance_id = str(row.instance_id)
        problem = (row.problem_statement or "").strip()
        if not problem:
            stats["drop_no_problem_statement"] += 1
            continue
        fail_to_pass = as_list(row.FAIL_TO_PASS)
        if not fail_to_pass:
            # With no failing test to flip, the reward is undefined: a do-nothing
            # rollout scores the same as a correct one.
            stats["drop_no_fail_to_pass"] += 1
            continue
        test_patch = str(row.test_patch or "")
        if not test_patch.strip():
            stats["drop_no_test_patch"] += 1
            continue
        pass_to_pass = as_list(row.PASS_TO_PASS)

        measured = outcomes.get(instance_id)
        if measured is None:
            stats["drop_no_rollout_coverage"] += 1
            continue
        resolved, total = measured
        pass_rate = resolved / total
        tier = tier_of(pass_rate)
        tier_counts[tier] += 1
        if tier not in wanted:
            stats[f"skipped_tier_{tier}"] += 1
            continue

        rates.append(pass_rate)
        repos[str(row.repo)] += 1
        verifiers.append({"instance_id": instance_id, "test_patch": test_patch,
                          "FAIL_TO_PASS": fail_to_pass, "PASS_TO_PASS": pass_to_pass})
        records.append({
            # What the agent is actually shown. `hints_text` is deliberately absent --
            # see the module docstring -- and so is the gold patch.
            "prompt": problem,
            "label": instance_id,
            "metadata": {
                "instance_id": instance_id,
                # Each SWE-bench instance is its own prompt -- a distinct issue at a
                # distinct commit -- so the group is the instance, as in the termigen
                # pool. `repo` is kept separately below because it is the right key for
                # reasoning about contamination and image reuse, not for grouping: 2,438
                # instances share only 11 repos, and treating those as 11 groups would
                # make a group-disjoint split throw away 9 % of the pool per held-out repo.
                "task_group_id": instance_id,
                "repo": str(row.repo),
                "base_commit": str(row.base_commit),
                "version": str(row.version),
                "environment_setup": "swebench harness (repo + base_commit + version); "
                                     "this pool ships no Dockerfile",
                # The verifier spec itself lives in `verifier_spec.parquet`, keyed by
                # instance_id, not inline here. PASS_TO_PASS averages 751 test names and
                # peaks at 29,737, which is 173 MiB of the 191 MiB an inline jsonl came
                # to -- and JSON is the wrong container for a column of string lists
                # that compresses ~10x. It also restores this repo's convention that
                # rl_tasks.jsonl carries prompt plus metadata while the verifier sits
                # beside it (there, in task dirs).
                "verifier_spec": "verifier_spec.parquet",
                "n_fail_to_pass": len(fail_to_pass),
                "n_pass_to_pass": len(pass_to_pass),
                "verifier_sha256": hashlib.sha256(json.dumps(
                    {"test_patch": test_patch, "FAIL_TO_PASS": fail_to_pass,
                     "PASS_TO_PASS": pass_to_pass}, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "tier": tier,
                "empirical_pass_rate": round(pass_rate, 4),
                "n_reference_trials": total,
                "hints_text_available_upstream": bool((row.hints_text or "").strip()),
                "gold_patch_excluded": True,
                "problem_statement_sha256": hashlib.sha256(
                    problem.encode("utf-8")).hexdigest(),
                "source_dataset": SOURCE_POOL,
                "pass_rate_source": SOURCE_ROLLOUTS,
            },
        })
        stats["selected"] += 1

    if not records:
        sys.exit("no tasks survived the gates")

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out / "rl_tasks.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda r: r["label"]):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[write] {jsonl} tasks={len(records):,} "
          f"({jsonl.stat().st_size / 2**20:.1f} MiB)", flush=True)

    spec = args.out / "verifier_spec.parquet"
    pd.DataFrame(sorted(verifiers, key=lambda v: v["instance_id"])).to_parquet(
        spec, index=False)
    print(f"[write] {spec} rows={len(verifiers):,} "
          f"({spec.stat().st_size / 2**20:.1f} MiB)", flush=True)

    manifest = {
        "source_pool": SOURCE_POOL,
        "source_rollouts": SOURCE_ROLLOUTS,
        "source_license": SOURCE_LICENSE,
        "pool_instances": len(pool),
        "not_sft_data": "SWE-bench format -- instance_id, problem_statement, patch, "
                        "test_patch, FAIL_TO_PASS. No messages or conversations column, "
                        "so there is nothing to supervise on.",
        "sft_trajectories_exist_but_were_not_converted": {
            "where": "SWE-Gym/OpenHands-SFT-Trajectories (491 rows, == the resolved "
                     "subset of the 6,055 sampled rollouts)",
            "why_not": "OpenHands/CodeAct action space (execute_bash + "
                       "str_replace_editor function-call XML). Rewriting it as the "
                       "terminus-2 {analysis, plan, commands} contract would mean "
                       "inventing the analysis and plan text and re-expressing "
                       "structured edits as shell heredocs. Converted faithfully it "
                       "would be 491 rows -- 4 steps/epoch at GBS 128 -- in an action "
                       "space this repo's Harbor/terminus-2 eval cannot drive, "
                       "distilled from gpt-4o-2024-08-06.",
        },
        "dpo_verdict": {
            "built": False,
            "reason": "of 2,438 instances only 188 (7.7 %) have both a resolved and an "
                      "unresolved rollout, giving 291 pairs at min(resolved, failed) "
                      "per instance against the existing DPO stage's 2,673. Thinner "
                      "than TMax, and over an action space that is not this repo's.",
            "pairable_instances": 188,
            "total_instances": 2438,
            "pairs_at_min_per_instance": 291,
        },
        "tasks_selected": len(records),
        "tier_requested": args.tier,
        "tier_counts_all_instances": dict(sorted(tier_counts.items())),
        "tier_bands": {name: [low, high] for name, low, high in TIERS},
        "zero_gradient_fraction": round(
            tier_counts["hard"] / max(1, sum(tier_counts.values())), 4),
        "rollouts_used": sum(v[1] for v in outcomes.values()),
        "rollout_policy": "gpt-4o-2024-08-06 (pass rates measure difficulty for THAT "
                          "policy; they transfer as an ordering, not as absolutes)",
        "rollout_run_ids": dict(runs.most_common()),
        "pass_rate_summary": {
            "mean": statistics.mean(rates),
            "p50": statistics.median(rates),
            "min": min(rates),
            "max": max(rates),
        },
        "distinct_repos": len(repos),
        "repos": dict(repos.most_common()),
        "gold_patch_excluded": True,
        "hints_text_excluded_from_prompt": True,
        "hints_text_available_upstream_count": int(
            sum(1 for h in pool.hints_text if (h or "").strip())),
        "drop_counters": dict(sorted(stats.items())),
        "prompt_data": str(jsonl),
        "verifier_spec": str(spec),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "tasks_selected", "tier_counts_all_instances", "zero_gradient_fraction",
        "rollouts_used", "distinct_repos", "pass_rate_summary")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
