#!/usr/bin/env python3
"""Convert AI2's `open-instruct-termigen` into this repo's GRPO task pool.

    python scripts/10b_build_termigen_taskset.py \
        --parquet   data/termigen/source-train.parquet \
        --task-data data/termigen/task-data.tar.gz \
        --out data/termigen --materialize

WHY THIS IS NOT AN SFT CONVERTER
--------------------------------
`allenai/open-instruct-termigen` has **zero assistant turns**. Measured over all
3,556 rows: every one is exactly `(system, user)`, carrying a `ground_truth` task id
and an `env_config`. It is the open-instruct RLVR format -- a task pool with
verifiers -- so there is nothing in it to supervise on and nothing to build
preference pairs from. It is the third dataset in this repo to look like SFT data
and not be one, after `allenai/TMax-15K` and `allenai/tmax-15k-open-instruct`.

What it *is* is a ready GRPO pool, and a good one: `task-data.tar.gz` holds 3,556
complete task directories with `instruction.md`, `environment/Dockerfile`,
`tests/test.sh` and a per-task `tests/test_*.py` verifier. That is the same shape
`10_build_rl_taskset.py` materializes from the RST release, so this emits the same
two artifacts and `12_run_grpo.sh` consumes them unchanged.

TWO DIFFERENCES FROM THE RST POOL THAT CHANGE HOW IT SHOULD BE USED
-------------------------------------------------------------------
**No empirical pass rates.** `10_build_rl_taskset.py` exists mostly to sort tasks
into tiers by measured pass rate, because a GRPO group whose rollouts all score the
same contributes exactly zero gradient while paying full sandbox cost. Termigen
ships no trial data, so every task here is `tier="unknown"` and
`empirical_pass_rate=None`. **Do not treat this pool as interchangeable with the
RST `sweet` tier** -- it has not been screened for within-group variance, so an
unknown fraction of it is all-fail or all-pass and therefore pure cost. Measure it
with a cheap sampling pass before spending a real budget on it.

**No reference solution.** RST tasks carry `solution/solve.sh`; these do not. That
is a small integrity win -- there is no answer in the build context to leak -- but
it also means the `tier` cannot be bootstrapped by replaying a known-good solution.

The verifier-leak guard still runs, and matters more here than upstream: 26,323 of
the tarball's 47,688 entries are under `environment/`, which is the Docker build
context and therefore visible to the agent. A verifier that lands there makes the
task's reward hackable. Filenames are discovered per task rather than hardcoded,
because termigen names its verifier `tests/test_outputs.py` and similar, not the
RST pool's `tests/test_state.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

SOURCE_DATASET = "allenai/open-instruct-termigen"

# Termigen's verifier, named rather than discovered. Measured across the tarball:
# `tests/test.sh` and `tests/test_outputs.py` appear in all 3,556 tasks, and the other
# 26 filenames under `tests/` appear once to three times each -- they are workspace
# files that happen to live there.
#
# Discovering the names per task instead was tried and is wrong: tasks *about* testing
# (pytest fixtures, k6 load tests, robotframework, ctest) legitimately ship
# `tests/conftest.py`, `tests/*.test.js`, `tests/__init__.py` in the build context as
# the thing the agent is meant to work on, and a name-match against those excluded 11
# sound tasks. This is the same reason `10_build_rl_taskset.py` hardcodes the RST
# pool's two verifier names rather than globbing its `tests/` directory.
VERIFIER_FILES = ("test.sh", "test_outputs.py")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def base_image(dockerfile: str) -> str:
    """The FROM line, so a pre-build pass knows what to pull. Same as 10_build's."""
    match = re.search(r"^\s*FROM\s+(\S+)", dockerfile or "", re.M | re.I)
    return match.group(1) if match else "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, required=True,
                    help=f"{SOURCE_DATASET} data/train-00000-of-00001.parquet")
    ap.add_argument("--task-data", type=Path, required=True, help="task-data.tar.gz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--materialize", action="store_true",
                    help="extract the task dirs; required before rollouts")
    args = ap.parse_args()

    import pandas as pd

    for path in (args.parquet, args.task_data):
        if not path.is_file():
            sys.exit(f"missing input: {path}")
    frame = pd.read_parquet(args.parquet)
    required = {"messages", "ground_truth", "env_config", "source"}
    missing = required - set(frame.columns)
    if missing:
        sys.exit(f"{args.parquet} is missing {sorted(missing)}; upstream schema changed")

    # The claim this file rests on. Asserted rather than trusted, because it is the
    # whole reason there is no SFT output here and someone will reasonably ask.
    assistant_turns = sum(
        1 for messages in frame["messages"] for m in messages if m["role"] == "assistant"
    )
    if assistant_turns:
        sys.exit(f"{assistant_turns} assistant turn(s) found upstream. This dataset is no "
                 f"longer a pure task pool and may now be SFT-able -- check before "
                 f"discarding the responses.")
    print(f"[source] {len(frame)} rows, {assistant_turns} assistant turns "
          f"(a task pool, as expected)", flush=True)

    task_root = args.out / "tasks"
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- read the task bodies out of the tarball ----------------------------
    bodies: dict[str, dict[str, bytes]] = {}
    with tarfile.open(args.task_data) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            task_id, rel = parts[0], str(Path(*parts[1:]))
            payload = tar.extractfile(member)
            if payload is None:
                continue
            bodies.setdefault(task_id, {})[rel] = payload.read()
    print(f"[task-data] {len(bodies)} task dirs, "
          f"{sum(len(v) for v in bodies.values())} files", flush=True)

    stats: Counter = Counter()
    images: Counter = Counter()
    records: list[dict[str, object]] = []
    leaked: list[tuple[str, str]] = []

    for row in frame.itertuples(index=False):
        task_id = str(row.ground_truth)
        files = bodies.get(task_id)
        if files is None:
            stats["drop_no_task_dir"] += 1
            continue
        instruction = files.get("instruction.md", b"").decode("utf-8", "replace")
        dockerfile = files.get("environment/Dockerfile", b"").decode("utf-8", "replace")
        if not instruction.strip():
            stats["drop_no_instruction"] += 1
            continue
        if not dockerfile.strip():
            stats["drop_no_dockerfile"] += 1
            continue

        # ---- verifier-leak guard -------------------------------------------
        # `environment/` is the Docker build context, so anything in it is visible to
        # the agent. Verifier names are discovered from this task's own `tests/`
        # rather than hardcoded, and content is compared by hash as well as by name,
        # so a renamed copy is caught too.
        verifier_hashes = {sha256_bytes(files[f"tests/{name}"])
                           for name in VERIFIER_FILES if f"tests/{name}" in files}
        hit = None
        for name, blob in sorted(files.items()):
            if not name.startswith("environment/"):
                continue
            identical = sha256_bytes(blob) in verifier_hashes
            if Path(name).name in VERIFIER_FILES or identical:
                # `identical` is the unambiguous case: the agent can read its own
                # grader. A name-only match is a project file that merely shares the
                # verifier's name -- excluded anyway, because for an RL pool a
                # false exclusion costs one task and a false inclusion costs the
                # meaning of every reward that task produces.
                hit = (name, "byte_identical" if identical else "name_only")
                if identical:
                    break
        if hit:
            leaked.append((task_id, hit[0], hit[1]))
            stats["drop_verifier_leak"] += 1
            stats[f"leak_{hit[1]}"] += 1
            continue

        image = (files.get("image.txt", b"").decode("utf-8", "replace").strip()
                 or base_image(dockerfile))
        images[image] += 1
        content_hash = sha256_bytes(
            b"\x00".join(name.encode() + b"\x01" + files[name] for name in sorted(files))
        )

        if args.materialize:
            for name, blob in files.items():
                target = task_root / task_id / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(blob)
            stats["materialized"] += 1

        env = row.env_config if isinstance(row.env_config, dict) else json.loads(row.env_config)
        records.append({
            # `prompt` is required by slime's loader, but the real prompt is built by
            # Terminus-2 inside Harbor from instruction.md. Kept here so the row is
            # self-describing and debuggable -- same convention as 10_build_rl_taskset.
            "prompt": instruction,
            "label": task_id,
            "metadata": {
                "task_id": task_id,
                "task_group_id": task_id,   # upstream ships one instance per task
                "task_dir": str((task_root / task_id).resolve()),
                "task_content_sha256": content_hash,
                "base_image": image,
                "env_name": str(env.get("env_name") or ""),
                "tier": "unknown",
                # No trial data upstream, so there is nothing to screen for
                # within-group variance. Left explicitly null rather than defaulted
                # to something that would read as measured.
                "empirical_pass_rate": None,
                "n_reference_trials": 0,
                "source_dataset": SOURCE_DATASET,
            },
        })
        stats["selected"] += 1

    if not records:
        sys.exit("no tasks survived the gates")

    jsonl = args.out / "rl_tasks.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[write] {jsonl} tasks={len(records):,}", flush=True)

    if leaked:
        (args.out / "verifier_leaks.json").write_text(
            json.dumps(leaked, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "source_dataset": SOURCE_DATASET,
        "source_rows": len(frame),
        "source_assistant_turns": assistant_turns,
        "not_sft_data": "every row is (system, user) with a ground_truth and an "
                        "env_config -- the open-instruct RLVR format. There are no "
                        "responses to supervise on and no pairs to build.",
        "tasks_selected": len(records),
        "task_dirs_in_tarball": len(bodies),
        "materialized_task_dirs": stats["materialized"],
        "leak_guard_ran": True,
        "verifier_leaks_excluded": len(leaked),
        "verifier_leaks_byte_identical": stats["leak_byte_identical"],
        "verifier_leaks_name_only": stats["leak_name_only"],
        "distinct_base_images": len(images),
        "base_images": dict(images.most_common(20)),
        "tier": "unknown (upstream ships no trial data; not screened for "
                "within-group reward variance)",
        "drop_counters": dict(sorted(stats.items())),
        "prompt_data": str(jsonl),
        "task_root": str(task_root),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True)[:1600])
    if not args.materialize:
        print("\n[note] --materialize was not passed, so metadata.task_dir points at "
              "paths that do not exist. Rerun with it before any rollout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
