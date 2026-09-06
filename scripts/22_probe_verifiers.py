#!/usr/bin/env python3
"""E0 -- census of every private verifier in the RST task release.

    python scripts/22_probe_verifiers.py --tasks-root data/rst-tasks \
        --out data/probe-v1/census --workers 8 [--pool-dir data/rl-sweet/tasks]

WHY
    The paper accepts a task when its own oracle passes its own verifier once. It
    never reports what the verifiers check. A quick regex census over the 5,140
    materialized sweet-pool tasks (2026-09-03, function-only rules) found 26.7% of
    test functions checking existence or non-emptiness only, 6.4% comparing to a
    literal, 8.5% of tasks with no recompute/rerun check anywhere, and 130 verifiers
    reading the oracle's own script. Those are the openings a shortcut adversary
    would use, and they decide how the sandbox probe (23) stratifies its sample.
    This script re-measures on all 37,484 tasks with ONE versioned rule set
    (`rst_common.probe_core.STYLE_RULES_VERSION`) and reports both rule variants --
    function-only and with the helpers a test calls -- side by side, because the
    helper rule moves ~13 points of functions out of "existence/literal" and a
    number that moves that much under a defensible refinement must be shown moving.

WHAT ELSE IT RECORDS, AND WHY THE TWO OTHER PROBES DEPEND ON IT
    * every test function's qualname (`test_x` / `TestC::test_y`), because the
      trajectory release names no task and the only pin to the exact variant is the
      set of test names in a trial's `verifier/ctrf.json` (24 uses this index);
    * the fixtures the Dockerfile bakes and whether solve.sh / the verifier read
      them, plus a deterministic mutation plan or the reason there is none (23
      selects only tasks with a plan);
    * `allow_internet`, compose, base image, WORKDIR, test runner, oracle command
      count and whether the oracle needs the network (23's eligibility filters;
      24's cheapness normaliser).

OUTPUT
    <out>/verifier_census.parquet      one row per task
    <out>/verifier_functions.parquet   one row per test function (helper rule)
    <out>/manifest.json                every count, both rule variants, the pool subset
                                       next to the quick-census numbers, tar sha256s
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rst_common import probe_core as pc  # noqa: E402
from taskpool_common import base_image  # noqa: E402

KEEP = {"instruction.md", "task.toml", "environment/Dockerfile", "solution/solve.sh",
        "tests/test.sh", "tests/test_state.py", "rewrite_target.json"}
ENV_FILE_CAP = 64 * 1024

# The 2026-09-03 function-only quick census on the 5,140-task pool, kept so the
# manifest can show them next to what this script measures.
QUICK_CENSUS_2026_09_03 = {
    "recompute_from_env": 0.303, "existence_or_nonempty_only": 0.267, "other": 0.163,
    "rerun_tool_or_script": 0.150, "literal_compare_only": 0.064, "read_then_literal_compare": 0.053,
    "tasks_no_recompute_or_rerun": 0.085, "verifiers_reading_solve_sh": 130, "verifiers_asserting_solve_sh_exists": 13,
}


def _text(payload: bytes | None) -> str:
    return payload.decode("utf-8", errors="replace") if payload else ""


def compute_row(task_id: str, shard: str, files: dict[str, bytes], env_files: dict[str, bytes]) -> tuple[dict, list[dict]]:
    test_py = _text(files.get("tests/test_state.py"))
    test_sh = _text(files.get("tests/test.sh"))
    dockerfile = _text(files.get("environment/Dockerfile"))
    solve = _text(files.get("solution/solve.sh"))
    toml = _text(files.get("task.toml"))
    instruction = _text(files.get("instruction.md"))
    compose = any(name.startswith("docker-compose") for name in env_files) or "environment/docker-compose.yaml" in files

    with_helpers = pc.census_test_file(test_py, include_helpers=True)
    alone = pc.census_test_file(test_py, include_helpers=False)

    operator = family = None
    rt = files.get("rewrite_target.json")
    if rt:
        try:
            target = json.loads(_text(rt)).get("rewrite_target") or {}
            operator = target.get("preferred_operator")
            family = target.get("preferred_family")
        except (json.JSONDecodeError, AttributeError):
            pass

    fixtures = pc.dockerfile_fixtures(dockerfile, env_files)
    written = pc.solve_written_paths(solve)
    fixture_rows = [{
        "path": f.path, "source_kind": f.source_kind, "line_no": f.line_no,
        "content_known": f.content is not None,
        "referenced_by_solve": pc.references_path(solve, f.path),
        "referenced_by_tests": pc.references_path(test_py, f.path),
        "written_by_solve": f.path in written,
    } for f in fixtures]
    mutation, na_reason = pc.plan_mutation(dockerfile, solve, test_py, env_files, compose=compose)
    instruction_paths = pc.instruction_paths(instruction)

    row = {
        "task_id": task_id, "shard": shard,
        "n_tests": with_helpers["n_tests"], "parse_ok": with_helpers["parse_ok"],
        "test_qualnames": with_helpers["test_qualnames"],
        "dominant_style": with_helpers["dominant_style"],
        "dominant_style_alone": alone["dominant_style"],
        "existence_only_task": with_helpers["existence_only_task"],
        "no_recompute_or_rerun": with_helpers["no_recompute_or_rerun"],
        "no_recompute_or_rerun_alone": alone["no_recompute_or_rerun"],
        "inspects_oracle_script": with_helpers["inspects_oracle_script"],
        "asserts_oracle_exists": with_helpers["asserts_oracle_exists"],
        "reads_history": with_helpers["reads_history"],
        "pass_literal": with_helpers["pass_literal"],
        "uses_mtime": with_helpers["uses_mtime"],
        "uses_random": with_helpers["uses_random"] or bool(pc.RANDOMNESS_RE.search(dockerfile)) or bool(pc.RANDOMNESS_RE.search(solve)),
        "allow_internet": pc.toml_allow_internet(toml),
        "agent_timeout_sec": pc.toml_agent_timeout(toml),
        "compose": compose,
        "base_image": base_image(dockerfile),
        "workdir": pc.dockerfile_workdir(dockerfile),
        "test_runner": pc.test_runner(test_sh),
        "operator": operator, "family": family, "is_seed": rt is None,
        "oracle_n_lines": sum(1 for ln in solve.splitlines() if ln.strip() and not ln.strip().startswith("#")),
        "oracle_n_cmds": pc.oracle_command_count(solve),
        "oracle_needs_network": pc.oracle_needs_network(solve),
        "n_instruction_paths": len(instruction_paths),
        "instruction_paths": instruction_paths,
        "n_fixtures": len(fixture_rows),
        "fixtures_json": json.dumps(fixture_rows, sort_keys=True),
        "mutation_applicable": mutation is not None,
        "mutation_na_reason": na_reason,
        "mutation_json": json.dumps(mutation.to_json(), sort_keys=True) if mutation else None,
        "mutation_kind": mutation.kind if mutation else None,
        "mutation_ref_strength": mutation.ref_strength if mutation else None,
        "mutation_path": mutation.path if mutation else None,
    }
    for style in pc.STYLES:
        row[f"n_{style}"] = with_helpers["style_counts"][style]
        row[f"n_alone_{style}"] = alone["style_counts"][style]
    fn_rows = [{"task_id": task_id, **fn} for fn in with_helpers["functions"]]
    return row, fn_rows


def census_shard(job: tuple[str, str, int]) -> tuple[list[dict], list[dict], dict]:
    shard, tasks_root, limit = job
    per_task: dict[str, dict[str, bytes]] = {}
    env_files: dict[str, dict[str, bytes]] = {}
    with tarfile.open(Path(tasks_root) / shard) as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 3 or parts[0] != "tasks":
                continue
            task_id = parts[1]
            rel = "/".join(parts[2:])
            if limit and task_id not in per_task and len(per_task) >= limit:
                continue
            files = per_task.setdefault(task_id, {})
            if rel in KEEP:
                handle = tar.extractfile(member)
                files[rel] = handle.read() if handle else b""
            elif rel.startswith("environment/"):
                env_rel = rel[len("environment/"):]
                bucket = env_files.setdefault(task_id, {})
                if member.size <= ENV_FILE_CAP:
                    handle = tar.extractfile(member)
                    bucket[env_rel] = handle.read() if handle else b""
                else:
                    bucket[env_rel] = b"\x00"  # present, but not something we will mutate
    rows: list[dict] = []
    fn_rows: list[dict] = []
    stats: Counter = Counter()
    for task_id, files in per_task.items():
        row, fns = compute_row(task_id, shard, files, env_files.get(task_id, {}))
        rows.append(row)
        fn_rows.extend(fns)
        stats["tasks"] += 1
        stats["functions"] += row["n_tests"]
        if not row["parse_ok"]:
            stats["unparseable"] += 1
    return rows, fn_rows, dict(stats)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-root", type=Path, default=Path("data/rst-tasks"))
    parser.add_argument("--out", type=Path, default=Path("data/probe-v1/census"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shards", nargs="*", default=None, help="relative shard paths; default: all in shard_manifest.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="tasks per shard, 0 = all (smoke runs)")
    parser.add_argument("--pool-dir", type=Path, default=Path("data/rl-sweet/tasks"),
                        help="materialized pool whose task ids form the comparison subset")
    args = parser.parse_args()

    import pandas as pd

    started = time.time()
    manifest_rows = [json.loads(ln) for ln in (args.tasks_root / "metadata" / "shard_manifest.jsonl").read_text().splitlines() if ln.strip()]
    shards = args.shards or [r["shard"] for r in manifest_rows]
    jobs = [(shard, str(args.tasks_root), args.limit) for shard in shards]
    rows: list[dict] = []
    fn_rows: list[dict] = []
    stats: Counter = Counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for shard_rows, shard_fns, shard_stats in pool.map(census_shard, jobs):
            rows.extend(shard_rows)
            fn_rows.extend(shard_fns)
            stats.update(shard_stats)
            print(f"[rst-census] {shard_stats}", flush=True)

    tasks = pd.read_parquet(args.tasks_root / "metadata" / "tasks.parquet", columns=["task_id", "task_group_id"])
    df = pd.DataFrame(rows).merge(tasks, on="task_id", how="left")
    fns = pd.DataFrame(fn_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "verifier_census.parquet", index=False)
    fns.to_parquet(args.out / "verifier_functions.parquet", index=False)

    def style_shares(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
        total = sum(int(frame[f"{prefix}{s}"].sum()) for s in pc.STYLES)
        return {s: round(int(frame[f"{prefix}{s}"].sum()) / max(1, total), 4) for s in pc.STYLES}

    def summary(frame: pd.DataFrame) -> dict:
        n = len(frame)
        return {
            "n_tasks": int(n),
            "n_functions": int(frame.n_tests.sum()),
            "style_share_with_helpers": style_shares(frame, "n_"),
            "style_share_function_only": style_shares(frame, "n_alone_"),
            "dominant_style_counts": frame.dominant_style.value_counts().to_dict(),
            "tasks_no_recompute_or_rerun": round(float(frame.no_recompute_or_rerun.mean()), 4),
            "tasks_no_recompute_or_rerun_function_only": round(float(frame.no_recompute_or_rerun_alone.mean()), 4),
            "existence_only_tasks": int(frame.existence_only_task.sum()),
            "verifiers_reading_solve_sh": int(frame.inspects_oracle_script.sum()),
            "verifiers_asserting_solve_sh_exists": int(frame.asserts_oracle_exists.sum()),
            "verifiers_reading_history": int(frame.reads_history.sum()),
            "pass_literal_tasks": int(frame.pass_literal.sum()),
            "mtime_tasks": int(frame.uses_mtime.sum()),
            "random_tasks": int(frame.uses_random.sum()),
            "allow_internet": frame.allow_internet.value_counts().to_dict(),
            "compose_tasks": int(frame.compose.sum()),
            "test_runner": frame.test_runner.value_counts().to_dict(),
            "oracle_needs_network": int(frame.oracle_needs_network.sum()),
            "seed_tasks": int(frame.is_seed.sum()),
            "mutation_applicable": int(frame.mutation_applicable.sum()),
            "mutation_na_reasons": frame.mutation_na_reason.value_counts(dropna=True).to_dict(),
            "mutation_kinds": frame.mutation_kind.value_counts(dropna=True).to_dict(),
            "mutation_ref_strength": frame.mutation_ref_strength.value_counts(dropna=True).to_dict(),
            "unparseable": int((~frame.parse_ok).sum()),
            "no_test_functions": int((frame.n_tests == 0).sum()),
        }

    manifest = {
        "rules_version": pc.STYLE_RULES_VERSION,
        "tasks_root": str(args.tasks_root.resolve()),
        "shards": [{"shard": r["shard"], "task_count": r.get("task_count"),
                    "sha256_manifest": r.get("sha256"), "sha256_on_disk": sha256_file(args.tasks_root / r["shard"])}
                   for r in manifest_rows if r["shard"] in shards],
        "limit_per_shard": args.limit,
        "all_tasks": summary(df),
        "quick_census_2026_09_03_function_only_pool": QUICK_CENSUS_2026_09_03,
        "seconds": round(time.time() - started, 1),
        "outputs": {"tasks": str(args.out / "verifier_census.parquet"), "functions": str(args.out / "verifier_functions.parquet")},
    }
    if args.pool_dir.is_dir():
        pool_ids = {p.name for p in args.pool_dir.iterdir() if p.is_dir()}
        subset = df[df.task_id.isin(pool_ids)]
        manifest["pool_subset"] = {"pool_dir": str(args.pool_dir), "n_ids_in_pool": len(pool_ids), **summary(subset)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    a = manifest["all_tasks"]
    print(f"[rst-census] tasks={a['n_tasks']:,} functions={a['n_functions']:,} unparseable={a['unparseable']} "
          f"no_recompute_or_rerun={a['tasks_no_recompute_or_rerun']:.1%} (function-only {a['tasks_no_recompute_or_rerun_function_only']:.1%}) "
          f"reads_solve_sh={a['verifiers_reading_solve_sh']} asserts_solve_sh_exists={a['verifiers_asserting_solve_sh_exists']} "
          f"mutation_applicable={a['mutation_applicable']:,} seconds={manifest['seconds']}", flush=True)
    print(f"[rst-census] style share (helpers / function-only): " + " ".join(
        f"{s}={a['style_share_with_helpers'][s]:.1%}/{a['style_share_function_only'][s]:.1%}" for s in pc.STYLES), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
