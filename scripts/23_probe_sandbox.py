#!/usr/bin/env python3
"""E1 -- sandbox probe of RST tasks: is the oracle a general solution, does the verifier recompute?

    source scripts/00b_setup_sandbox.sh          # DOCKER_HOST, RST_HARBOR_ENV_KWARGS
    python scripts/23_probe_sandbox.py smoke  --task /tmp/tbtasks/terminal-bench/password-recovery
    python scripts/23_probe_sandbox.py select --n 20
    python scripts/23_probe_sandbox.py plan-mutation
    python scripts/23_probe_sandbox.py run    --concurrency 4
    python scripts/23_probe_sandbox.py report

WHAT ONE TASK GOES THROUGH (five Harbor trials, sequential)
    nop            an agent that does nothing, on the original task. Control: the
                   verifier must score 0. A 1 is a vacuous verifier, and a finding.
    oracle         `SnapshotOracleAgent`: Harbor's own oracle behaviour (upload
                   solution/, run solve.sh) plus a before/after listing of the
                   filesystem, so the files the oracle created or changed are known
                   and tarred. Baseline: must score 1 or the local harness does not
                   reproduce the paper's acceptance and the task is uninterpretable.
    touch          `TouchPathsAgent`: create every absolute path the instruction
                   names that does not already exist, with empty content. The
                   budget-1, black-box adversary; its pass rate is the empirical
                   share of verifiers satisfied by existence alone.
    oracle_mut     the same oracle on a COPY of the task whose Dockerfile gained one
                   layer rewriting one input fixture by one token (22's plan). Tests
                   whether solve.sh is a general solution or a script for one instance.
    artifacts_mut  `ReplayArtifactsAgent`: the ORIGINAL oracle's output files,
                   extracted into the mutated environment, nothing executed. Tests
                   whether the verifier recomputes from the instance or compares to
                   constants. The mutated fixture itself is never restored.

    (oracle_mut, artifacts_mut) = (1,0) parametric_ok · (1,1) verifier_fixture_insensitive
    · (0,0) oracle_brittle · (0,1) oracle_brittle_and_verifier_constant; anything with a
    failed control or an infra outcome is `uninterpretable:<why>`, never a cell.

WHY HARBOR TRIALS AND NOT `docker exec`
    Every trial gets Harbor's verifier, the same `result.json`/`ctrf.json` shape as
    every other rollout in this repo, and `rst_common.harbor.read_reward`'s
    infra-vs-reward classification: a failed apt build is "unmeasured", never a 0.
    Every command line goes through `rst_common.harbor.run_argv` (tests enforce it).

WHAT THE NUMBERS MEAN, AND DO NOT
    * `cpu_enforcement_policy=ignore` is required on this box (rootless docker,
      cgroup `cpu` not delegated): tasks get the whole machine instead of their
      declared cpus. Recorded in the manifest; it can only help an agent.
    * The pool is the sweet band (10-90% solver pass rate), not the release: the
      sample says nothing about all-fail or saturated tasks.
    * A `(1,1)` cell means "verifier ignores the fixture" only when the fixture is
      actually consumed; the plan's `ref_strength` (both / tests_only / solve_only)
      is carried into the report so the cell can be read at the right confidence.
    * Images are not pruned automatically (975 GB free); `docker system df` before
      and after is recorded instead.

OUTPUT (<out> = data/probe-v1/sandbox)
    selection.json, mutations.json, mutated/<task_id>/ (task copies), jobs/<task>/<arm>/,
    outcomes.jsonl (append-only, resumable), outcomes.parquet, manifest.json,
    and the E1 section of reports/probe_v1_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rst_common import probe_core as pc  # noqa: E402
from rst_common.harbor import (  # noqa: E402
    PROBE_AGENT_IMPORT_PATHS,
    apply_proxy_policy,
    custom_agent_env,
    harbor_python,
    read_reward,
    refine_with_stdout,
    result_json_candidates,
    run_argv,
)

ARMS = list(pc.ARMS)
DEFAULT_HARBOR = str(Path.home() / ".venvs" / "AgentGen" / "bin" / "harbor")
CPU_CAVEAT = ("cpu_enforcement_policy=ignore: cgroup v2 'cpu' is not delegated to this user, so the "
              "rootless daemon cannot apply each task's declared cpus= limit. Tasks run with the whole "
              "box's CPU instead of their declared budget; this can only help the agent.")
STRATUM_OF = {
    "existence_or_nonempty_only": "existence",
    "literal_compare_only": "literal",
    "read_then_literal_compare": "literal",
    "recompute_from_env": "recompute",
    "rerun_tool_or_script": "rerun",
    "mixed": "mixed",
    "other": "mixed",
}
DEFAULT_STRATA = ["existence", "literal", "recompute", "rerun", "mixed"]


# ------------------------------------------------------------------ context

class ProbeContext:
    """Everything an arm needs to launch one Harbor trial and read it back."""

    def __init__(self, args):
        self.out = Path(args.out)
        self.harbor_bin = args.harbor_bin
        self.harbor_env = args.harbor_env
        self.trial_timeout = args.trial_timeout
        self.env_kwargs = harbor_env_kwargs(args)
        self.env = dict(os.environ)
        docker_host = args.docker_host or os.environ.get("RST_DOCKER_HOST") or os.environ.get("DOCKER_HOST")
        if docker_host:
            self.env["DOCKER_HOST"] = docker_host
        self.env.setdefault("RST_BUILD_CMD", "docker")
        apply_proxy_policy(self.env, self.harbor_env, "")
        custom_agent_env(self.env, ROOT)
        self.lock = threading.Lock()
        self.outcomes_path = self.out / "outcomes.jsonl"

    def preflight(self) -> None:
        if "DOCKER_HOST" not in self.env:
            raise SystemExit("no DOCKER_HOST: `source scripts/00b_setup_sandbox.sh` or pass --docker-host")
        if not any(k.startswith("cpu_enforcement_policy=") for k in self.env_kwargs):
            print(f"[rst-probe] WARNING: no cpu_enforcement_policy kwarg; on this box containers will not start", flush=True)
        py = harbor_python(self.harbor_bin)
        proc = subprocess.run([str(py), "-c", "import rst_common.probe_agents"], env=self.env,
                              capture_output=True, text=True, timeout=120, check=False)
        if proc.returncode != 0:
            raise SystemExit(f"probe agents do not import in harbor's interpreter ({py}):\n{proc.stderr[-2000:]}")
        print(f"[rst-probe] preflight ok: harbor={self.harbor_bin} python={py} DOCKER_HOST={self.env['DOCKER_HOST']} "
              f"env_kwargs={self.env_kwargs}", flush=True)

    def append_outcome(self, record: dict) -> None:
        with self.lock:
            self.outcomes_path.parent.mkdir(parents=True, exist_ok=True)
            with self.outcomes_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def harbor_env_kwargs(args) -> list[str]:
    """`--environment-kwarg` values: the flag, then RST_HARBOR_ENV_KWARGS (flag wins per key)."""
    pairs = list(getattr(args, "harbor_env_kwarg", None) or [])
    pairs += [tok for tok in os.environ.get("RST_HARBOR_ENV_KWARGS", "").split() if "=" in tok]
    seen: set[str] = set()
    out: list[str] = []
    for pair in pairs:
        key = pair.split("=", 1)[0]
        if key not in seen:
            seen.add(key)
            out.append(pair)
    return out


def load_outcomes(path: Path) -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                latest[(rec["task_id"], rec["arm"])] = rec
    return latest


# --------------------------------------------------------------------- arms

def arm_spec(arm: str, task_id: str, orig_dir: Path, mut_dir: Path | None, mutation: dict | None,
             oracle_tgz: Path | None, agent_timeout: float | None) -> tuple[Path, str, list[str]]:
    """`(task_dir, agent, agent_kwargs)` for one arm. Everything an agent needs is a kwarg."""
    fixture = (mutation or {}).get("path") or ""
    timeout = str(int(agent_timeout or 900))
    if arm == "nop":
        return orig_dir, "nop", []
    if arm == "oracle_builtin":
        return orig_dir, "oracle", []
    if arm == "oracle":
        return orig_dir, PROBE_AGENT_IMPORT_PATHS["snapshot_oracle"], [
            f"task_dir={orig_dir.resolve()}", f"fixture_paths={fixture}", f"agent_timeout_sec={timeout}"]
    if arm == "touch":
        return orig_dir, PROBE_AGENT_IMPORT_PATHS["touch_paths"], ["mode=empty"]
    if arm == "oracle_mut":
        if mut_dir is None:
            raise ValueError(f"{task_id}: oracle_mut needs a mutated copy")
        return mut_dir, PROBE_AGENT_IMPORT_PATHS["snapshot_oracle"], [
            f"task_dir={mut_dir.resolve()}", f"fixture_paths={fixture}", f"agent_timeout_sec={timeout}"]
    if arm == "artifacts_mut":
        if mut_dir is None or oracle_tgz is None:
            raise ValueError(f"{task_id}: artifacts_mut needs a mutated copy and the oracle's tarball")
        return mut_dir, PROBE_AGENT_IMPORT_PATHS["replay_artifacts"], [
            f"artifacts_tgz={oracle_tgz.resolve()}", f"exclude_paths={fixture}"]
    raise ValueError(f"unknown arm {arm}")


def trial_dir_of(job_dir: Path) -> Path | None:
    trials = [p.parent for p in result_json_candidates(job_dir) if p.parent != job_dir]
    return trials[0] if trials else None


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_arm(ctx: ProbeContext, task_id: str, arm: str, task_dir: Path, agent: str,
            agent_kwargs: list[str], label: str | None = None) -> dict:
    """One Harbor trial. Returns the outcome record (also appended to outcomes.jsonl)."""
    label = label or arm
    job_root = ctx.out / "jobs" / task_id
    jobs_dir = job_root / label
    if jobs_dir.exists():
        shutil.rmtree(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_name = f"{task_id}-{label}"[:96]
    argv = run_argv(harbor_bin=ctx.harbor_bin, task_dir=task_dir, agent=agent, model=None,
                    env=ctx.harbor_env, jobs_dir=jobs_dir, job_name=job_name,
                    env_kwargs=ctx.env_kwargs, agent_kwargs=agent_kwargs)
    started = time.time()
    stdout = ""
    try:
        proc = subprocess.run(argv, env=ctx.env, capture_output=True, text=True,
                              timeout=ctx.trial_timeout, check=False)
        stdout = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = ((exc.stdout or b"") if isinstance(exc.stdout, bytes) else (exc.stdout or "")) or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        stdout += f"\n[rst-probe] harbor subprocess timed out after {ctx.trial_timeout}s"
        rc = -9
    seconds = round(time.time() - started, 1)
    job_dir = jobs_dir / job_name
    outcome = refine_with_stdout(read_reward(job_dir), stdout)
    trial = trial_dir_of(job_dir)
    record = {
        "task_id": task_id, "arm": label, "agent": agent, "task_dir": str(task_dir),
        "reward": outcome.reward, "kind": outcome.kind, "reason": outcome.reason,
        "scorable": outcome.scorable, "seconds": seconds, "harbor_rc": rc,
        "job_dir": str(job_dir), "trial_dir": str(trial) if trial else None,
        "stdout_tail": stdout[-1500:],
    }
    if trial:
        ctrf = read_json(trial / "verifier" / "ctrf.json")
        summary = pc.ctrf_summary(ctrf)
        record.update({"ctrf_total": summary["total"], "ctrf_passed": summary["passed"],
                       "ctrf_failed_names": summary["failed_names"]})
        test_stdout = trial / "verifier" / "test-stdout.txt"
        text = test_stdout.read_text(errors="replace") if test_stdout.is_file() else ""
        record["verifier_bootstrap_ok"] = ("passed" in text or "failed" in text or "error" in text) and "pytest" in text.lower() or bool(ctrf)
        agent_dir = trial / "agent"
        exit_code = agent_dir / "exit-code.txt"
        record["agent_exit_code"] = int(exit_code.read_text().strip()) if exit_code.is_file() else 0
        for name in ("snapshot.json", "touch.json", "replay.json", "replay-artifacts.json"):
            payload = read_json(agent_dir / name)
            if payload is not None:
                record[name.replace(".json", "").replace("-", "_")] = payload
        sha = agent_dir / "fixture-before.sha256"
        if sha.is_file():
            record["fixture_sha256"] = sha.read_text().split()[0] if sha.read_text().strip() else None
        tgz = agent_dir / "oracle-artifacts.tgz"
        record["artifacts_tgz"] = str(tgz) if tgz.is_file() else None
    ctx.append_outcome(record)
    print(f"[rst-probe] {task_id} {label:14s} reward={outcome.reward} kind={outcome.kind or '-'} "
          f"{seconds:6.0f}s {('reason=' + str(outcome.reason)) if outcome.reason else ''}", flush=True)
    return record


def run_task(ctx: ProbeContext, task_id: str, orig_dir: Path, mut_dir: Path | None, mutation: dict | None,
             arms: list[str], existing: dict[tuple[str, str], dict], force: bool, rerun_infra: bool,
             agent_timeout: float | None) -> None:
    oracle_tgz: Path | None = None
    prev = existing.get((task_id, "oracle"))
    if prev and prev.get("artifacts_tgz"):
        oracle_tgz = Path(prev["artifacts_tgz"])
    for arm in arms:
        prior = existing.get((task_id, arm))
        if prior and not force and not (rerun_infra and not prior.get("scorable")):
            if arm == "oracle" and prior.get("artifacts_tgz"):
                oracle_tgz = Path(prior["artifacts_tgz"])
            continue
        if arm in ("oracle_mut", "artifacts_mut") and mut_dir is None:
            ctx.append_outcome({"task_id": task_id, "arm": arm, "reward": None, "kind": None,
                                "reason": "no_mutation_plan", "scorable": False, "skipped": True})
            continue
        if arm == "artifacts_mut" and (oracle_tgz is None or not oracle_tgz.is_file()):
            ctx.append_outcome({"task_id": task_id, "arm": arm, "reward": None, "kind": None,
                                "reason": "no_oracle_artifacts", "scorable": False, "skipped": True})
            continue
        task_dir, agent, kwargs = arm_spec(arm, task_id, orig_dir, mut_dir, mutation, oracle_tgz, agent_timeout)
        record = run_arm(ctx, task_id, arm, task_dir, agent, kwargs)
        if arm == "oracle" and record.get("artifacts_tgz"):
            oracle_tgz = Path(record["artifacts_tgz"])


# ---------------------------------------------------------------- selection

def select_tasks(census, pool_dir: Path, n: int, seed: int, strata: list[str], top_images: int) -> dict:
    """Stratified, deterministic, one task per group. Returns the selection with its funnel."""
    df = census.copy()
    pool_ids = {p.name for p in pool_dir.iterdir() if p.is_dir()} if pool_dir.is_dir() else set()
    funnel: dict[str, int] = {"census": len(df)}
    df = df[df.task_id.isin(pool_ids)]
    funnel["in_pool"] = len(df)
    steps = [
        ("not_compose", ~df.compose),
        ("allow_internet_not_false", df.allow_internet != "false"),
        ("oracle_offline", ~df.oracle_needs_network),
        ("has_tests", df.n_tests >= 1),
        ("known_runner", df.test_runner.isin(["uvx", "pytest", "python -m pytest"])),
        ("mutation_applicable", df.mutation_applicable),
    ]
    for name, mask in steps:
        df = df[mask.loc[df.index]]
        funnel[name] = len(df)
    top = df.base_image.value_counts().head(top_images).index.tolist()
    df = df[df.base_image.isin(top)]
    funnel[f"top{top_images}_base_images"] = len(df)
    df = df.assign(stratum=df.dominant_style.map(STRATUM_OF).fillna("mixed"),
                   order=[hashlib.sha256(f"{seed}:{t}".encode()).hexdigest() for t in df.task_id])
    df = df.sort_values("order").drop_duplicates("task_group_id")
    funnel["one_per_group"] = len(df)
    per = max(1, n // len(strata))
    chosen: list[dict] = []
    short: dict[str, int] = {}
    for stratum in strata:
        rows = df[df.stratum == stratum].head(per)
        short[stratum] = per - len(rows)
        chosen += rows.to_dict("records")
    if len(chosen) < n:  # fill from whatever is left, still deterministic
        taken = {r["task_id"] for r in chosen}
        rest = df[~df.task_id.isin(taken)].head(n - len(chosen))
        chosen += rest.to_dict("records")
    chosen = chosen[:n]
    return {
        "n": n, "seed": seed, "strata": strata, "per_stratum": per, "stratum_shortfall": short,
        "top_base_images": top, "funnel": funnel,
        "tasks": [{"task_id": r["task_id"], "task_group_id": r["task_group_id"], "stratum": r["stratum"],
                   "dominant_style": r["dominant_style"], "base_image": r["base_image"],
                   "mutation_kind": r.get("mutation_kind"), "mutation_ref_strength": r.get("mutation_ref_strength"),
                   "agent_timeout_sec": r.get("agent_timeout_sec"), "operator": r.get("operator")}
                  for r in chosen],
    }


def materialize_mutation(pool_dir: Path, out_dir: Path, task_id: str, mutation: dict) -> Path:
    """Copy the task and append the mutation layer; the copy keeps the task's dir name."""
    src = pool_dir / task_id
    dst = out_dir / task_id
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    dockerfile = dst / "environment" / "Dockerfile"
    mut = pc.Mutation(**mutation)
    dockerfile.write_text(pc.patch_dockerfile(dockerfile.read_text(encoding="utf-8", errors="replace"), mut), encoding="utf-8")
    (dst / "mutation.json").write_text(json.dumps(mutation, indent=2, sort_keys=True), encoding="utf-8")
    return dst


def docker_df(env: dict[str, str]) -> str:
    try:
        return subprocess.run(["docker", "system", "df"], env=env, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def harbor_version(harbor_bin: str) -> str:
    try:
        return subprocess.run([harbor_bin, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "?"


# ------------------------------------------------------------------ report

def build_report(out: Path, census_path: Path | None) -> dict:
    import pandas as pd

    records = list(load_outcomes(out / "outcomes.jsonl").values())
    if not records:
        raise SystemExit("no outcomes yet")
    df = pd.DataFrame(records)
    selection = read_json(out / "selection.json") or {}
    sel_by_id = {t["task_id"]: t for t in selection.get("tasks", [])}
    mutations = read_json(out / "mutations.json") or {}
    per_task: list[dict] = []
    for task_id, group in df.groupby("task_id"):
        arms: dict[str, tuple] = {}
        shas: dict[str, str | None] = {}
        for rec in group.to_dict("records"):
            arms[rec["arm"]] = (rec.get("reward"), bool(rec.get("scorable")))
            shas[rec["arm"]] = rec.get("fixture_sha256")
        applied = bool(shas.get("oracle") and shas.get("oracle_mut") and shas["oracle"] != shas["oracle_mut"])
        cell = pc.classify_task_outcome(arms, applied) if all(a in arms for a in ARMS) else "uninterpretable:incomplete"
        meta = sel_by_id.get(task_id, {})
        mut = (mutations.get("plans") or {}).get(task_id) or {}
        per_task.append({
            "task_id": task_id, "cell": cell, "mutation_applied": applied,
            "stratum": meta.get("stratum"), "dominant_style": meta.get("dominant_style"),
            "mutation_kind": mut.get("kind"), "ref_strength": mut.get("ref_strength"), "fixture": mut.get("path"),
            **{f"{a}_reward": arms.get(a, (None, False))[0] for a in ARMS},
            **{f"{a}_scorable": arms.get(a, (None, False))[1] for a in ARMS},
        })
    tasks = pd.DataFrame(per_task)
    df.to_parquet(out / "outcomes.parquet", index=False)
    tasks.to_parquet(out / "task_cells.parquet", index=False)

    scorable = df[df.scorable.fillna(False)]
    by_arm = {}
    for arm, g in df.groupby("arm"):
        s = g[g.scorable.fillna(False)]
        by_arm[arm] = {"n": int(len(g)), "scorable": int(len(s)), "infra": int((g.kind == "harness_infra").sum()),
                       "budget": int((g.kind == "agent_budget").sum()), "skipped": int(g.get("skipped", pd.Series(dtype=bool)).fillna(False).sum()) if "skipped" in g else 0,
                       "pass_rate": (round(float((s.reward >= 1).mean()), 3) if len(s) else None),
                       "median_seconds": (float(g.seconds.median()) if "seconds" in g and g.seconds.notna().any() else None)}
    cells = tasks.cell.value_counts().to_dict()
    interpretable = tasks[~tasks.cell.str.startswith("uninterpretable")]
    touch = df[(df.arm == "touch") & df.scorable.fillna(False)].merge(tasks[["task_id", "dominant_style", "stratum"]], on="task_id", how="left")
    manifest = {
        "arms": by_arm,
        "cells": cells,
        "cells_by_ref_strength": {k: v.cell.value_counts().to_dict() for k, v in interpretable.groupby("ref_strength")} if len(interpretable) else {},
        "cells_by_mutation_kind": {k: v.cell.value_counts().to_dict() for k, v in interpretable.groupby("mutation_kind")} if len(interpretable) else {},
        "touch_pass_rate": (round(float((touch.reward >= 1).mean()), 3) if len(touch) else None),
        "touch_pass_by_stratum": {k: round(float((v.reward >= 1).mean()), 3) for k, v in touch.groupby("stratum")} if len(touch) else {},
        "nop_passes": int(((df.arm == "nop") & (df.reward >= 1)).sum()),
        "n_tasks": int(tasks.task_id.nunique()),
        "n_trials": int(len(df)),
    }
    lines = [
        "## E1 -- sandbox probe (oracle generality, verifier recomputation, budget-1 adversary)", "",
        f"Tasks: {manifest['n_tasks']} (sweet pool, stratified by verifier style) · trials: {manifest['n_trials']} · "
        f"harbor `--env docker` on rootless docker with `cpu_enforcement_policy=ignore` (whole-box CPU; can only help the agent).", "",
        "| arm | trials | scorable | infra | pass rate | median s |", "|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        a = by_arm.get(arm)
        if a:
            lines.append(f"| {arm} | {a['n']} | {a['scorable']} | {a['infra']} | {a['pass_rate']} | {a['median_seconds']} |")
    lines += ["", "2x2 over interpretable tasks (nop=0, oracle=1, mutation landed, all arms scorable):", ""]
    for cell, count in sorted(cells.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{cell}`: {count}")
    lines += ["", f"Touch adversary pass rate: {manifest['touch_pass_rate']} (by stratum: {manifest['touch_pass_by_stratum']}) · "
              f"nop passes: {manifest['nop_passes']}", ""]
    if len(interpretable):
        lines += ["| task | stratum | fixture | kind | ref | cell |", "|---|---|---|---|---|---|"]
        for r in interpretable.sort_values("cell").to_dict("records"):
            lines.append(f"| {r['task_id']} | {r['stratum']} | `{r['fixture']}` | {r['mutation_kind']} | {r['ref_strength']} | {r['cell']} |")
    (out / "report_E1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


# -------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/probe-v1/sandbox"))
    parser.add_argument("--census", type=Path, default=Path("data/probe-v1/census/verifier_census.parquet"))
    parser.add_argument("--pool", type=Path, default=Path("data/rl-sweet/tasks"))
    parser.add_argument("--harbor-bin", default=os.environ.get("HARBOR_BIN", DEFAULT_HARBOR))
    parser.add_argument("--harbor-env", default=os.environ.get("RST_HARBOR_ENV", "docker"))
    parser.add_argument("--harbor-env-kwarg", action="append", default=None)
    parser.add_argument("--docker-host", default=None)
    parser.add_argument("--trial-timeout", type=int, default=2400, help="seconds per harbor subprocess incl. image build")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select");           s.add_argument("--n", type=int, default=20); s.add_argument("--seed", type=int, default=20260903)
    s.add_argument("--strata", default=",".join(DEFAULT_STRATA)); s.add_argument("--top-images", type=int, default=8)
    p = sub.add_parser("plan-mutation");    p.add_argument("--task-ids", default=None)
    r = sub.add_parser("run");              r.add_argument("--arms", default=",".join(ARMS)); r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--task-ids", default=None); r.add_argument("--force", action="store_true"); r.add_argument("--rerun-infra", action="store_true")
    sub.add_parser("report")
    k = sub.add_parser("smoke");            k.add_argument("--task", type=Path, required=True)
    k.add_argument("--arms", default="nop,oracle_builtin,oracle,touch"); k.add_argument("--keystrokes", default=None,
                   help="JSON list of keystrokes to replay as an extra arm")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.cmd == "select":
        import pandas as pd
        census = pd.read_parquet(args.census)
        strata = [x for x in args.strata.split(",") if x]
        selection = select_tasks(census, args.pool, args.n, args.seed, strata, args.top_images)
        (args.out / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(f"[rst-probe] selected {len(selection['tasks'])} tasks; funnel={selection['funnel']} shortfall={selection['stratum_shortfall']}")
        for t in selection["tasks"]:
            print(f"  {t['task_id']} {t['stratum']:10s} {t['dominant_style']:28s} {t['base_image']:22s} {t['mutation_kind']} {t['mutation_ref_strength']}")
        return 0

    if args.cmd == "plan-mutation":
        import pandas as pd
        selection = read_json(args.out / "selection.json") or {"tasks": []}
        ids = args.task_ids.split(",") if args.task_ids else [t["task_id"] for t in selection["tasks"]]
        census = pd.read_parquet(args.census).set_index("task_id")
        plans: dict[str, dict] = {}
        na: Counter = Counter()
        for tid in ids:
            row = census.loc[tid] if tid in census.index else None
            mutation_json = row.mutation_json if row is not None else None
            if not mutation_json:
                na[(row.mutation_na_reason if row is not None else "not_in_census")] += 1
                continue
            mutation = json.loads(mutation_json)
            materialize_mutation(args.pool, args.out / "mutated", tid, mutation)
            plans[tid] = mutation
            print(f"[rst-probe] {tid}: {mutation['path']} line {mutation['line_no']} {mutation['before']!r} -> {mutation['after']!r} "
                  f"({mutation['kind']}, {mutation['ref_strength']})")
        (args.out / "mutations.json").write_text(json.dumps({"plans": plans, "na": dict(na)}, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[rst-probe] mutation plans: {len(plans)} applicable, na={dict(na)}")
        return 0

    if args.cmd in ("run", "smoke"):
        ctx = ProbeContext(args)
        ctx.preflight()
        started = time.time()
        df_before = docker_df(ctx.env)
        if args.cmd == "smoke":
            task_dir = args.task.resolve()
            task_id = task_dir.name
            toml = (task_dir / "task.toml").read_text(errors="replace") if (task_dir / "task.toml").is_file() else ""
            timeout = pc.toml_agent_timeout(toml)
            existing: dict = {}
            for arm in [a for a in args.arms.split(",") if a]:
                spec = arm_spec(arm, task_id, task_dir, None, None, None, timeout)
                run_arm(ctx, task_id, arm, *spec, label=f"smoke-{arm}")
            if args.keystrokes:
                ks_path = args.out / "smoke-keystrokes.json"
                ks_path.write_text(args.keystrokes, encoding="utf-8")
                run_arm(ctx, task_id, "replay", task_dir, PROBE_AGENT_IMPORT_PATHS["replay_keystrokes"],
                        [f"keystrokes_json={ks_path.resolve()}", "workdir=/app"], label="smoke-replay")
        else:
            selection = read_json(args.out / "selection.json") or {"tasks": []}
            plans = (read_json(args.out / "mutations.json") or {}).get("plans", {})
            ids = args.task_ids.split(",") if args.task_ids else [t["task_id"] for t in selection["tasks"]]
            timeouts = {t["task_id"]: t.get("agent_timeout_sec") for t in selection["tasks"]}
            arms = [a for a in args.arms.split(",") if a]
            existing = load_outcomes(ctx.outcomes_path)
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = []
                for tid in ids:
                    orig = args.pool / tid
                    if not orig.is_dir():
                        print(f"[rst-probe] {tid}: not in pool, skipped", flush=True)
                        continue
                    mut_dir = (args.out / "mutated" / tid) if tid in plans else None
                    futures.append(pool.submit(run_task, ctx, tid, orig, mut_dir, plans.get(tid), arms,
                                               existing, args.force, args.rerun_infra, timeouts.get(tid)))
                for f in futures:
                    f.result()
        manifest = {
            "cmd": args.cmd, "harbor": harbor_version(args.harbor_bin), "harbor_bin": args.harbor_bin,
            "docker_host": ctx.env.get("DOCKER_HOST"), "env_kwargs": ctx.env_kwargs, "cpu_caveat": CPU_CAVEAT,
            "trial_timeout": args.trial_timeout, "seconds": round(time.time() - started, 1),
            "docker_system_df_before": df_before, "docker_system_df_after": docker_df(ctx.env),
            "rules_version": pc.STYLE_RULES_VERSION,
        }
        (args.out / f"run_manifest_{args.cmd}_{int(started)}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[rst-probe] {args.cmd} done in {manifest['seconds']}s", flush=True)
        return 0

    if args.cmd == "report":
        manifest = build_report(args.out, args.census)
        manifest.update({"cpu_caveat": CPU_CAVEAT, "rules_version": pc.STYLE_RULES_VERSION,
                         "selection": read_json(args.out / "selection.json"),
                         "mutation_na": (read_json(args.out / "mutations.json") or {}).get("na")})
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(f"[rst-probe] cells={manifest['cells']} touch_pass_rate={manifest['touch_pass_rate']} nop_passes={manifest['nop_passes']}")
        print(f"[rst-probe] report -> {args.out / 'report_E1.md'}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
