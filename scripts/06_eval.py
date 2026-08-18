#!/usr/bin/env python3
"""Benchmark a checkpoint with SGLang + Harbor + Terminus-2 on Docker.

    python scripts/06_eval.py --model-path /shared/rst/out-hf-full --tp 4 \
        --benchmarks tb-hard,tb2 --runs 3 --out $BASE_FOLDER/eval/mine

Mirrors the paper's protocol (Appendix F): a local OpenAI-compatible SGLang
endpoint, Harbor for orchestration, Terminus-2 as the harness, three runs
reported as mean +- std. Daytona is replaced by Docker.

WHAT THIS DOES THAT A NAIVE PASS-RATE SCRIPT DOES NOT
-----------------------------------------------------
1. **Separates infrastructure failures from model failures.** A Docker build
   failure is not a wrong answer. Infra failures are excluded from the pass-rate
   denominator and reported as their own rate; a high rate invalidates the run
   rather than lowering the score. Conflating them silently deflates results and
   is the single easiest way to draw a false conclusion here.
2. **Reports mean +- std over independent runs**, because that is what the paper
   reports and a single run on ~100 tasks has a std of several points.
3. **Records per-task outcomes**, so a regression can be localized instead of
   guessed at.
4. **Refuses to invent numbers.** Benchmarks whose verifiers are unavailable are
   reported as `unscorable`, never as 0.

BENCHMARK AVAILABILITY (measured 2026-08-18)
  tb-hard  100 tasks, verifiers ship          -> scorable
  tb2       89 tasks, verifiers ship          -> scorable
  lhtb      46 tasks, verifiers are WITHHELD  -> NOT scorable locally.
           The dataset card states grading uses "hidden, rebuild-from-artifact
           verifiers"; 0/46 tasks contain tests/. Do not report an LHTB number
           from this harness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Narrow marker list: a plain reward-0 must never be classified as infrastructure.
INFRA_MARKERS = (
    "connection reset by peer", "could not resolve host", "docker daemon",
    "docker compose command failed", "error during connect", "failed to pull image",
    "name or service not known", "no space left on device",
    "temporary failure in name resolution", "unexpected eof while reading",
    "command too long", "rate limit exceeded", "timed out", "unable to locate package",
)

EXPECTED_TASK_COUNTS = {"tb-hard": 100, "tb2": 89}
UNSCORABLE = {"lhtb": "verifiers withheld upstream (0/46 tasks ship tests/)"}

# Paper Tables 3-4, pass rate %. LHTB kept for reference only.
PAPER = {
    "base":    {"tb2": 41.20, "tb-hard": 22.67, "lhtb": 18.10},
    "sft_r1":  {"tb2": 42.32, "tb-hard": 23.00, "lhtb": 21.32},
    "sft_r3":  {"tb2": 47.94, "tb-hard": 28.33, "lhtb": 22.44},
    "rl":      {"tb2": 49.44, "tb-hard": 32.00, "lhtb": 22.07},
}


@dataclass
class TaskOutcome:
    benchmark: str
    run: int
    task_id: str
    reward: float | None = None
    infra_reason: str | None = None
    seconds: float = 0.0
    returncode: int = 0

    @property
    def scorable(self) -> bool:
        return self.reward is not None and self.infra_reason is None

    @property
    def solved(self) -> bool:
        return self.scorable and self.reward >= 1.0


@dataclass
class BenchmarkResult:
    name: str
    per_run_pass_rate: list[float] = field(default_factory=list)
    outcomes: list[TaskOutcome] = field(default_factory=list)
    unscorable_reason: str | None = None


def classify_infra(text: str) -> str | None:
    lowered = text.lower()
    return next((m for m in INFRA_MARKERS if m in lowered), None)


def read_reward(job_dir: Path) -> tuple[float | None, str | None]:
    """Read Harbor's trial result. Layout per terminalevo/runner/harbor.py."""
    candidates = sorted({*(job_dir / "trials").glob("*/result.json"), *job_dir.glob("*/result.json"),
                         *job_dir.glob("result.json")})
    if not candidates:
        return None, "job artifacts incomplete: no result.json"
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"unreadable result.json: {exc}"
        if payload.get("exception_info"):
            return None, classify_infra(json.dumps(payload["exception_info"])) or "harness exception"
        value = ((payload.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        if isinstance(value, (int, float)):
            return float(value), None
    return None, "no verifier reward in result.json"


def discover_tasks(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    direct = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "instruction.md").is_file()]
    if direct:
        return direct
    nested = root / "tasks"
    if nested.is_dir():
        return [p for p in sorted(nested.iterdir()) if p.is_dir() and (p / "instruction.md").is_file()]
    return []


async def run_task(task_dir: Path, benchmark: str, run: int, args, sem: asyncio.Semaphore,
                   served_name: str) -> TaskOutcome:
    task_id = task_dir.name
    job_name = re.sub(r"[^A-Za-z0-9_.-]", "-", f"{benchmark}-r{run}-{task_id}")[:96].lstrip("-")
    jobs_dir = Path(args.out) / "jobs" / benchmark / f"run{run}"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    outcome = TaskOutcome(benchmark=benchmark, run=run, task_id=task_id)
    started = time.time()

    argv = [
        args.harbor_bin, "run", "--path", str(task_dir.resolve()),
        "--agent", "terminus-2", "--model", f"hosted_vllm/{served_name}",
        "--env", "docker", "--n-attempts", "1", "--n-concurrent", "1",
        "--max-retries", "0", "--jobs-dir", str(jobs_dir), "--job-name", job_name, "--quiet",
    ]
    env = dict(os.environ)
    env.update({
        "HOSTED_VLLM_API_BASE": args.endpoint,
        "HOSTED_VLLM_API_KEY": "eval",
        "OPENAI_BASE_URL": args.endpoint,
        "OPENAI_API_KEY": "eval",
    })
    if args.docker_host:
        env["DOCKER_HOST"] = args.docker_host
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(key, None)

    async with sem:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=args.task_timeout)
        except asyncio.TimeoutError:
            proc.kill(); await proc.wait()
            outcome.infra_reason = "task wall-clock timeout"
            outcome.seconds = time.time() - started
            return outcome

    outcome.returncode = proc.returncode or 0
    outcome.seconds = time.time() - started
    text = (stdout or b"").decode("utf-8", "replace")
    job_dir = jobs_dir / job_name
    reward, infra = read_reward(job_dir if job_dir.is_dir() else jobs_dir)
    outcome.reward, outcome.infra_reason = reward, infra
    if reward is None and infra is not None:
        outcome.infra_reason = classify_infra(text) or infra
    if not args.keep_jobs:
        shutil.rmtree(job_dir, ignore_errors=True)
    return outcome


async def eval_benchmark(name: str, root: Path, args, served_name: str) -> BenchmarkResult:
    result = BenchmarkResult(name=name)
    if name in UNSCORABLE:
        result.unscorable_reason = UNSCORABLE[name]
        print(f"[{name}] UNSCORABLE: {result.unscorable_reason}")
        return result
    tasks = discover_tasks(root)
    if not tasks:
        result.unscorable_reason = f"no task dirs with instruction.md under {root}"
        print(f"[{name}] SKIP: {result.unscorable_reason}")
        return result
    expected = EXPECTED_TASK_COUNTS.get(name)
    if expected and len(tasks) != expected:
        print(f"[{name}] WARNING: found {len(tasks)} tasks, expected {expected}")
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    sem = asyncio.Semaphore(args.n_concurrent)
    for run in range(1, args.runs + 1):
        print(f"[{name}] run {run}/{args.runs}: {len(tasks)} tasks, concurrency {args.n_concurrent}", flush=True)
        outcomes = await asyncio.gather(
            *(run_task(t, name, run, args, sem, served_name) for t in tasks)
        )
        result.outcomes.extend(outcomes)
        scorable = [o for o in outcomes if o.scorable]
        infra = [o for o in outcomes if o.infra_reason]
        rate = 100.0 * sum(o.solved for o in scorable) / len(scorable) if scorable else 0.0
        result.per_run_pass_rate.append(rate)
        print(f"[{name}] run {run}: pass={rate:.2f}% scorable={len(scorable)}/{len(outcomes)} "
              f"infra_failed={len(infra)}", flush=True)
    return result


def summarize(results: list[BenchmarkResult]) -> dict:
    out: dict = {"benchmarks": {}, "protocol": {}}
    for r in results:
        if r.unscorable_reason:
            out["benchmarks"][r.name] = {"status": "unscorable", "reason": r.unscorable_reason}
            continue
        rates = r.per_run_pass_rate
        total = len(r.outcomes)
        infra = [o for o in r.outcomes if o.infra_reason]
        scorable = [o for o in r.outcomes if o.scorable]
        per_task: dict[str, dict] = {}
        for o in r.outcomes:
            entry = per_task.setdefault(o.task_id, {"attempts": 0, "solved": 0, "infra": 0})
            entry["attempts"] += 1
            entry["solved"] += int(o.solved)
            entry["infra"] += int(bool(o.infra_reason))
        infra_reasons: dict[str, int] = {}
        for o in infra:
            infra_reasons[o.infra_reason] = infra_reasons.get(o.infra_reason, 0) + 1
        out["benchmarks"][r.name] = {
            "status": "scored",
            "pass_rate_mean": round(statistics.mean(rates), 2) if rates else None,
            "pass_rate_std": round(statistics.stdev(rates), 2) if len(rates) > 1 else 0.0,
            "per_run_pass_rate": [round(x, 2) for x in rates],
            "runs": len(rates),
            "trials_total": total,
            "trials_scorable": len(scorable),
            "infra_failures": len(infra),
            "infra_failure_rate": round(100.0 * len(infra) / total, 2) if total else 0.0,
            "infra_reasons": dict(sorted(infra_reasons.items(), key=lambda kv: -kv[1])),
            # pass@k over the independent runs (k = number of runs)
            "pass_at_k": round(
                100.0 * sum(1 for v in per_task.values() if v["solved"] > 0) / len(per_task), 2
            ) if per_task else None,
            "tasks_never_solved": sorted(t for t, v in per_task.items() if v["solved"] == 0),
            "per_task": per_task,
        }
    return out


async def main_async(args) -> int:
    served_name = args.served_name
    server = None
    if args.model_path:
        args.endpoint = f"http://127.0.0.1:{args.port}/v1"
        log = Path(args.out) / "sglang.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", str(args.model_path), "--served-model-name", served_name,
            "--tp", str(args.tp), "--port", str(args.port), "--host", "127.0.0.1",
            "--mem-fraction-static", "0.85", "--context-length", str(args.context_length),
            "--disable-radix-cache",              # hybrid KV-cache layouts + prefix caching are unreliable
            "--mamba-scheduler-strategy", "extra_buffer",   # required for the gated-delta-net layers
        ]
        # Some checkpoints ship a template whose *generation* prompt does not match
        # what training produced (Qwen3.5-0.8B defaults thinking off, so its prompt
        # already closes the think block while the trained target opens with
        # "\n</think>\n\n"). Serving with an explicit template fixes the alignment.
        if args.chat_template:
            cmd += ["--chat-template", str(args.chat_template)]
        print("[serve] " + " ".join(cmd), flush=True)
        with log.open("wb") as fh:
            server = await asyncio.create_subprocess_exec(*cmd, stdout=fh, stderr=asyncio.subprocess.STDOUT)
        # readiness = a real chat completion, not /health
        ok = False
        for _ in range(args.serve_timeout // 10):
            await asyncio.sleep(10)
            probe = await asyncio.create_subprocess_exec(
                "curl", "-sS", "-m", "10", f"{args.endpoint}/chat/completions",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({"model": served_name,
                                  "messages": [{"role": "user", "content": "say ok"}],
                                  "max_tokens": 8}),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            out, _ = await probe.communicate()
            if b"choices" in (out or b""):
                ok = True
                break
        if not ok:
            print("[serve] server never became ready; tail of log:", file=sys.stderr)
            print(log.read_text(errors="replace")[-3000:], file=sys.stderr)
            if server:
                server.kill()
            return 1
        print("[serve] ready", flush=True)

    try:
        roots = {
            "tb-hard": Path(args.tb_hard_tasks) if args.tb_hard_tasks else None,
            "tb2": Path(args.tb2_tasks) if args.tb2_tasks else None,
            "lhtb": Path(args.lhtb_tasks) if args.lhtb_tasks else None,
        }
        results: list[BenchmarkResult] = []
        for name in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
            root = roots.get(name)
            results.append(await eval_benchmark(name, root or Path("/nonexistent"), args, served_name))
    finally:
        if server:
            server.kill()
            await server.wait()

    summary = summarize(results)
    summary["protocol"] = {
        "model_path": str(args.model_path) if args.model_path else None,
        "endpoint": args.endpoint,
        "served_name": served_name,
        "agent": "terminus-2",
        "sandbox": "docker",
        "runs": args.runs,
        "n_concurrent": args.n_concurrent,
        "task_timeout_sec": args.task_timeout,
        "label": args.label,
        "paper_reference": PAPER,
        "note": "Infra failures are excluded from the pass-rate denominator and "
                "reported separately. LHTB is unscorable locally: its verifiers are withheld.",
    }
    out_path = Path(args.out) / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== SUMMARY ===")
    for name, b in summary["benchmarks"].items():
        if b["status"] != "scored":
            print(f"  {name:8s} UNSCORABLE ({b['reason']})")
        else:
            print(f"  {name:8s} {b['pass_rate_mean']:.2f} +- {b['pass_rate_std']:.2f} %   "
                  f"(runs={b['per_run_pass_rate']}, infra={b['infra_failure_rate']}%)")
    print(f"\nwrote {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", type=Path, help="HF dir to serve; omit to use --endpoint")
    p.add_argument("--endpoint", default="", help="existing OpenAI-compatible /v1 endpoint")
    p.add_argument("--served-name", default="rst-eval")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--context-length", type=int, default=65536)
    p.add_argument("--serve-timeout", type=int, default=1800)
    p.add_argument("--benchmarks", default="tb-hard,tb2")
    p.add_argument("--tb-hard-tasks", default=os.environ.get("TB_HARD_TASKS", ""))
    p.add_argument("--tb2-tasks", default=os.environ.get("TB2_TASKS", ""))
    p.add_argument("--lhtb-tasks", default=os.environ.get("LHTB_TASKS", ""))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--n-concurrent", type=int, default=8)
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--task-timeout", type=int, default=1800)
    p.add_argument("--harbor-bin", default="harbor")
    p.add_argument("--docker-host", default=os.environ.get("RST_DOCKER_HOST", ""))
    p.add_argument("--keep-jobs", action="store_true")
    p.add_argument("--chat-template", default=os.environ.get("SERVE_CHAT_TEMPLATE", ""),
                   help="override the served chat template (see configs/models.json "
                        "serve_chat_template_repo)")
    p.add_argument("--label", default="candidate", help="e.g. base / sft / reference")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if not args.model_path and not args.endpoint:
        sys.exit("give --model-path or --endpoint")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
