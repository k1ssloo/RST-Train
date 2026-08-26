#!/usr/bin/env python3
"""Benchmark a checkpoint with SGLang + Harbor + Terminus-2 on Docker.

    python scripts/06_eval.py --model-path /shared/rst/out-hf-full --tp 4 \
        --benchmarks tb-hard,tb2 --runs 3 --out $BASE_FOLDER/eval/mine

Mirrors the paper's protocol (Appendix F): a local OpenAI-compatible SGLang
endpoint, Harbor for orchestration, Terminus-2 as the harness, three runs
reported as mean +- std. Daytona is replaced by Docker.

WHAT THIS DOES THAT A NAIVE PASS-RATE SCRIPT DOES NOT
-----------------------------------------------------
1. **Separates infrastructure failures from model failures, in BOTH directions.**
   A Docker build failure is not a wrong answer: harness-infrastructure failures
   are excluded from the pass-rate denominator and reported as their own rate, so
   a high rate invalidates the run rather than lowering the score. But the
   converse matters just as much -- an agent that burns its wall clock, or emits a
   command the harness refuses as too long, has *failed the task*. Those count as
   reward 0 inside the denominator and are reported separately. Excluding them
   would inflate the score most on the hardest tasks, which is where it would do
   the most damage. The taxonomy lives in `rst_common/harbor.py` and is shared
   with the RL rollout so the two sets of numbers are comparable.
2. **Reports mean +- std over independent runs**, because that is what the paper
   reports and a single run on ~100 tasks has a std of several points. The
   sampling parameters that make those runs independent are recorded in
   `protocol.sampling` -- including, explicitly, when this script could not set
   them, because "3 runs of a greedy decode" is one run reported three times.
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
import functools
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rst_common.harbor import (  # noqa: E402  (path shim must run first)
    AGENT_BUDGET,
    AGENT_BUDGET_MARKERS,
    HARNESS_INFRA,
    HARNESS_INFRA_MARKERS,
    Outcome,
    apply_proxy_policy,
    read_reward,
    refine_with_stdout,
    wall_clock_timeout,
)
from rst_common.paper import PAPER  # noqa: E402

EXPECTED_TASK_COUNTS = {"tb-hard": 100, "tb2": 89}
UNSCORABLE = {"lhtb": "verifiers withheld upstream (0/46 tasks ship tests/)"}


@dataclass
class TaskOutcome:
    benchmark: str
    run: int
    task_id: str
    reward: float | None = None
    infra_reason: str | None = None
    budget_reason: str | None = None
    seconds: float = 0.0
    returncode: int = 0

    def apply(self, outcome: Outcome) -> None:
        self.reward = outcome.reward
        self.infra_reason = outcome.infra_reason
        self.budget_reason = outcome.budget_reason

    @property
    def scorable(self) -> bool:
        """In the denominator: a real score, or a budget failure scored as 0."""
        return self.reward is not None and self.infra_reason is None

    @property
    def solved(self) -> bool:
        return self.scorable and (self.reward or 0.0) >= 1.0


@dataclass
class BenchmarkResult:
    name: str
    per_run_pass_rate: list[float] = field(default_factory=list)
    per_run_pass_rate_budget_excluded: list[float] = field(default_factory=list)
    outcomes: list[TaskOutcome] = field(default_factory=list)
    unscorable_reason: str | None = None


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


def harbor_env_kwargs(args) -> list[str]:
    """`--environment-kwarg` values, from the flag and from RST_HARBOR_ENV_KWARGS.

    The environment variable is how 00b_setup_sandbox.sh passes backend options
    it discovered (a k8s namespace, a .sif cache dir) without every caller having
    to know about them.
    """
    pairs = list(args.harbor_env_kwarg or [])
    pairs += [tok for tok in os.environ.get("RST_HARBOR_ENV_KWARGS", "").split() if "=" in tok]
    seen: set[str] = set()
    out = []
    for pair in pairs:
        key = pair.split("=", 1)[0]
        if key not in seen:          # an explicit flag wins over the environment
            seen.add(key)
            out.append(pair)
    return out


def harbor_process_env(args) -> dict[str, str]:
    """Environment for the harbor subprocess. Proxy policy is shared with RL."""
    env = dict(os.environ)
    env.update({
        "HOSTED_VLLM_API_BASE": args.endpoint,
        "HOSTED_VLLM_API_KEY": "eval",
        "OPENAI_BASE_URL": args.endpoint,
        "OPENAI_API_KEY": "eval",
    })
    if args.docker_host:
        env["DOCKER_HOST"] = args.docker_host
    apply_proxy_policy(env, args.harbor_env, args.endpoint)
    return env


@functools.lru_cache(maxsize=4)
def harbor_run_flags(harbor_bin: str) -> frozenset[str]:
    """Long options `harbor run` accepts, scraped from its own --help.

    Used to answer one question honestly: can this script pin the sampling
    parameters, or is it at the mercy of whatever Terminus-2/LiteLLM defaults to?
    A wrong guess either crashes every trial or silently reports three runs of a
    greedy decode as mean +- std, so we ask instead of assuming.
    """
    try:
        proc = subprocess.run([harbor_bin, "run", "--help"], capture_output=True,
                              text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(re.findall(r"--[a-z0-9][a-z0-9-]+", (proc.stdout or "") + (proc.stderr or "")))


def agent_kwargs_argv(args) -> tuple[list[str], dict]:
    """`--agent-kwarg model_info=...`, without which harbor 0.21.0 runs zero tasks.

    Terminus-2 calls the model through LiteLLM, which looks up context window and
    per-token price in its own model map. `hosted_vllm/<served-name>` is not in that
    map -- the name is whatever `--served-model-name` we chose -- so the lookup misses
    and harbor refuses to start:

        ValueError: hosted_vllm models require model_info

    Measured against harbor 0.21.0: every task fails this way, before the agent
    issues a single command. It is a harness-infrastructure failure, so the taxonomy
    correctly keeps it out of the pass-rate denominator -- which means the run
    "succeeds" with a denominator of zero unless this is passed. That is exactly the
    shape of failure this script exists to make impossible to misread.

    The four keys are the minimum LiteLLM needs:

      max_input_tokens   derived from the server's own --context-length minus the
                         output reserve, NOT hardcoded. If it exceeded what sglang
                         was launched with, LiteLLM would happily send a prompt the
                         server then rejects, and the task would fail for a reason
                         that looks nothing like "the context is too small".
      max_output_tokens  the cap Terminus-2 requests per turn.
      *_cost_per_token   zero. Local weights have no price, but LiteLLM still
                         computes spend and needs the keys to exist.
    """
    reserve = max(1, args.max_output_tokens)
    model_info = {
        "max_input_tokens": max(1, args.context_length - reserve),
        "max_output_tokens": reserve,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    }
    record: dict = {"model_info": model_info, "forwarded": False, "control": ""}
    if "--agent-kwarg" not in harbor_run_flags(args.harbor_bin):
        record["control"] = (
            "NOT FORWARDED: this harbor build has no --agent-kwarg, so model_info "
            "could not be supplied. If it is a build that requires model_info for "
            "hosted_vllm (0.21.0 does), every task will fail in the harness before "
            "the agent starts and the pass rate will be computed over an empty "
            "denominator. Check the per-task errors before believing any number here."
        )
        return [], record
    record["forwarded"] = True
    record["control"] = (
        f"forwarded as --agent-kwarg model_info=... with max_input_tokens="
        f"{model_info['max_input_tokens']} (context_length {args.context_length} minus "
        f"{reserve} reserved for output)"
    )
    return ["--agent-kwarg", "model_info=" + json.dumps(model_info, separators=(",", ":"))], record


def sampling_argv(args) -> tuple[list[str], dict]:
    """`harbor run` sampling flags plus the record that goes into results.json."""
    wanted = {"temperature": args.temperature, "top_p": args.top_p}
    requested = {k: v for k, v in wanted.items() if v is not None}
    record: dict = {"requested": requested or None, "forwarded": {}, "control": ""}
    if not requested:
        record["control"] = (
            "NOT SET by this script: the harness/LiteLLM default was used, and this "
            "script does not know what it is. If that default is greedy, the 3 runs "
            "are not independent and std is meaningless -- pass --temperature to fix it."
        )
        return [], record
    flags = harbor_run_flags(args.harbor_bin)
    argv: list[str] = []
    unsupported = []
    for key, value in requested.items():
        flag = "--" + key.replace("_", "-")
        if flag in flags:
            argv += [flag, str(value)]
            record["forwarded"][key] = value
        else:
            unsupported.append(flag)
    record["control"] = (
        f"forwarded to harbor: {record['forwarded'] or 'nothing'}"
        + (f"; NOT SUPPORTED by this harbor build and therefore NOT APPLIED: "
           f"{unsupported} -- the numbers were produced at the harness default, "
           f"whatever that is" if unsupported else "")
    )
    return argv, record


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
        "--env", args.harbor_env, "--n-attempts", "1", "--n-concurrent", "1",
        "--max-retries", "0", "--jobs-dir", str(jobs_dir), "--job-name", job_name, "--quiet",
    ]
    argv += args.sampling_argv
    argv += args.agent_kwargs_argv
    for kwarg in harbor_env_kwargs(args):
        argv += ["--environment-kwarg", kwarg]
    env = harbor_process_env(args)

    async with sem:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=args.task_timeout)
        except asyncio.TimeoutError:
            proc.kill(); await proc.wait()
            # Our wall clock is the only thing capping how long the agent may keep
            # issuing commands, so this is the budget being enforced: reward 0, in
            # the denominator. See rst_common/harbor.py for why, and for the caveat
            # that a hung sandbox is indistinguishable from here.
            outcome.apply(wall_clock_timeout(args.task_timeout))
            outcome.seconds = time.time() - started
            return outcome

    outcome.returncode = proc.returncode or 0
    outcome.seconds = time.time() - started
    text = (stdout or b"").decode("utf-8", "replace")
    job_dir = jobs_dir / job_name
    result = read_reward(job_dir if job_dir.is_dir() else jobs_dir)
    outcome.apply(refine_with_stdout(result, text))
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
        budget = [o for o in outcomes if o.budget_reason]
        rate = 100.0 * sum(o.solved for o in scorable) / len(scorable) if scorable else 0.0
        # The same numerator over the stricter denominator, i.e. what the old
        # taxonomy would have reported. Both are written out so the effect of the
        # choice is visible instead of being argued about.
        verified = [o for o in scorable if not o.budget_reason]
        rate_excl = (100.0 * sum(o.solved for o in verified) / len(verified)) if verified else 0.0
        result.per_run_pass_rate.append(rate)
        result.per_run_pass_rate_budget_excluded.append(rate_excl)
        print(f"[{name}] run {run}: pass={rate:.2f}% scorable={len(scorable)}/{len(outcomes)} "
              f"agent_budget_failed={len(budget)} infra_failed={len(infra)} "
              f"(pass excl. budget={rate_excl:.2f}%)", flush=True)
    return result


def summarize(results: list[BenchmarkResult]) -> dict:
    out: dict = {"benchmarks": {}, "protocol": {}}
    for r in results:
        if r.unscorable_reason:
            out["benchmarks"][r.name] = {"status": "unscorable", "reason": r.unscorable_reason}
            continue
        rates = r.per_run_pass_rate
        rates_excl = r.per_run_pass_rate_budget_excluded
        total = len(r.outcomes)
        infra = [o for o in r.outcomes if o.infra_reason]
        budget = [o for o in r.outcomes if o.budget_reason]
        scorable = [o for o in r.outcomes if o.scorable]
        per_task: dict[str, dict] = {}
        for o in r.outcomes:
            entry = per_task.setdefault(o.task_id,
                                        {"attempts": 0, "solved": 0, "infra": 0, "budget": 0})
            entry["attempts"] += 1
            entry["solved"] += int(o.solved)
            entry["infra"] += int(bool(o.infra_reason))
            entry["budget"] += int(bool(o.budget_reason))
        infra_reasons: dict[str, int] = {}
        for o in infra:
            infra_reasons[o.infra_reason] = infra_reasons.get(o.infra_reason, 0) + 1
        budget_reasons: dict[str, int] = {}
        for o in budget:
            budget_reasons[o.budget_reason] = budget_reasons.get(o.budget_reason, 0) + 1
        out["benchmarks"][r.name] = {
            "status": "scored",
            "pass_rate_mean": round(statistics.mean(rates), 2) if rates else None,
            "pass_rate_std": round(statistics.stdev(rates), 2) if len(rates) > 1 else 0.0,
            "per_run_pass_rate": [round(x, 2) for x in rates],
            "runs": len(rates),
            "trials_total": total,
            "trials_scorable": len(scorable),
            # Agent-budget failures ARE scorable (reward 0). Reported on their own
            # because a score dominated by timeouts is a different claim from a
            # score dominated by wrong commands, and because the second denominator
            # below is what the old "budget == infra" taxonomy would have used.
            "agent_budget_failures": len(budget),
            "agent_budget_rate": round(100.0 * len(budget) / total, 2) if total else 0.0,
            "agent_budget_reasons": dict(sorted(budget_reasons.items(), key=lambda kv: -kv[1])),
            "pass_rate_mean_budget_excluded":
                round(statistics.mean(rates_excl), 2) if rates_excl else None,
            "per_run_pass_rate_budget_excluded": [round(x, 2) for x in rates_excl],
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
    args.sampling_argv, args.sampling_record = sampling_argv(args)
    print(f"[protocol] sampling: {args.sampling_record['control']}", flush=True)
    args.agent_kwargs_argv, args.agent_kwargs_record = agent_kwargs_argv(args)
    print(f"[protocol] model_info: {args.agent_kwargs_record['control']}", flush=True)
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
        "sandbox": args.harbor_env,
        "runs": args.runs,
        "n_concurrent": args.n_concurrent,
        "task_timeout_sec": args.task_timeout,
        "label": args.label,
        # What made the runs (in)dependent. Recorded even when it is "we could not
        # set this", because that is the case a reader most needs to know about.
        "sampling": args.sampling_record,
        # Without this, harbor 0.21.0 fails every task before the agent starts.
        # Recorded so a zero-denominator run is diagnosable from results.json alone.
        "agent_model_info": args.agent_kwargs_record,
        "failure_taxonomy": {
            HARNESS_INFRA: list(HARNESS_INFRA_MARKERS),
            AGENT_BUDGET: list(AGENT_BUDGET_MARKERS),
        },
        "paper_reference": PAPER,
        "note": "Harness-infrastructure failures are excluded from the pass-rate "
                "denominator and reported separately; agent-budget failures (wall clock "
                "spent, command too long) count as reward 0 INSIDE it, because those are "
                "the policy failing. pass_rate_mean_budget_excluded is the same numerator "
                "over the stricter denominator. LHTB is unscorable locally: its verifiers "
                "are withheld.",
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
                  f"(runs={b['per_run_pass_rate']}, agent_budget={b['agent_budget_rate']}%, "
                  f"infra={b['infra_failure_rate']}%)")
            if b["runs"] > 1 and b["pass_rate_std"] == 0.0:
                print("           NOTE: std is exactly 0 over multiple runs -- the runs are "
                      "probably not independent (greedy decode?). See protocol.sampling.")
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
    p.add_argument("--max-output-tokens", type=int, default=4096,
                   help="per-turn generation cap handed to LiteLLM as "
                        "model_info.max_output_tokens; also reserved out of "
                        "--context-length when computing max_input_tokens")
    p.add_argument("--serve-timeout", type=int, default=1800)
    p.add_argument("--benchmarks", default="tb-hard,tb2")
    p.add_argument("--tb-hard-tasks", default=os.environ.get("TB_HARD_TASKS", ""))
    p.add_argument("--tb2-tasks", default=os.environ.get("TB2_TASKS", ""))
    p.add_argument("--lhtb-tasks", default=os.environ.get("LHTB_TASKS", ""))
    p.add_argument("--runs", type=int, default=3)
    # "mean +- std over 3 runs" is only a claim if the runs are independent. These are
    # forwarded to `harbor run` when that build supports them, and recorded either way;
    # see sampling_argv() and protocol.sampling in results.json.
    p.add_argument("--temperature", type=float, default=None,
                   help="sampling temperature for the agent's model calls. Leave unset to "
                        "accept the harness default -- which may be greedy, in which case "
                        "the 3 runs are one run reported three times.")
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--n-concurrent", type=int, default=8)
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--task-timeout", type=int, default=1800)
    p.add_argument("--harbor-bin", default="harbor")
    p.add_argument("--docker-host", default=os.environ.get("RST_DOCKER_HOST", ""))
    # Where the task container runs. "docker" means a local (or remote) Docker API
    # endpoint; daytona/e2b/modal build environment/Dockerfile on the provider's
    # side and need no container privilege here at all. scripts/00b_setup_sandbox.sh
    # picks one and exports RST_HARBOR_ENV; see it for why that matters.
    p.add_argument("--harbor-env", default=os.environ.get("RST_HARBOR_ENV", "docker"),
                   help="harbor --env value: docker|daytona|e2b|modal|gke|ack|openshift|...")
    p.add_argument("--harbor-env-kwarg", action="append", default=None, metavar="K=V",
                   help="repeatable; forwarded as harbor --environment-kwarg K=V")
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
