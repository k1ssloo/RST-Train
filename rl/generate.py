"""Agentic GRPO rollout for RST terminal tasks.

Wire in via:
    --custom-generate-function-path rl.generate.generate

Architecture
------------
Harbor + Terminus-2 already implement the exact agent loop the SFT data was
generated with (JSON action protocol, tmux keystrokes, container verifier). We
reuse it verbatim instead of reimplementing it, so the RL prompt distribution
matches the SFT prompt distribution. slime's ``OpenAIAdapter`` sits between
Harbor and SGLang purely to capture *exact sampled token ids + logprobs*:

    Harbor(Terminus-2) --OpenAI /v1/chat/completions--> OpenAIAdapter --> SGLang
       (host process)          Bearer <session_id>        (token capture)

    Docker container: runs shell commands only. NO network, never talks to the model.

The adapter resolves the session from ``Authorization: Bearer <sid>`` (see
``slime/agent/adapters/common.py::_request_session_id``), so we pass the session
id as Harbor's API key. That is what keeps concurrent rollouts separated on one
shared adapter port.

FOUR CORRECTNESS RULES THAT ARE EASY TO GET WRONG
-------------------------------------------------
1. **Never normalize the model's JSON here.** The SFT pipeline rewrites fenced
   ```json blocks; doing that in RL would train on tokens the policy did not
   emit, silently breaking the importance ratio. The adapter returns the sampled
   ids; pass them through untouched.
2. **An infrastructure failure is not a reward of 0.** A Docker build failure, an
   image pull timeout or a DNS error must ABORT the sample (``remove_sample``),
   not teach the policy that its actions were bad. Conflating the two is the
   fastest way to a silently poisoned run.
3. **And the converse: a policy failure is not an infrastructure failure.** An
   agent that spends its whole wall clock, or emits a command the harness refuses
   as too long, has failed the task. Dropping those samples would remove the most
   informative negatives from the GRPO group -- which changes the group baseline
   and therefore every advantage in it. They score 0 and stay. The taxonomy is
   ``rst_common/harbor.py``, shared with ``scripts/06_eval.py`` so that RL rewards
   and eval numbers mean the same thing.
4. **Reward comes only from the task's own verifier**, read out of Harbor's
   ``result.json`` after the agent exits. Never let the agent see ``tests/``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slime.agent.adapters import OpenAIAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

# 12_run_grpo.sh puts the repo root on PYTHONPATH (that is how `rl.generate` is
# importable at all), but ray workers have been known to start from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rst_common.harbor import (  # noqa: E402
    HARNESS_INFRA,
    Outcome,
    apply_proxy_policy,
    read_reward,
    refine_with_stdout,
    wall_clock_timeout,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutConfig:
    harbor_bin: str
    model_name: str
    adapter_public_host: str | None
    adapter_bind_host: str
    adapter_port: int
    docker_host: str          # Docker-API-compatible socket (docker OR podman)
    harbor_env: str           # harbor --env: docker | daytona | e2b | modal | gke | ...
    harbor_env_kwargs: tuple[str, ...]
    agent: str
    agent_timeout_sec: int
    guard_sec: int
    jobs_root: Path
    keep_jobs: bool
    max_concurrent_sandboxes: int

    @classmethod
    def from_env(cls) -> RolloutConfig:
        agent_timeout = int(os.environ.get("RST_AGENT_TIMEOUT_SEC", "1800"))
        return cls(
            harbor_bin=os.environ.get("RST_HARBOR_BIN", "harbor"),
            model_name=os.environ.get("RST_SERVED_MODEL", "hosted_vllm/rst-policy"),
            adapter_public_host=os.environ.get("ADAPTER_PUBLIC_HOST"),
            adapter_bind_host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "18101")),
            docker_host=os.environ.get("RST_DOCKER_HOST", ""),
            harbor_env=os.environ.get("RST_HARBOR_ENV", "docker"),
            harbor_env_kwargs=tuple(
                tok for tok in os.environ.get("RST_HARBOR_ENV_KWARGS", "").split() if "=" in tok
            ),
            agent=os.environ.get("RST_AGENT", "terminus-2"),
            agent_timeout_sec=agent_timeout,
            guard_sec=int(os.environ.get("RST_ROLLOUT_GUARD_SEC", "0") or 0) or (agent_timeout + 600),
            jobs_root=Path(os.environ.get("RST_JOBS_ROOT", "/tmp/rst-rl-jobs")),
            keep_jobs=os.environ.get("RST_KEEP_JOBS", "0") == "1",
            max_concurrent_sandboxes=int(os.environ.get("RST_MAX_SANDBOXES", "8")),
        )


CONFIG = RolloutConfig.from_env()
_SANDBOX_SEM = asyncio.Semaphore(CONFIG.max_concurrent_sandboxes)


class _AdapterService(metaclass=SingletonMeta):
    """One adapter + one aiohttp thread per worker process."""

    def __init__(self, args) -> None:
        if not CONFIG.adapter_public_host:
            raise RuntimeError(
                "ADAPTER_PUBLIC_HOST is not set. Harbor runs as a host process and "
                "dials the adapter over TCP; set it to an IP reachable from that "
                "process (the node IP, not 127.0.0.1, if Harbor may run elsewhere)."
            )
        if CONFIG.harbor_env == "docker" and not CONFIG.docker_host:
            raise RuntimeError(
                "RST_HARBOR_ENV=docker but RST_DOCKER_HOST is not set. Run "
                "`source scripts/00b_setup_sandbox.sh` first: it picks a place for the "
                "sandbox to live and exports both. On a cluster without Docker permission "
                "the answer is usually rootless podman, whose Docker-compatible API socket "
                "Harbor uses unchanged; if this machine cannot mount(2) at all (an AppArmor "
                "or SELinux policy denial -- run the script with --diagnose), the answer is "
                "an off-machine backend such as RST_HARBOR_ENV=daytona, which builds the "
                "task's Dockerfile on the provider's side and needs no container privilege "
                "here. Task Dockerfiles are untrusted third-party build scripts, so they "
                "must never be built on a shared root Docker daemon -- both of those "
                "satisfy that more strongly than a dedicated daemon would."
            )
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        self.adapter = OpenAIAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
            reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
        )
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=CONFIG.adapter_bind_host,
            port=CONFIG.adapter_port,
            thread_name="rst-openai-adapter",
            runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
        )
        self.adapter_url = f"http://{CONFIG.adapter_public_host}:{self.app_handle.port}"
        CONFIG.jobs_root.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[rst-rl] adapter=%s sglang=%s harbor_env=%s docker_host=%s agent=%s "
            "max_sandboxes=%d",
            self.adapter_url, sglang_url, CONFIG.harbor_env, CONFIG.docker_host or "-",
            CONFIG.agent, CONFIG.max_concurrent_sandboxes,
        )


# --------------------------------------------------------------------- harbor

async def _run_harbor(state: _AdapterService, session_id: str, task_dir: Path,
                      job_name: str) -> tuple[Outcome, int]:
    """Run one Terminus-2 rollout against the adapter. Returns (outcome, returncode).

    The taxonomy lives in ``rst_common.harbor`` and is shared with eval, so a
    budget failure here scores 0 and stays in the GRPO group while a broken
    sandbox comes back unmeasured.
    """
    jobs_dir = CONFIG.jobs_root / job_name
    jobs_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        CONFIG.harbor_bin, "run",
        "--path", str(task_dir.resolve()),
        "--agent", CONFIG.agent,
        "--model", CONFIG.model_name,
        "--env", CONFIG.harbor_env,
        "--n-attempts", "1",
        "--n-concurrent", "1",
        "--max-retries", "0",
        "--jobs-dir", str(jobs_dir),
        "--job-name", job_name,
        "--quiet",
    ]
    for kwarg in CONFIG.harbor_env_kwargs:
        argv += ["--environment-kwarg", kwarg]
    env = dict(os.environ)
    env.update(
        {
            # LiteLLM `hosted_vllm/` provider -> these two env vars. The API key
            # IS the session id: the adapter reads it from the Bearer header.
            "HOSTED_VLLM_API_BASE": f"{state.adapter_url}/v1",
            "HOSTED_VLLM_API_KEY": session_id,
            "OPENAI_BASE_URL": f"{state.adapter_url}/v1",
            "OPENAI_API_KEY": session_id,
        }
    )
    if CONFIG.docker_host:
        env["DOCKER_HOST"] = CONFIG.docker_host
    apply_proxy_policy(env, CONFIG.harbor_env, state.adapter_url)

    async with _SANDBOX_SEM:
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=CONFIG.agent_timeout_sec)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            # Our own wall clock IS the agent's budget: nothing else caps how long
            # Terminus-2 may keep issuing commands. Reward 0, kept in the group.
            return wall_clock_timeout(CONFIG.agent_timeout_sec), -1

    output = (stdout or b"").decode("utf-8", "replace")
    job_dir = jobs_dir / job_name
    outcome = read_reward(job_dir if job_dir.is_dir() else jobs_dir)
    # Harbor's own stdout often names the cause more precisely than result.json.
    outcome = refine_with_stdout(outcome, output)
    if outcome.kind is not None:
        logger.warning("[rst-rl] %s %s=%s rc=%s tail=%s", job_name, outcome.kind, outcome.reason,
                       process.returncode, output[-400:].replace("\n", " | "))
    if not CONFIG.keep_jobs:
        shutil.rmtree(jobs_dir, ignore_errors=True)
    return outcome, process.returncode or 0


# -------------------------------------------------------------------- results

def _session_id(sample: Sample, task_id: str) -> str:
    if sample.session_id:
        return sample.session_id
    if sample.index is not None and sample.group_index is not None:
        return f"rst-{task_id}-{sample.index}-{sample.group_index}"
    return f"rst-{task_id}-{secrets.token_hex(8)}"


def _placeholder(sample: Sample, *, reward: float, status, reason: str,
                 extra: dict[str, Any]) -> list[Sample]:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = reward
    sample.remove_sample = True
    sample.status = status
    sample.metadata = {**(sample.metadata or {}), "rst_reason": reason, **extra}
    return [sample]


def _abort(sample: Sample, reason: str, extra: dict[str, Any] | None = None) -> list[Sample]:
    """Drop the sample entirely. Used for INFRASTRUCTURE failures only."""
    logger.warning("[rst-rl] abort %s: %s", sample.label, reason)
    return _placeholder(sample, reward=0.0, status=Sample.Status.ABORTED,
                        reason=reason, extra={"infrastructure_failure": True, **(extra or {})})


# ------------------------------------------------------------------- generate

async def generate(args, base_sample: Sample, sampling_params: dict[str, Any],
                   evaluation: bool = False):
    state = _AdapterService(args)
    metadata = base_sample.metadata or {}
    task_id = metadata.get("task_id") or base_sample.label or "unknown"
    task_dir_raw = metadata.get("task_dir")
    if not task_dir_raw:
        return _abort(base_sample, "metadata.task_dir missing")
    task_dir = Path(task_dir_raw)
    if not (task_dir / "instruction.md").is_file():
        return _abort(base_sample, f"task dir not materialized: {task_dir}")

    session_id = base_sample.session_id = _session_id(base_sample, task_id)
    # Harbor validates job_name as [A-Za-z0-9][A-Za-z0-9_.-]*
    job_name = re.sub(r"[^A-Za-z0-9_.-]", "-", session_id)[:96].lstrip("-") or "rst-job"

    state.adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        max_context_tokens=state.max_context_len,
    )
    started = time.time()
    try:
        async with asyncio.timeout(CONFIG.guard_sec):
            outcome, returncode = await _run_harbor(state, session_id, task_dir, job_name)

            if outcome.kind == HARNESS_INFRA:
                # Unmeasured, not unsuccessful. Dropping it keeps the group honest.
                return _abort(base_sample, f"infrastructure:{outcome.reason}",
                              {"harbor_returncode": returncode})
            assert outcome.reward is not None
            reward = outcome.reward
            budget_reason = outcome.budget_reason

            elapsed = time.time() - started
            if evaluation:
                return _placeholder(base_sample, reward=reward, status=Sample.Status.COMPLETED,
                                    reason="eval", extra={"task_id": task_id, "elapsed_sec": elapsed,
                                                          "agent_budget_failure": budget_reason})

            samples = await state.adapter.finish_session(
                session_id,
                base_sample=base_sample,
                reward=reward,
                extra_metadata={
                    "task_id": task_id,
                    "task_group_id": metadata.get("task_group_id"),
                    "tier": metadata.get("tier"),
                    "empirical_pass_rate": metadata.get("empirical_pass_rate"),
                    "solved": reward >= 1.0,
                    "elapsed_sec": elapsed,
                    # A budget failure is a REAL reward-0 sample and stays in the
                    # group; this key only exists so the run is auditable.
                    "agent_budget_failure": budget_reason,
                },
            )
            if not samples:
                # No trainable tokens at all: the policy never got one completion
                # back, so there is nothing to reward 0. That is a harness-side
                # failure (adapter or server stalled), not a policy failure --
                # even when it arrived wearing a budget-timeout hat.
                return _abort(base_sample, "adapter_session_empty",
                              {"harbor_returncode": returncode,
                               "agent_budget_failure": budget_reason})
            logger.info("[rst-rl] %s reward=%.2f turns_as_segments=%d elapsed=%.0fs%s",
                        task_id, reward, len(samples), elapsed,
                        f" budget={budget_reason}" if budget_reason else "")
            return samples

    except asyncio.TimeoutError:
        # The outer guard, agent_timeout + 600s: harbor ignored its own kill, so
        # something below us is genuinely wedged. Infrastructure, not budget.
        return _abort(base_sample, f"wall_clock_timeout>{CONFIG.guard_sec}s")
    except Exception as exc:  # noqa: BLE001 - a rollout must never kill the trainer
        logger.warning("[rst-rl] %s failed: %s\n%s", task_id, exc, traceback.format_exc())
        return _abort(base_sample, f"exception:{type(exc).__name__}")
    finally:
        await state.adapter.drop_session(session_id, wait_timeout=30)
