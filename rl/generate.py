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

THREE CORRECTNESS RULES THAT ARE EASY TO GET WRONG
--------------------------------------------------
1. **Never normalize the model's JSON here.** The SFT pipeline rewrites fenced
   ```json blocks; doing that in RL would train on tokens the policy did not
   emit, silently breaking the importance ratio. The adapter returns the sampled
   ids; pass them through untouched.
2. **An infrastructure failure is not a reward of 0.** A Docker build failure, an
   image pull timeout or a DNS error must ABORT the sample (``remove_sample``),
   not teach the policy that its actions were bad. Conflating the two is the
   fastest way to a silently poisoned run.
3. **Reward comes only from the task's own verifier**, read out of Harbor's
   ``result.json`` after the agent exits. Never let the agent see ``tests/``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
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

logger = logging.getLogger(__name__)

# Markers that mean "the sandbox/harness broke", not "the policy failed".
# Kept deliberately narrow: a plain reward-0 must never land here.
_INFRA_MARKERS = (
    "connection reset by peer",
    "could not resolve host",
    "docker daemon",
    "docker compose command failed",
    "error during connect",
    "failed to pull image",
    "name or service not known",
    "no space left on device",
    "temporary failure in name resolution",
    "unexpected eof while reading",
    "command too long",
    "rate limit exceeded",
    "timed out",
    "unable to locate package",
)


@dataclass(frozen=True)
class RolloutConfig:
    harbor_bin: str
    model_name: str
    adapter_public_host: str | None
    adapter_bind_host: str
    adapter_port: int
    docker_host: str
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
        if not CONFIG.docker_host:
            raise RuntimeError(
                "RST_DOCKER_HOST is not set. RST task Dockerfiles are untrusted "
                "third-party build scripts; they must not be built on the host's "
                "default Docker daemon. Point this at a dedicated/rootless daemon "
                "socket (e.g. unix:///run/user/1000/docker.sock)."
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
            "[rst-rl] adapter=%s sglang=%s docker_host=%s agent=%s max_sandboxes=%d",
            self.adapter_url, sglang_url, CONFIG.docker_host, CONFIG.agent,
            CONFIG.max_concurrent_sandboxes,
        )


# --------------------------------------------------------------------- harbor

def _classify_infra(text: str) -> str | None:
    lowered = text.lower()
    return next((m for m in _INFRA_MARKERS if m in lowered), None)


def _read_reward(job_dir: Path) -> tuple[float | None, str | None]:
    """Return ``(reward, infra_reason)`` from a finished Harbor job directory.

    Mirrors terminalevo/runner/harbor.py: trial results live at
    ``<job>/trials/*/result.json`` or ``<job>/*/result.json``.
    """
    candidates = sorted({*(job_dir / "trials").glob("*/result.json"), *job_dir.glob("*/result.json")})
    if not candidates:
        return None, "job artifacts incomplete: no trial result.json"
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"unreadable trial result: {exc}"
        exception_info = payload.get("exception_info")
        if exception_info:
            reason = _classify_infra(json.dumps(exception_info)) or "harness exception"
            return None, reason
        rewards = (payload.get("verifier_result") or {}).get("rewards") or {}
        value = rewards.get("reward")
        if isinstance(value, (int, float)):
            return float(value), None
    return None, "no verifier reward in trial results"


async def _run_harbor(state: _AdapterService, session_id: str, task_dir: Path,
                      job_name: str) -> tuple[float | None, str | None, int]:
    """Run one Terminus-2 rollout against the adapter. Returns (reward, infra, rc)."""
    jobs_dir = CONFIG.jobs_root / job_name
    jobs_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        CONFIG.harbor_bin, "run",
        "--path", str(task_dir.resolve()),
        "--agent", CONFIG.agent,
        "--model", CONFIG.model_name,
        "--env", "docker",
        "--n-attempts", "1",
        "--n-concurrent", "1",
        "--max-retries", "0",
        "--jobs-dir", str(jobs_dir),
        "--job-name", job_name,
        "--quiet",
    ]
    env = dict(os.environ)
    env.update(
        {
            # LiteLLM `hosted_vllm/` provider -> these two env vars. The API key
            # IS the session id: the adapter reads it from the Bearer header.
            "HOSTED_VLLM_API_BASE": f"{state.adapter_url}/v1",
            "HOSTED_VLLM_API_KEY": session_id,
            "OPENAI_BASE_URL": f"{state.adapter_url}/v1",
            "OPENAI_API_KEY": session_id,
            "DOCKER_HOST": CONFIG.docker_host,
        }
    )
    env.pop("HTTP_PROXY", None); env.pop("HTTPS_PROXY", None)
    env.pop("http_proxy", None); env.pop("https_proxy", None)

    async with _SANDBOX_SEM:
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=CONFIG.agent_timeout_sec)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return None, "agent time budget exceeded", -1

    output = (stdout or b"").decode("utf-8", "replace")
    job_dir = jobs_dir / job_name
    reward, infra = _read_reward(job_dir if job_dir.is_dir() else jobs_dir)
    if reward is None and infra is not None:
        # Harbor's own stdout often names the real cause more precisely.
        infra = _classify_infra(output) or infra
        logger.warning("[rst-rl] %s infra=%s rc=%s tail=%s",
                       job_name, infra, process.returncode, output[-400:].replace("\n", " | "))
    if not CONFIG.keep_jobs:
        shutil.rmtree(jobs_dir, ignore_errors=True)
    return reward, infra, process.returncode or 0


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
            reward, infra, returncode = await _run_harbor(state, session_id, task_dir, job_name)

            if infra is not None:
                return _abort(base_sample, f"infrastructure:{infra}",
                              {"harbor_returncode": returncode})
            assert reward is not None

            elapsed = time.time() - started
            if evaluation:
                return _placeholder(base_sample, reward=reward, status=Sample.Status.COMPLETED,
                                    reason="eval", extra={"task_id": task_id, "elapsed_sec": elapsed})

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
                },
            )
            if not samples:
                # Session produced no trainable tokens: the agent never issued a
                # completion (usually a harness-side failure), so this is infra.
                return _abort(base_sample, "adapter_session_empty",
                              {"harbor_returncode": returncode})
            logger.info("[rst-rl] %s reward=%.2f turns_as_segments=%d elapsed=%.0fs",
                        task_id, reward, len(samples), elapsed)
            return samples

    except asyncio.TimeoutError:
        return _abort(base_sample, "wall_clock_timeout")
    except Exception as exc:  # noqa: BLE001 - a rollout must never kill the trainer
        logger.warning("[rst-rl] %s failed: %s\n%s", task_id, exc, traceback.format_exc())
        return _abort(base_sample, f"exception:{type(exc).__name__}")
    finally:
        await state.adapter.drop_session(session_id, wait_timeout=30)
