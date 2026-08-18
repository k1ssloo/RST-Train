"""Harbor + Terminus-2 agentic rollout as a verl AgentLoop.

Register with:
    actor_rollout_ref.rollout.agent.agent_loop_config_path=verl_backend/agent_loop_config.yaml
and select `agent_name: harbor_terminus` on the dataset rows.

STATUS: UNVERIFIED. The interface below is written against the real verl API
(`verl/experimental/agent_loop/agent_loop.py` @ main: `AgentLoopBase.__init__`,
`@register`, `async def run(sampling_params, **kwargs) -> AgentLoopOutput`,
`AgentLoopOutput.response_mask`), but none of it has been executed. Treat every
claim here as a hypothesis with a named test in RL_PLAN.md.

THE PROBLEM THIS FILE SOLVES
----------------------------
verl's AgentLoop talks to the policy through `self.server_manager.generate(
request_id, prompt_ids=[...], sampling_params=...) -> TokenOutput`. That is
token-ids-in, token-ids-out, in-process. It is *not* an HTTP endpoint.

Harbor speaks OpenAI `/v1/chat/completions` over HTTP, because that is how
Terminus-2 drives a model. So something has to sit in between. slime ships exactly
this (`slime.agent.adapters.OpenAIAdapter`); **verl does not**, which is the main
concrete cost of the verl path for a Harbor-driven rollout.

We must not close the gap by re-tokenizing Harbor's response text. The whole point
of going through `server_manager` is that we get back the *sampled* token ids; if
we re-tokenized the text instead, the ids we train on could differ from the ids
that were sampled, the importance ratio would be wrong, and the run would be
silently off-policy. So the shim below records ids on the way through.

ARCHITECTURE
------------
    Harbor(Terminus-2)  --HTTP /v1/chat/completions-->  _OpenAIShim (this file)
                                                             |
                                                     apply_chat_template
                                                             v
                                          server_manager.generate(prompt_ids=...)
                                                             |
                                                       TokenOutput (exact ids)
                                                             v
                          OpenAI-shaped JSON back to Harbor + ids recorded per turn

    Docker container: shell commands only. No network. Never talks to the model.

After Harbor exits we assemble one AgentLoopOutput: the concatenated prompt/response
ids with `response_mask=1` on model-generated tokens and 0 on observations, and the
reward read from the task's own verifier.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The launcher puts the repo root on PYTHONPATH (that is how this module is
# importable at all), but a ray worker may start from anywhere -- make the shared
# import independent of that.
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


# --------------------------------------------------------------------- session

@dataclass
class Turn:
    """One model call: the prompt it saw and the ids it actually produced."""
    prompt_ids: list[int]
    response_ids: list[int]


@dataclass
class Session:
    session_id: str
    turns: list[Turn] = field(default_factory=list)

    def assemble(self) -> tuple[list[int], list[int], int]:
        """Return (token_ids, response_mask, num_turns).

        Terminus-2 re-sends the whole conversation each turn, so turn N's prompt
        contains turn N-1's response. We therefore walk the turns and only append
        what is new: the un-covered prompt prefix (mask 0 -- harness text and
        terminal observations) followed by the sampled response (mask 1).
        """
        token_ids: list[int] = []
        mask: list[int] = []
        for turn in self.turns:
            prompt = turn.prompt_ids
            # How much of this prompt is already in token_ids? Prompts grow by
            # appending, so the common prefix is what we have; anything beyond it is
            # new observation text.
            common = 0
            limit = min(len(token_ids), len(prompt))
            while common < limit and token_ids[common] == prompt[common]:
                common += 1
            if common < len(token_ids):
                # The prompt diverged from what we recorded. That means the harness
                # rewrote history (summarization/compaction). Refuse rather than
                # emit a sequence that never existed.
                raise ValueError(
                    f"session {self.session_id}: prompt diverged from recorded history at "
                    f"token {common} (recorded {len(token_ids)}). The harness likely "
                    f"compacted the context; this rollout cannot be assembled into one "
                    f"linear sequence and must be dropped."
                )
            new_prompt = prompt[common:]
            token_ids.extend(new_prompt)
            mask.extend([0] * len(new_prompt))
            token_ids.extend(turn.response_ids)
            mask.extend([1] * len(turn.response_ids))
        return token_ids, mask, len(self.turns)


# ------------------------------------------------------------------ http shim

class _OpenAIShim:
    """Minimal OpenAI /v1/chat/completions endpoint backed by verl's server_manager.

    One shim per worker process, shared by all concurrent rollouts. Sessions are
    keyed off the Bearer token, which is how Harbor's LiteLLM client passes its API
    key -- the same trick slime's adapter uses.
    """

    def __init__(self, tokenizer, server_manager, *, host: str, port: int,
                 apply_chat_template_kwargs: dict[str, Any] | None = None) -> None:
        self.tokenizer = tokenizer
        self.server_manager = server_manager
        self.host, self.port = host, port
        self.act_kwargs = apply_chat_template_kwargs or {}
        self.sessions: dict[str, Session] = {}
        self.sampling: dict[str, dict[str, Any]] = {}
        self._runner = None
        self._site = None

    def open_session(self, session_id: str, sampling_params: dict[str, Any]) -> None:
        self.sessions[session_id] = Session(session_id)
        self.sampling[session_id] = dict(sampling_params)

    def close_session(self, session_id: str) -> Session | None:
        self.sampling.pop(session_id, None)
        return self.sessions.pop(session_id, None)

    async def start(self) -> None:
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handle)
        app.router.add_get("/v1/models", self._models)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        logger.info("[harbor-shim] listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _models(self, _request):
        from aiohttp import web
        return web.json_response({"object": "list", "data": [{"id": "rst-policy", "object": "model"}]})

    @staticmethod
    def _session_id_from(request, body: dict) -> str:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        meta = body.get("metadata") or {}
        return str(meta.get("session_id") or body.get("user") or "default")

    async def _handle(self, request):
        from aiohttp import web

        body = await request.json()
        session_id = self._session_id_from(request, body)
        session = self.sessions.get(session_id)
        if session is None:
            return web.json_response({"error": {"message": f"unknown session {session_id}"}}, status=400)

        messages = body.get("messages") or []
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False,
            **self.act_kwargs,
        )

        sampling = dict(self.sampling.get(session_id) or {})
        for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("max_tokens", "max_new_tokens")):
            if body.get(src) is not None:
                sampling[dst] = body[src]

        out = await self.server_manager.generate(
            request_id=f"{session_id}-{len(session.turns)}-{uuid.uuid4().hex[:8]}",
            prompt_ids=prompt_ids,
            sampling_params=sampling,
        )
        response_ids = list(getattr(out, "token_ids", None) or getattr(out, "response_ids", []))
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        session.turns.append(Turn(prompt_ids=list(prompt_ids), response_ids=response_ids))

        return web.json_response({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "rst-policy"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(response_ids),
                "total_tokens": len(prompt_ids) + len(response_ids),
            },
        })


# ------------------------------------------------------------------ agent loop

try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
    _HAVE_VERL = True
except ImportError:  # pragma: no cover - importable for review without verl
    _HAVE_VERL = False

    class AgentLoopBase:  # type: ignore[no-redef]
        pass

    class AgentLoopOutput:  # type: ignore[no-redef]
        pass

    def register(name):  # type: ignore[no-redef]
        return lambda cls: cls


@register("harbor_terminus")
class HarborTerminusAgentLoop(AgentLoopBase):
    """One Harbor/Terminus-2 rollout in a Docker sandbox -> one AgentLoopOutput."""

    _shim: _OpenAIShim | None = None
    _shim_lock: asyncio.Lock | None = None

    async def _ensure_shim(self) -> _OpenAIShim:
        cls = type(self)
        if cls._shim_lock is None:
            cls._shim_lock = asyncio.Lock()
        async with cls._shim_lock:
            if cls._shim is None:
                host = os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0")
                port = int(os.environ.get("ADAPTER_PORT", "18101"))
                shim = _OpenAIShim(
                    self.tokenizer, self.server_manager, host=host, port=port,
                    apply_chat_template_kwargs=getattr(self, "apply_chat_template_kwargs", {}),
                )
                await shim.start()
                cls._shim = shim
            return cls._shim

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        metadata = kwargs.get("metadata") or {}
        task_id = metadata.get("task_id") or kwargs.get("label") or "unknown"
        task_dir = metadata.get("task_dir")

        harbor_env = os.environ.get("RST_HARBOR_ENV", "docker")
        docker_host = os.environ.get("RST_DOCKER_HOST", "")
        if harbor_env == "docker" and not docker_host:
            raise RuntimeError(
                "RST_HARBOR_ENV=docker but RST_DOCKER_HOST is not set. Run "
                "`source scripts/00b_setup_sandbox.sh` first: it picks a place for the "
                "sandbox to live and exports both. On a cluster without Docker permission "
                "the answer is usually rootless podman, whose Docker-compatible API socket "
                "Harbor uses unchanged; if this machine cannot mount(2) at all (an AppArmor "
                "or SELinux policy denial -- run the script with --diagnose), set "
                "RST_HARBOR_ENV=daytona (or e2b / modal) instead. Those build the task's "
                "Dockerfile on the provider's side, so this process needs no container "
                "privilege -- and because Terminus-2 is a HOST-side agent, the agent loop "
                "and every model call still run here, against the local shim. Task "
                "Dockerfiles are untrusted third-party build scripts and must never be "
                "built on a shared root Docker daemon; both options satisfy that."
            )
        if not task_dir or not (Path(task_dir) / "instruction.md").is_file():
            raise RuntimeError(f"task not materialized: {task_dir} "
                               f"(run scripts/10_build_rl_taskset.py --materialize)")

        shim = await self._ensure_shim()
        public_host = os.environ.get("ADAPTER_PUBLIC_HOST")
        if not public_host:
            raise RuntimeError("ADAPTER_PUBLIC_HOST is not set; Harbor cannot reach the shim")

        session_id = f"rst-{task_id}-{uuid.uuid4().hex[:8]}"
        job_name = re.sub(r"[^A-Za-z0-9_.-]", "-", session_id)[:96].lstrip("-") or "rst-job"
        shim.open_session(session_id, sampling_params)

        jobs_root = Path(os.environ.get("RST_JOBS_ROOT", "/tmp/rst-verl-jobs")) / job_name
        jobs_root.mkdir(parents=True, exist_ok=True)
        started = time.time()
        try:
            outcome = await self._run_harbor(
                task_dir=Path(task_dir), job_name=job_name, jobs_root=jobs_root,
                session_id=session_id,
                base_url=f"http://{public_host}:{shim.port}/v1",
                harbor_env=harbor_env,
                docker_host=docker_host,
            )
            session = shim.close_session(session_id)
            if outcome.kind == HARNESS_INFRA:
                # Infrastructure failure. NOT a reward of 0 -- returning 0 here would
                # teach the policy its actions were bad because Docker broke.
                raise RuntimeError(f"infrastructure:{outcome.reason}")
            if session is None or not session.turns:
                raise RuntimeError("no model turns recorded; harness never called the shim")
            # A budget failure (wall clock spent, command too long) IS a reward-0
            # sample and stays in the GRPO group -- see rst_common/harbor.py.
            assert outcome.reward is not None
            reward = outcome.reward

            token_ids, response_mask, num_turns = session.assemble()
            prompt_len = len(session.turns[0].prompt_ids)
            logger.info("[harbor-verl] %s reward=%.2f turns=%d tokens=%d elapsed=%.0fs%s",
                        task_id, reward, num_turns, len(token_ids), time.time() - started,
                        f" budget={outcome.budget_reason}" if outcome.budget_reason else "")
            return AgentLoopOutput(
                prompt_ids=token_ids[:prompt_len],
                response_ids=token_ids[prompt_len:],
                response_mask=response_mask[prompt_len:],
                num_turns=num_turns,
                reward_score=reward,
                metrics={"agent_budget_failure": float(outcome.budget_reason is not None)},
            )
        finally:
            shim.close_session(session_id)
            if os.environ.get("RST_KEEP_JOBS", "0") != "1":
                shutil.rmtree(jobs_root, ignore_errors=True)

    async def _run_harbor(self, *, task_dir: Path, job_name: str, jobs_root: Path,
                          session_id: str, base_url: str, harbor_env: str,
                          docker_host: str) -> Outcome:
        argv = [
            os.environ.get("RST_HARBOR_BIN", "harbor"), "run",
            "--path", str(task_dir.resolve()),
            "--agent", os.environ.get("RST_AGENT", "terminus-2"),
            "--model", os.environ.get("RST_SERVED_MODEL", "hosted_vllm/rst-policy"),
            "--env", harbor_env, "--n-attempts", "1", "--n-concurrent", "1",
            "--max-retries", "0", "--jobs-dir", str(jobs_root),
            "--job-name", job_name, "--quiet",
        ]
        for kwarg in os.environ.get("RST_HARBOR_ENV_KWARGS", "").split():
            if "=" in kwarg:
                argv += ["--environment-kwarg", kwarg]
        env = dict(os.environ)
        # The API key IS the session id: the shim reads it from the Bearer header.
        env.update({
            "HOSTED_VLLM_API_BASE": base_url, "HOSTED_VLLM_API_KEY": session_id,
            "OPENAI_BASE_URL": base_url, "OPENAI_API_KEY": session_id,
        })
        if docker_host:
            env["DOCKER_HOST"] = docker_host
        apply_proxy_policy(env, harbor_env, base_url)

        timeout = int(os.environ.get("RST_AGENT_TIMEOUT_SEC", "1800"))
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env)
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            # This timeout IS the agent's budget: nothing else caps how long
            # Terminus-2 may keep issuing commands. Reward 0, not "unmeasured".
            return wall_clock_timeout(timeout)

        text = (stdout or b"").decode("utf-8", "replace")
        job_dir = jobs_root / job_name
        outcome = read_reward(job_dir if job_dir.is_dir() else jobs_root)
        outcome = refine_with_stdout(outcome, text)
        if outcome.kind is not None:
            logger.warning("[harbor-verl] %s %s=%s rc=%s tail=%s", job_name, outcome.kind,
                           outcome.reason, proc.returncode, text[-400:].replace("\n", " | "))
        return outcome
