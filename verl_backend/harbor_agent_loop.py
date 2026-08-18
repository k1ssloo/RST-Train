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
import json
import logging
import os
import re
import shutil
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Narrow list: a plain reward-0 must never be classified as infrastructure.
INFRA_MARKERS = (
    "connection reset by peer", "could not resolve host", "docker daemon",
    "docker compose command failed", "error during connect", "failed to pull image",
    "name or service not known", "no space left on device",
    "temporary failure in name resolution", "unexpected eof while reading",
    "command too long", "rate limit exceeded", "timed out", "unable to locate package",
)


def classify_infra(text: str) -> str | None:
    low = text.lower()
    return next((m for m in INFRA_MARKERS if m in low), None)


_PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")


def _apply_proxy_policy(env: dict[str, str], harbor_env: str, shim_url: str) -> None:
    """Decide what the harbor subprocess should do with proxy variables.

    With `--env docker` the sandbox and the shim are both on this host, so a
    proxy can only get in the way -- drop it. With an off-machine backend,
    harbor has to reach the provider's API over HTTPS, which on a locked-down
    cluster is exactly what the proxy is for; keep it, and put the shim's own
    host in NO_PROXY so rollout traffic still goes direct.
    """
    if harbor_env == "docker":
        for key in _PROXY_KEYS:
            env.pop(key, None)
        return
    if not any(env.get(key) for key in _PROXY_KEYS):
        return
    host = urllib.parse.urlsplit(shim_url).hostname or ""
    extra = [h for h in (host, "127.0.0.1", "localhost") if h]
    for key in ("NO_PROXY", "no_proxy"):
        current = [tok for tok in env.get(key, "").split(",") if tok]
        env[key] = ",".join(dict.fromkeys(current + extra))


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
            reward, infra = await self._run_harbor(
                task_dir=Path(task_dir), job_name=job_name, jobs_root=jobs_root,
                session_id=session_id,
                base_url=f"http://{public_host}:{shim.port}/v1",
                harbor_env=harbor_env,
                docker_host=docker_host,
            )
            session = shim.close_session(session_id)
            if infra is not None:
                # Infrastructure failure. NOT a reward of 0 -- returning 0 here would
                # teach the policy its actions were bad because Docker broke.
                raise RuntimeError(f"infrastructure:{infra}")
            if session is None or not session.turns:
                raise RuntimeError("no model turns recorded; harness never called the shim")

            token_ids, response_mask, num_turns = session.assemble()
            prompt_len = len(session.turns[0].prompt_ids)
            logger.info("[harbor-verl] %s reward=%.2f turns=%d tokens=%d elapsed=%.0fs",
                        task_id, reward, num_turns, len(token_ids), time.time() - started)
            return AgentLoopOutput(
                prompt_ids=token_ids[:prompt_len],
                response_ids=token_ids[prompt_len:],
                response_mask=response_mask[prompt_len:],
                num_turns=num_turns,
                reward_score=reward,
                metrics={},
            )
        finally:
            shim.close_session(session_id)
            if os.environ.get("RST_KEEP_JOBS", "0") != "1":
                shutil.rmtree(jobs_root, ignore_errors=True)

    async def _run_harbor(self, *, task_dir: Path, job_name: str, jobs_root: Path,
                          session_id: str, base_url: str, harbor_env: str,
                          docker_host: str) -> tuple[float, str | None]:
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
        _apply_proxy_policy(env, harbor_env, base_url)

        timeout = int(os.environ.get("RST_AGENT_TIMEOUT_SEC", "1800"))
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env)
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill(); await proc.wait()
            return 0.0, "agent time budget exceeded"

        text = (stdout or b"").decode("utf-8", "replace")
        job_dir = jobs_root / job_name
        reward, infra = self._read_reward(job_dir if job_dir.is_dir() else jobs_root)
        if reward is None:
            return 0.0, classify_infra(text) or (infra or "no verifier reward")
        return reward, None

    @staticmethod
    def _read_reward(job_dir: Path) -> tuple[float | None, str | None]:
        candidates = sorted({*(job_dir / "trials").glob("*/result.json"),
                             *job_dir.glob("*/result.json"), *job_dir.glob("result.json")})
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
