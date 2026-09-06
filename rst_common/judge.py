"""LLM supervision for the data loop: one client, one cache, one budget, one contract.

WHERE AN LLM JUDGE BELONGS IN THIS PIPELINE, AND WHERE IT DOES NOT
------------------------------------------------------------------
The verifier is the reward. Every number this repo reports -- pass rates, the DPO
implicit reward, the GRPO advantage -- comes from a task's own tests, and an LLM
opinion must never leak into those or the runs stop being comparable with the paper
and with each other. So a judge is *never* consulted

  * inside a rollout (the reward is the verifier's; a judge call per turn would also
    put an HTTP round-trip on the hot path of every sandbox);
  * in the loss mask, the normalizer or the tokenizer contract (deterministic, and the
    two ports of the mask are already checked against each other);
  * in eval scoring.

It IS worth consulting, off the critical path and on the *borderline* only, where the
verifier is necessary but not sufficient:

  * curating verifier-passed trajectories before they become SFT data
    (`scripts/03g_curate_sft.py`): a reward-1 episode that thrashed for forty turns
    teaches thrashing. Cheap heuristics decide the clear cases; the judge sees the
    band in between, once, cached.
  * cold-start difficulty and clarity for task pools that have no rollouts yet
    (termigen ships none), so the first GRPO batch is not spent on unsolvable or
    ambiguous prompts;
  * which success/failure pairs are informative for DPO (a decision, not luck).

THE CONTRACT
------------
Any OpenAI-compatible `/chat/completions` endpoint: OpenAI, Azure OpenAI's `/openai/v1`,
a LiteLLM proxy, or a local sglang/vLLM server -- the latter being the cost-free option
on a cluster that is already serving a model. Configured from the environment only, so
no script grows a provider flag:

    RST_JUDGE_BASE_URL     e.g. https://api.openai.com/v1  or  http://127.0.0.1:30000/v1
    RST_JUDGE_MODEL        the served model name
    RST_JUDGE_API_KEY      may be empty for a local server
    RST_JUDGE_AUTH_HEADER  bearer (default) | api-key  (Azure)
    RST_JUDGE_MAX_CALLS    budget of *uncached* calls per process, default 2000
    RST_JUDGE_CONCURRENCY  parallel requests, default 8
    RST_JUDGE_TIMEOUT_SEC  per request, default 120
    RST_JUDGE_CACHE_DIR    default $BASE_FOLDER/.judge-cache or ./.judge-cache
    RST_JUDGE_JSON_MODE    1 = send response_format json_object (not every server has it)
    RST_JUDGE_MAX_TOKENS   completion cap, default 600 (raise it for a thinking model)
    RST_JUDGE_EXTRA_BODY   JSON object merged into every request body, e.g.
                           {"chat_template_kwargs": {"enable_thinking": false}} for a local
                           Qwen3.5 server, so the judge answers instead of thinking aloud

Unset `RST_JUDGE_BASE_URL` or `RST_JUDGE_MODEL` and `judge_from_env()` returns a
`NullJudge`: every consumer must run to completion without an API and say in its
manifest that it did (`judge.enabled: false`), never silently pretend the borderline
was reviewed.

Every verdict is cached on disk under a content hash of (model, prompts, schema), so a
rerun of a curation pass costs nothing and a rebuild reproduces the same decisions.
Every call counts against a budget; once spent, remaining items come back as
`budget_exhausted` verdicts the consumer counts, rather than an exception that loses
the work already paid for. Responses must be a JSON object with the keys the caller
named; one retry asks for JSON only, then the item is `unparseable`. Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]

RETRY_STATUSES = (408, 409, 425, 429, 500, 502, 503, 504)
RETRY_BACKOFF_SEC = (0.5, 2.0, 6.0)


class JudgeError(Exception):
    """Configuration or transport failure that no retry will fix."""


@dataclass(frozen=True)
class JudgeConfig:
    base_url: str
    model: str
    api_key: str = ""
    auth_header: str = "bearer"          # bearer | api-key
    max_calls: int = 2000
    concurrency: int = 8
    timeout_sec: float = 120.0
    cache_dir: Path = Path(".judge-cache")
    json_mode: bool = False
    temperature: float = 0.0
    max_tokens: int = 600
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> JudgeConfig | None:
        env = os.environ if env is None else env
        base_url = env.get("RST_JUDGE_BASE_URL", "").strip()
        model = env.get("RST_JUDGE_MODEL", "").strip()
        if not base_url or not model:
            return None
        auth = env.get("RST_JUDGE_AUTH_HEADER", "bearer").strip().lower()
        if auth not in ("bearer", "api-key"):
            raise JudgeError(f"RST_JUDGE_AUTH_HEADER must be bearer or api-key, not {auth!r}")
        cache_default = Path(env.get("BASE_FOLDER", ".")) / ".judge-cache"
        return cls(
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=env.get("RST_JUDGE_API_KEY", ""),
            auth_header=auth,
            max_calls=int(env.get("RST_JUDGE_MAX_CALLS", "2000")),
            concurrency=max(1, int(env.get("RST_JUDGE_CONCURRENCY", "8"))),
            timeout_sec=float(env.get("RST_JUDGE_TIMEOUT_SEC", "120")),
            cache_dir=Path(env.get("RST_JUDGE_CACHE_DIR", str(cache_default))),
            json_mode=env.get("RST_JUDGE_JSON_MODE", "0") == "1",
            max_tokens=int(env.get("RST_JUDGE_MAX_TOKENS", "600")),
            extra_body=json.loads(env.get("RST_JUDGE_EXTRA_BODY", "{}") or "{}"),
        )


@dataclass(frozen=True)
class JudgeRequest:
    item_id: str
    system: str
    user: str
    required_keys: tuple[str, ...]


@dataclass
class Verdict:
    item_id: str
    ok: bool
    data: dict[str, Any] | None = None
    raw: str = ""
    cached: bool = False
    error: str | None = None      # budget_exhausted | unparseable | http:<status> | transport | judge_disabled
    usage: dict[str, int] = field(default_factory=dict)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """First JSON object in `text`, tolerating a ```json fence or prose around it."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start >= 0:
        try:
            value, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def _urllib_transport(url: str, headers: dict[str, str], payload: bytes,
                      timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - caller-configured URL
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class Judge:
    """Cached, budgeted, concurrent access to one OpenAI-compatible chat model."""

    enabled = True

    def __init__(self, config: JudgeConfig, transport: Transport | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.config = config
        self._transport = transport or _urllib_transport
        self._sleep = sleep
        self._lock = threading.Lock()
        self.calls = 0            # uncached HTTP round-trips that produced a verdict
        self.cache_hits = 0
        self.failures = 0
        self.budget_refusals = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        config.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public

    def judge_many(self, requests: Sequence[JudgeRequest]) -> list[Verdict]:
        """Verdicts in the order of `requests`; cache first, then the network."""
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            return list(pool.map(self.judge_one, requests))

    def judge_one(self, request: JudgeRequest) -> Verdict:
        key = self.cache_key(request)
        cached = self._read_cache(key)
        if cached is not None:
            with self._lock:
                self.cache_hits += 1
            return Verdict(item_id=request.item_id, ok=True, data=cached["data"],
                           raw=cached.get("raw", ""), cached=True, usage=cached.get("usage", {}))

        with self._lock:
            if self.calls >= self.config.max_calls:
                self.budget_refusals += 1
                return Verdict(item_id=request.item_id, ok=False, error="budget_exhausted")
            self.calls += 1

        messages = [{"role": "system", "content": request.system},
                    {"role": "user", "content": request.user}]
        raw, usage, error = self._complete(messages)
        data = extract_json_object(raw) if raw else None
        if data is not None and not all(k in data for k in request.required_keys):
            data = None
        if data is None and error is None:
            # One retry, asking for the object and nothing else. Counted as the same
            # budget unit: the item cost two round-trips, not two verdicts.
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                             "Return ONLY a JSON object with exactly these keys: "
                             + ", ".join(request.required_keys) + ". No prose, no fences."})
            raw2, usage2, error = self._complete(messages)
            for name in ("prompt_tokens", "completion_tokens"):
                usage[name] = usage.get(name, 0) + usage2.get(name, 0)
            data = extract_json_object(raw2) if raw2 else None
            if data is not None and not all(k in data for k in request.required_keys):
                data = None
            raw = raw2 or raw
        with self._lock:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
        if data is None:
            with self._lock:
                self.failures += 1
            return Verdict(item_id=request.item_id, ok=False, raw=raw,
                           error=error or "unparseable", usage=usage)
        self._write_cache(key, {"data": data, "raw": raw, "usage": usage,
                                "model": self.config.model, "item_id": request.item_id})
        return Verdict(item_id=request.item_id, ok=True, data=data, raw=raw, usage=usage)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "budget_max_calls": self.config.max_calls,
            "budget_refusals": self.budget_refusals,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_dir": str(self.config.cache_dir),
        }

    # ----------------------------------------------------------------- cache

    def cache_key(self, request: JudgeRequest) -> str:
        blob = json.dumps({"model": self.config.model, "system": request.system,
                           "user": request.user, "keys": list(request.required_keys)},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.config.cache_dir / key[:2] / f"{key}.json"

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload.get("data"), dict) else None

    def _write_cache(self, key: str, payload: dict[str, Any]) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    # --------------------------------------------------------------- network

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            if self.config.auth_header == "api-key":
                headers["api-key"] = self.config.api_key
            else:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _complete(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, int], str | None]:
        """One chat completion with bounded retries. Returns (text, usage, error)."""
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.config.extra_body)
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        url = self.config.base_url + "/chat/completions"
        last_error = "transport"
        for attempt in range(len(RETRY_BACKOFF_SEC) + 1):
            try:
                status, raw = self._transport(url, self._headers(), payload, self.config.timeout_sec)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_error = f"transport:{type(exc).__name__}"
                status, raw = 0, b""
            if status == 200:
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                    text = parsed["choices"][0]["message"]["content"] or ""
                except (ValueError, KeyError, IndexError, TypeError):
                    return "", {}, "malformed_response"
                usage = parsed.get("usage") or {}
                return text, {"prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                              "completion_tokens": int(usage.get("completion_tokens", 0) or 0)}, None
            if status:
                last_error = f"http:{status}"
                if status not in RETRY_STATUSES:
                    return "", {}, last_error
            if attempt < len(RETRY_BACKOFF_SEC):
                self._sleep(RETRY_BACKOFF_SEC[attempt])
        return "", {}, last_error


class NullJudge:
    """What every consumer gets when no endpoint is configured. Runs; decides nothing."""

    enabled = False

    def judge_many(self, requests: Sequence[JudgeRequest]) -> list[Verdict]:
        return [Verdict(item_id=r.item_id, ok=False, error="judge_disabled") for r in requests]

    def judge_one(self, request: JudgeRequest) -> Verdict:
        return Verdict(item_id=request.item_id, ok=False, error="judge_disabled")

    def stats(self) -> dict[str, Any]:
        return {"enabled": False,
                "why": "RST_JUDGE_BASE_URL / RST_JUDGE_MODEL not set; borderline items were "
                       "decided by heuristics alone"}


def judge_from_env(env: dict[str, str] | None = None, **kwargs: Any) -> Judge | NullJudge:
    config = JudgeConfig.from_env(env)
    if config is None:
        return NullJudge()
    return Judge(config, **kwargs)
