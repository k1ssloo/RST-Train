"""`rst_common/judge.py` -- the LLM-supervision client, exercised through a fake transport.

No network. The transport is a function the tests hand in, so what is pinned is the
contract every consumer relies on: cache before network, a budget that degrades
rather than raises, one JSON-only retry, the two auth header styles, and a disabled
judge that still answers.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import load_repo_module  # noqa: E402

judge_mod = load_repo_module("rst_common.judge")


def _config(tmp, **over):
    base = dict(base_url="http://fake/v1", model="m", api_key="k", cache_dir=Path(tmp), max_calls=5)
    base.update(over)
    return judge_mod.JudgeConfig(**base)


def _reply(text, prompt=10, completion=5):
    return json.dumps({"choices": [{"message": {"content": text}}],
                       "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}).encode()


class _Transport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, url, headers, payload, timeout):
        self.seen.append((url, headers, json.loads(payload.decode())))
        status, body = self.replies.pop(0)
        return status, body


REQ = judge_mod.JudgeRequest("t1", "you judge", "trajectory ...", ("keep", "reason"))


def test_a_verdict_is_parsed_cached_and_never_fetched_twice():
    with tempfile.TemporaryDirectory() as tmp:
        transport = _Transport([(200, _reply('{"keep": true, "reason": "fine"}'))])
        judge = judge_mod.Judge(_config(tmp), transport=transport, sleep=lambda _s: None)
        first = judge.judge_one(REQ)
        assert first.ok and first.data == {"keep": True, "reason": "fine"} and not first.cached
        second = judge_mod.Judge(_config(tmp), transport=transport).judge_one(REQ)
        assert second.ok and second.cached and second.data == first.data
        assert len(transport.seen) == 1, "the second judge read the disk, not the network"
        url, headers, body = transport.seen[0]
        assert url == "http://fake/v1/chat/completions"
        assert headers["Authorization"] == "Bearer k"
        assert body["temperature"] == 0.0 and body["messages"][0]["role"] == "system"
        stats = judge.stats()
        assert stats["calls"] == 1 and stats["prompt_tokens"] == 10


def test_azure_style_auth_uses_the_api_key_header():
    with tempfile.TemporaryDirectory() as tmp:
        transport = _Transport([(200, _reply('{"keep": false, "reason": "x"}'))])
        judge_mod.Judge(_config(tmp, auth_header="api-key"), transport=transport).judge_one(REQ)
        headers = transport.seen[0][1]
        assert headers["api-key"] == "k" and "Authorization" not in headers


def test_prose_around_the_object_and_one_json_only_retry():
    with tempfile.TemporaryDirectory() as tmp:
        transport = _Transport([
            (200, _reply("Sure! ```json\n{\"keep\": true, \"reason\": \"ok\"}\n```")),
        ])
        v = judge_mod.Judge(_config(tmp), transport=transport).judge_one(REQ)
        assert v.ok and v.data["keep"] is True, "a fenced object is still an object"

    with tempfile.TemporaryDirectory() as tmp:
        transport = _Transport([
            (200, _reply("I think we should keep it.")),                 # no object at all
            (200, _reply('{"keep": true, "reason": "retry"}')),           # after the JSON-only nudge
        ])
        judge = judge_mod.Judge(_config(tmp), transport=transport)
        v = judge.judge_one(REQ)
        assert v.ok and v.data["reason"] == "retry"
        assert len(transport.seen) == 2 and judge.calls == 1, "two round-trips, one budget unit"
        nudge = transport.seen[1][2]["messages"][-1]["content"]
        assert "ONLY a JSON object" in nudge and "keep, reason" in nudge

    with tempfile.TemporaryDirectory() as tmp:
        transport = _Transport([(200, _reply('{"keep": true}')), (200, _reply('{"nope": 1}'))])
        judge = judge_mod.Judge(_config(tmp), transport=transport)
        v = judge.judge_one(REQ)
        assert not v.ok and v.error == "unparseable", "missing required keys is not a verdict"
        assert judge.failures == 1
        assert not any(Path(tmp).rglob("*.json")), "failures are not cached"


def test_the_budget_degrades_to_counted_refusals_instead_of_raising():
    with tempfile.TemporaryDirectory() as tmp:
        transport = _Transport([(200, _reply('{"keep": true, "reason": "a"}'))] * 2)
        judge = judge_mod.Judge(_config(tmp, max_calls=2), transport=transport)
        reqs = [judge_mod.JudgeRequest(f"t{i}", "s", f"u{i}", ("keep", "reason")) for i in range(4)]
        verdicts = judge.judge_many(reqs)
        assert [v.item_id for v in verdicts] == ["t0", "t1", "t2", "t3"], "order preserved"
        assert sum(v.ok for v in verdicts) == 2
        assert [v.error for v in verdicts if not v.ok] == ["budget_exhausted"] * 2
        assert judge.stats()["budget_refusals"] == 2


def test_retryable_http_statuses_are_retried_and_others_are_not():
    with tempfile.TemporaryDirectory() as tmp:
        transport = _Transport([(429, b"slow down"), (200, _reply('{"keep": true, "reason": "a"}'))])
        naps = []
        judge = judge_mod.Judge(_config(tmp), transport=transport, sleep=naps.append)
        assert judge.judge_one(REQ).ok and naps == [0.5]
    with tempfile.TemporaryDirectory() as tmp:
        transport = _Transport([(401, b"who are you")])
        v = judge_mod.Judge(_config(tmp), transport=transport, sleep=lambda _s: None).judge_one(REQ)
        assert not v.ok and v.error == "http:401" and len(transport.seen) == 1


def test_unset_environment_gives_a_null_judge_that_still_answers():
    judge = judge_mod.judge_from_env(env={})
    assert isinstance(judge, judge_mod.NullJudge) and judge.enabled is False
    verdicts = judge.judge_many([REQ])
    assert verdicts[0].error == "judge_disabled" and not verdicts[0].ok
    assert judge.stats()["enabled"] is False
    cfg = judge_mod.JudgeConfig.from_env({"RST_JUDGE_BASE_URL": "http://x/v1/", "RST_JUDGE_MODEL": "m",
                                          "RST_JUDGE_AUTH_HEADER": "api-key", "BASE_FOLDER": "/b"})
    assert cfg.base_url == "http://x/v1" and cfg.auth_header == "api-key"
    assert cfg.cache_dir == Path("/b/.judge-cache")


def test_extract_json_object_skips_arrays_and_broken_braces():
    ex = judge_mod.extract_json_object
    assert ex('[1,2] then {"a": 1}') == {"a": 1}
    assert ex('{ not json { "a": 2 }') == {"a": 2}
    assert ex("nothing here") is None


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
