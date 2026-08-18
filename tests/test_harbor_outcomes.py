"""`rst_common.harbor` decides what a finished rollout *means*.

Every number this repo reports -- eval pass rate, GRPO advantage, DPO pair yield --
is downstream of the two-kind split in here: `HARNESS_INFRA` is unmeasured and
leaves the denominator, `AGENT_BUDGET` is a policy failure worth reward 0 and stays
in it. Getting that backwards does not crash anything; it inflates the pass rate
exactly on the hardest tasks and removes the most informative negatives from the
GRPO group. So it gets tests.

Standard library only: no torch, no transformers, no sandbox.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import load_repo_module  # noqa: E402

H = load_repo_module("rst_common.harbor")


def write_job(root: Path, layout: str, payload: dict) -> Path:
    """Materialize one of the three result.json layouts Harbor 0.21 produces."""
    job = root / layout
    target = {
        "trials": job / "trials" / "trial-1" / "result.json",
        "nested": job / "trial-1" / "result.json",
        "flat": job / "result.json",
    }[layout]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return job


def reward_payload(value: float) -> dict:
    return {"verifier_result": {"rewards": {"reward": value}}}


def test_infra_markers_win_over_budget_markers():
    """An image pull that dies says both "failed to pull image" and "timed out"."""
    kind, marker = H.classify("Error: failed to pull image ... operation timed out")
    assert kind == H.HARNESS_INFRA, "a dead pull was attributed to the agent"
    assert marker == "failed to pull image"


def test_bare_timeout_is_the_agents_budget_not_infrastructure():
    kind, marker = H.classify("agent loop timed out after 1800s")
    assert kind == H.AGENT_BUDGET
    assert marker == "timed out"
    outcome = H.budget(marker)
    assert outcome.reward == 0.0, "a budget failure must score 0, not be dropped"
    assert outcome.scorable, "a budget failure must stay in the denominator"
    assert not outcome.solved


def test_clean_text_classifies_as_nothing():
    assert H.classify("all tests passed, reward 1.0") == (None, None)
    assert H.classify_infra("command too long") is None, "budget marker leaked into infra"


def test_infra_outcome_is_unmeasured_not_zero():
    outcome = H.infra("no space left on device")
    assert outcome.reward is None
    assert not outcome.scorable, "an unmeasured trial must leave the denominator"
    assert outcome.infra_reason and outcome.budget_reason is None


def test_read_reward_handles_all_three_layouts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for layout in ("trials", "nested", "flat"):
            job = write_job(root, layout, reward_payload(1.0))
            outcome = H.read_reward(job)
            assert outcome.reward == 1.0, f"{layout} layout was not read"
            assert outcome.solved and outcome.kind is None


def test_missing_or_corrupt_artifacts_are_infrastructure():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        empty = root / "empty"
        empty.mkdir()
        assert H.read_reward(empty).kind == H.HARNESS_INFRA
        assert H.read_reward(empty).reward is None, "a missing artifact scored as 0"

        broken = root / "broken"
        broken.mkdir()
        (broken / "result.json").write_text("{not json", encoding="utf-8")
        outcome = H.read_reward(broken)
        assert outcome.kind == H.HARNESS_INFRA and outcome.reward is None


def test_a_real_zero_reward_stays_a_zero_reward():
    with tempfile.TemporaryDirectory() as tmp:
        job = write_job(Path(tmp), "flat", reward_payload(0.0))
        outcome = H.read_reward(job)
        assert outcome.reward == 0.0 and outcome.kind is None
        assert outcome.scorable and not outcome.solved


def test_exception_info_is_split_by_kind():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        infra_job = write_job(root, "flat", {"exception_info": {"msg": "docker daemon is down"}})
        assert H.read_reward(infra_job).kind == H.HARNESS_INFRA

        budget_root = root / "b"
        budget_root.mkdir()
        budget_job = write_job(budget_root, "flat", {"exception_info": {"msg": "command too long"}})
        outcome = H.read_reward(budget_job)
        assert outcome.kind == H.AGENT_BUDGET and outcome.reward == 0.0


def test_refine_with_stdout_only_ever_escalates():
    scored = H.Outcome(reward=1.0)
    assert H.refine_with_stdout(scored, "docker daemon died later on") is scored, \
        "a verifier score was overwritten by stray log text"

    vague = H.infra("harness exception")
    refined = H.refine_with_stdout(vague, "could not resolve host registry.example")
    assert refined.kind == H.HARNESS_INFRA and refined.reason == "could not resolve host"

    # infra must not be downgraded to a budget failure by a stray "timed out"
    stayed = H.refine_with_stdout(H.infra("no space left on device"), "... timed out ...")
    assert stayed.kind == H.HARNESS_INFRA


def test_wall_clock_timeout_is_a_budget_failure_and_says_it_is_ambiguous():
    outcome = H.wall_clock_timeout(1800)
    assert outcome.kind == H.AGENT_BUDGET and outcome.reward == 0.0
    assert "hung sandbox" in (outcome.reason or ""), \
        "the ambiguity must stay in the reason string, not be silently resolved"


def test_proxy_policy_drops_the_proxy_for_a_local_docker_sandbox():
    env = {"HTTP_PROXY": "http://proxy:8080", "https_proxy": "http://proxy:8080", "PATH": "/bin"}
    H.apply_proxy_policy(env, "docker", "http://127.0.0.1:30000/v1")
    assert not any(key in env for key in H.PROXY_KEYS), "a local sandbox kept the proxy"
    assert env["PATH"] == "/bin", "unrelated environment was disturbed"


def test_proxy_policy_keeps_the_proxy_off_machine_but_excludes_the_local_endpoint():
    env = {"HTTPS_PROXY": "http://proxy:8080"}
    H.apply_proxy_policy(env, "daytona", "http://10.0.0.5:30000/v1")
    assert env["HTTPS_PROXY"] == "http://proxy:8080", "an off-machine backend lost its proxy"
    for key in ("NO_PROXY", "no_proxy"):
        entries = env[key].split(",")
        assert "10.0.0.5" in entries and "127.0.0.1" in entries and "localhost" in entries
        assert len(entries) == len(set(entries)), "duplicate NO_PROXY entries"


def test_proxy_policy_is_a_no_op_when_there_is_no_proxy():
    env = {"PATH": "/bin"}
    H.apply_proxy_policy(env, "daytona", "http://127.0.0.1:30000/v1")
    assert env == {"PATH": "/bin"}


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
