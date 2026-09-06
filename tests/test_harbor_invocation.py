"""One `harbor run` command line, and the switch that makes its trajectories data.

BUG.md BUG-19: Terminus-2's default ATIF holds the harness's "Analysis:/Plan:"
rendering of each turn and drops the model's completion. A rollout kept for training
without `trajectory_config.raw_content` is a file that looks like data and cannot
become any. The kwarg lives in one place and every caller that can feed the data
loop reaches it through `run_argv`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, load_repo_module, load_script  # noqa: E402

harbor = load_repo_module("rst_common.harbor")


def test_run_argv_carries_the_single_attempt_contract_and_the_caller_extras():
    argv = harbor.run_argv(
        harbor_bin="harbor", task_dir=Path("/tmp/t"), agent="terminus-2",
        model="hosted_vllm/x", env="docker", jobs_dir=Path("/tmp/j"), job_name="job",
        env_kwargs=["cpu_enforcement_policy=ignore"], agent_kwargs=["a=1"],
        extra=["--temperature", "0.7"])
    text = " ".join(argv)
    assert argv[:2] == ["harbor", "run"]
    for flag in ("--n-attempts 1", "--n-concurrent 1", "--max-retries 0", "--quiet",
                 "--environment-kwarg cpu_enforcement_policy=ignore", "--agent-kwarg a=1",
                 "--temperature 0.7", "--jobs-dir /tmp/j", "--job-name job"):
        assert flag in text, flag


def test_a_model_free_agent_gets_no_model_flag_and_can_be_an_import_path():
    argv = harbor.run_argv(
        harbor_bin="harbor", task_dir=Path("/tmp/t"),
        agent=harbor.PROBE_AGENT_IMPORT_PATHS["snapshot_oracle"], model=None, env="docker",
        jobs_dir=Path("/tmp/j"), job_name="probe",
        agent_kwargs=["task_dir=/tmp/t", "fixture_paths=/app/x"],
        env_kwargs=["cpu_enforcement_policy=ignore"])
    text = " ".join(argv)
    assert "--model" not in argv, "oracle/nop/probe agents make no model call"
    assert "--agent rst_common.probe_agents:SnapshotOracleAgent" in text
    assert "--agent-kwarg task_dir=/tmp/t" in text and "--n-attempts 1" in text
    with_model = harbor.run_argv(harbor_bin="harbor", task_dir=Path("/tmp/t"), agent="terminus-2",
                                 model="hosted_vllm/x", env="docker", jobs_dir=Path("/tmp/j"), job_name="j")
    assert with_model[with_model.index("--model") + 1] == "hosted_vllm/x"


def test_custom_agent_env_puts_the_repo_root_first_on_pythonpath_once():
    env = {"PYTHONPATH": "/elsewhere:" + str(Path("/repo").resolve())}
    harbor.custom_agent_env(env, Path("/repo"))
    assert env["PYTHONPATH"].split(":") == [str(Path("/repo").resolve()), "/elsewhere"]
    empty: dict[str, str] = {}
    harbor.custom_agent_env(empty, Path("/repo"))
    assert empty["PYTHONPATH"] == str(Path("/repo").resolve())
    assert harbor.harbor_python("/x/venv/bin/harbor") == Path("/x/venv/bin/python")


def test_the_export_kwarg_asks_for_raw_content_and_linear_history():
    kwargs = harbor.export_agent_kwargs()
    assert len(kwargs) == 1 and kwargs[0].startswith("trajectory_config=")
    parsed = json.loads(kwargs[0].split("=", 1)[1])
    assert parsed == {"raw_content": True, "linear_history": True}
    assert parsed is not harbor.TRAJECTORY_EXPORT_CONFIG, "callers get a copy in text form"


def test_every_rollout_launcher_builds_its_command_through_run_argv():
    for rel in ("scripts/06_eval.py", "rl/generate.py", "verl_backend/harbor_agent_loop.py"):
        source = (ROOT / rel).read_text(encoding="utf-8")
        assert "run_argv(" in source, rel
        assert '"--n-attempts"' not in source, f"{rel} spells the harbor command out again"
        assert "export_agent_kwargs" in source, f"{rel} cannot request exportable trajectories"


def test_eval_refuses_to_export_on_a_harbor_without_agent_kwarg():
    ev = load_script("06_eval")

    class Args:
        export_trajectories = True
        harbor_bin = "harbor"
        keep_jobs = False
        out = "/tmp/out"

    original = ev.harbor_run_flags
    ev.harbor_run_flags = lambda _bin: frozenset()
    try:
        try:
            ev.trajectory_export_argv(Args())
        except SystemExit as exc:
            assert "--agent-kwarg" in str(exc)
        else:
            raise AssertionError("must refuse: the trajectories would be unusable")
    finally:
        ev.harbor_run_flags = original


def test_eval_export_forces_keep_jobs_and_records_what_it_forwarded():
    ev = load_script("06_eval")

    class Args:
        export_trajectories = True
        harbor_bin = "harbor"
        keep_jobs = False
        out = "/tmp/out"

    original = ev.harbor_run_flags
    ev.harbor_run_flags = lambda _bin: frozenset({"--agent-kwarg"})
    try:
        args = Args()
        kwargs, record = ev.trajectory_export_argv(args)
        assert args.keep_jobs is True, "deleting the job dir would delete the data"
        assert record["enabled"] is True and record["agent_kwargs"] == kwargs
        assert re.search(r"raw_content", kwargs[0])
        off = Args(); off.export_trajectories = False
        assert ev.trajectory_export_argv(off) == ([], {"enabled": False, "agent_kwargs": [],
                                                       "control": ev.trajectory_export_argv(off)[1]["control"]})
        assert off.keep_jobs is False
    finally:
        ev.harbor_run_flags = original


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
