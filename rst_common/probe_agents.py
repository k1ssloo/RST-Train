"""Custom Harbor agents for the RST release probe (E1 sandbox arms, E2 replay).

WHY CUSTOM AGENTS AND NOT `docker exec`
    Harbor owns the trial: it builds the image, mounts `/logs/{agent,verifier,
    artifacts}` from the trial dir, runs the agent, then uploads `tests/` and runs
    the private verifier, writing `verifier/{reward.txt,ctrf.json}` and
    `result.json`. Anything that runs *as the agent* gets the verifier, the reward
    parsing (`rst_common.harbor.read_reward`) and the infra/budget classification
    for free, in the same shape every other rollout in this repo has. Driving a
    container by hand would mean re-implementing that and then arguing the numbers
    are comparable. So each probe arm is an agent:

      SnapshotOracleAgent   run the reference solution like Harbor's own oracle
                            agent does, but record what it changed on disk
      ReplayArtifactsAgent  extract a previous oracle run's output files into a
                            (mutated) environment and run NOTHING else
      TouchPathsAgent       create the absolute paths the instruction names and
                            nothing else -- the budget-1 black-box adversary
      ReplayKeystrokesAgent replay a recorded trajectory's keystrokes in order

WHAT HARBOR DOES AND DOES NOT HAND A CUSTOM AGENT
    `harbor/trial/trial.py` passes `task_dir`, `trial_paths` and the agent timeout
    ONLY to its built-in oracle agent. A custom agent gets `logs_dir` (the host side
    of `/logs/agent`), `model_name`, `logger`, `extra_env` and whatever
    `--agent-kwarg key=value` supplied -- all as strings. Every input below is
    therefore a kwarg, and every artifact is written under `self.logs_dir`, which
    the docker environment bind-mounts, so the host sees it without a download.

IMPORT BOUNDARY (tested)
    This module runs inside harbor's interpreter, which has no pandas. It imports
    the standard library, `harbor.*` and `rst_common.probe_core` only.
"""

from __future__ import annotations

import json
import shlex
import shutil
import time
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.scripts import build_execution_command, needs_chmod, quote_shell_arg

from rst_common import probe_core

DEFAULT_SCAN_ROOTS = "/app,/root,/home,/etc,/opt,/srv,/var/www,/workspace,/repo,/data,/project,/tmp,/mnt"
_EXCLUDE_FIND = (
    "! -path '/logs/*' ! -path '/tests/*' ! -path '/solution/*' ! -path '*/__pycache__/*' "
    "! -path '/root/.cache/*' ! -path '/var/lib/apt/*' ! -path '/tmp/replay*' ! -path '/proc/*'"
)
PER_FILE_CAP = 50 * 1024 * 1024
TOTAL_CAP = 512 * 1024 * 1024


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class _ProbeAgent(BaseAgent):
    """Common plumbing: JSON notes into logs_dir, a root shell helper, no model."""

    SUPPORTS_WINDOWS = False
    PROBE_NAME = "probe"

    @staticmethod
    def name() -> str:  # overridden per class through PROBE_NAME
        return "rst-probe"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return

    def _note(self, filename: str, payload: dict) -> None:
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.logs_dir) / filename).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    async def _sh(self, environment: BaseEnvironment, command: str, timeout_sec: int | None = None):
        return await environment.exec(command=f"bash -lc {shlex.quote(command)}", user="root", timeout_sec=timeout_sec)

    def _host(self, filename: str) -> Path:
        return Path(self.logs_dir) / filename


class SnapshotOracleAgent(_ProbeAgent):
    """Run `solution/solve.sh` like Harbor's oracle agent, and snapshot what it wrote.

    Kwargs: `task_dir` (host path), `fixture_paths` (csv of absolute container paths
    that the mutation targets; excluded from the artifact tarball so a later replay
    cannot restore the original instance), `scan_roots` (csv), `agent_timeout_sec`.
    Writes under /logs/agent: fs-before.tsv, fs-after.tsv, fixture-before.sha256,
    oracle.txt, exit-code.txt, delta.list, oracle-artifacts.tgz, snapshot.json.
    """

    @staticmethod
    def name() -> str:
        return "rst-snapshot-oracle"

    def __init__(self, logs_dir: Path, model_name: str | None = None, *, task_dir: str = "",
                 fixture_paths: str = "", scan_roots: str = DEFAULT_SCAN_ROOTS,
                 agent_timeout_sec: str | float | None = None, **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if not task_dir:
            raise ValueError("SnapshotOracleAgent needs --agent-kwarg task_dir=<host task dir>")
        self._task_dir = Path(task_dir)
        self._fixtures = [p for p in fixture_paths.split(",") if p.strip()]
        self._roots = [r for r in scan_roots.split(",") if r.strip()]
        self._timeout = int(float(agent_timeout_sec)) if agent_timeout_sec else None

    def _listing_cmd(self, out: str) -> str:
        roots = " ".join(shlex.quote(r) for r in self._roots)
        return (f"for r in {roots}; do [ -d \"$r\" ] && find \"$r\" -xdev -type f {_EXCLUDE_FIND} "
                f"-printf '%p\\t%s\\t%T@\\n' 2>/dev/null; done | sort > {out}")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        env_paths = EnvironmentPaths()
        started = time.time()
        solution_dir = self._task_dir / "solution"
        solve = solution_dir / "solve.sh"
        if not solve.is_file():
            raise FileNotFoundError(f"no solution/solve.sh under {self._task_dir}")
        await environment.upload_dir(source_dir=solution_dir, target_dir=str(env_paths.solution_dir))
        container_solve = str(env_paths.solution_dir / "solve.sh")
        if needs_chmod(container_solve):
            await environment.exec(command=f"chmod +x {quote_shell_arg(container_solve, None)}", user="root")

        await self._sh(environment, self._listing_cmd("/logs/agent/fs-before.tsv"))
        if self._fixtures:
            quoted = " ".join(shlex.quote(p) for p in self._fixtures)
            await self._sh(environment, f"sha256sum {quoted} > /logs/agent/fixture-before.sha256 2>/dev/null || true")

        command = build_execution_command(container_solve, stdout_path="/logs/agent/oracle.txt")
        result = await environment.exec(command=command, env={"DEBIAN_FRONTEND": "noninteractive"},
                                        timeout_sec=self._timeout)
        if result.return_code != 0:
            self._host("exit-code.txt").write_text(str(result.return_code))

        await self._sh(environment, self._listing_cmd("/logs/agent/fs-after.tsv"))
        before = self._read_host("fs-before.tsv")
        after = self._read_host("fs-after.tsv")
        created, modified = probe_core.artifact_delta(before, after)
        excluded = set(self._fixtures)
        delta = [p for p in created + modified if p not in excluded]
        sizes = probe_core.parse_listing(after)
        kept: list[str] = []
        skipped: list[str] = []
        total = 0
        for p in delta:
            size = sizes.get(p, (0, 0.0))[0]
            if size > PER_FILE_CAP or total + size > TOTAL_CAP:
                skipped.append(p)
                continue
            kept.append(p)
            total += size
        self._host("delta.list").write_text("".join(f"{p}\n" for p in kept), encoding="utf-8")
        tar_rc = None
        if kept:
            tar = await self._sh(environment, "tar -czf /logs/agent/oracle-artifacts.tgz --ignore-failed-read "
                                              "-T /logs/agent/delta.list 2>/logs/agent/tar.err")
            tar_rc = tar.return_code
        self._note("snapshot.json", {
            "exit_code": result.return_code,
            "created": created, "modified": modified,
            "excluded_fixtures": sorted(excluded & set(created + modified)),
            "kept": len(kept), "skipped_for_size": skipped, "bytes": total,
            "truncated": bool(skipped), "tar_rc": tar_rc,
            "seconds": round(time.time() - started, 1),
        })

    def _read_host(self, filename: str) -> str:
        path = self._host(filename)
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


class ReplayArtifactsAgent(_ProbeAgent):
    """Extract a recorded oracle run's OUTPUT files into this environment; run nothing.

    Kwargs: `artifacts_tgz` (host path produced by SnapshotOracleAgent),
    `exclude_paths` (csv of absolute paths never to restore -- the mutated fixture).
    This is the hard-coded-answer adversary with perfect knowledge of the answer: if
    the verifier still passes in a mutated environment, it compares against constants.
    """

    @staticmethod
    def name() -> str:
        return "rst-replay-artifacts"

    def __init__(self, logs_dir: Path, model_name: str | None = None, *, artifacts_tgz: str = "",
                 exclude_paths: str = "", **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if not artifacts_tgz:
            raise ValueError("ReplayArtifactsAgent needs --agent-kwarg artifacts_tgz=<host path>")
        self._tgz = Path(artifacts_tgz)
        self._exclude = [p for p in exclude_paths.split(",") if p.strip()]

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if not self._tgz.is_file():
            self._note("replay-artifacts.json", {"status": "missing_tgz", "path": str(self._tgz)})
            return
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._tgz, self._host("incoming.tgz"))
        excludes = " ".join(f"--exclude={shlex.quote(p.lstrip('/'))}" for p in self._exclude)
        result = await self._sh(environment,
                                f"tar -xzf /logs/agent/incoming.tgz -C / --no-same-owner {excludes} "
                                f"2>/logs/agent/untar.err && tar -tzf /logs/agent/incoming.tgz > /logs/agent/extracted.txt")
        self._note("replay-artifacts.json", {"status": "extracted" if result.return_code == 0 else "tar_failed",
                                             "return_code": result.return_code, "excluded": self._exclude})


class TouchPathsAgent(_ProbeAgent):
    """Create every absolute path the instruction names that does not already exist.

    Kwargs: `mode` = `empty` (zero-byte files) or `plausible` (`{}` for .json, `[]`
    for .yaml, a csv header, `ok` otherwise). Existing paths are inputs and are left
    alone; the record says which were created and which were skipped.
    """

    @staticmethod
    def name() -> str:
        return "rst-touch-paths"

    def __init__(self, logs_dir: Path, model_name: str | None = None, *, mode: str = "empty", **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._mode = mode

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        paths = probe_core.instruction_paths(instruction)
        created: list[str] = []
        skipped: list[str] = []
        for p in paths:
            q = shlex.quote(p)
            content = probe_core.plausible_content(p) if self._mode == "plausible" else ""
            script = (f"if [ -e {q} ]; then echo EXISTS; else mkdir -p \"$(dirname {q})\" && "
                      f"printf %s {shlex.quote(content)} > {q} && echo CREATED; fi")
            result = await self._sh(environment, script)
            out = (result.stdout or "").strip()
            (created if out.endswith("CREATED") else skipped).append(p)
        self._note("touch.json", {"mode": self._mode, "paths": paths, "created": created, "skipped_existing": skipped})


class ReplayKeystrokesAgent(_ProbeAgent):
    """Replay a recorded trajectory's keystrokes, in order, in one login shell.

    Kwargs: `keystrokes_json` (host path to a JSON list of strings), `per_cmd_cap_sec`
    (default 60, Terminus-2's own cap), `workdir` (default `/app`). Keystrokes that
    are tmux special keys or bare editor/REPL launches cannot be replayed through a
    script; the whole trajectory is then recorded as `non_replayable` and nothing
    runs. A tmux-faithful replay is the phase-2 upgrade if that bucket dominates.
    """

    @staticmethod
    def name() -> str:
        return "rst-replay-keystrokes"

    def __init__(self, logs_dir: Path, model_name: str | None = None, *, keystrokes_json: str = "",
                 per_cmd_cap_sec: str | int = 60, workdir: str = "/app", **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if not keystrokes_json:
            raise ValueError("ReplayKeystrokesAgent needs --agent-kwarg keystrokes_json=<host path>")
        self._keystrokes = json.loads(Path(keystrokes_json).read_text(encoding="utf-8"))
        self._cap = int(per_cmd_cap_sec)
        self._workdir = workdir

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        keys = [str(k) for k in self._keystrokes]
        flags = probe_core.interactive_flags(keys)
        if flags:
            self._note("replay.json", {"status": "non_replayable", "interactive": flags, "n_keystrokes": len(keys)})
            return
        lines = ["#!/bin/bash", "set +e", f"cd {shlex.quote(self._workdir)} 2>/dev/null || cd /"]
        for i, k in enumerate(keys):
            lines.append(k.rstrip("\n"))
            lines.append(f'echo "__RC__ {i} $?" >> /logs/agent/replay.log')
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        self._host("replay.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
        started = time.time()
        timeout = max(60, self._cap * max(1, len(keys)))
        status = "completed"
        rc = None
        try:
            result = await environment.exec(command="bash /logs/agent/replay.sh > /logs/agent/replay.out 2>&1",
                                            user="root", timeout_sec=timeout)
            rc = result.return_code
        except Exception as exc:  # harbor raises on timeout
            status = "timeout" if "time" in str(exc).lower() else f"error:{type(exc).__name__}"
        sent = 0
        log = self._host("replay.log")
        if log.is_file():
            sent = sum(1 for ln in log.read_text(errors="replace").splitlines() if ln.startswith("__RC__"))
        self._note("replay.json", {"status": status, "return_code": rc, "n_keystrokes": len(keys),
                                   "n_completed": sent, "seconds": round(time.time() - started, 1)})
