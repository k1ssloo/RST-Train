"""One definition of what a finished Harbor job means.

`scripts/06_eval.py`, `rl/generate.py` and `verl_backend/harbor_agent_loop.py` each
used to carry a private copy of the marker list, the classifier, the result reader
and the proxy policy -- and the copies had already drifted (`rl/generate.py` never
globbed the flat `<job>/result.json`, so a layout the other two handled looked to
RL like "job artifacts incomplete"). This module exists so that eval and RL score
the same event the same way; otherwise the two sets of numbers are not comparable
and nobody can tell why.

TWO KINDS OF FAILURE, AND WHY THE SPLIT IS LOAD-BEARING
-------------------------------------------------------
`HARNESS_INFRA` -- the sandbox broke: Docker daemon, registry, DNS, disk. This is
    *not* a reward of 0. In eval it is excluded from the pass-rate denominator and
    reported as its own rate; in RL the sample is dropped (`remove_sample`).
    Teaching the policy that its actions were bad because a pull timed out is the
    fastest way to a silently poisoned run.

`AGENT_BUDGET` -- the agent spent its wall clock, or emitted a command the harness
    refused because it was too long. That is the *policy* failing, and on
    long-horizon tasks it is the single most common way a weak policy fails: it
    thrashes until the budget runs out. Both of these used to sit in the infra
    list, which had two consequences, both bad:
      * eval: the hardest tasks were dropped from the denominator most often, so
        the pass rate was inflated exactly where it mattered;
      * RL: the most informative negatives were removed from the GRPO group,
        which changes the group baseline and therefore every advantage in it.
    So a budget failure scores as **reward 0, inside the denominator**, and is
    counted separately so it stays visible.

Order matters in `classify()`: harness markers are tested first, because an image
pull that dies says both "failed to pull image" and "timed out" and the first one
is the real cause. `"timed out"` on its own is ambiguous by construction -- the
harness cannot tell "the agent kept going" from "the sandbox hung" from a string
-- so it is attributed to the agent and the count is reported, rather than being
quietly excluded from the science.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

HARNESS_INFRA = "harness_infra"
AGENT_BUDGET = "agent_budget"

# The sandbox or the harness broke. Deliberately narrow: a plain reward-0 must
# never land here, and neither must anything the policy could have caused.
HARNESS_INFRA_MARKERS = (
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
    "rate limit exceeded",
    "unable to locate package",
)

# The policy ran out of room. Counted as reward 0, never excluded.
AGENT_BUDGET_MARKERS = (
    "command too long",
    "timed out",
)

ALL_MARKERS = HARNESS_INFRA_MARKERS + AGENT_BUDGET_MARKERS


def classify(text: str) -> tuple[str | None, str | None]:
    """Return ``(kind, marker)`` for a blob of harness text; ``(None, None)`` if clean."""
    low = text.lower()
    for marker in HARNESS_INFRA_MARKERS:
        if marker in low:
            return HARNESS_INFRA, marker
    for marker in AGENT_BUDGET_MARKERS:
        if marker in low:
            return AGENT_BUDGET, marker
    return None, None


def classify_infra(text: str) -> str | None:
    """Harness-infrastructure marker only, or None. Budget markers do not count."""
    kind, marker = classify(text)
    return marker if kind == HARNESS_INFRA else None


@dataclass(frozen=True)
class Outcome:
    """What one finished rollout means.

    Exactly one of these three shapes:
      * ``reward=<float>, kind=None``            -- the verifier ran and scored it
      * ``reward=0.0,     kind=AGENT_BUDGET``    -- policy failure, counts as 0
      * ``reward=None,    kind=HARNESS_INFRA``   -- unmeasured, excluded / dropped
    """

    reward: float | None
    kind: str | None = None
    reason: str | None = None

    @property
    def infra_reason(self) -> str | None:
        return self.reason if self.kind == HARNESS_INFRA else None

    @property
    def budget_reason(self) -> str | None:
        return self.reason if self.kind == AGENT_BUDGET else None

    @property
    def scorable(self) -> bool:
        """True when this trial belongs in the pass-rate denominator."""
        return self.reward is not None and self.kind != HARNESS_INFRA

    @property
    def solved(self) -> bool:
        return self.scorable and (self.reward or 0.0) >= 1.0


def infra(reason: str) -> Outcome:
    return Outcome(reward=None, kind=HARNESS_INFRA, reason=reason)


def budget(reason: str) -> Outcome:
    return Outcome(reward=0.0, kind=AGENT_BUDGET, reason=reason)


def wall_clock_timeout(seconds: float) -> Outcome:
    """We killed the harbor subprocess at our own wall clock.

    Attributed to the agent's budget, not to infrastructure: nothing else in this
    pipeline caps how long Terminus-2 may keep issuing commands, so this timeout
    *is* the budget being enforced. A hung sandbox produces the same symptom, which
    is why the reason string says so and the count is reported separately.
    """
    return budget(f"agent wall-clock budget exceeded ({seconds:.0f}s; "
                  f"a hung sandbox looks identical from here)")


def result_json_candidates(job_dir: Path) -> list[Path]:
    """Every result.json Harbor 0.21 might have written, in a stable order.

    Layouts seen in the wild: ``<job>/trials/<trial>/result.json``,
    ``<job>/<trial>/result.json`` and a flat ``<job>/result.json``. Missing the
    flat one is what the drift between the three old copies came down to.
    """
    return sorted({
        *(job_dir / "trials").glob("*/result.json"),
        *job_dir.glob("*/result.json"),
        *job_dir.glob("result.json"),
    })


def read_reward(job_dir: Path) -> Outcome:
    """Read the verifier's reward out of a finished Harbor job directory.

    Never raises: an unreadable or absent artifact is an infrastructure outcome,
    because "we could not measure this" is not "the policy scored 0".
    """
    candidates = result_json_candidates(job_dir)
    if not candidates:
        return infra("job artifacts incomplete: no result.json")
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return infra(f"unreadable result.json: {exc}")
        exception_info = payload.get("exception_info")
        if exception_info:
            kind, marker = classify(json.dumps(exception_info))
            if kind == AGENT_BUDGET:
                return budget(marker or "agent budget exceeded")
            return infra(marker or "harness exception")
        value = ((payload.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        if isinstance(value, (int, float)):
            return Outcome(reward=float(value))
    return infra("no verifier reward in result.json")


def refine_with_stdout(outcome: Outcome, text: str) -> Outcome:
    """Let Harbor's own stdout name the cause more precisely than result.json did.

    Only ever escalates: a `HARNESS_INFRA` verdict is never downgraded to a budget
    failure on the strength of a stray "timed out" further up the log, and a real
    verifier score is never overwritten at all.
    """
    if outcome.reward is not None and outcome.kind is None:
        return outcome
    kind, marker = classify(text)
    if kind is None or marker is None:
        return outcome
    if outcome.kind == HARNESS_INFRA and kind == AGENT_BUDGET:
        return outcome
    return infra(marker) if kind == HARNESS_INFRA else budget(marker)


PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")


def apply_proxy_policy(env: dict[str, str], harbor_env: str, local_url: str) -> None:
    """Decide what the harbor subprocess should do with proxy variables, in place.

    Two situations, and conflating them breaks one of them:

    * ``--env docker``: sandbox, harness and the local endpoint are all on this
      host. A proxy can only get in the way of reaching the endpoint -- drop it.
    * an off-machine backend (daytona/e2b/modal/k8s): harbor must reach the
      provider's HTTPS API, which on a locked-down cluster is exactly what the
      proxy is for. Keep it, and put the local endpoint in NO_PROXY so rollout
      traffic still goes direct.
    """
    if harbor_env == "docker":
        for key in PROXY_KEYS:
            env.pop(key, None)
        return
    if not any(env.get(key) for key in PROXY_KEYS):
        return
    host = urllib.parse.urlsplit(local_url).hostname or ""
    extra = [h for h in (host, "127.0.0.1", "localhost") if h]
    for key in ("NO_PROXY", "no_proxy"):
        current = [tok for tok in env.get(key, "").split(",") if tok]
        env[key] = ",".join(dict.fromkeys(current + extra))


# ----------------------------------------------------------------- invocation

# Terminus-2 writes its ATIF trajectory in one of two shapes, chosen by this agent
# kwarg. Without it (Harbor 0.21's default) each agent step's `message` is the
# harness's OWN rendering -- "Analysis: ...\nPlan: ..." -- and the commands live in
# `tool_calls`; the model's actual completion is gone. With `raw_content` the step
# holds the completion verbatim, which is the shape the RST release has and the only
# shape `normalize_assistant` can turn into a training target. `linear_history`
# splits the file at every context compaction, so each file is exactly the history
# the model was shown rather than a stitched-together view it never saw.
# Every rollout meant to feed the data loop -- eval with --export-trajectories, RL
# with RST_EXPORT_TRAJECTORIES=1 -- must carry this. See BUG.md BUG-19.
TRAJECTORY_EXPORT_CONFIG: dict[str, bool] = {"raw_content": True, "linear_history": True}


def export_agent_kwargs() -> list[str]:
    """`--agent-kwarg` values that make a Terminus-2 trajectory SFT-exportable."""
    return ["trajectory_config=" + json.dumps(TRAJECTORY_EXPORT_CONFIG, separators=(",", ":"))]


def run_argv(
    *,
    harbor_bin: str,
    task_dir: Path,
    agent: str,
    model: str | None,
    env: str,
    jobs_dir: Path,
    job_name: str,
    env_kwargs: Iterable[str] = (),
    agent_kwargs: Iterable[str] = (),
    extra: Iterable[str] = (),
) -> list[str]:
    """The one `harbor run` command line eval, both RL paths and the release probe launch.

    One attempt, one concurrent trial, no retries: the caller owns concurrency and
    the retry policy (a retried infra failure would otherwise be scored as if the
    first attempt had never happened). `--quiet` keeps Harbor's progress UI out of
    the stdout the caller classifies with `refine_with_stdout`.

    `model=None` omits `--model`: Harbor's `oracle` and `nop` agents and the probe's
    custom agents (`PROBE_AGENT_IMPORT_PATHS`) make no model call, and Harbor 0.21
    accepts `--agent module.path:ClassName` in the same `--agent` slot.
    """
    argv = [
        harbor_bin, "run",
        "--path", str(Path(task_dir).resolve()),
        "--agent", agent,
    ]
    if model is not None:
        argv += ["--model", model]
    argv += [
        "--env", env,
        "--n-attempts", "1",
        "--n-concurrent", "1",
        "--max-retries", "0",
        "--jobs-dir", str(jobs_dir),
        "--job-name", job_name,
        "--quiet",
    ]
    argv += list(extra)
    for kwarg in agent_kwargs:
        argv += ["--agent-kwarg", kwarg]
    for kwarg in env_kwargs:
        argv += ["--environment-kwarg", kwarg]
    return argv


# The release probe's agents (rst_common/probe_agents.py). Harbor imports the module
# with `importlib.import_module` inside ITS OWN interpreter, which is why the module
# is stdlib+harbor only and why `custom_agent_env` has to put the repo root on
# PYTHONPATH -- a bad import otherwise surfaces as a generic `exception_info` on
# every trial, i.e. as 100% infra failures.
PROBE_AGENT_IMPORT_PATHS = {
    "snapshot_oracle": "rst_common.probe_agents:SnapshotOracleAgent",
    "replay_artifacts": "rst_common.probe_agents:ReplayArtifactsAgent",
    "touch_paths": "rst_common.probe_agents:TouchPathsAgent",
    "replay_keystrokes": "rst_common.probe_agents:ReplayKeystrokesAgent",
}


def custom_agent_env(env: dict[str, str], repo_root: Path) -> None:
    """Make `rst_common.probe_agents` importable by the harbor subprocess, in place."""
    root = str(Path(repo_root).resolve())
    current = [tok for tok in env.get("PYTHONPATH", "").split(":") if tok and tok != root]
    env["PYTHONPATH"] = ":".join([root, *current])


def harbor_python(harbor_bin: str) -> Path:
    """The interpreter behind a `harbor` entry point, for an import pre-flight."""
    return Path(harbor_bin).resolve().parent / "python"
