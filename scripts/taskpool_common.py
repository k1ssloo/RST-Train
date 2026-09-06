#!/usr/bin/env python3
"""What every RL task-pool builder must agree on: the tiers, the image, the leak rule.

Three pools feed one GRPO launcher -- the RST release (`10_build_rl_taskset.py`),
termigen (`10b`) and SWE-Gym (`10c`) -- and "sweet" has to mean the same band in all
of them or the launcher's budget split means nothing. The tier table, the tier lookup
and the Dockerfile `FROM` reader were duplicated across the three; the verifier-leak
DECISION was implemented twice with two different shapes (a directory on disk, a dict
of bytes in memory). The decision is now one function fed by either shape.

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from taskpool_common import TIERS, tier_of, base_image, find_verifier_leak
"""

from __future__ import annotations

import hashlib
import re
import tarfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath

# (name, low, high): a pass rate p lands in the tier with low <= p < high.
TIERS = (
    ("sweet", 0.10, 0.90),   # primary GRPO pool: reliable within-group variance
    ("hard", 0.00, 0.10),    # exploration: only worth it once the policy improves
    ("easy", 0.90, 1.01),    # near-saturated: keep a trickle to avoid regression
)

# The six files that make an RST task; a materialized task dir must have them all.
TRACKED = (
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "solution/solve.sh",
    "tests/test.sh",
    "tests/test_state.py",
)

# The private verifier of an RST task lives under tests/. Anything with one of these
# names -- or byte-identical to one of them -- inside environment/ (the Docker build
# context, i.e. visible to the agent) makes the task's reward hackable.
RST_VERIFIER_FILES = ("test.sh", "test_state.py")


def tier_of(pass_rate: float) -> str:
    for name, low, high in TIERS:
        if low <= pass_rate < high:
            return name
    raise AssertionError(f"pass rate {pass_rate} fell outside every tier")


def base_image(dockerfile: str) -> str:
    """The `FROM` line, so a pre-build pass knows what to pull; `?` when absent."""
    match = re.search(r"^\s*FROM\s+(\S+)", dockerfile or "", re.M | re.I)
    return match.group(1) if match else "?"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def find_verifier_leak(
    build_context: Iterable[tuple[str, str]],
    verifier_hashes: set[str],
    verifier_names: Iterable[str] = RST_VERIFIER_FILES,
) -> tuple[str, str] | None:
    """Return `(path, kind)` for the first file in the build context that leaks the
    verifier, or None. `kind` is `byte_identical` or `name_only`.

    `build_context` is `(relative_path, sha256)` for every file under `environment/`.
    A byte-identical hit is the unambiguous case: the agent can read its own grader.
    A name-only hit is a project file that merely shares the verifier's name -- still
    excluded, because for an RL pool a false exclusion costs one task and a false
    inclusion costs the meaning of every reward that task produces. Byte-identical
    wins over name-only when both occur, so the reported kind is the stronger one.
    """
    names = set(verifier_names)
    hit: tuple[str, str] | None = None
    for path, digest in sorted(build_context):
        if digest in verifier_hashes:
            return path, "byte_identical"
        if hit is None and PurePosixPath(path).name in names:
            hit = (path, "name_only")
    return hit


def materialize_tasks(tasks_root: Path, wanted: Mapping[str, Mapping[str, str]], task_root: Path,
                      *, log: Callable[[str], None] = print) -> None:
    """Extract task dirs out of the release tars: `task_root/<task_id>/<member path>`.

    `wanted` is `{shard: {member_prefix_without_slash: task_id}}`, exactly what
    `10_build_rl_taskset.py` built inline before this was lifted out. One sequential
    pass per shard (no `getmembers()`, no random access -- the tars are 3.55 GiB), the
    first matching prefix wins, and a member that would escape its task dir aborts the
    run rather than being skipped: a pool with one silently missing file is a pool
    whose verifier may be missing.
    """
    for shard, members in sorted(wanted.items()):
        prefix_to_id = {p + "/": tid for p, tid in members.items()}
        with tarfile.open(Path(tasks_root) / shard) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                for prefix, tid in prefix_to_id.items():
                    if not member.name.startswith(prefix):
                        continue
                    rel = member.name[len(prefix) :]
                    # path-safety: never write outside the task dir
                    if rel.startswith("/") or ".." in Path(rel).parts:
                        raise SystemExit(f"unsafe member path: {member.name}")
                    dest = Path(task_root) / tid / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    handle = tar.extractfile(member)
                    if handle is not None:
                        dest.write_bytes(handle.read())
                    break
        log(f"  materialized from {shard}")


def verify_tracked(task_root: Path, task_ids: Iterable[str]) -> tuple[int, list[tuple[str, list[str]]]]:
    """`(complete_count, [(task_id, missing_files), ...])` against the six-file contract."""
    complete = 0
    incomplete: list[tuple[str, list[str]]] = []
    for tid in task_ids:
        missing = [f for f in TRACKED if not (Path(task_root) / tid / f).is_file()]
        if missing:
            incomplete.append((tid, missing))
        else:
            complete += 1
    return complete, incomplete
