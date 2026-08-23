"""`scripts/10b_build_termigen_taskset.py` -- the GRPO pool, and its leak guard.

WHY THIS IS A TASKSET BUILDER AND NOT AN SFT CONVERTER
    `allenai/open-instruct-termigen` has zero assistant turns. Measured over all
    3,556 rows: every one is exactly `(system, user)` plus a `ground_truth` and an
    `env_config` -- the open-instruct RLVR format. It is the third dataset this repo
    has been handed that looks like SFT data and is not, after `allenai/TMax-15K`
    and `allenai/tmax-15k-open-instruct`, so the converter asserts the absence
    rather than assuming it: if upstream ever adds responses, the run fails loudly
    instead of silently discarding them.

WHAT THE TESTS ARE ACTUALLY FOR
    The leak guard. `environment/` is the Docker build context, so anything in it is
    readable by the agent, and a verifier that lands there makes the task's reward
    hackable -- a failure that shows up as a suspiciously good GRPO curve and
    nothing else.

    The guard went through a wrong version first, and both versions are pinned here
    because the wrong one looked more general:

        discovering verifier names from each task's own `tests/` directory excluded
        11 sound tasks, because tasks *about* testing (pytest fixtures, k6 load
        tests, robotframework, ctest) legitimately ship `tests/conftest.py`,
        `tests/*.test.js`, `tests/__init__.py` in the build context as the very
        thing the agent is meant to work on.

    Measured across the tarball: `tests/test.sh` and `tests/test_outputs.py` appear
    in all 3,556 tasks; the other 26 filenames under `tests/` appear one to three
    times each. So the verifier is named, not globbed -- the same choice
    `10_build_rl_taskset.py` makes for the RST pool's two names.

    Of the 15 tasks excluded, **10 ship a byte-identical copy of the verifier** and
    5 are name-only matches. Both are excluded: for an RL pool a false exclusion
    costs one task, and a false inclusion costs the meaning of every reward that
    task produces.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, load_script  # noqa: E402

builder = load_script("10b_build_termigen_taskset")


def test_the_verifier_is_named_not_globbed():
    """The regression. Globbing `tests/` excluded 11 sound tasks."""
    assert builder.VERIFIER_FILES == ("test.sh", "test_outputs.py")


def test_base_image_reads_the_from_line():
    assert builder.base_image("FROM python:3.11-slim\nRUN pip install x") == "python:3.11-slim"
    assert builder.base_image("# comment\n\n  from  ubuntu:22.04  \n") == "ubuntu:22.04"
    assert builder.base_image("") == "?"
    assert builder.base_image("RUN echo no from line") == "?"


def test_sha256_is_over_bytes_so_encoding_cannot_shift_it():
    payload = "verifier body\n".encode("utf-8")
    assert builder.sha256_bytes(payload) == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------- the leak guard
#
# The guard lives inline in main(), so these reproduce its two rules against the
# same constant it uses. Reproduced rather than refactored because the rules are
# three lines and extracting them for testability would be the only reason to.


def _leak(files: dict[str, bytes]) -> tuple[str, str] | None:
    verifier_hashes = {builder.sha256_bytes(files[f"tests/{name}"])
                       for name in builder.VERIFIER_FILES if f"tests/{name}" in files}
    hit = None
    for name, blob in sorted(files.items()):
        if not name.startswith("environment/"):
            continue
        identical = builder.sha256_bytes(blob) in verifier_hashes
        if Path(name).name in builder.VERIFIER_FILES or identical:
            hit = (name, "byte_identical" if identical else "name_only")
            if identical:
                break
    return hit


def test_a_byte_identical_verifier_in_the_build_context_is_caught():
    verifier = b"#!/bin/bash\npytest tests/test_outputs.py\n"
    files = {
        "instruction.md": b"do the thing",
        "environment/Dockerfile": b"FROM python:3.11",
        "tests/test.sh": verifier,
        "environment/setup/run.sh": verifier,     # renamed copy of the verifier
    }
    hit = _leak(files)
    assert hit is not None and hit[1] == "byte_identical"


def test_a_verifier_by_name_anywhere_in_the_build_context_is_caught():
    files = {
        "instruction.md": b"x",
        "environment/Dockerfile": b"FROM python:3.11",
        "tests/test.sh": b"the real verifier",
        "environment/legacy_app/test.sh": b"a different script that shares the name",
    }
    hit = _leak(files)
    assert hit is not None and hit[1] == "name_only"


def test_a_clean_task_is_not_flagged():
    files = {
        "instruction.md": b"x",
        "environment/Dockerfile": b"FROM python:3.11",
        "environment/app/main.py": b"print(1)",
        "tests/test.sh": b"verifier",
        "tests/test_outputs.py": b"assert True",
    }
    assert _leak(files) is None


def test_workspace_test_files_are_not_mistaken_for_the_verifier():
    """The 11 tasks the first version of the guard wrongly excluded.

    A pytest-fixture task ships `environment/tests/conftest.py` because fixing the
    fixtures IS the task. Under the globbed guard that name came from the task's own
    `tests/`, so the task was dropped.
    """
    files = {
        "instruction.md": b"the conftest.py fixtures are broken; fix them",
        "environment/Dockerfile": b"FROM python:3.11",
        "environment/tests/conftest.py": b"import pytest  # broken fixture",
        "environment/tests/__init__.py": b"",
        "environment/tests/test_calculator.py": b"def test_add(): assert add(1,1)==2",
        "tests/test.sh": b"the real verifier",
        "tests/test_outputs.py": b"assert True",
        "tests/conftest.py": b"import pytest  # broken fixture",   # same name, and same bytes
    }
    assert _leak(files) is None, "a workspace fixture was mistaken for the verifier"


def test_the_verifier_itself_is_not_reported_as_its_own_leak():
    # `tests/` is not the build context, so a verifier sitting where it belongs must
    # not trip the hash rule against itself.
    files = {"instruction.md": b"x", "environment/Dockerfile": b"FROM x",
             "tests/test.sh": b"v", "tests/test_outputs.py": b"w"}
    assert _leak(files) is None


# ---------------------------------------------------------------- wiring


def test_the_absence_of_responses_is_asserted_not_assumed():
    source = (ROOT / "scripts" / "10b_build_termigen_taskset.py").read_text(encoding="utf-8")
    assert "assistant_turns" in source
    assert "may now be SFT-able" in source, (
        "upstream adding responses should fail loudly, not silently discard them")


def test_the_missing_pass_rates_are_recorded_as_null_not_defaulted():
    """The trap this pool sets.

    `10_build_rl_taskset.py` exists mostly to tier tasks by measured pass rate,
    because a GRPO group whose rollouts all score the same contributes zero gradient
    at full sandbox cost. Termigen ships no trial data, so a default of 0.0 or 0.5
    here would read as measured and get the pool spent like a screened one.
    """
    source = (ROOT / "scripts" / "10b_build_termigen_taskset.py").read_text(encoding="utf-8")
    assert '"empirical_pass_rate": None' in source
    assert '"n_reference_trials": 0' in source
    assert '"tier": "unknown"' in source


def test_the_output_matches_the_schema_grpo_actually_consumes():
    # 12_run_grpo.sh passes --prompt-data rl_tasks.jsonl --label-key label, so these
    # three keys are the contract, not a convention.
    source = (ROOT / "scripts" / "10b_build_termigen_taskset.py").read_text(encoding="utf-8")
    for key in ('"prompt":', '"label":', '"metadata":', "rl_tasks.jsonl"):
        assert key in source, key


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
