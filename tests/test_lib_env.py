"""What `scripts/lib_env.sh` decides BEFORE it touches any environment.

The fast path in `rst_enter_env` asks "is the current `python` already the
training env?" by looking for a handful of modules. On an image whose system
interpreter already ships those modules, the answer is yes for an interpreter
that is not the built env, so nothing enters it and every later stage runs on
whatever that image happens to carry. The backend's own package is the
discriminator, and these tests pin when it is added and when it is not.

No environment is entered here: the tests put a `python` on PATH that reports
every module missing, so the fast path cannot match and the function falls
through to its own error handling. Only the requirement list is inspected.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "scripts" / "lib_env.sh"


def _run(body: str, env_prefix: str = "", tmp_bin: bool = True) -> str:
    """Run a bash snippet that sources lib_env.sh, and return its stdout."""
    setup = ""
    if tmp_bin:
        # A `python` that finds nothing, so rst_env_ok is always false, and a PATH
        # with no package manager on it, so no real env can be entered either.
        setup = textwrap.dedent(
            """
            tmp="$(mktemp -d)"
            printf '#!/bin/sh\\nexit 1\\n' > "$tmp/python"
            chmod +x "$tmp/python"
            PATH="$tmp:/usr/bin:/bin"
            """
        )
    script = f"set -euo pipefail\n{setup}\n{env_prefix}\nsource {LIB}\n{body}\n"
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=120
    )
    return out.stdout


def test_the_verl_backend_requires_its_own_package_before_trusting_the_interpreter():
    out = _run('rst_enter_env >/dev/null 2>&1 || true; echo "$RST_ENV_REQUIRE"')
    assert "verl" in out.split(), out


def test_an_explicit_requirement_list_is_left_exactly_as_the_caller_wrote_it():
    out = _run(
        'rst_enter_env >/dev/null 2>&1 || true; echo "$RST_ENV_REQUIRE"',
        env_prefix='export RST_ENV_REQUIRE="torch transformers"',
    )
    assert out.strip() == "torch transformers", out


def test_the_slime_backend_is_not_asked_for_a_verl_install_it_never_makes():
    out = _run('rst_enter_env slime >/dev/null 2>&1 || true; echo "$RST_ENV_REQUIRE"')
    assert "verl" not in out.split(), out


def test_sourcing_the_library_cannot_abort_a_set_e_caller():
    # `[[ ... ]] && x=1` at file scope returns non-zero when the test is false,
    # which under `set -e` kills the sourcing script before it runs anything.
    out = _run('echo "sourced-ok"', tmp_bin=False)
    assert "sourced-ok" in out, out


def test_sourcing_twice_does_not_mistake_our_own_default_for_the_callers_choice():
    out = _run(
        f'source {LIB}; rst_enter_env >/dev/null 2>&1 || true; echo "$RST_ENV_REQUIRE"'
    )
    assert "verl" in out.split(), out
