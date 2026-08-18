#!/usr/bin/env python3
"""Run the test suite with or without pytest.

    python -m pytest tests/ -q        # if pytest is installed
    python tests/run_tests.py         # if it is not (a fresh cluster env often is not)

Both runners execute the same functions. There are no fixtures anywhere in
`tests/` precisely so that this fallback stays a few lines long and cannot drift
from what pytest does.

Exit codes: 0 all passed (skips allowed), 1 at least one failure.
A skip is printed, counted and named -- a run that skipped everything must not be
mistakable for a run that verified everything.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _util import Skipped  # noqa: E402


def run_module(module: ModuleType) -> int:
    """Run every `test_*` function in one module. Returns the failure count."""
    names = sorted(n for n in dir(module) if n.startswith("test_"))
    failures = 0
    for name in names:
        func = getattr(module, name)
        if not callable(func):
            continue
        label = f"{module.__name__.split('.')[-1]}::{name}"
        try:
            func()
        except Skipped as exc:
            print(f"  SKIP {label}: {exc}")
        except BaseException:  # noqa: BLE001 - a test runner reports, it does not filter
            failures += 1
            print(f"  FAIL {label}")
            traceback.print_exc()
        else:
            print(f"  ok   {label}")
    return failures


def main() -> int:
    import importlib

    failures = 0
    modules = sorted(p.stem for p in HERE.glob("test_*.py"))
    if len(sys.argv) > 1:
        wanted = {Path(a).stem for a in sys.argv[1:]}
        modules = [m for m in modules if m in wanted]
        if not modules:
            print(f"no test module matched {sorted(wanted)}")
            return 1
    for stem in modules:
        print(f"[{stem}]")
        try:
            module = importlib.import_module(stem)
        except Skipped as exc:
            print(f"  SKIP whole module: {exc}")
            continue
        except BaseException:  # noqa: BLE001
            failures += 1
            print(f"  FAIL could not import {stem}")
            traceback.print_exc()
            continue
        failures += run_module(module)
    print(f"\n{'FAILED' if failures else 'OK'}: {failures} failure(s) across "
          f"{len(modules)} module(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
