"""Test helpers. Deliberately depends on nothing but the standard library.

Two things every test here needs:

`load_script("15_export_pretokenized")`
    `scripts/` is not a package and its modules start with a digit, so
    `import scripts.15_export_pretokenized` is a syntax error. This loads the file
    by path instead. The repo root is put on `sys.path` first, because the scripts
    themselves do `from rst_common... import` after inserting it.

`need("torch")`
    A skip that works under pytest *and* under `python tests/run_tests.py`. The
    heavy tests (real checkpoints, real tokenizers) must not turn into failures on
    a laptop that has neither -- they must say "skipped" out loud, so a green run
    is never mistaken for a complete one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent


class Skipped(Exception):
    """Raised when a test cannot run here. Caught by run_tests.py."""


def skip(reason: str):
    """Skip the current test, under whichever runner is executing it.

    The runner is detected by `"pytest" in sys.modules`, not by whether pytest is
    installable: pytest may well be installed in an environment where the suite is
    being driven by `run_tests.py`, and then `pytest.skip()` raises an error about
    being called outside a test instead of skipping.
    """
    if "pytest" in sys.modules:
        sys.modules["pytest"].skip(reason, allow_module_level=True)
    raise Skipped(reason)


def need(module_name: str) -> ModuleType:
    """Import a module or skip the test."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return skip(f"{module_name} is not installed in this environment")


_cache: dict[str, ModuleType] = {}


def load_script(stem: str) -> ModuleType:
    """Import `scripts/<stem>.py` under a legal module name."""
    if stem in _cache:
        return _cache[stem]
    path = ROOT / "scripts" / f"{stem}.py"
    if not path.is_file():
        raise AssertionError(f"missing script: {path}")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    name = "rst_script_" + stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _cache[stem] = module
    return module


def load_repo_module(dotted: str) -> ModuleType:
    """Import a real package module from the repo root (`rst_common.harbor`, ...)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module(dotted)
