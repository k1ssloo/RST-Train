"""Load a sibling script from `scripts/` by path, once, under a legal module name.

`scripts/` is not a package and its files start with a digit, so
`import scripts.15_export_pretokenized` is a syntax error. Five scripts used to carry
their own copy of the `spec_from_file_location` dance to get around that
(03d/03e/03f/17/06b), each with a different module name, a different cache and a
different answer to "register it in sys.modules or not". The loader is how this
repo keeps one definition of `normalize_assistant` and one of `qwen3_5_mask` -- so
the loader itself should not have five definitions either.

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ on the path
    from siblings import load_script

    builder = load_script("03_build_sft_data")
    builder.normalize_assistant(raw)

Every caller gets the SAME module object for a given stem (cached), so state such as
compiled regexes is shared, and `sys.modules["rst_script_<stem>"]` is populated so
`pickle` can find the module when a loaded function is shipped to a worker process.
`tests/_util.py::load_script` is a thin wrapper over this.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent

_cache: dict[str, ModuleType] = {}


def module_name_for(stem: str) -> str:
    """`03_build_sft_data` -> `rst_script_03_build_sft_data` (a legal identifier)."""
    return "rst_script_" + stem.replace("-", "_")


def load_script(stem: str) -> ModuleType:
    """Import `scripts/<stem>.py` by path. Raises ImportError if the file is absent."""
    if stem in _cache:
        return _cache[stem]
    name = module_name_for(stem)
    already = sys.modules.get(name)
    if already is not None:
        _cache[stem] = already
        return already
    path = SCRIPTS_DIR / f"{stem}.py"
    if not path.is_file():
        raise ImportError(f"no such script: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _cache[stem] = module
    return module
