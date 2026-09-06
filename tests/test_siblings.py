"""`scripts/siblings.py` -- one loader for the digit-prefixed scripts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, load_script  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))

import siblings  # noqa: E402


def test_the_test_helper_and_the_scripts_get_the_same_module_object():
    via_util = load_script("03_build_sft_data")
    via_scripts = siblings.load_script("03_build_sft_data")
    assert via_util is via_scripts
    convert = load_script("03d_build_openthoughts_sft")
    assert convert.load_builder() is via_util, "the converter shares the builder, not a copy"


def test_a_missing_script_is_an_import_error_with_the_path_in_it():
    try:
        siblings.load_script("99_does_not_exist")
    except ImportError as exc:
        assert "99_does_not_exist.py" in str(exc)
    else:
        raise AssertionError("expected ImportError")


def test_loaded_scripts_are_registered_so_worker_processes_can_unpickle_them():
    module = siblings.load_script("15_export_pretokenized")
    assert sys.modules[siblings.module_name_for("15_export_pretokenized")] is module


def test_no_script_carries_its_own_spec_from_file_location_dance():
    for path in sorted((ROOT / "scripts").glob("*.py")):
        if path.name == "siblings.py":
            continue
        assert "spec_from_file_location" not in path.read_text(encoding="utf-8"), path.name


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
