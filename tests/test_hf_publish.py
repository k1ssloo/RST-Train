"""`scripts/hf_publish.py` -- the one upload path behind the six `13*` scripts.

No network: `publish` is exercised in `--dry-run` shape, and the six scripts are
checked at the source level for using it rather than a private copy.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))

import hf_publish  # noqa: E402


def _src(tmp, manifest):
    (Path(tmp) / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (Path(tmp) / "data.parquet").write_bytes(b"x" * 10)
    return Path(tmp)


def test_check_card_counts_every_disagreement_and_follows_nested_keys():
    with tempfile.TemporaryDirectory() as tmp:
        src = _src(tmp, {"rows": 5, "token_stats": {"total": 100}})
        claims = [("rows", "manifest.json", ("rows",), 5),
                  ("total", "manifest.json", ("token_stats", "total"), 100),
                  ("wrong", "manifest.json", ("rows",), 6)]
        assert hf_publish.check_card(src, claims) == 1


def test_a_missing_input_is_refused_before_anything_is_created():
    with tempfile.TemporaryDirectory() as tmp:
        src = _src(tmp, {})
        try:
            hf_publish.publish(repo="x/y", files=[(src / "absent.parquet", "a")], card="c",
                               private=True, dry_run=False)
        except SystemExit as exc:
            assert "missing input" in str(exc)
        else:
            raise AssertionError("a missing file must stop the publish")


def test_dry_run_prints_the_plan_and_returns_without_a_token(capsys=None):
    with tempfile.TemporaryDirectory() as tmp:
        src = _src(tmp, {})
        rc = hf_publish.publish(repo="x/y", files=[(src / "data.parquet", "data/a.parquet")],
                                card="c", private=False, dry_run=True, kind="SFT",
                                notes=("bodies not uploaded",))
        assert rc == 0, "dry-run must not need HF_TOKEN"


def test_the_six_upload_scripts_use_the_shared_path_and_carry_no_private_copy():
    for stem in ("13_upload_hf", "13c_upload_openthoughts_hf", "13d_upload_tmax_hf",
                 "13e_upload_termigen_hf", "13f_upload_nemotron_hf", "13g_upload_swegym_hf"):
        source = (ROOT / "scripts" / f"{stem}.py").read_text(encoding="utf-8")
        assert "from hf_publish import" in source, stem
        assert "def check_card" not in source, f"{stem} grew its own check_card again"
        assert "HfApi" not in source, f"{stem} talks to the Hub directly again"


def test_every_card_gated_script_declares_claims_against_a_named_manifest():
    for stem in ("13c_upload_openthoughts_hf", "13d_upload_tmax_hf", "13e_upload_termigen_hf",
                 "13f_upload_nemotron_hf", "13g_upload_swegym_hf"):
        source = (ROOT / "scripts" / f"{stem}.py").read_text(encoding="utf-8")
        assert "CLAIMS: list[tuple[str, str, tuple[str, ...], object]]" in source, stem
        assert "check_card(args.src, CLAIMS)" in source, stem


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
