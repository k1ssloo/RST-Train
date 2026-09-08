"""BUG-25: protocol-specific supervision must match the exact released dataset."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from sft_data_gate import validate_mask_fraction  # noqa: E402


def _release(directory: str) -> tuple[Path, Path]:
    parquet = Path(directory) / "pretokenized_train.parquet"
    parquet.write_bytes(b"a verified native-tool training export")
    manifest = Path(directory) / "release_manifest.json"
    manifest.write_text(json.dumps({
        "format": "rst-sft-release-v1",
        "artifacts": {"data/pretokenized/train.parquet": {
            "sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
            "rows": 976, "total_tokens": 9836475, "trained_tokens": 4919395,
        }},
    }))
    return parquet, manifest


def _refuses(call, message: str) -> None:
    try:
        call()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError("the data gate accepted an invalid contract")


def test_legacy_band_still_refuses_unreviewed_native_masks():
    for trained in (25, 32, 45):
        validate_mask_fraction("unused", rows=1, total_tokens=100, trained_tokens=trained)
    for trained in (24, 46, 50):
        _refuses(lambda: validate_mask_fraction(
            "unused", rows=1, total_tokens=100, trained_tokens=trained), "legacy RST")


def test_verified_seta_fraction_passes_without_changing_masks():
    with tempfile.TemporaryDirectory() as tmp:
        parquet, manifest = _release(tmp)
        assert "50.01%" in validate_mask_fraction(
            parquet, rows=976, total_tokens=9836475, trained_tokens=4919395,
            manifest=manifest,
        )


def test_same_counts_with_different_file_bytes_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        parquet, manifest = _release(tmp)
        parquet.write_bytes(b"different content with the same claimed counts")
        _refuses(lambda: validate_mask_fraction(
            parquet, rows=976, total_tokens=9836475, trained_tokens=4919395,
            manifest=manifest), "SHA-256")


def test_each_stale_count_is_refused_even_with_matching_file_hash():
    with tempfile.TemporaryDirectory() as tmp:
        parquet, manifest = _release(tmp)
        for key in ("rows", "total_tokens", "trained_tokens"):
            counts = dict(rows=976, total_tokens=9836475, trained_tokens=4919395)
            counts[key] += 1
            _refuses(lambda: validate_mask_fraction(
                parquet, **counts, manifest=manifest), key + " mismatch")


def test_invalid_manifest_does_not_fall_back_to_legacy_band():
    with tempfile.TemporaryDirectory() as tmp:
        parquet, manifest = _release(tmp)
        for payload in ("not json", "{}", "[]", '{"format":"wrong"}'):
            manifest.write_text(payload)
            _refuses(lambda: validate_mask_fraction(
                parquet, rows=1, total_tokens=100, trained_tokens=32,
                manifest=manifest), "manifest")


def test_empty_or_fully_supervised_data_is_refused():
    for rows, tokens, trained in ((0, 100, 32), (1, 0, 0), (1, 100, 100), (1, 100, 101)):
        _refuses(lambda: validate_mask_fraction(
            "unused", rows=rows, total_tokens=tokens, trained_tokens=trained),
            "invalid supervised-token totals")


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
