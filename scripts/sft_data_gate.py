"""Check dataset-specific mask statistics against a hash-pinned release (BUG-25).

The launcher checks tensor invariants first. This check then either preserves its
legacy RST fraction band or verifies the exact file and counts from a reviewed
release manifest. A different assistant protocol needs its own measured contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def validate_mask_fraction(
    parquet: str | Path,
    *,
    rows: int,
    total_tokens: int,
    trained_tokens: int,
    manifest: str | Path | None = None,
    legacy_band: bool = True,
) -> str:
    """Return a gate description, or raise ValueError without weakening the checks."""
    if rows <= 0 or not 0 < trained_tokens < total_tokens:
        raise ValueError("empty data or invalid supervised-token totals")
    fraction = trained_tokens / total_tokens
    if not manifest:
        if not legacy_band:
            return f"target-tokenizer export; measured trained fraction {fraction:.2%} (no Qwen-specific band)"
        if not 0.25 <= fraction <= 0.45:
            raise ValueError(
                f"trained fraction {fraction:.2%} is outside the legacy RST 0.25-0.45 "
                "band. For a separately validated corpus, set SFT_DATA_MANIFEST to "
                "its reviewed, revision-pinned release_manifest.json; do not alter masks."
            )
        return f"legacy RST trained fraction {fraction:.2%}"

    try:
        release = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if release["format"] != "rst-sft-release-v1":
            raise ValueError("unsupported SFT release manifest format")
        record = release["artifacts"]["data/pretokenized/train.parquet"]
        for key, actual in (("rows", rows), ("total_tokens", total_tokens),
                            ("trained_tokens", trained_tokens)):
            expected = record[key]
            if type(expected) is not int or expected != actual:
                raise ValueError(f"release {key} mismatch: expected {expected!r}, got {actual}")
        with Path(parquet).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if record["sha256"] != digest:
            raise ValueError("training parquet SHA-256 differs from the reviewed release")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot validate SFT release manifest: {exc}") from exc

    return f"release SHA-256 and exact counts match; trained fraction {fraction:.2%}"
