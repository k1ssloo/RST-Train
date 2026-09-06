#!/usr/bin/env python3
"""The one upload path every `13*_upload_*_hf.py` script uses.

Six publish scripts used to carry byte-identical copies of `check_card` and of the
create-repo / upload-files / upload-README sequence; what differed between them was
the dataset CARD, the CLAIMS the card makes, and the file list. Those stay in each
script -- they ARE the script. The machinery lives here once.

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hf_publish import check_card, publish

`check_card` is the gate that keeps a card honest: every number the prose quotes is
listed as a CLAIM `(label, manifest_file, key_path, expected)` and compared against
the manifest actually on disk. A card that disagrees with its manifest is refused,
because the alternative is a public page stating a count that the data does not
have.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# (label, manifest filename, key path into that manifest, value the card states)
Claim = tuple[str, str, tuple[str, ...], object]


def check_card(src_dir: Path, claims: Sequence[Claim]) -> int:
    """Return how many card claims disagree with the manifests under `src_dir`."""
    cache: dict[str, dict] = {}
    bad = 0
    for label, filename, keys, expected in claims:
        if filename not in cache:
            cache[filename] = json.loads((src_dir / filename).read_text(encoding="utf-8"))
        node: Any = cache[filename]
        for key in keys:
            assert isinstance(node, dict), f"{label}: {keys} is not a path into {filename}"
            node = node[key]
        if node != expected:
            print(f"  MISMATCH {label}: card says {expected}, {filename} says {node}")
            bad += 1
    print(f"[check] {len(claims) - bad}/{len(claims)} card claims match the manifests")
    return bad


def publish(
    *,
    repo: str,
    files: Sequence[tuple[Path, str]],
    card: str,
    private: bool,
    dry_run: bool,
    kind: str = "dataset",
    notes: Sequence[str] = (),
    missing_hint: str = "",
) -> int:
    """Print the plan, then (unless `dry_run`) create the repo and upload everything.

    Every source path is checked for existence BEFORE anything is created on the Hub,
    so a typo cannot leave a half-populated public repo behind. `HF_TOKEN` is only
    required once the plan is approved for upload.
    """
    for src, _dst in files:
        if not src.is_file():
            sys.exit(f"missing input: {src}" + (f". {missing_hint}" if missing_hint else ""))
    visibility = "private" if private else "public"
    total = sum(src.stat().st_size for src, _ in files)
    print(f"{kind} -> {repo}  ({visibility.upper() if private else visibility}, "
          f"{total / 2**20:.1f} MB in {len(files)} files)")
    for src, dst in files:
        print(f"   {src}  ->  {dst}  ({src.stat().st_size / 2**20:.1f} MB)")
    for note in notes:
        print(f"   ({note})")
    if dry_run:
        print("\n--dry-run: nothing uploaded")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN")
    from huggingface_hub import HfApi  # noqa: PLC0415 - only needed for a real upload

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    print(f"\n[{repo}] created/exists ({visibility})")
    for src, dst in files:
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                        repo_id=repo, repo_type="dataset")
        print(f"  uploaded {dst}")
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    print("  uploaded README.md")
    print(f"\nhttps://huggingface.co/datasets/{repo}" + ("  (private)" if private else ""))
    return 0
