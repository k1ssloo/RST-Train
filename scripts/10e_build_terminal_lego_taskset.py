#!/usr/bin/env python3
"""Build a pinned Terminal-Lego task pool using PrimeIntellect's exclusion IDs.

    python scripts/10e_build_terminal_lego_taskset.py --download \
        --source-root data/terminal-lego/source --out data/terminal-lego/train-v1

The source is a shallow Git checkout, not HF repo-info.siblings (that listing
is incomplete for this large repository). Verify each selected Git blob and
resolve LFS assets by SHA-256. Preserve the original task files and Dockerfiles;
the third-party mirror's platform-specific images are not needed. BUG-22.

Outputs are rl_tasks.jsonl, tasks/, source_audit.jsonl and manifest.json. These
are static training candidates, not model trajectories or local oracle passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from taskpool_common import base_image, find_verifier_leak, sha256_bytes

SOURCE_REPO = "Lego-X/Terminal-Lego-15k"
SOURCE_REVISION = "9c197f1c2e87b64cc316b1a5bfcef57b584929f0"
EXCLUSION_REPO = "PrimeIntellect/Terminal-Lego-15k"
EXCLUSION_REVISION = "92e6b5f577610cec9b040250a94ce66cfce24839"
EXCLUSION_FILE = "prime-data-excluded-tasks.jsonl"
EXCLUSION_SHA256 = "c8bc7e64d77f23f242f8c754574b00fac65299d5585bc20e562feba2b6e4e64d"
REQUIRED = ("instruction.md", "task.toml", "environment/Dockerfile", "solution/solve.sh",
            "tests/test.sh", "tests/test_outputs.py")
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
Files = dict[str, tuple[bytes, int]]


def task_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"task_[0-9]{5}", value):
        raise ValueError(f"invalid Terminal-Lego task ID: {value!r}")
    return value


def safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or path.as_posix() != name or ".git" in path.parts):
        raise ValueError(f"unsafe source path: {name!r}")
    return path


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def download_source(root: Path) -> None:
    """Fetch the fixed commit only into a new directory; never reset user changes."""
    if (root / ".git").is_dir():
        try:
            head = git(root, "rev-parse", "HEAD").decode().strip()
        except subprocess.CalledProcessError:
            head = None
        if head == SOURCE_REVISION:
            return
        if head is not None or any(p.name != ".git" for p in root.iterdir()):
            raise ValueError("source checkout is not the pinned revision; use a fresh source-root")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("source-root must be empty or an existing pinned Git checkout")
        subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "fetch", "--quiet", "--depth=1",
                    f"https://huggingface.co/datasets/{SOURCE_REPO}", SOURCE_REVISION], check=True)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    subprocess.run(["git", "-C", str(root), "-c", "filter.lfs.smudge=",
                    "-c", "filter.lfs.required=false", "checkout", "--quiet", "--detach",
                    SOURCE_REVISION], env=env, check=True)


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def parse_exclusions(payload: bytes) -> dict[str, str]:
    result = {}
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tid = task_id(row["task_id"])
        if tid in result or not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError(f"duplicate/invalid exclusion: {tid}")
        result[tid] = row["reason"]
    return result


def load_exclusions(path: Path, download: bool, expected_sha: str) -> dict[str, str]:
    if not path.is_file():
        if not download:
            raise ValueError(f"missing {path}; use --download")
        payload = fetch(f"https://huggingface.co/datasets/{EXCLUSION_REPO}/resolve/"
                        f"{EXCLUSION_REVISION}/{EXCLUSION_FILE}")
        if sha256_bytes(payload) != expected_sha:
            raise ValueError("exclusion download SHA-256 mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    payload = path.read_bytes()
    if sha256_bytes(payload) != expected_sha:
        raise ValueError("exclusion file SHA-256 mismatch")
    return parse_exclusions(payload)


def git_index(root: Path, revision: str) -> dict[str, dict[str, tuple[str, int]]]:
    """Read the complete commit tree, retaining Git mode and blob identity."""
    if git(root, "rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError(f"source-root must be at {revision}")
    groups: dict[str, dict[str, tuple[str, int]]] = {}
    for item in git(root, "ls-tree", "-rz", "--full-tree", revision).split(b"\0"):
        if not item:
            continue
        header, path_bytes = item.split(b"\t", 1)
        name = path_bytes.decode("utf-8")
        parts = safe_path(name).parts
        if not parts[0].startswith("task_"):
            continue
        tid = task_id(parts[0])
        if len(parts) < 2:
            raise ValueError(f"task root is not a directory: {name}")
        mode, kind, oid = header.decode().split()
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"non-regular tracked task file: {name} ({mode})")
        rel = PurePosixPath(*parts[1:]).as_posix()
        groups.setdefault(tid, {})[rel] = (oid, int(mode, 8) & 0o777)
    if not groups:
        raise ValueError("no task directories in the pinned tree")
    return groups


def lfs_pointer(payload: bytes) -> tuple[str, int] | None:
    if not payload.startswith(LFS_PREFIX):
        return None
    match = re.fullmatch(LFS_PREFIX + rb"oid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?", payload)
    if match is None:
        raise ValueError("unsupported/malformed LFS pointer")
    return match[1].decode(), int(match[2])


def resolve_lfs(payload: bytes, *, name: str, cache: Path, revision: str,
                download: bool) -> tuple[bytes, str | None]:
    pointer = lfs_pointer(payload)
    if pointer is None:
        return payload, None
    digest, size = pointer
    path = cache / digest
    if not path.is_file():
        if not download:
            raise ValueError(f"missing LFS asset {name}; use --download")
        body = fetch(f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/{revision}/"
                     + urllib.parse.quote(name, safe="/"))
        if len(body) != size or sha256_bytes(body) != digest:
            raise ValueError(f"LFS download hash/size mismatch: {name}")
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    body = path.read_bytes()
    if len(body) != size or sha256_bytes(body) != digest:
        raise ValueError(f"LFS cache hash/size mismatch: {name}")
    return body, digest


def read_task(root: Path, tid: str, entries: dict[str, tuple[str, int]], *,
              cache: Path, revision: str, download: bool) -> tuple[Files, list[dict]]:
    files: Files = {}
    assets = []
    for rel, (oid, mode) in sorted(entries.items()):
        name = f"{tid}/{rel}"
        path = root / safe_path(name)
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"symlink or escaped source file: {name}")
        payload = path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if actual != oid:
            raise ValueError(f"source Git blob mismatch: {name}")
        body, lfs_sha = resolve_lfs(payload, name=name, cache=cache,
                                    revision=revision, download=download)
        files[rel] = body, mode
        if lfs_sha:
            assets.append({"path": rel, "sha256": lfs_sha, "size_bytes": len(body)})
    return files, assets


def content_hash(files: Files) -> str:
    digest = hashlib.sha256()
    for name, (payload, mode) in sorted(files.items()):
        digest.update(name.encode() + b"\0" + str(mode).encode() + b"\0" + payload + b"\0")
    return digest.hexdigest()


def group_id(tid: str, url: str) -> str:
    match = re.match(r"https://stackoverflow\.com/questions/([0-9]+)(?:/|$)", url)
    return f"terminal_lego_so_{match[1]}" if match else f"terminal_lego_{tid}"


def audit_task(tid: str, files: Files) -> dict[str, Any]:
    audit: dict[str, Any] = {"source_task_id": tid, "content_sha256": content_hash(files),
                              "rejection": None}
    missing = [name for name in REQUIRED if not files.get(name, (b"", 0))[0].strip()]
    if missing:
        audit.update(rejection="missing_required_files", missing=missing)
        return audit
    try:
        config = tomllib.loads(files["task.toml"][0].decode("utf-8"))
        for name in REQUIRED:
            files[name][0].decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        audit.update(rejection="invalid_task_text_or_toml", detail=str(exc))
        return audit
    metadata = config.get("metadata", {})
    if not isinstance(metadata, dict) or not isinstance(metadata.get("source_url"), str):
        audit["rejection"] = "missing_source_url"
        return audit
    if any(not isinstance(config.get(section), dict)
           for section in ("environment", "agent", "verifier")):
        audit["rejection"] = "invalid_config_sections"
        return audit
    audit.update(source_url=metadata["source_url"], category=metadata.get("category"),
                 difficulty=metadata.get("difficulty"))
    # The pinned release uses this COPY instruction even when Git contains no
    # task_file directory. Harbor parses such tasks, but the image cannot build.
    # Do not infer missing fixtures or synthesize an empty source directory.
    dockerfile = files["environment/Dockerfile"][0].decode("utf-8")
    if (re.search(r"(?im)^\s*COPY\s+\./task_file\s+/app/task_file\s*$", dockerfile)
            and not any(name.startswith("environment/task_file/") for name in files)):
        audit.update(rejection="missing_build_context", missing=["environment/task_file/"],
                     detail="COPY source has no tracked files in the pinned Git tree")
        return audit
    # Tests and the reference solution must stay outside the agent's build context.
    private = {sha256_bytes(blob) for name, (blob, _) in files.items()
               if (name.startswith("tests/") or name.startswith("solution/")) and blob.strip()}
    context = ((name, sha256_bytes(blob)) for name, (blob, _) in files.items()
               if name.startswith("environment/"))
    hit = find_verifier_leak(context, private, ("test.sh", "test_outputs.py", "solve.sh"))
    if hit:
        audit.update(rejection="verifier_or_solution_leak", leak={"path": hit[0], "kind": hit[1]})
    return audit


def build(args: argparse.Namespace, *, revision: str = SOURCE_REVISION,
          exclusion_sha: str = EXCLUSION_SHA256) -> dict[str, Any]:
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        raise ValueError(f"output must be new or empty: {args.out}")
    if args.max_tasks < 0:
        raise ValueError("max-tasks must be nonnegative")
    if args.download:
        download_source(args.source_root)
    excluded = load_exclusions(args.exclusions, args.download, exclusion_sha)
    index = git_index(args.source_root, revision)
    if excluded.keys() - index.keys():
        raise ValueError("exclusion IDs are absent from the pinned source tree")
    selected = sorted(index.keys() - excluded.keys())
    total_candidates = len(selected)
    if args.task_ids:
        wanted = {task_id(tid) for tid in args.task_ids}
        if wanted - set(selected):
            raise ValueError("requested task IDs are absent or excluded")
        selected = sorted(wanted)
    if args.max_tasks:
        selected = selected[:args.max_tasks]
    print(f"[select] {len(index)} source - {len(excluded)} exclusions = "
          f"{total_candidates} candidates; auditing {len(selected)}", flush=True)
    audits = []
    packages: dict[str, Files] = {}
    records = []
    lfs_objects: dict[str, int] = {}
    blob_count = 0
    for number, tid in enumerate(selected, 1):
        files, assets = read_task(args.source_root, tid, index[tid], cache=args.lfs_cache,
                                  revision=revision, download=args.download)
        blob_count += len(files)
        for asset in assets:
            lfs_objects[asset["sha256"]] = asset["size_bytes"]
        audit = audit_task(tid, files)
        audit["lfs_assets"] = assets
        audits.append(audit)
        if audit["rejection"]:
            continue
        label = f"terminal_lego_{tid}"
        packages[label] = files
        records.append({
            "prompt": files["instruction.md"][0].decode("utf-8"), "label": label,
            "metadata": {
                "task_id": label, "task_group_id": group_id(tid, audit["source_url"]),
                "task_dir": str((args.out / "tasks" / label).resolve()),
                "source_task_id": tid, "source_dataset": SOURCE_REPO,
                "source_revision": revision, "source_url": audit["source_url"],
                "task_content_sha256": audit["content_sha256"],
                "base_image": base_image(files["environment/Dockerfile"][0].decode()),
                "category": audit["category"], "upstream_difficulty": audit["difficulty"],
                "tier": "unknown", "empirical_pass_rate": None, "n_reference_trials": 0,
                "upstream_exclusion_filter": f"{EXCLUSION_REPO}@{EXCLUSION_REVISION}",
                "upstream_validation_revision_matched": False,
                "local_validation": "static_only", "lfs_assets_resolved": len(assets),
            },
        })
        if number % 500 == 0 or number == len(selected):
            print(f"[audit] {number}/{len(selected)} kept={len(records)} "
                  f"lfs_objects={len(lfs_objects)}", flush=True)
    if not records:
        raise ValueError("no tasks survived the checks")

    args.out.mkdir(parents=True, exist_ok=True)
    for label, files in packages.items():
        for name, (payload, mode) in files.items():
            dest = args.out / "tasks" / label / safe_path(name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
            dest.chmod(mode)
    for name, rows in (("rl_tasks.jsonl", records), ("source_audit.jsonl", audits)):
        (args.out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                     encoding="utf-8")
    manifest = {
        "source_dataset": SOURCE_REPO, "source_revision": revision,
        "source_root": str(args.source_root.resolve()), "source_tasks": len(index),
        "source_listing": "complete pinned Git tree; not repo-info.siblings",
        "exclusion_source": {"dataset": EXCLUSION_REPO, "revision": EXCLUSION_REVISION,
                             "file": EXCLUSION_FILE, "sha256": exclusion_sha},
        "upstream_excluded": len(excluded), "exclusion_reasons": dict(Counter(excluded.values())),
        "candidates_before_cap": total_candidates, "candidates_checked": len(selected),
        "requested_task_ids": args.task_ids, "max_tasks": args.max_tasks,
        "tasks_selected": len(records), "materialized_task_dirs": len(packages),
        "task_groups": len({r["metadata"]["task_group_id"] for r in records}),
        "drop_counters": dict(Counter(a["rejection"] for a in audits if a["rejection"])),
        "verified_git_blobs": blob_count, "lfs_objects": lfs_objects,
        "lfs_references": sum(len(a["lfs_assets"]) for a in audits),
        "lfs_bytes": sum(lfs_objects.values()),
        "categories": dict(Counter(r["metadata"]["category"] for r in records)),
        "base_images": dict(Counter(r["metadata"]["base_image"] for r in records)),
        "source_files_modified": False, "canary_comments_preserved": True,
        "config_repairs": 0, "tier": "unknown", "local_oracle_trials": 0,
        "local_model_trials": 0, "sft_rows_written": 0,
        "validation_scope": "Git/LFS integrity, required files, TOML, Terminal-Lego COPY source "
                            "and verifier/solution leak checks. Runtime controls are recorded "
                            "separately.",
        "upstream_validation_note": "Third-party exclusions are historical hints; neither "
                                    "policy performance nor local runtime validity is inferred.",
        "benchmark_overlap_checked": False,
        "prompt_data": str(args.out / "rl_tasks.jsonl"),
        "task_root": str((args.out / "tasks").resolve()),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(f"[write] {len(records)} static candidates; drops={manifest['drop_counters']}; "
          f"{args.out / 'manifest.json'}", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/terminal-lego/source"))
    parser.add_argument("--exclusions", type=Path,
                        default=Path("data/terminal-lego") / EXCLUSION_FILE)
    parser.add_argument("--lfs-cache", type=Path, default=Path("data/terminal-lego/lfs"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--task-ids", nargs="+", default=None)
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
