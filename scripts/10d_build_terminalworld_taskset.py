#!/usr/bin/env python3
"""Import TerminalWorld task packages into the RST Harbor/RL task-pool format.

    # The official 20-task evaluation sample (downloads only selected artifacts).
    python scripts/10d_build_terminalworld_taskset.py --source official \
        --source-root data/terminalworld/source --split sample --download \
        --out data/terminalworld/official-sample

    # Repaired third-party packages; train_ready minus official verified tasks.
    python scripts/10d_build_terminalworld_taskset.py --source seeds-clean \
        --source-root data/terminalworld/seeds-clean-source \
        --official-root data/terminalworld/source --download \
        --out data/terminalworld/train-candidates

These releases contain tasks, oracle shell scripts and graders, not model
conversations. Output is rl_tasks.jsonl + tasks/ + manifest.json, never SFT rows.
Collect raw Terminus-2 trajectories before using 03h_build_rollout_sft.py.

BUG-20: the pinned Seeds-Clean release has a stale shard_manifest.jsonl and
stale instruction/dockerfile/solution columns. Authenticate files against the
pinned Hub commit, use archive contents for prompts, and report metadata drift.
The upstream train_ready list is a selection hint, not a local validation result.
Its pass_at_5.csv contains solved_attempts / graded_attempts, not a pass@5
estimator. Those observations are kept as upstream statistics; local difficulty
stays unknown until measured with the target policy on these exact packages.

Source files and canary comments are preserved. In materialized task.toml files,
redundant legacy memory/storage aliases are removed when canonical *_mb fields
exist; the canonical numeric limits are unchanged. --resource-policy preserve
keeps the original bytes for auditing older runners. Root-level resource settings
that modern Harbor ignores are excluded rather than silently applying defaults.
The train split excludes all official verified IDs. That ID check does not prove
independence from the RST corpus, whose synthesis seeds included TerminalWorld.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import tarfile
import tomllib
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from taskpool_common import (  # noqa: E402
    RST_VERIFIER_FILES,
    TRACKED,
    base_image,
    find_verifier_leak,
    sha256_bytes,
    tier_of,
)

SOURCES = {
    "official": ("EuniAI/TerminalWorld", "dda7c099cc076735aef28c03bf8d3624dc0564e1"),
    "seeds-clean": (
        "andylizf/TerminalWorld-Seeds-Clean", "e033a42eaf1b6748607bb563fe9f621b5e1452f9"
    ),
}
TEXT_COLUMNS = {
    "instruction.md": "instruction",
    "task.toml": "task_toml",
    "environment/Dockerfile": "dockerfile",
    "solution/solve.sh": "solution",
}
Files = dict[str, tuple[bytes, int]]


def safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe path: {name!r}")
    return path


def task_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"tw_[0-9]+", value):
        raise ValueError(f"invalid TerminalWorld task ID: {value!r}")
    return value


def verify_payload(payload: bytes, entry: dict[str, Any]) -> str:
    """Verify either an LFS SHA-256 or a Git blob ID from the pinned Hub commit."""
    if len(payload) != entry["size"]:
        raise ValueError(f"size mismatch: {entry['rfilename']}")
    digest = sha256_bytes(payload)
    if entry.get("lfs"):
        matches = digest == entry["lfs"]["sha256"]
    else:
        blob = b"blob " + str(len(payload)).encode() + b"\0" + payload
        matches = hashlib.sha1(blob).hexdigest() == entry["blobId"]
    if not matches:
        raise ValueError(f"hash mismatch: {entry['rfilename']}")
    return digest


class Source:
    def __init__(self, root: Path, name: str, download: bool):
        self.root = root
        self.repo, self.revision = SOURCES[name]
        self.download = download
        self.receipts: dict[str, dict[str, Any]] = {}
        index = root / "hub_info.json"
        if not index.is_file():
            if not download:
                raise ValueError(f"missing {index}; use --download to fetch pinned inputs")
            url = (f"https://huggingface.co/api/datasets/{self.repo}/revision/"
                   f"{self.revision}?blobs=true")
            payload = self.fetch(url)
            root.mkdir(parents=True, exist_ok=True)
            index.write_bytes(payload)
        info = json.loads(index.read_text(encoding="utf-8"))
        if info.get("id") != self.repo or info.get("sha") != self.revision:
            raise ValueError(f"{index}: expected {self.repo}@{self.revision}")
        self.entries = {row["rfilename"]: row for row in info["siblings"]}

    @staticmethod
    def fetch(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()

    def file(self, name: str) -> Path:
        path = self.root / safe_path(name)
        entry = self.entries.get(name)
        if entry is None:
            raise ValueError(f"{name} is absent from {self.repo}@{self.revision}")
        if not path.is_file():
            if not self.download:
                raise ValueError(f"missing {path}; use --download")
            payload = self.fetch(
                f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{name}"
            )
            verify_payload(payload, entry)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        digest = verify_payload(path.read_bytes(), entry)
        self.receipts[name] = {"sha256": digest, "size_bytes": path.stat().st_size}
        return path


def jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        tid = task_id(row["task_id"])
        if tid in indexed:
            raise ValueError(f"duplicate task ID: {tid}")
        if {"messages", "conversations", "steps"} & row.keys():
            raise ValueError("source now contains conversation fields; audit before discarding them")
        indexed[tid] = row
    return indexed


def select_ids(rows: dict[str, dict[str, Any]], *, source: str, split: str,
               verified: set[str], sample: set[str], ready: set[str]
               ) -> tuple[list[str], Counter]:
    stats: Counter = Counter()
    if source == "seeds-clean" and ready - rows.keys():
        raise ValueError("train_ready_ids.txt names tasks absent from the parquet")
    selected = []
    for tid, row in sorted(rows.items()):
        if split == "train":
            if source == "seeds-clean":
                if tid not in ready:
                    stats["not_train_ready"] += 1
                    continue
                if row.get("reward_verdict") != "pass":
                    stats["train_ready_without_passing_oracle"] += 1
                    continue
            if tid in verified:
                stats["official_verified_reserved"] += 1
                continue
        elif split == "verified" and tid not in verified:
            continue
        elif split == "sample" and tid not in sample:
            continue
        selected.append(tid)
    return selected, stats


def read_archive(path: Path, prefixes: dict[str, str]) -> dict[str, Files]:
    """Read regular files without extractall; reject traversal, links and duplicates."""
    roots = {safe_path(prefix).parts: task_id(tid) for prefix, tid in prefixes.items()}
    if len(roots) != len(prefixes) or len(set(roots.values())) != len(roots):
        raise ValueError("duplicate archive task prefix or ID")
    depths = sorted({len(parts) for parts in roots})
    result: dict[str, Files] = {tid: {} for tid in roots.values()}
    with tarfile.open(path) as archive:
        for member in archive:
            parts = safe_path(member.name).parts
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"non-regular archive member: {member.name}")
            for depth in depths:
                tid = roots.get(parts[:depth])
                if tid is None:
                    continue
                rel = PurePosixPath(*parts[depth:]).as_posix()
                if rel == "." or rel in result[tid]:
                    raise ValueError(f"duplicate/invalid archive member: {member.name}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"unreadable archive member: {member.name}")
                result[tid][rel] = (handle.read(), member.mode & 0o777)
                break
    return result


def content_hash(files: Files) -> str:
    """Same sorted (relative path, NUL, bytes, NUL) hash as TerminalWorld-Seeds."""
    digest = hashlib.sha256()
    for name, (payload, _) in sorted(files.items()):
        digest.update(name.encode("utf-8") + b"\0" + payload + b"\0")
    return digest.hexdigest()


def audit_task(row: dict[str, Any], files: Files) -> dict[str, Any]:
    missing = [name for name in TRACKED if not files.get(name, (b"", 0))[0].strip()]
    audit: dict[str, Any] = {
        "task_id": row["task_id"], "missing_or_empty_files": missing,
        "task_content_sha256": content_hash(files), "metadata_text_mismatches": [],
    }
    if missing:
        audit["rejection"] = "missing_required_files"
        return audit
    for name, column in TEXT_COLUMNS.items():
        text = files[name][0].decode("utf-8")
        if column in row and text != row[column]:
            audit["metadata_text_mismatches"].append(column)
    config = tomllib.loads(files["task.toml"][0].decode("utf-8"))
    if not isinstance(config.get("environment", {}), dict):
        raise ValueError(f"{row['task_id']}: environment must be a TOML table")
    # Harbor permits omitted tables, but ignores these old root-level settings.
    # Preserve and report them rather than silently changing the task's limits.
    audit["legacy_root_config_keys"] = sorted(
        {"timeout_sec", "build_timeout_sec", "memory_mb", "storage_mb", "cpus", "allow_internet"}
        & config.keys()
    )
    audit["upstream_content_hash_mismatch"] = (
        row.get("task_content_sha256") is not None
        and row["task_content_sha256"] != audit["task_content_sha256"]
    )
    private = {sha256_bytes(files[f"tests/{name}"][0]) for name in RST_VERIFIER_FILES}
    context = ((name, sha256_bytes(blob)) for name, (blob, _) in files.items()
               if name.startswith("environment/"))
    hit = find_verifier_leak(context, private)
    audit["verifier_leak"] = {"path": hit[0], "kind": hit[1]} if hit else None
    audit["rejection"] = ("verifier_leak" if hit else
                          "legacy_root_config" if audit["legacy_root_config_keys"] else None)
    return audit


def normalize_resources(files: Files, policy: str) -> tuple[Files, list[dict[str, Any]]]:
    """Keep canonical MB limits; remove only their redundant old aliases.

    Harbor rejects memory='2G' together with memory_mb=4096 before it can start
    the task. Prefer the explicit modern field and archive both values. Edit
    individual lines to preserve comments and every other TOML setting.
    """
    if policy == "preserve":
        return files, []
    raw = files["task.toml"][0].decode("utf-8")
    environment = tomllib.loads(raw).get("environment", {})
    repairs = [{"removed_key": alias, "removed_value": environment[alias],
                "kept_key": f"{alias}_mb", "kept_value": environment[f"{alias}_mb"]}
               for alias in ("memory", "storage")
               if alias in environment and f"{alias}_mb" in environment]
    if not repairs:
        return files, []
    keys = {repair["removed_key"] for repair in repairs}
    lines = []
    in_environment = False
    removed = set()
    for line in raw.splitlines(keepends=True):
        if line.lstrip().startswith("["):
            in_environment = bool(re.fullmatch(r"\s*\[environment\]\s*(?:#.*)?", line.strip()))
        match = re.match(r"\s*(memory|storage)\s*=", line) if in_environment else None
        if match and match[1] in keys:
            removed.add(match[1])
        else:
            lines.append(line)
    if removed != keys:
        raise ValueError("cannot safely remove resource aliases from this task.toml syntax")
    text = "".join(lines)
    expected = tomllib.loads(raw)
    for key in keys:
        del expected["environment"][key]
    if tomllib.loads(text) != expected:
        raise ValueError("resource normalization changed unrelated TOML settings")
    result = dict(files)
    result["task.toml"] = (text.encode("utf-8"), files["task.toml"][1])
    return result, repairs


def load_upstream_rates(path: Path) -> dict[str, dict[str, Any]]:
    rates = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tid = task_id(row["task_id"])
            n, k = int(row["graded_attempts"]), int(row["solved_attempts"])
            if tid in rates or n <= 0 or not 0 <= k <= n:
                raise ValueError(f"invalid/duplicate upstream attempt counts: {tid}")
            rates[tid] = {
                "graded_attempts": n, "solved_attempts": k,
                "success_fraction": k / n, "tier": tier_of(k / n) if n >= 4 else "unknown",
            }
    return rates


def make_record(row: dict[str, Any], files: Files, audit: dict[str, Any], *,
                out: Path, source: Source, split: str, verified: set[str],
                upstream_rates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tid = task_id(row["task_id"])
    return {
        # Harbor reads this same instruction.md; never use a stale parquet prompt.
        "prompt": files["instruction.md"][0].decode("utf-8"),
        "label": tid,
        "metadata": {
            "task_id": tid, "task_group_id": tid,
            "task_dir": str((out / "tasks" / tid).resolve()),
            "task_content_sha256": content_hash(files),
            "source_task_content_sha256": audit["task_content_sha256"],
            "source_dataset": source.repo, "source_revision": source.revision,
            "original_dataset": SOURCES["official"][0], "split": split,
            "official_verified": tid in verified,
            "terminal_domain": row.get("terminal_domain"),
            "base_image": base_image(files["environment/Dockerfile"][0].decode("utf-8")),
            "tier": "unknown", "empirical_pass_rate": None, "n_reference_trials": 0,
            "upstream_reward_verdict": row.get("reward_verdict"),
            "upstream_solver_statistics": upstream_rates.get(tid),
            "upstream_statistics_revision_matched": False,
            "metadata_text_mismatches": audit["metadata_text_mismatches"],
            "resource_normalizations": audit.get("resource_normalizations", []),
            "local_validation": "static_only",
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        raise ValueError(f"output must be a new or empty directory: {args.out}")
    if args.max_tasks < 0:
        raise ValueError("--max-tasks must be nonnegative")
    source = Source(args.source_root, args.source, args.download)
    official = source if args.source == "official" else Source(
        args.official_root or args.source_root.parent / "source", "official", args.download
    )
    full = index_rows(jsonl_gz(official.file("data/full.jsonl.gz")))
    verified = set(index_rows(jsonl_gz(official.file("data/verified.jsonl.gz"))))
    sample = set(index_rows(jsonl_gz(official.file("data/sample.jsonl.gz"))))
    if not sample <= verified <= full.keys():
        raise ValueError("official sample/verified/full subsets are inconsistent")
    ready: set[str] = set()
    rates: dict[str, dict[str, Any]] = {}
    shard_claims: list[dict[str, Any]] = []
    if args.source == "official":
        rows = full
    else:
        import pandas as pd

        rows = index_rows(pd.read_parquet(source.file("metadata/tasks.parquet"))
                          .to_dict(orient="records"))
        if not rows.keys() <= full.keys():
            raise ValueError("seed packages contain IDs absent from the official release")
        ready = {task_id(tid) for tid in source.file("metadata/train_ready_ids.txt")
                 .read_text(encoding="utf-8").split()}
        rates = load_upstream_rates(source.file("metadata/pass_at_5.csv"))
        shard_claims = [json.loads(line) for line in
                        source.file("metadata/shard_manifest.jsonl").read_text().splitlines()
                        if line.strip()]
    selected, selection_stats = select_ids(
        rows, source=args.source, split=args.split, verified=verified, sample=sample, ready=ready
    )
    candidates_before_cap = len(selected)
    if args.max_tasks:
        selected = selected[:args.max_tasks]
    if not selected:
        raise ValueError("no tasks selected")
    print(f"[select] {len(rows)} source tasks -> {len(selected)} {args.split} candidates",
          flush=True)

    # Seeds-Clean has small shared shards: audit every task, including excluded ones.
    # Official artifacts are individual downloads, so audit only selected packages.
    bodies: dict[str, Files] = {}
    if args.source == "official":
        for tid in selected:
            archive = rows[tid]["artifact_path"]
            if archive != f"artifacts/{tid}.tar.gz":
                raise ValueError(f"unexpected artifact path for {tid}: {archive}")
            bodies.update(read_archive(source.file(archive), {tid: tid}))
    else:
        shards: dict[str, dict[str, str]] = {}
        for tid, row in rows.items():
            if row["member_prefix"] != f"tasks/{tid}":
                raise ValueError(f"unexpected member prefix for {tid}")
            shards.setdefault(row["shard"], {})[row["member_prefix"]] = tid
        for shard, prefixes in sorted(shards.items()):
            bodies.update(read_archive(source.file(shard), prefixes))

    shard_drift = []
    for claim in shard_claims:
        receipt = source.receipts.get(claim["shard"])
        if receipt is None or any(claim[key] != receipt[key] for key in ("sha256", "size_bytes")):
            shard_drift.append({"shard": claim["shard"], "declared": claim, "actual": receipt})
    audits = {tid: audit_task(rows[tid], files) for tid, files in sorted(bodies.items())}
    kept = [tid for tid in selected if not audits[tid]["rejection"]]
    if not kept:
        raise ValueError("no tasks survived package and verifier-leak checks")
    emitted = {}
    for tid in kept:
        emitted[tid], repairs = normalize_resources(bodies[tid], args.resource_policy)
        audits[tid]["resource_normalizations"] = repairs
    records = [make_record(rows[tid], emitted[tid], audits[tid], out=args.out, source=source,
                           split=args.split, verified=verified, upstream_rates=rates)
               for tid in kept]

    # Write only after all reads/checks succeeded, into a fresh directory so a
    # narrower rerun cannot leave stale tasks that Harbor would discover by glob.
    args.out.mkdir(parents=True, exist_ok=True)
    for tid in kept:
        for name, (payload, mode) in emitted[tid].items():
            dest = args.out / "tasks" / tid / safe_path(name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
            dest.chmod(mode)
    with (args.out / "rl_tasks.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(args.out / "source_audit.json", list(audits.values()))
    mismatch_counts = Counter(col for audit in audits.values()
                              for col in audit["metadata_text_mismatches"])
    manifest = {
        "source_dataset": source.repo, "source_revision": source.revision,
        "original_dataset": official.repo, "original_revision": official.revision,
        "source_rows": len(rows), "source_train_ready_ids": len(ready),
        "split": args.split, "selection_counters": dict(selection_stats),
        "candidates_before_cap": candidates_before_cap, "max_tasks": args.max_tasks,
        "candidates_checked": len(selected), "tasks_selected": len(kept),
        "materialized_task_dirs": len(kept), "source_tasks_audited": len(audits),
        "official_verified_total": len(verified),
        "official_verified_overlap": len(set(kept) & verified),
        "drop_counters": dict(Counter(audits[tid]["rejection"] for tid in selected
                                      if audits[tid]["rejection"])),
        "metadata_text_mismatches": dict(mismatch_counts),
        "source_content_hash_mismatches": sum(audit.get("upstream_content_hash_mismatch", False)
                                              for audit in audits.values()),
        "selected_tasks_with_legacy_root_config": [
            tid for tid in kept if audits[tid].get("legacy_root_config_keys")
        ],
        "upstream_shard_manifest_mismatches": shard_drift,
        "source_files": source.receipts, "official_files": official.receipts,
        "leak_guard_ran": True,
        "verifier_leaks_excluded": sum(audits[tid]["rejection"] == "verifier_leak"
                                       for tid in selected),
        "base_images": dict(Counter(r["metadata"]["base_image"] for r in records)),
        "terminal_domains": dict(Counter(rows[tid]["terminal_domain"] for tid in kept)),
        "tier": "unknown", "local_oracle_trials": 0, "local_model_trials": 0,
        "upstream_statistics_note": "Historical solved/graded fractions; policy and task "
                                    "revision are not matched to local rollouts.",
        "upstream_solver_tiers": dict(Counter(rates.get(tid, {}).get("tier", "unknown")
                                              for tid in kept)),
        "sft_rows_written": 0,
        "not_sft_data": "Task packages and oracle scripts contain no model conversations. "
                        "Collect raw Terminus-2 trajectories before building SFT or DPO data.",
        "source_files_modified": False, "canary_comments_preserved": True,
        "resource_policy": args.resource_policy,
        "task_configs_normalized": sum(bool(audits[tid]["resource_normalizations"]) for tid in kept),
        "resource_normalization_note": "Removed old memory/storage aliases when canonical "
                                       "*_mb fields exist; kept canonical numeric limits.",
        "validation_scope": "Static files, TOML, pinned Hub hashes, verifier-copy guard. "
                            "No environment builds or oracle/model executions in this builder.",
        "benchmark_note": "Train excludes official verified IDs. This does not establish "
                          "independence from RST descendants or other semantically related tasks.",
        "prompt_data": str(args.out / "rl_tasks.jsonl"),
        "task_root": str((args.out / "tasks").resolve()),
    }
    write_json(args.out / "manifest.json", manifest)
    print(f"[write] {len(kept)} tasks; verified overlap={manifest['official_verified_overlap']}; "
          f"drops={manifest['drop_counters']}; SFT rows=0", flush=True)
    print(f"[audit] metadata drift={dict(mismatch_counts)}; "
          f"stale shard manifests={len(shard_drift)}; {args.out / 'manifest.json'}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=tuple(SOURCES), default="seeds-clean")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path,
                        help="official cache for the verified IDs (default: sibling source/)")
    parser.add_argument("--split", choices=("train", "verified", "sample", "full"), default="train")
    parser.add_argument("--max-tasks", type=int, default=0, help="sorted ID cap; 0 = all selected")
    parser.add_argument("--download", action="store_true", help="fetch missing pinned inputs")
    parser.add_argument("--resource-policy", choices=("canonical-mb", "preserve"),
                        default="canonical-mb", help="resolve duplicate Harbor resource fields")
    parser.add_argument("--out", type=Path, required=True, help="new or empty output directory")
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, OSError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f"[terminalworld] {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
