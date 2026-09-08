"""Validate policy identities and complete ATIF histories before SFT reconstruction."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


def actual_model(requested: str | None, *observed: str | None) -> str:
    names = {name for name in observed if name and name != "unknown"}
    if len(names) > 1 or (requested and names and requested not in names):
        raise ValueError("rollout model identity mismatch; --model-name cannot relabel evidence")
    return next(iter(names)) if names else requested or "unknown"


def content_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_export_bundle(episodes_path: Path) -> dict[str, Any]:
    """Verify a portable v3 export and its frozen split before reading training rows."""
    root = episodes_path.resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 3 or episodes_path.name not in manifest.get("files", {}):
        raise ValueError("version 3 episodes must belong to a complete export bundle")
    for name, expected in manifest["files"].items():
        source = root / name
        if (source.is_symlink() or not source.resolve().is_relative_to(root)
                or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected):
            raise ValueError("export bundle hash or path is invalid")
    splits = json.loads((root / "splits.json").read_text())
    seeds = {row["root_lineage_id"]: row for row in splits["seeds"]}
    if len(seeds) != len(splits["seeds"]):
        raise ValueError("frozen split contains duplicate root lineages")
    groups: dict[str, str] = {}
    for row in seeds.values():
        group = row["overlap_group_id"]
        if groups.setdefault(group, row["split"]) != row["split"]:
            raise ValueError("frozen overlap group leaks across splits")
    policy_path = root / "student-policy.json"
    policy = json.loads(policy_path.read_text()) if policy_path.is_file() else None
    if manifest["purpose"] == "training" and (
            policy is None or "student-policy.json" not in manifest["files"]):
        raise ValueError("production export lacks its checkpoint policy manifest")
    excluded: set[str] = set()
    exclusion_path = root / "seed-exclusions.json"
    if exclusion_path.exists():
        if "seed-exclusions.json" not in manifest["files"]:
            raise ValueError("seed family exclusions are absent from export integrity checks")
        exclusions = json.loads(exclusion_path.read_text())
        if exclusions.get("seed_manifest_sha256") != manifest["files"]["splits.json"]:
            raise ValueError("seed family exclusions refer to another frozen split")
        for row in exclusions["excluded_roots"]:
            source, holdout = seeds.get(row["root_lineage_id"]), seeds.get(row["related_holdout_root"])
            if (source is None or holdout is None or source["split"] != "train"
                    or holdout["split"] != "holdout" or not row.get("reason")
                    or source.get("bundle_digest") != row["bundle_digest"]
                    or holdout.get("bundle_digest") != row["related_holdout_bundle_digest"]):
                raise ValueError("invalid seed family exclusion evidence")
            excluded.add(row["root_lineage_id"])
    return {"manifest": manifest, "seeds": seeds, "policy": policy, "excluded_roots": excluded}


def validate_export_membership(episode: dict[str, Any], bundle: dict[str, Any]) -> None:
    task, manifest = episode["task"], bundle["manifest"]
    seed = bundle["seeds"].get(task["root_lineage_id"])
    if (task["root_lineage_id"] in bundle.get("excluded_roots", ())
            or task.get("seed_exclusions_sha256") != manifest["files"].get("seed-exclusions.json")):
        raise ValueError("episode contradicts frozen seed family exclusions")
    if (seed is None or seed["split"] != "train" or task["split"] != seed["split"]
            or task["overlap_group_id"] != seed["overlap_group_id"]
            or episode["dataset_purpose"] != manifest["purpose"]
            or episode["policy_id"] != manifest["student_policy_id"]
            or episode["rollout"]["model_name"] != manifest["student_model"]):
        raise ValueError("episode contradicts its frozen split or export policy")
    policy = bundle["policy"]
    if policy and (episode.get("student_profile_sha256") != manifest["files"]["student-policy.json"]
                   or episode["inference_digest"] != policy["agent_profile_sha256"]
                   or episode["policy_id"] != policy["policy_id"]
                   or episode["rollout"]["model_name"] != policy["model_name"]):
        raise ValueError("episode contradicts its checkpoint inference profile")


def _link(value: Any) -> str | None:
    if value is None:
        return None
    if (not isinstance(value, str) or not value or PurePosixPath(value).is_absolute()
            or ".." in PurePosixPath(value).parts or "\\" in value):
        raise ValueError("unsafe ATIF continuation reference")
    return value


def validate_segments(
    segments: list[dict[str, Any]], names: list[str], *, strict: bool = False,
) -> None:
    if not segments or len(segments) != len(names) or len(names) != len(set(names)):
        raise ValueError("incomplete or duplicate ATIF continuation segments")
    identity = None
    for index, record in enumerate(segments):
        if not isinstance(record, dict) or not isinstance(record.get("steps"), list):
            raise ValueError("ATIF segment has no steps")
        current = (record.get("session_id"), (record.get("agent") or {}).get("model_name"))
        if ((strict or len(segments) > 1) and not all(current)) or (
                identity is not None and identity != current):
            raise ValueError("ATIF continuation changes model or session")
        identity = current
        expected = names[index + 1] if index + 1 < len(names) else None
        if _link(record.get("continued_trajectory_ref")) != expected:
            raise ValueError("ATIF continuation chain is missing, cyclic or out of order")
        step_ids = [step.get("step_id") for step in record["steps"]]
        if any(step_id is None for step_id in step_ids) or len(step_ids) != len(set(step_ids)):
            raise ValueError("ATIF segment has missing or duplicate step IDs")


def disk_segments(first: Path) -> list[tuple[Path, dict[str, Any]]]:
    directory = first.resolve().parent
    path = first
    result: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    while True:
        resolved = path.resolve()
        if (path.is_symlink() or not resolved.is_relative_to(directory) or resolved in seen
                or not path.is_file()):
            raise ValueError("missing, unsafe or cyclic ATIF continuation")
        seen.add(resolved)
        record = json.loads(path.read_text(encoding="utf-8"))
        result.append((resolved, record))
        link = _link(record.get("continued_trajectory_ref"))
        if link is None:
            break
        path = directory / link
    if any(path.resolve() not in seen for path in directory.glob("trajectory.cont-*.json")):
        raise ValueError("orphan ATIF continuation; explicit links are required")
    validate_segments([record for _, record in result],
                      [str(path.relative_to(directory)) for path, _ in result])
    return result


def episode_segments(episode: dict[str, Any]) -> list[dict[str, Any]]:
    rollout = episode.get("rollout") or {}
    first = rollout.get("trajectory")
    if not isinstance(first, dict):
        raise ValueError("episode has no trajectory")
    segments = rollout.get("trajectory_segments", [first])
    names = rollout.get("trajectory_segment_names", ["trajectory.json"])
    if not isinstance(segments, list) or not isinstance(names, list) or segments[0] != first:
        raise ValueError("episode initial trajectory disagrees with its segments")
    strict = episode.get("schema_version") == 3
    validate_segments(segments, names, strict=strict)
    if strict and rollout.get("trajectory_content_sha256") != [content_digest(s) for s in segments]:
        raise ValueError("embedded ATIF content hashes differ")
    return segments


def evolution_metadata(episode: dict[str, Any], *, allow_smoke: bool) -> dict[str, Any]:
    task, rollout = episode.get("task") or {}, episode.get("rollout") or {}
    ancestry = task.get("ancestry") or []
    if (len(ancestry) < 2 or len(set(ancestry)) != len(ancestry)
            or ancestry[-1] != task.get("task_id")
            or ancestry[-2] != task.get("parent_task_id")
            or ancestry[0] != task.get("root_lineage_id")
            or len(ancestry) - 1 != task.get("generation")
            or not task.get("bundle_digest") or not task.get("semantic_cluster_id")
            or not task.get("overlap_group_id")):
        raise ValueError("version 3 episode has invalid lineage or task provenance")
    purpose = episode.get("dataset_purpose")
    if purpose not in {"training", "smoke"} or (purpose == "smoke" and not allow_smoke):
        raise ValueError("smoke data requires explicit --allow-smoke and is not production data")
    if (rollout.get("role") != "student" or not episode.get("policy_id")
            or not episode.get("inference_digest") or not episode.get("admission_sha256")
            or task.get("split") != "train"):
        raise ValueError("version 3 episode lacks student identity or frozen train membership")
    return {
        "task_id": task["task_id"], "task_group_id": ancestry[0],
        "root_lineage_id": ancestry[0], "bundle_digest": task["bundle_digest"],
        "semantic_cluster_id": task["semantic_cluster_id"],
        "overlap_group_id": task["overlap_group_id"],
        "source_split": task["split"], "dataset_purpose": purpose,
        "policy_id": episode["policy_id"], "inference_digest": episode["inference_digest"],
        "training_subset": task.get("training_subset"),
        "student_calibration": task.get("student_calibration"),
    }


def cap_foundation_records(
    records: list[dict[str, Any]], *, fraction: float, seed: int,
) -> tuple[list[dict[str, Any]], int]:
    """Limit all-success rows within calibrated production SFT, preserving raw evidence."""
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("foundation fraction must be between zero and one")
    production = [row for row in records if row.get("dataset_purpose") == "training"]
    if any(row.get("training_subset") not in {"foundation", "mixed"} for row in production):
        raise ValueError("foundation quota requires measured production calibration metadata")
    foundation = [row for row in production if row["training_subset"] == "foundation"]
    mixed_count = len(production) - len(foundation)
    limit = len(foundation) if fraction == 1 else math.floor(
        fraction * mixed_count / (1 - fraction) + 1e-10)
    ranked = sorted(foundation, key=lambda row: hashlib.sha256(
        f"{seed}/{row.get('task_id')}/{row['trajectory_id']}".encode()).hexdigest())
    selected = {id(row) for row in ranked[:limit]}
    kept = [row for row in records if row.get("dataset_purpose") != "training"
            or row.get("training_subset") != "foundation" or id(row) in selected]
    return kept, len(records) - len(kept)
