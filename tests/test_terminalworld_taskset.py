"""TerminalWorld import integrity and BUG-20: stale metadata must not become prompts.

Offline fixtures exercise the same full build path as downloaded packages. No
containers or model calls: a successful import must never claim either ran.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import load_script, need  # noqa: E402

builder = load_script("10d_build_terminalworld_taskset")


def package(tid: str) -> dict[str, tuple[bytes, int]]:
    return {
        "instruction.md": (f"<!-- harbor-canary fixture -->\nWrite {tid} to /app/out.\n".encode(), 0o644),
        "task.toml": (b'version = "1.0"\n[environment]\ncpus = 1\nmemory_mb = 1024\n', 0o644),
        "environment/Dockerfile": (b"FROM ubuntu:22.04\nWORKDIR /app\n", 0o644),
        "solution/solve.sh": (f"#!/bin/sh\nprintf {tid} > /app/out\n".encode(), 0o755),
        "tests/test.sh": (b"#!/bin/sh\npytest /tests/test_state.py\n", 0o755),
        "tests/test_state.py": (b"def test_output():\n    assert open('/app/out').read()\n", 0o644),
    }


def tar_payload(tasks: dict[str, dict], prefix: str = "") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for tid, files in tasks.items():
            for name, (blob, mode) in files.items():
                member = tarfile.TarInfo(f"{prefix}{tid}/{name}")
                member.mode, member.size = mode, len(blob)
                tar.addfile(member, io.BytesIO(blob))
    return buffer.getvalue()


def cached_source(root: Path, source: str, payloads: dict[str, bytes]) -> None:
    entries = []
    for name, payload in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append({"rfilename": name, "size": len(payload),
                        "blobId": hashlib.sha1(b"blob " + str(len(payload)).encode()
                                               + b"\0" + payload).hexdigest()})
    repo, revision = builder.SOURCES[source]
    (root / "hub_info.json").write_text(json.dumps({
        "id": repo, "sha": revision, "siblings": entries,
    }))


def official_cache(root: Path, ids: list[str], verified: set[str]) -> None:
    rows = [{"task_id": tid, "instruction": "stale metadata instruction",
             "artifact_path": f"artifacts/{tid}.tar.gz", "terminal_domain": "Scripting"}
            for tid in ids]
    payloads = {}
    for split, values in [("full", rows), ("verified", [r for r in rows if r["task_id"] in verified]),
                          ("sample", [r for r in rows if r["task_id"] in verified])]:
        body = "".join(json.dumps(row) + "\n" for row in values).encode()
        payloads[f"data/{split}.jsonl.gz"] = gzip.compress(body, mtime=0)
    for tid in ids:
        payloads[f"artifacts/{tid}.tar.gz"] = gzip.compress(tar_payload({tid: package(tid)}), mtime=0)
    cached_source(root, "official", payloads)


def assert_rejected(fn, text: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"expected rejection containing {text!r}")


def test_official_build_uses_archive_instruction_and_reserves_verified():
    """BUG-20 occurs in the official JSONL too, not just the derived parquet."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        official_cache(root / "source", ["tw_1", "tw_2"], {"tw_1"})
        out = root / "out"
        manifest = builder.build(argparse.Namespace(
            source="official", source_root=root / "source", official_root=None,
            split="train", max_tasks=0, download=False, out=out, resource_policy="canonical-mb",
        ))
        rows = [json.loads(line) for line in (out / "rl_tasks.jsonl").read_text().splitlines()]
        assert [row["label"] for row in rows] == ["tw_2"]
        assert manifest["official_verified_overlap"] == 0
        assert manifest["selection_counters"] == {"official_verified_reserved": 1}
        task = Path(rows[0]["metadata"]["task_dir"])
        assert rows[0]["prompt"] == (task / "instruction.md").read_text()
        assert "harbor-canary" in rows[0]["prompt"]
        assert "stale metadata" not in rows[0]["prompt"]
        for name, (blob, mode) in package("tw_2").items():
            assert (task / name).read_bytes() == blob
            assert (task / name).stat().st_mode & 0o777 == mode
        assert manifest["local_oracle_trials"] == manifest["sft_rows_written"] == 0
        assert rows[0]["metadata"]["empirical_pass_rate"] is None


def test_clean_build_filters_verdicts_leaks_and_reports_stale_shard_manifest():
    """BUG-20: verified bytes may differ from the stale checksum INSIDE the release."""
    pd = need("pandas")
    need("pyarrow")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ids = [f"tw_{i}" for i in range(1, 6)]
        official_cache(root / "official", ids, {"tw_1"})
        tasks = {tid: package(tid) for tid in ids}
        rows = []
        for tid in ids:
            row = {"task_id": tid, "task_group_id": tid, "terminal_domain": "Scripting",
                   "shard": "data/tasks-00000.tar", "member_prefix": f"tasks/{tid}",
                   "task_content_sha256": builder.content_hash(tasks[tid]),
                   "reward_verdict": "fail" if tid in {"tw_3", "tw_4"} else "pass"}
            row.update({column: tasks[tid][name][0].decode()
                        for name, column in builder.TEXT_COLUMNS.items()})
            rows.append(row)
        tasks["tw_2"]["instruction.md"] = (b"Actual updated instruction.\n", 0o644)
        tasks["tw_5"]["environment/renamed.py"] = tasks["tw_5"]["tests/test_state.py"]
        parquet = io.BytesIO()
        pd.DataFrame(rows).to_parquet(parquet, index=False)
        payloads = {
            "metadata/tasks.parquet": parquet.getvalue(),
            "metadata/train_ready_ids.txt": b"tw_1\ntw_2\ntw_4\ntw_5\n",
            "metadata/pass_at_5.csv": b"task_id,graded_attempts,solved_attempts,pass_at_5\ntw_2,5,4,0.8\n",
            "data/tasks-00000.tar": tar_payload(tasks, "tasks/"),
            "metadata/shard_manifest.jsonl": json.dumps({
                "shard": "data/tasks-00000.tar", "size_bytes": 1, "sha256": "stale"
            }).encode(),
        }
        cached_source(root / "clean", "seeds-clean", payloads)
        out = root / "out"
        manifest = builder.build(argparse.Namespace(
            source="seeds-clean", source_root=root / "clean", official_root=root / "official",
            split="train", max_tasks=0, download=False, out=out, resource_policy="canonical-mb",
        ))
        assert manifest["tasks_selected"] == 1
        assert manifest["selection_counters"] == {
            "not_train_ready": 1, "official_verified_reserved": 1,
            "train_ready_without_passing_oracle": 1,
        }
        assert manifest["drop_counters"] == {"verifier_leak": 1}
        assert manifest["metadata_text_mismatches"] == {"instruction": 1}
        assert manifest["source_content_hash_mismatches"] == 2
        drift = manifest["upstream_shard_manifest_mismatches"]
        assert len(drift) == 1
        assert drift[0]["actual"]["sha256"] == hashlib.sha256(payloads["data/tasks-00000.tar"]).hexdigest()
        row = json.loads((out / "rl_tasks.jsonl").read_text())
        assert row["label"] == "tw_2" and row["prompt"] == "Actual updated instruction.\n"
        assert row["metadata"]["upstream_solver_statistics"]["success_fraction"] == 0.8
        assert row["metadata"]["tier"] == "unknown"
        assert row["metadata"]["n_reference_trials"] == 0
        assert row["metadata"]["empirical_pass_rate"] is None
        assert not row["metadata"]["upstream_statistics_revision_matched"]
        assert not (out / "tasks/tw_1").exists()
        assert not (out / "tasks/tw_5").exists()


def test_pinned_file_integrity_checks_both_git_and_lfs_and_detects_tampering():
    payload = b"abcdef"
    for entry in [
        {"rfilename": "a", "size": 6, "lfs": {"sha256": hashlib.sha256(payload).hexdigest()}},
        {"rfilename": "a", "size": 6, "blobId": hashlib.sha1(b"blob 6\0" + payload).hexdigest()},
    ]:
        assert builder.verify_payload(payload, entry) == hashlib.sha256(payload).hexdigest()
        assert_rejected(lambda: builder.verify_payload(b"abcdeg", entry), "hash mismatch")
        assert_rejected(lambda: builder.verify_payload(b"short", entry), "size mismatch")


def test_source_rejects_a_different_commit_and_modified_cached_files():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cached_source(root, "official", {"data/a": b"abcdef"})
        source = builder.Source(root, "official", False)
        (root / "data/a").write_bytes(b"abcdeg")
        assert_rejected(lambda: source.file("data/a"), "hash mismatch")
        index = json.loads((root / "hub_info.json").read_text())
        index["sha"] = "different revision"
        (root / "hub_info.json").write_text(json.dumps(index))
        assert_rejected(lambda: builder.Source(root, "official", False), "expected EuniAI")


def test_archive_rejects_traversal_links_and_duplicate_members():
    with tempfile.TemporaryDirectory() as directory:
        archive_path = Path(directory) / "input.tar"
        for name, kind, duplicate in [
            ("tw_1/../../escape", tarfile.REGTYPE, False),
            ("/tmp/escape", tarfile.REGTYPE, False),
            ("tw_1/link", tarfile.SYMTYPE, False),
            ("tw_1/link", tarfile.LNKTYPE, False),
            ("tw_1/file", tarfile.REGTYPE, True),
        ]:
            with tarfile.open(archive_path, "w") as tar:
                for _ in range(2 if duplicate else 1):
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = "/tmp/escape" if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE) else ""
                    tar.addfile(member)
            reason = "duplicate/invalid" if duplicate else (
                "non-regular" if kind != tarfile.REGTYPE else "unsafe path"
            )
            assert_rejected(lambda: builder.read_archive(archive_path, {"tw_1": "tw_1"}), reason)


def test_missing_grader_is_rejected_and_project_tests_are_allowed():
    files = package("tw_1")
    files["environment/tests/test_project.py"] = (b"def test_project(): pass\n", 0o644)
    assert builder.audit_task({"task_id": "tw_1"}, files)["rejection"] is None
    del files["tests/test_state.py"]
    audit = builder.audit_task({"task_id": "tw_1"}, files)
    assert audit["rejection"] == "missing_required_files"
    assert audit["missing_or_empty_files"] == ["tests/test_state.py"]


def test_legacy_root_resource_settings_are_excluded_instead_of_ignored():
    files = package("tw_1")
    files["task.toml"] = (b'memory_mb = 4096\nallow_internet = false\n', 0o644)
    original = dict(files)
    audit = builder.audit_task({"task_id": "tw_1"}, files)
    assert audit["rejection"] == "legacy_root_config"  # Harbor would silently apply defaults.
    assert audit["legacy_root_config_keys"] == ["allow_internet", "memory_mb"]
    assert files == original


def test_resource_normalization_keeps_canonical_limits_and_other_sections():
    """BUG-20: conflicting aliases prevent current Harbor from loading 451 candidates."""
    import tomllib

    files = package("tw_1")
    raw = ('# harbor-canary fixture\nversion = "1.0"\n[environment]\n'
           'memory = "2G"\nmemory_mb = 4096\nstorage = "5G"\nstorage_mb = 10240\n'
           'allow_internet = false\n[metadata]\nmemory = "leave this alone"\n')
    files["task.toml"] = (raw.encode(), 0o644)
    fixed, repairs = builder.normalize_resources(files, "canonical-mb")
    assert files["task.toml"][0] == raw.encode()
    config = tomllib.loads(fixed["task.toml"][0].decode())
    assert config["environment"] == {"memory_mb": 4096, "storage_mb": 10240, "allow_internet": False}
    assert config["metadata"]["memory"] == "leave this alone"
    assert "harbor-canary fixture" in fixed["task.toml"][0].decode()
    assert len(repairs) == 2
    assert builder.content_hash(files) != builder.content_hash(fixed)
    assert builder.normalize_resources(files, "preserve") == (files, [])
    assert builder.normalize_resources(fixed, "canonical-mb") == (fixed, [])


def test_upstream_rate_is_the_observed_fraction_and_bad_counts_fail():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rates.csv"
        header = "task_id,graded_attempts,solved_attempts,pass_at_5\n"
        path.write_text(header + "tw_1,5,2,1\n")
        assert builder.load_upstream_rates(path)["tw_1"]["success_fraction"] == 0.4
        for body in ["tw_1,0,0,0\n", "tw_1,5,6,1\n", "tw_1,5,2,0.4\ntw_1,5,2,0.4\n"]:
            path.write_text(header + body)
            assert_rejected(lambda: builder.load_upstream_rates(path), "invalid/duplicate")


def test_nonempty_output_is_never_overwritten():
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        sentinel = out / "rl_tasks.jsonl"
        sentinel.write_text("existing user data")
        assert_rejected(lambda: builder.build(argparse.Namespace(out=out)), "new or empty")
        assert sentinel.read_text() == "existing user data"


def test_duplicate_ids_and_new_conversation_fields_require_an_audit():
    assert_rejected(lambda: builder.index_rows([{"task_id": "tw_1"}] * 2), "duplicate")
    assert_rejected(lambda: builder.index_rows([{"task_id": "tw_1", "messages": []}]),
                    "conversation fields")
