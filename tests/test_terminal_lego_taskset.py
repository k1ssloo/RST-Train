"""BUG-22: complete Git snapshots, LFS bytes and honest Terminal-Lego task pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import load_script

builder = load_script("10e_build_terminal_lego_taskset")


def package(tid: str = "task_00001") -> dict[str, tuple[bytes, int]]:
    return {
        "instruction.md": (f"# terminal-bench-canary fixture\nWrite {tid} to /app/out.\n".encode(), 0o644),
        "task.toml": (b'version = "1.0"\n[metadata]\ncategory = "shell-scripting"\n'
                      b'difficulty = "easy"\nsource_url = "https://stackoverflow.com/questions/123/test"\n'
                      b'[environment]\ncpus = 1\nmemory = "1G"\nstorage = "5G"\n'
                      b'[agent]\ntimeout_sec = 60\n[verifier]\ntimeout_sec = 30\n', 0o644),
        "environment/Dockerfile": (b"FROM ubuntu:22.04\nWORKDIR /app\n", 0o644),
        "solution/solve.sh": (b"#!/bin/sh\nprintf solved > /app/out\n", 0o644),
        "tests/test.sh": (b"#!/bin/sh\npytest /tests/test_outputs.py\n", 0o644),
        "tests/test_outputs.py": (b"def test_output():\n    assert open('/app/out').read() == 'solved'\n", 0o644),
    }


def fixture_repo(root: Path, packages: dict[str, dict]) -> str:
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    for tid, files in packages.items():
        for name, (body, mode) in files.items():
            path = root / tid / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            path.chmod(mode)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture",
                    "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
                    "commit", "-qm", "Fixture"], check=True, capture_output=True)
    return builder.git(root, "rev-parse", "HEAD").decode().strip()


def reject(fn, message: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"expected rejection containing {message!r}")


def test_full_build_filters_exclusions_and_uses_original_namespaced_tasks():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        packages = {tid: package(tid) for tid in ("task_00001", "task_00002", "task_00003")}
        del packages["task_00003"]["tests/test_outputs.py"]
        revision = fixture_repo(root / "source", packages)
        exclusion = json.dumps({"task_id": "task_00002", "reason": "no_op_pass"}).encode()
        (root / "excluded.jsonl").write_bytes(exclusion)
        args = argparse.Namespace(source_root=root / "source", exclusions=root / "excluded.jsonl",
                                  lfs_cache=root / "lfs", out=root / "out", download=False,
                                  max_tasks=0, task_ids=None)
        manifest = builder.build(args, revision=revision,
                                 exclusion_sha=hashlib.sha256(exclusion).hexdigest())
        assert manifest["source_tasks"] == 3 and manifest["upstream_excluded"] == 1
        assert manifest["tasks_selected"] == 1
        assert manifest["drop_counters"] == {"missing_required_files": 1}
        row = json.loads((args.out / "rl_tasks.jsonl").read_text())
        assert row["label"] == "terminal_lego_task_00001"
        assert row["metadata"]["task_group_id"] == "terminal_lego_so_123"
        task = Path(row["metadata"]["task_dir"])
        assert row["prompt"] == (task / "instruction.md").read_text()
        for name, (body, mode) in packages["task_00001"].items():
            assert (task / name).read_bytes() == body
            assert (task / name).stat().st_mode & 0o777 == mode
        assert row["metadata"]["empirical_pass_rate"] is None
        assert row["metadata"]["local_validation"] == "static_only"
        assert manifest["local_oracle_trials"] == manifest["sft_rows_written"] == 0
        assert not row["metadata"]["upstream_validation_revision_matched"]
        assert not (args.out / "tasks/terminal_lego_task_00002").exists()


def test_git_index_checks_revision_and_rejects_changed_task_bytes():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        revision = fixture_repo(root / "source", {"task_00001": package()})
        reject(lambda: builder.git_index(root / "source", "0" * 40), "must be at")
        entries = builder.git_index(root / "source", revision)["task_00001"]
        (root / "source/task_00001/instruction.md").write_text("changed prompt")
        reject(lambda: builder.read_task(root / "source", "task_00001", entries,
                                          cache=root / "lfs", revision=revision, download=False),
               "Git blob mismatch")


def test_lfs_pointer_is_materialized_from_verified_cache():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        asset = b"\x89PNG\r\nfixture"
        digest = hashlib.sha256(asset).hexdigest()
        pointer = (builder.LFS_PREFIX + f"oid sha256:{digest}\nsize {len(asset)}\n".encode())
        files = package()
        files["environment/image.png"] = pointer, 0o644
        revision = fixture_repo(root / "source", {"task_00001": files})
        cache = root / "lfs"
        cache.mkdir()
        (cache / digest).write_bytes(asset)
        entries = builder.git_index(root / "source", revision)["task_00001"]
        resolved, assets = builder.read_task(root / "source", "task_00001", entries,
                                             cache=cache, revision=revision, download=False)
        assert resolved["environment/image.png"][0] == asset
        assert assets == [{"path": "environment/image.png", "sha256": digest, "size_bytes": len(asset)}]
        assert (root / "source/task_00001/environment/image.png").read_bytes() == pointer
        (cache / digest).write_bytes(b"tampered")
        reject(lambda: builder.resolve_lfs(pointer, name="task_00001/environment/image.png",
                                           cache=cache, revision=revision, download=False),
               "LFS cache hash/size mismatch")


def test_missing_and_malformed_lfs_are_errors_not_shipped_pointer_text():
    with tempfile.TemporaryDirectory() as directory:
        pointer = builder.LFS_PREFIX + b"oid sha256:" + b"0" * 64 + b"\nsize 1\n"
        reject(lambda: builder.resolve_lfs(pointer, name="asset.png", cache=Path(directory),
                                           revision="fixture", download=False), "missing LFS")
        reject(lambda: builder.lfs_pointer(builder.LFS_PREFIX + b"invalid"), "malformed LFS")
        assert builder.lfs_pointer(b"ordinary fixture data") is None


def test_exclusion_integrity_duplicates_and_invalid_ids():
    valid = b'{"task_id":"task_00001","reason":"no_op_pass"}\n'
    assert builder.parse_exclusions(valid) == {"task_00001": "no_op_pass"}
    reject(lambda: builder.parse_exclusions(valid * 2), "duplicate/invalid")
    reject(lambda: builder.parse_exclusions(b'{"task_id":"../escape","reason":"bad"}'),
           "invalid Terminal-Lego")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "excluded.jsonl"
        path.write_bytes(valid)
        reject(lambda: builder.load_exclusions(path, False, "0" * 64), "SHA-256 mismatch")


def test_git_symlinks_cannot_enter_the_pool():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        revision = fixture_repo(root, {"task_00001": package()})
        # The filesystem check must also catch symlinks added after fetching a
        # clean commit, even if their target happens to have the expected bytes.
        entries = builder.git_index(root, revision)["task_00001"]
        path = root / "task_00001/instruction.md"
        path.rename(root / "original")
        path.symlink_to(root / "original")
        reject(lambda: builder.read_task(root, "task_00001", entries, cache=root / "lfs",
                                          revision=revision, download=False), "symlink")


def test_verifier_and_solution_copies_in_environment_are_excluded():
    for private in ("tests/test_outputs.py", "solution/solve.sh"):
        files = package()
        files["environment/renamed.txt"] = files[private]
        audit = builder.audit_task("task_00001", files)
        assert audit["rejection"] == "verifier_or_solution_leak"
        assert audit["leak"]["kind"] == "byte_identical"
    files = package()
    files["environment/tests/test_project.py"] = b"def test_project(): pass\n", 0o644
    assert builder.audit_task("task_00001", files)["rejection"] is None


def test_missing_grader_and_malformed_config_are_rejected():
    files = package()
    del files["tests/test_outputs.py"]
    assert builder.audit_task("task_00001", files)["rejection"] == "missing_required_files"
    files = package()
    files["task.toml"] = b"[invalid", 0o644
    assert builder.audit_task("task_00001", files)["rejection"] == "invalid_task_text_or_toml"


def test_missing_docker_copy_directory_is_not_treated_as_a_runnable_task():
    files = package()
    files["environment/Dockerfile"] = (
        b"FROM python:3.13-slim-bookworm\nCOPY ./task_file /app/task_file\n", 0o644,
    )
    audit = builder.audit_task("task_00001", files)
    assert audit["rejection"] == "missing_build_context"
    assert audit["missing"] == ["environment/task_file/"]
    # A tracked empty file is sufficient to preserve its parent directory.
    files["environment/task_file/.keep"] = b"", 0o644
    assert builder.audit_task("task_00001", files)["rejection"] is None
    # Tasks that build their workspace in RUN do not need that COPY source.
    files = package()
    files["environment/Dockerfile"] = b"FROM ubuntu:22.04\nRUN mkdir -p /app/task_file\n", 0o644
    assert builder.audit_task("task_00001", files)["rejection"] is None


def test_nonempty_output_and_unsafe_paths_are_refused():
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        (out / "existing").write_text("keep")
        reject(lambda: builder.build(argparse.Namespace(out=out)), "new or empty")
        assert (out / "existing").read_text() == "keep"
    for path in ("/tmp/escape", "../escape", "a/../../escape", "a\\escape", "a//b", ".git/config"):
        reject(lambda: builder.safe_path(path), "unsafe source path")


def test_source_question_groups_do_not_depend_on_task_directory_number():
    a = builder.group_id("task_00001", "https://stackoverflow.com/questions/123/old-title")
    b = builder.group_id("task_00002", "https://stackoverflow.com/questions/123/new-title")
    assert a == b == "terminal_lego_so_123"
    assert builder.group_id("task_00001", "https://other.example/123") == "terminal_lego_task_00001"


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
