"""`scripts/taskpool_common.py` -- what the three RL pool builders must agree on."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, load_script  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))

import taskpool_common as tp  # noqa: E402


def test_the_three_pool_builders_share_one_tier_table_object():
    rst = load_script("10_build_rl_taskset")
    swegym = load_script("10c_build_swegym_taskset")
    assert rst.TIERS is tp.TIERS and swegym.TIERS is tp.TIERS, (
        "a copied tuple can drift; the same object cannot")
    assert swegym.tier_of is tp.tier_of


def test_the_from_line_reader_is_the_one_both_pools_use():
    termigen = load_script("10b_build_termigen_taskset")
    rst = load_script("10_build_rl_taskset")
    assert termigen.base_image is tp.base_image and rst.base_image is tp.base_image
    assert tp.base_image("FROM python:3.11-slim\nRUN x") == "python:3.11-slim"
    assert tp.base_image("") == "?"


def test_a_byte_identical_verifier_in_the_build_context_is_the_strong_verdict():
    hit = tp.find_verifier_leak(
        [("environment/data/grader.py", "deadbeef"), ("environment/Dockerfile", "0000")],
        verifier_hashes={"deadbeef"})
    assert hit == ("environment/data/grader.py", "byte_identical"), (
        "renamed copies are caught by content, not by name")


def test_a_name_only_match_is_still_excluded_and_labelled_as_the_weaker_case():
    hit = tp.find_verifier_leak(
        [("environment/tests/test.sh", "1111")], verifier_hashes={"deadbeef"})
    assert hit == ("environment/tests/test.sh", "name_only")


def test_byte_identical_wins_over_name_only_whatever_the_order():
    context = [("environment/a/test.sh", "1111"), ("environment/z/other.py", "deadbeef")]
    assert tp.find_verifier_leak(context, {"deadbeef"}) == ("environment/z/other.py", "byte_identical")
    assert tp.find_verifier_leak(reversed(context), {"deadbeef"})[1] == "byte_identical"


def test_a_clean_build_context_is_none():
    assert tp.find_verifier_leak([("environment/Dockerfile", "0000")], {"deadbeef"}) is None
    assert tp.find_verifier_leak([], {"deadbeef"}) is None


def _make_shard(path: Path, members: dict[str, bytes]) -> None:
    import io
    import tarfile

    with tarfile.open(path, "w") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def _old_materialize_loop(tasks_root: Path, wanted, task_root: Path, log) -> None:
    """Verbatim copy of `10_build_rl_taskset.py:165-188` as it was before the lift."""
    import tarfile

    for shard, members in sorted(wanted.items()):
        prefix_to_id = {p + "/": tid for p, tid in members.items()}
        with tarfile.open(tasks_root / shard) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                for prefix, tid in prefix_to_id.items():
                    if not member.name.startswith(prefix):
                        continue
                    rel = member.name[len(prefix) :]
                    if rel.startswith("/") or ".." in Path(rel).parts:
                        raise SystemExit(f"unsafe member path: {member.name}")
                    dest = task_root / tid / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    handle = tar.extractfile(member)
                    if handle is not None:
                        dest.write_bytes(handle.read())
                    break
        log(f"  materialized from {shard}")


def test_materialize_tasks_is_byte_identical_to_the_loop_it_replaced():
    import tempfile

    members = {
        "tasks/rts_task_a/instruction.md": b"do A\n",
        "tasks/rts_task_a/tests/test_state.py": b"def test_x(): pass\n",
        "tasks/rts_task_a/environment/Dockerfile": b"FROM x\n",
        "tasks/rts_task_b/instruction.md": b"do B\n",
        "tasks/other/ignored.txt": b"not wanted\n",
    }
    wanted = {"data/tasks-00000.tar": {"tasks/rts_task_a": "rts_task_a", "tasks/rts_task_b": "rts_task_b"}}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data").mkdir()
        _make_shard(root / "data" / "tasks-00000.tar", members)
        new_out, old_out = root / "new", root / "old"
        new_log: list[str] = []
        old_log: list[str] = []
        tp.materialize_tasks(root, wanted, new_out, log=new_log.append)
        _old_materialize_loop(root, wanted, old_out, old_log.append)
        assert new_log == old_log == ["  materialized from data/tasks-00000.tar"]
        new_files = sorted(p.relative_to(new_out) for p in new_out.rglob("*") if p.is_file())
        old_files = sorted(p.relative_to(old_out) for p in old_out.rglob("*") if p.is_file())
        assert new_files == old_files and new_files, "same tree"
        for rel in new_files:
            assert (new_out / rel).read_bytes() == (old_out / rel).read_bytes(), rel
        assert not (new_out / "other").exists(), "unwanted prefixes are never extracted"
        complete, incomplete = tp.verify_tracked(new_out, ["rts_task_a", "rts_task_b"])
        assert complete == 0 and [t for t, _ in incomplete] == ["rts_task_a", "rts_task_b"]
        assert "task.toml" in incomplete[0][1]


def test_materialize_refuses_a_member_that_escapes_its_task_dir():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data").mkdir()
        _make_shard(root / "data" / "evil.tar", {"tasks/rts_task_a/../../escape": b"x"})
        try:
            tp.materialize_tasks(root, {"data/evil.tar": {"tasks/rts_task_a": "rts_task_a"}}, root / "out")
        except SystemExit as exc:
            assert "unsafe member path" in str(exc)
        else:
            raise AssertionError("a traversal member must abort the run")


def test_the_rl_pool_builder_extracts_through_the_shared_materializer():
    rst = load_script("10_build_rl_taskset")
    assert rst.materialize_tasks is tp.materialize_tasks and rst.verify_tracked is tp.verify_tracked
    source = (ROOT / "scripts" / "10_build_rl_taskset.py").read_text(encoding="utf-8")
    assert "tar.extractfile" not in source, "the loop grew a private copy back"


def test_the_termigen_pool_names_its_own_verifier_files():
    # termigen's grader is test_outputs.py, not test_state.py; the shared decision
    # must take the pool's names rather than assume the RST ones.
    hit = tp.find_verifier_leak([("environment/x/test_outputs.py", "1")], set(),
                                verifier_names=("test.sh", "test_outputs.py"))
    assert hit == ("environment/x/test_outputs.py", "name_only")
    assert tp.find_verifier_leak([("environment/x/test_outputs.py", "1")], set()) is None, (
        "under the RST names test_outputs.py is a project file")


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
