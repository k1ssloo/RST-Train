"""`scripts/08_prepare_eval_ckpt.sh`'s shard gate, run rather than eyeballed.

THE FAILURE IT PREVENTS
    `verl.model_merger` loads whichever `model_world_size_N_rank_*.pt` files it finds. A
    missing rank does not make it fail -- it leaves that rank's slice of EVERY tensor at
    its initialization value, and the result loads, serves and evaluates. Nothing
    downstream can tell: the loss was never recomputed, the vision-restore gate only looks
    at tensor NAMES, and a partly-random 4B still emits fluent text.

    So the shard count has to be checked before the merge, from `fsdp_config.json` (which
    records the world size) and the shard filenames. That is pure filename arithmetic, so
    it is testable with empty files and no GPU.

WHY THE TEST EXTRACTS THE BLOCK FROM THE SHELL SCRIPT
    Copying the logic into the test would test the copy. `gate_source()` pulls the real
    heredoc out of the launcher, so an edit to the launcher that breaks the gate breaks
    this test. It also means a syntax error in an embedded heredoc -- which `bash -n`
    cannot see, because to bash it is just a string -- fails here instead of on the
    cluster after a 9 GB download.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT  # noqa: E402

LAUNCHER = ROOT / "scripts" / "08_prepare_eval_ckpt.sh"
SHARD_GATE_MARKER = "model_world_size_*_rank_*.pt shards"


def blocks() -> list[str]:
    """Every `python - ... <<'EOF_PY'` heredoc in the launcher, in order."""
    text = LAUNCHER.read_text(encoding="utf-8")
    found = re.findall(r"<<'EOF_PY'[^\n]*\n(.*?)\nEOF_PY", text, re.S)
    assert found, "no EOF_PY heredocs found; did the launcher's quoting change?"
    return found


def gate_source() -> str:
    matching = [b for b in blocks() if SHARD_GATE_MARKER in b]
    assert len(matching) == 1, (
        f"expected exactly one shard gate, found {len(matching)}. Anchor the search on a "
        f"different string if the gate was split."
    )
    return matching[0]


def run_gate(ckpt: Path) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(gate_source())
        script = handle.name
    try:
        return subprocess.run([sys.executable, script, str(ckpt)],
                              capture_output=True, text=True, check=False)
    finally:
        Path(script).unlink(missing_ok=True)


def make_ckpt(root: Path, *, world: int, ranks=None, declared=None, size: int = 4096) -> Path:
    """A verl-shaped checkpoint dir of empty files. Content is irrelevant to the gate."""
    root.mkdir(parents=True, exist_ok=True)
    for rank in (range(world) if ranks is None else ranks):
        (root / f"model_world_size_{world}_rank_{rank}.pt").write_bytes(b"\0" * size)
        (root / f"optim_world_size_{world}_rank_{rank}.pt").write_bytes(b"\0")
    if declared is not None:
        (root / "fsdp_config.json").write_text(
            json.dumps({"FSDP_version": 2, "world_size": declared}))
    return root


def test_every_embedded_python_block_compiles():
    # bash -n cannot see inside a heredoc, so without this a typo in the gate surfaces on
    # the cluster, after the download and the merge.
    for index, block in enumerate(blocks()):
        try:
            compile(block, f"<08_prepare_eval_ckpt block {index}>", "exec")
        except SyntaxError as exc:  # pragma: no cover - only on a real breakage
            raise AssertionError(
                f"embedded python block {index} does not compile: line {exc.lineno}: "
                f"{exc.msg}\n    {exc.text}") from None


def test_a_complete_shard_set_passes():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = make_ckpt(Path(tmp) / "global_step_82", world=8, declared=8)
        done = run_gate(ckpt)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "8/8 present" in done.stdout, done.stdout


def test_a_missing_rank_is_refused():
    # The whole point. This is the case that otherwise produces a partly-random model
    # which evaluates without complaint.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = make_ckpt(Path(tmp) / "global_step_82", world=8,
                         ranks=[0, 1, 2, 3, 4, 5, 7], declared=8)
        done = run_gate(ckpt)
        assert done.returncode != 0
        out = done.stdout + done.stderr
        assert "REFUSING TO MERGE" in out and "[6]" in out, out
        assert "initialization value" in out, "the message does not say why it matters"


def test_shards_from_two_world_sizes_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = make_ckpt(Path(tmp) / "global_step_82", world=8, declared=8)
        (ckpt / "model_world_size_4_rank_0.pt").write_bytes(b"\0")
        done = run_gate(ckpt)
        assert done.returncode != 0
        assert "more than one world size" in (done.stdout + done.stderr)


def test_a_declared_world_size_that_contradicts_the_filenames_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = make_ckpt(Path(tmp) / "global_step_82", world=8, declared=16)
        done = run_gate(ckpt)
        assert done.returncode != 0
        out = done.stdout + done.stderr
        assert "fsdp_config.json says world_size=16" in out, out


def test_a_missing_fsdp_config_falls_back_to_the_filenames():
    # Not every verl version writes it, and refusing there would block a valid checkpoint.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = make_ckpt(Path(tmp) / "global_step_82", world=8, declared=None)
        done = run_gate(ckpt)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "8/8 present" in done.stdout


def test_wildly_uneven_shards_are_warned_about():
    # A truncated upload looks exactly like this. FSDP shards are within a few percent.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = make_ckpt(Path(tmp) / "global_step_82", world=4, declared=4, size=4096)
        (ckpt / "model_world_size_4_rank_3.pt").write_bytes(b"\0" * 16)
        done = run_gate(ckpt)
        assert done.returncode == 0, "an uneven shard set is suspicious, not fatal"
        assert "WARNING" in done.stdout and "truncated upload" in done.stdout, done.stdout


def test_an_already_merged_hf_directory_is_recognized_and_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "out-hf"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")
        (ckpt / "model-00001-of-00001.safetensors").write_bytes(b"\0")
        done = run_gate(ckpt)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "already an HF checkpoint" in done.stdout


def test_a_directory_that_is_neither_says_what_it_expected():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "empty"
        ckpt.mkdir()
        done = run_gate(ckpt)
        assert done.returncode != 0
        out = done.stdout + done.stderr
        assert "model_world_size_8_rank_0.pt" in out, "no example of the expected layout"
        assert "global_step_" in out, "does not suggest the most likely mistake"


def test_the_launcher_restores_the_vision_tower_and_diffs_against_the_base():
    # Two steps that are easy to drop when someone "simplifies" the script, and whose
    # absence is invisible: without the restore the checkpoint will not load as a
    # ConditionalGeneration model, and without the diff a merge that reproduced the base
    # would be evaluated as the trained model.
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "07_restore_vision.py" in text, "the vision tower is never restored"
    assert "REFUSING: every sampled text tensor is bit-identical to the base" in text, (
        "the launcher no longer checks that training changed the weights"
    )


def test_the_diff_gate_separates_bf16_rounding_from_a_partial_merge():
    """The distinction the gate got wrong the first time it ran for real.

    Training keeps fp32 masters; the merge saves bf16. bf16 has 8 mantissa bits, so at a
    norm weight's magnitude the spacing is ~2e-3 while 82 steps at lr 3e-6 move it ~6e-5 --
    it rounds straight back to the base value. Measured on the 4B step-82 checkpoint:

        layers.26.input_layernorm.weight   fp32 |d| 5.77e-05 -> identical after bf16
        layers.26.mlp.down_proj.weight     fp32 |d| 9.24e-05 -> NOT identical after bf16

    So "some tensors identical" is only an alarm for MULTI-DIM weights. Treating a 1-D norm
    as evidence of a partial merge refuses a perfectly good checkpoint, which is what it did.
    """
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "still_matrix" in text and "still_flat" in text, (
        "the gate no longer distinguishes 1-D tensors from matrices, so bf16 rounding on a "
        "norm weight will be reported as a partial merge again"
    )
    assert "below bf16 resolution" in text, (
        "the 1-D case is refused or reported without saying why it is expected"
    )
    assert "cannot round back to the base wholesale" in text, (
        "the matrix case no longer explains why IT is the informative one"
    )
    # The sample must actually contain matrices, or the alarm can never fire.
    assert "spread(matrices, 16)" in text, "the sample no longer guarantees matrix coverage"


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
