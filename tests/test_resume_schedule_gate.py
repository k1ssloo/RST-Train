"""Whether a relaunch that silently restarts the lr schedule is refused and reported.

khazic/rst-qwen3.5-4b-tmax-sft was trained by two launches of the same RUN_NAME:

    run bic28b9c  total_epochs=1  step 42  loss 0.19885  lr 3.000e-07   <- min_lr floor
    run l6op97sl  total_epochs=3  step 43  loss 0.19539  lr 2.355e-06   <- 7.8x back up

Both exited 0 and neither warned. `trainer.resume_mode` defaults to `auto`, so the
second launch loaded `global_step_42` and kept counting, while `total_training_steps`
was re-derived from the new epoch count and the cosine rebuilt over 126 steps instead
of 42. The extra 42 steps moved the loss 0.1954 -> 0.1965 -- an anneal restarted at
7.8x its own floor buys nothing, and the only trace is the lr column.

Two defences, both pinned here: a launcher gate that refuses the resume before torchrun
starts, and a report check that catches it afterwards from the log alone.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _util import ROOT  # noqa: E402
from resume_guard import (  # noqa: E402
    ESCAPE_HATCH,
    FINGERPRINT_FILE,
    check,
    compare,
    latest_checkpoint_step,
    schedule_fingerprint,
)

LAUNCHER = ROOT / "scripts" / "30_run_sft_verl.sh"
TEXT = LAUNCHER.read_text(encoding="utf-8")

# The tmax run's overrides, trimmed to the keys that matter, exactly as
# 30_run_sft_verl.sh spells them.
RUN1 = [
    "data.train_files=/data/tmax/pretokenized_train.parquet",
    "data.train_batch_size=64",
    "data.max_token_len_per_gpu=32768",
    "model.path=/data/Qwen3.5-4B",
    "optim.lr=3e-6",
    "optim.lr_scheduler_type=cosine",
    "optim.min_lr_ratio=0.1",
    "optim.lr_warmup_steps_ratio=0.03",
    "optim.weight_decay=0.1",
    "trainer.total_epochs=1",
    "trainer.save_freq=20",
]
RUN2 = ["trainer.total_epochs=3" if a.startswith("trainer.total_epochs") else a for a in RUN1]


def _run_dir(tmp: str, *, steps: tuple[int, ...] = ()) -> Path:
    run_dir = Path(tmp) / "qwen3.5-4b-tmax-sft"
    run_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (run_dir / f"global_step_{step}" / "huggingface").mkdir(parents=True)
    return run_dir


def test_the_observed_relaunch_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _run_dir(tmp)
        code, _ = check(run_dir, RUN1, world_size=8)
        assert code == 0, "the first launch is a fresh run and must be allowed"
        # run 1 wrote global_step_20/40/42 before finishing.
        for step in (20, 40, 42):
            (run_dir / f"global_step_{step}").mkdir()
        code, message = check(run_dir, RUN2, world_size=8)
        assert code == 2, "the epoch change on a resume was allowed through"
        assert "global_step_42" in message, "the message does not say what would be resumed"
        assert "trainer.total_epochs: '1' -> '3'" in message, message
        assert "3.0e-07" in message and "2.35e-06" in message, (
            "the message does not quote the measured lr jump, so it reads as bookkeeping "
            "rather than as a wrong curve"
        )
        assert "RUN_NAME=" in message and "resume_mode=disable" in message, (
            "a gate that refuses without naming the two correct ways forward gets the "
            "escape hatch instead"
        )


def test_an_honest_resume_is_allowed():
    # Same schedule, restarted after a node failure: this is what resume_mode=auto is
    # for, and the gate must not stand in front of it.
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _run_dir(tmp)
        check(run_dir, RUN1, world_size=8)
        (run_dir / "global_step_20").mkdir()
        code, message = check(run_dir, RUN1, world_size=8)
        assert code == 0, message
        assert "same schedule" in message


def test_a_resume_that_predates_the_gate_is_allowed_and_says_so():
    # Refusing here would strand exactly the runs already in flight, and there is
    # nothing to compare against -- but the earlier curve is unverified and the operator
    # has to be told which column to look at.
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _run_dir(tmp, steps=(42,))
        code, message = check(run_dir, RUN2, world_size=8)
        assert code == 0, message
        assert "train/lr" in message, "no pointer to the evidence the gate could not check"
        assert (run_dir / FINGERPRINT_FILE).is_file(), "the schedule was not adopted"


def test_the_escape_hatch_records_the_new_schedule_rather_than_ignoring_it():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _run_dir(tmp)
        check(run_dir, RUN1, world_size=8)
        (run_dir / "global_step_42").mkdir()
        code, _ = check(run_dir, RUN2, world_size=8, allow=True)
        assert code == 0
        recorded = json.loads((run_dir / FINGERPRINT_FILE).read_text())["schedule"]
        assert recorded["trainer.total_epochs"] == "3", (
            "the override was accepted but not recorded, so the NEXT relaunch would be "
            "compared against a schedule that no longer ran"
        )


def test_every_knob_that_reshapes_the_curve_is_compared():
    base = schedule_fingerprint(RUN1, world_size=8)
    for key, changed in (
        ("optim.lr", "1e-5"),
        ("optim.min_lr_ratio", "0.0"),
        ("optim.lr_scheduler_type", "constant"),
        ("optim.lr_warmup_steps_ratio", "0.1"),
        ("data.train_batch_size", "32"),
        ("model.path", "/data/Qwen3.5-9B"),
        ("data.train_files", "/data/sft-v1-cap10/pretokenized_train.parquet"),
    ):
        other = dict(base, **{key: changed})
        assert compare(base, other), f"{key} changed without the gate noticing"
    # Rank count too: with dynamic batching the world size decides how many optimizer
    # steps an epoch takes, so the schedule length moves with it.
    assert compare(base, schedule_fingerprint(RUN1, world_size=16))


def test_knobs_that_do_not_touch_the_curve_are_not_compared():
    # A gate that fires on save_freq or the wandb project name is a gate that gets
    # disabled. Only the schedule and what a step means are in scope.
    noise = RUN1 + [
        "trainer.save_freq=40",
        "trainer.project_name=rst-qwen35-verl-2",
        "model.enable_gradient_checkpointing=False",
        "engine.fsdp_size=4",
    ]
    assert not compare(
        schedule_fingerprint(RUN1, world_size=8), schedule_fingerprint(noise, world_size=8)
    )


def test_a_knob_appearing_or_disappearing_is_a_difference():
    # Dropping `optim.min_lr_ratio=0.1` does not leave the floor where it was; it falls
    # back to verl's default. An absent key must not compare equal to a present one.
    without = [a for a in RUN1 if not a.startswith("optim.min_lr_ratio")]
    assert compare(
        schedule_fingerprint(RUN1, world_size=8), schedule_fingerprint(without, world_size=8)
    )


def test_the_latest_step_comes_from_the_directories_not_the_pointer_file():
    # The 27B run left latest_checkpointed_iteration.txt saying 82 with no such
    # directory. resume_mode=auto walks the directories, so the gate must too.
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _run_dir(tmp, steps=(20, 40, 42))
        (run_dir / "latest_checkpointed_iteration.txt").write_text("82")
        assert latest_checkpoint_step(run_dir) == 42
        assert latest_checkpoint_step(Path(tmp) / "never-trained") is None


def test_the_launcher_runs_the_gate_before_torchrun_with_the_args_it_will_pass():
    gate_at = TEXT.index("resume schedule gate")
    torchrun_at = TEXT.index("\ntorchrun")
    assert gate_at < torchrun_at, "the gate runs after the launch, which is too late"
    window = TEXT[gate_at:torchrun_at]
    assert "scripts/resume_guard.py" in window
    assert '--run-dir "$BASE_FOLDER/$RUN_NAME"' in window, (
        "the gate is not pointed at trainer.default_local_dir, so it inspects a "
        "directory verl will not resume from"
    )
    assert '"${VERL_ARGS[@]}"' in window, (
        "the gate is given a hand-written list instead of the overrides actually "
        "passed to the trainer -- that is how a launcher validates flags it does not use"
    )
    assert "NNODES * NGPUS" in window, "world size is not fingerprinted"
    assert 'NODE_RANK:-0' in window, (
        "every node would race to write the one fingerprint file"
    )
    assert ESCAPE_HATCH in (ROOT / "scripts" / "resume_guard.py").read_text(encoding="utf-8")


def test_the_report_catches_the_restart_from_the_log_alone():
    spec = importlib.util.spec_from_file_location(
        "make_report_lr", ROOT / "scripts" / "14_make_report.py"
    )
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)

    # The real curve: warmup, decay to the floor, then the relaunch.
    steps = [
        {"step": 1, "loss": 0.2624, "lr": 1.0e-6},
        {"step": 2, "loss": 0.2600, "lr": 3.0e-6},
        {"step": 20, "loss": 0.2100, "lr": 1.5e-6},
        {"step": 42, "loss": 0.19885, "lr": 3.0e-7},
        {"step": 43, "loss": 0.19539, "lr": 2.3546379237176016e-6},
        {"step": 84, "loss": 0.1965, "lr": 1.005e-6},
    ]
    hit = mr.find_lr_restart(steps)
    assert hit and hit["step"] == 43 and hit["prev_step"] == 42, hit
    assert 7.0 < hit["factor"] < 8.5, hit
    # Warmup is a rise and must not be reported as one.
    assert mr.find_lr_restart(steps[:4]) is None
    # Neither must logging jitter on a flat tail.
    flat = [{"step": i, "loss": 0.2, "lr": 3.0e-7 * (1.02 if i % 2 else 1.0)} for i in range(9)]
    assert mr.find_lr_restart(flat) is None

    findings = mr.Findings()
    mr.check_training(findings, {"steps": steps}, None, None)
    hits = [f for f in findings.rows if f[2] == "lr schedule"]
    assert hits and hits[0][0] == mr.WARN, findings.rows
    assert "resume_mode=auto" in hits[0][3], hits[0]


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
