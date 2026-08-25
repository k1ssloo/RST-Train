"""Refuse to resume a run whose learning-rate schedule has changed under it.

verl's SFT trainer defaults to `trainer.resume_mode: auto`: if
`trainer.default_local_dir` already holds a `global_step_*/`, it loads it and keeps
counting. Nothing in that path checks whether the *schedule* the checkpoint was
produced under is the schedule about to be applied. `total_training_steps` is
re-derived every launch from `total_epochs` and the batch size, and the cosine is
rebuilt over the new total -- so relaunching the same RUN_NAME with more epochs
restarts the curve part-way up.

Measured on khazic/rst-qwen3.5-4b-tmax-sft, two consecutive same-name launches:

    run bic28b9c  total_epochs=1  step 42  loss 0.19885  lr 3.000e-07   <- min_lr floor
    run l6op97sl  total_epochs=3  step 43  loss 0.19539  lr 2.355e-06   <- 7.8x back up

The second run finished exitcode 0 with no warning anywhere, and its 42 further steps
moved the loss 0.1954 -> 0.1965. A schedule that ends at `min_lr_ratio` and then jumps
7.8x above it is not a continuation of the first run and not a fresh run either; it is
two half-anneals, and the only evidence is the lr column of a log nobody diffs.

So this module is the diff. `schedule_fingerprint()` records the knobs that define the
curve, the CLI stores them next to the checkpoints as `rst_launch.json`, and a resume
whose fingerprint differs is a hard failure naming each changed knob.

Importable on purpose: the launcher is a shell script and this logic needs unit tests.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# The knobs that change the shape or length of the lr curve, or what a step means.
# Anything here differing across a resume makes the loaded optimizer state and the
# schedule about to be applied describe different runs.
#
# `data.train_files` and `model.path` are not schedule knobs, but resuming step 42 of
# one dataset into another dataset -- or into different weights -- is the same class of
# silent mismatch and the same fix, so they are fingerprinted together.
SCHEDULE_KEYS: tuple[str, ...] = (
    "trainer.total_epochs",
    "optim.lr",
    "optim.lr_scheduler_type",
    "optim.min_lr_ratio",
    "optim.lr_warmup_steps_ratio",
    "optim.weight_decay",
    "optim.betas",
    "data.train_batch_size",
    "data.train_files",
    "model.path",
)

# Why each one matters, quoted back to whoever hits the gate. Without this the message
# is "total_epochs changed", which reads like bookkeeping rather than a wrong curve.
_CONSEQUENCE: dict[str, str] = {
    "trainer.total_epochs": (
        "verl re-derives total_training_steps from this, so the cosine is rebuilt over "
        "a different total and the lr jumps back up at the resumed step"
    ),
    "optim.lr": "the peak of the cosine moves; the resumed steps anneal from elsewhere",
    "optim.lr_scheduler_type": "a different curve entirely is applied mid-run",
    "optim.min_lr_ratio": "the floor the previous steps decayed toward moves",
    "optim.lr_warmup_steps_ratio": (
        "warmup is re-derived from the new total, so the resumed step may land inside a "
        "warmup that already finished"
    ),
    "optim.weight_decay": "the loaded Adam state was accumulated under a different penalty",
    "optim.betas": "the loaded Adam moments were accumulated with different decay rates",
    "data.train_batch_size": (
        "one step no longer means the same number of examples, so both the step count "
        "and the schedule length change"
    ),
    "data.train_files": "step N of the checkpoint indexed a different dataset",
    "model.path": "the optimizer state belongs to different weights",
}

FINGERPRINT_FILE = "rst_launch.json"
ESCAPE_HATCH = "ALLOW_SCHEDULE_CHANGE_ON_RESUME"

_STEP_DIR = re.compile(r"^global_step_(\d+)$")


def schedule_fingerprint(overrides: list[str], *, world_size: int) -> dict[str, str]:
    """The schedule-defining subset of a hydra override list, plus the world size.

    Overrides are `key=value` strings exactly as passed to `verl.trainer.sft_trainer`.
    A key absent from the list is recorded as absent (verl's own default applies, and
    that default is stable for a fixed verl version) rather than guessed at, so a knob
    appearing or disappearing between launches is itself a difference.

    The world size is included because with `use_dynamic_bsz` the per-GPU token budget
    times the rank count is what decides how many optimizer steps an epoch takes.
    """
    seen: dict[str, str] = {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            continue
        key = key.lstrip("+~")  # hydra's append/delete prefixes
        if key in SCHEDULE_KEYS:
            seen[key] = value  # a later override of the same key wins, as in hydra
    fingerprint = {key: seen[key] for key in SCHEDULE_KEYS if key in seen}
    fingerprint["world_size"] = str(world_size)
    return fingerprint


def compare(previous: dict[str, str], current: dict[str, str]) -> list[str]:
    """Human-readable descriptions of every schedule knob that differs.

    Empty means the two launches describe the same curve, so resuming is honest.
    """
    changes: list[str] = []
    for key in (*SCHEDULE_KEYS, "world_size"):
        was, now = previous.get(key), current.get(key)
        if was == now:
            continue
        why = _CONSEQUENCE.get(
            key,
            "steps per epoch change with the rank count, so the schedule length changes",
        )
        changes.append(f"{key}: {was!r} -> {now!r}\n      {why}")
    return changes


def latest_checkpoint_step(run_dir: Path) -> int | None:
    """The highest `global_step_<n>` in a verl output dir, or None if there is none.

    This is what `resume_mode: auto` will pick up. `latest_checkpointed_iteration.txt`
    is deliberately not trusted: a stale one of those (27B, value 82, with no matching
    directory) is already on record.
    """
    steps = [
        int(m.group(1))
        for child in run_dir.iterdir()
        if child.is_dir() and (m := _STEP_DIR.match(child.name))
    ] if run_dir.is_dir() else []
    return max(steps) if steps else None


def check(
    run_dir: Path, overrides: list[str], *, world_size: int, allow: bool = False
) -> tuple[int, str]:
    """The gate. Returns `(exit_code, message)` and writes the fingerprint on success.

    Exit 2 -- the launcher's existing "refusing to train" code -- when a checkpoint is
    present and the recorded fingerprint differs. A resume with no recorded fingerprint
    (a run started before this gate existed) is allowed and adopts the current one:
    refusing there would strand exactly the runs the gate is meant to protect, and
    there is nothing to compare against.
    """
    current = schedule_fingerprint(overrides, world_size=world_size)
    step = latest_checkpoint_step(run_dir)
    record = run_dir / FINGERPRINT_FILE

    if step is None:
        _write(record, current)
        return 0, f"[gate] fresh run in {run_dir}; recorded {len(current)} schedule knobs"

    if not record.is_file():
        _write(record, current)
        return 0, (
            f"[gate] resuming from global_step_{step} with no recorded schedule "
            f"(pre-dates this gate); adopting the current one. The lr curve of the "
            f"earlier steps is unverified -- check train/lr in the log for a rise."
        )

    try:
        previous = json.loads(record.read_text(encoding="utf-8"))["schedule"]
    except (ValueError, KeyError, OSError) as exc:
        _write(record, current)
        return 0, f"[gate] {record} unreadable ({exc}); rewriting it"

    changes = compare(previous, current)
    if not changes:
        return 0, f"[gate] resuming from global_step_{step} under the same schedule"

    listed = "\n".join(f"    {c}" for c in changes)
    message = (
        f"REFUSING TO TRAIN: verl would resume from global_step_{step} in\n"
        f"  {run_dir}\n"
        f"  (trainer.resume_mode defaults to auto), but {len(changes)} schedule knob(s) "
        f"changed since\n  that checkpoint was written:\n{listed}\n\n"
        f"  A resume re-derives total_training_steps and rebuilds the lr schedule over "
        f"the new\n  total, so the resumed step does not continue the curve the "
        f"checkpoint was left on.\n  Measured: 42 steps at total_epochs=1 ended at lr "
        f"3.0e-07 (the min_lr floor); the same\n  run relaunched at total_epochs=3 "
        f"resumed step 43 at lr 2.35e-06 -- 7.8x back up --\n  and its 42 extra steps "
        f"moved the loss 0.1954 -> 0.1965.\n\n"
        f"  Pick one:\n"
        f"    * a new run:      RUN_NAME=<something-else> (trains the new schedule from "
        f"step 0)\n"
        f"    * a true resume:  restore the knobs above to the recorded values\n"
        f"    * no resume:      trainer.resume_mode=disable (restarts in the same dir)\n"
        f"    * on purpose:     {ESCAPE_HATCH}=1 (records the new schedule and continues)"
    )
    if allow:
        _write(record, current)
        return 0, f"[gate] {ESCAPE_HATCH}=1: schedule change accepted\n{listed}"
    return 2, message


def _write(record: Path, schedule: dict[str, str]) -> None:
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps({"schedule": schedule}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="trainer.default_local_dir -- where global_step_* land")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("overrides", nargs="*", help="the hydra overrides being passed")
    args = parser.parse_args(argv)

    code, message = check(
        args.run_dir,
        args.overrides,
        world_size=args.world_size,
        allow=os.environ.get(ESCAPE_HATCH, "0") == "1",
    )
    print(message, file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
