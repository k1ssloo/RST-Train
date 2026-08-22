"""Whether the report can find the loss curve of a run that actually happened.

Every verl report so far carried this WARN:

    no loss values scraped from logs; the curve could not be checked.
    Point --run-dir at the directory holding the trainer stdout.

while the trainer stdout sat in `$BASE_FOLDER/logs/run.log` with 110 usable steps in it.
Two causes, both ours. `--run-dir` was the trainer's *output* directory, which under
verl/FSDP holds only `global_step_*/` and no logs at all -- so the report asked a human to
point it at a path the launcher itself had chosen. And the appended `run.log` holds every
stage's stdout, so scraping it whole would have mixed the SFT curve with DPO's, whose loss
is log 2 by construction: 0.144 followed by 0.693147 reads as a run that diverged at the
end.

So the fix is explicit logs plus a stage slice, and these tests pin both against the real
4B OTA log format.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT  # noqa: E402

REPORT = ROOT / "scripts" / "14_make_report.py"
LAUNCHER = ROOT / "scripts" / "20_run_all.sh"


def _load():
    spec = importlib.util.spec_from_file_location("make_report", REPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mr = _load()

# Real lines from khazic/rst-qwen3.5-4b-ota-sft logs/run.log, trimmed to the fields that
# are scraped. The progress-bar prefix on the same line is the format verl emits.
VERL_STEP = (
    "Epoch 1/1:  99%|#########9| 109/110 [1:14:27<01:11, 71.47s/it]step:{step} - "
    "perf/max_memory_allocated_gb:22.72986888885498 - train/loss:{loss} - "
    "train/grad_norm:0.4677063822746277 - train/lr:{lr} - train/mfu:0.2\n"
)


def _run_log() -> str:
    """A `20_run_all.sh` log with the stages in the order the 4B run produced them."""
    text = "=== STAGE preflight  2026-08-20T15:22:12+08:00\n=== DONE preflight\n"
    text += "=== SKIP convert (verl/FSDP loads the HF checkpoint directly)\n"
    text += "=== STAGE train  2026-08-20T15:22:15+08:00\n"
    text += VERL_STEP.format(step=1, loss="0.30185991525650024", lr="1e-06")
    text += VERL_STEP.format(step=110, loss="0.14415396749973297", lr="3.0000000000001e-07")
    text += "=== DONE train\n=== STAGE export  2026-08-20T16:42:17+08:00\n=== DONE export\n"
    text += "=== STAGE dpo  2026-08-20T18:38:41+08:00\n"
    # DPO's step-0 loss is log 2 to the bit -- by construction, not by failing.
    text += "step:0  loss=0.693147  grad_norm=0.081\n"
    return text


def _scrape(text: str, *, stage: str = "train") -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "run.log"
        log.write_text(text, encoding="utf-8")
        return mr.parse_training_log(None, [log], stage=stage)


def test_the_verl_curve_is_scraped_from_an_explicit_log():
    steps = _scrape(_run_log())["steps"]
    assert [rec["step"] for rec in steps] == [1, 110], steps
    assert steps[0]["loss"] > steps[-1]["loss"], steps
    assert steps[-1]["grad_norm"] == 0.4677063822746277


def test_dpos_log_2_loss_stays_out_of_the_sft_curve():
    # The whole point of the slice: the SFT curve must end where SFT ended.
    steps = _scrape(_run_log())["steps"]
    assert all(rec["loss"] < 0.5 for rec in steps), steps
    # And asking for the DPO section gets the DPO line, so nothing is being dropped --
    # the sections are separated, not filtered.
    dpo = _scrape(_run_log(), stage="dpo")["steps"]
    assert [rec["loss"] for rec in dpo] == [0.693147], dpo


def test_verls_short_lr_spelling_is_scraped():
    # verl prints `train/lr:`, never `learning_rate`, so the long spelling alone found a
    # learning rate in none of these runs.
    steps = _scrape(_run_log())["steps"]
    assert steps[0]["lr"] == 1e-06, steps[0]
    assert steps[-1]["lr"] < steps[0]["lr"], "the cosine schedule decayed"


def test_a_log_with_no_stage_markers_is_read_whole():
    # slime/Megatron stdout, or any log not produced by 20_run_all.sh. Slicing a file that
    # has no sections must not silently return nothing.
    plain = "iteration 5 | lm loss: 1.25 | grad norm: 0.4 | learning rate: 2.0e-05\n"
    steps = _scrape(plain)["steps"]
    assert steps == [{"loss": 1.25, "step": 5, "grad_norm": 0.4, "lr": 2.0e-05}], steps


def test_a_verl_run_dir_alone_yields_nothing_which_is_why_logs_are_passed():
    # This is the observed failure, reproduced: the run dir exists, it is the right
    # directory, and it contains no log because verl writes checkpoints there and stdout
    # somewhere else entirely.
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "qwen3.5-4b-ota-sft-v1"
        (run_dir / "global_step_110" / "huggingface").mkdir(parents=True)
        (run_dir / "global_step_110" / "model_world_size_8_rank_0.pt").write_bytes(b"x")
        (run_dir / "latest_checkpointed_iteration.txt").write_text("110")
        assert mr.parse_training_log(run_dir)["steps"] == []


def test_missing_and_unreadable_logs_are_skipped_not_fatal():
    # The launcher passes whichever candidate paths exist, and a report must still be
    # produced when none do -- a missing curve is a WARN, not a crash.
    out = mr.parse_training_log(None, [Path("/nonexistent/run.log")])
    assert out["steps"] == [] and out["log_files"] == []


def test_the_launcher_passes_its_own_log_and_names_the_stage():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "collect_train_logs()" in text, (
        "the launcher no longer collects its own log path; the report goes back to "
        "telling a human to find a file the launcher chose"
    )
    # Both names in use: this script's own usage line, and the cluster's.
    assert '"$BASE_FOLDER/run_all.log"' in text and '"$BASE_FOLDER/logs/run.log"' in text
    assert 'RST_RUN_LOG' in text, "no escape hatch for a third log name"
    assert '"${TRAIN_LOG_ARGS[@]}" --train-stage train' in text, "SFT report"
    assert '"${TRAIN_LOG_ARGS[@]}" --train-stage rl' in text, (
        "the GRPO report must ask for the rl section, or it reports the SFT curve twice"
    )


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
