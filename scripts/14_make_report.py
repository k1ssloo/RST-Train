#!/usr/bin/env python3
"""Assemble a markdown experiment report and run mechanical anomaly checks.

    python scripts/14_make_report.py --run-dir $BASE_FOLDER/qwen35-27b-rst-sft-v1 \
        --data-manifest $BASE_FOLDER/sft-v1-cap10/manifest.json \
        --eval mine=$BASE_FOLDER/eval/mine/results.json \
        --eval reference=$BASE_FOLDER/eval/reference/results.json \
        --out $BASE_FOLDER/REPORT.md

This tool only reports what it can *check*. It deliberately does not attempt root
cause analysis -- it emits a findings table plus an "Analysis" section the
operator fills in. A machine can tell you the infra-failure rate was 34 %; only
the operator can tell you the rootless daemon ran out of disk at 02:10.

Every check has a stated threshold and a stated reason, so a WARN can be argued
with rather than obeyed blindly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# `python scripts/14_make_report.py` puts scripts/ on sys.path[0], not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# One copy of the published numbers, shared with 06_eval.py -- see rst_common/paper.py
# for why two copies of a reference value is one copy too many.
from rst_common.paper import (  # noqa: E402
    PAPER,
    PAPER_MODELS,
    REF_TARGET,
    has_paper_reference,
)

__all__ = ["PAPER", "PAPER_MODELS", "REF_TARGET", "has_paper_reference"]

FAIL, WARN, OK, INFO = "FAIL", "WARN", "OK", "INFO"

# FAILs that mean "this checkpoint was never measured on agentic benchmarks", as
# opposed to "this checkpoint is not a valid model". The distinction matters because
# two different downstream stages read this verdict and they do not have the same
# prerequisites:
#
#   GRPO (20_run_all.sh 10a) needs a task container per rollout -- exactly what agentic
#     eval needs -- so if coverage failed for want of a sandbox, RL could not have run
#     anyway and blocking it costs nothing.
#   DPO  (10b) needs NO container, NO network and NO privilege. It is the stage that
#     exists precisely for the pod that cannot run containers. Blocking it on a
#     coverage FAIL would mean the container-less pod -- the documented target -- can
#     produce an SFT checkpoint and then nothing else, which is the opposite of what
#     DPO_PLAN.md promises.
#
# So `in_range` keeps its narrow meaning (zero FAILs of any kind) and a second signal,
# `checkpoint_trustworthy`, answers the different question DPO actually needs answered:
# is there any finding suggesting this checkpoint is not a sound model? Being unmeasured
# on tb-hard says nothing about that. A broken loss mask, a non-finite loss or a
# regression against the measured base does, and all of those stay blocking.
#
# Keep this set tiny. Anything added here is a FAIL somebody decided to train on top of.
POSTTRAIN_EXEMPT_FAILS = {("eval", "benchmark coverage")}


class Findings:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, level: str, area: str, check: str, detail: str) -> None:
        self.rows.append((level, area, check, detail))

    def worst(self) -> str:
        for level in (FAIL, WARN, OK):
            if any(r[0] == level for r in self.rows):
                return level
        return INFO

    def count(self, level: str) -> int:
        return sum(1 for r in self.rows if r[0] == level)


# ----------------------------------------------------------------- ingestion

def read_json(path: Path | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


STAGE_MARKER = re.compile(r"^=== (?:STAGE|SKIP|DONE|FAILED) (\S+)", re.M)


def stage_section(text: str, stage: str) -> str:
    """The slice of a `20_run_all.sh` log belonging to one stage, or all of it.

    `20_run_all.sh` writes every stage's stdout to one appended `logs/run.log`, so a
    scrape of the whole file mixes trainers: the SFT curve, then GRPO, then DPO, whose
    loss sits at log 2 by construction. That is how a "loss increased" finding gets
    manufactured out of two healthy runs. A log with no `=== STAGE` markers is not one
    of ours and is returned whole.
    """
    starts = [m for m in STAGE_MARKER.finditer(text) if m.group(0).startswith("=== STAGE")]
    if not starts:
        return text
    parts: list[str] = []
    for match in starts:
        if match.group(1) != stage:
            continue
        following = STAGE_MARKER.search(text, match.end())
        parts.append(text[match.end():following.start() if following else len(text)])
    return "\n".join(parts)


def parse_training_log(run_dir: Path | None, extra_logs: list[Path] | None = None,
                       stage: str = "train") -> dict:
    """Best-effort scrape of the trainer's stdout for loss / grad-norm / lr.

    Log format is not a contract, so everything here is optional and the report
    says so rather than pretending a missing field is a zero.

    `run_dir` is the trainer's *output* directory, which under verl/FSDP holds only
    `global_step_*/` -- no logs at all. That is why this takes explicit files too: the
    launcher knows its log is `$BASE_FOLDER/logs/run.log` and passes it, instead of the
    report telling a human to "point --run-dir at the directory holding the trainer
    stdout" for a path the launcher itself chose.
    """
    out: dict = {"log_files": [], "steps": [], "warnings": []}
    candidates: list[Path] = []
    if run_dir is not None and run_dir.is_dir():
        candidates = sorted(
            [*run_dir.glob("*.log"), *run_dir.glob("logs/*.log"), *run_dir.glob("**/run.log")],
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
        )[-4:]
    for explicit in extra_logs or []:
        if explicit.is_file() and explicit not in candidates:
            candidates.append(explicit)
    if not candidates:
        return out
    pat_loss = re.compile(r"\b(?:lm[ _-]?loss|loss)\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
    pat_step = re.compile(r"\b(?:iteration|step)\s*[:=]?\s*(\d+)")
    pat_gnorm = re.compile(r"\bgrad[ _-]?norm\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
    # `lr` as well as `learning_rate`: verl prints `train/lr:2.2e-06` and nothing else,
    # so the long spelling alone scraped a learning rate off none of these runs.
    pat_lr = re.compile(
        r"\b(?:learning[ _-]?rate|lr)\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)"
    )
    for path in candidates:
        out["log_files"].append(str(path))
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for line in stage_section(text, stage).splitlines():
            low = line.lower()
            if "nan" in low and ("loss" in low or "grad" in low):
                out["warnings"].append(line.strip()[:200])
            m_loss = pat_loss.search(low)
            if not m_loss:
                continue
            rec: dict = {"loss": float(m_loss.group(1))}
            if (m := pat_step.search(low)):
                rec["step"] = int(m.group(1))
            if (m := pat_gnorm.search(low)):
                rec["grad_norm"] = float(m.group(1))
            if (m := pat_lr.search(low)):
                rec["lr"] = float(m.group(1))
            out["steps"].append(rec)

    # Order by step number, not by which file was read first. Up to four log files are
    # scraped -- typically several ranks, or one run's stdout plus an earlier attempt's
    # -- and concatenating them in mtime order made `losses[0]` and `losses[-1]` (the
    # "loss decreased" check) compare arbitrary points from different files. Lines
    # with no step number cannot be placed on that axis, so they are kept separately
    # instead of being wedged in at the end.
    numbered = [rec for rec in out["steps"] if "step" in rec]
    unnumbered = [rec for rec in out["steps"] if "step" not in rec]
    if numbered:
        first_per_step: dict[int, dict] = {}
        for rec in numbered:
            first_per_step.setdefault(rec["step"], rec)
        repeats = len(numbered) - len(first_per_step)
        out["steps"] = [first_per_step[key] for key in sorted(first_per_step)]
        out["steps_unnumbered"] = unnumbered
        if repeats:
            out["step_numbers_repeated"] = repeats
            out["note"] = (f"{repeats} scraped line(s) repeated a step number (several ranks "
                           f"or several log files); the first of each was kept.")
    return out


# -------------------------------------------------------------------- checks

def check_data(findings: Findings, manifest: dict | None) -> None:
    if not manifest:
        findings.add(WARN, "data", "manifest present",
                     "no data manifest supplied; provenance of the training set is unverified")
        return
    final = manifest.get("final_examples")
    cap = manifest.get("per_group_cap")
    findings.add(INFO, "data", "examples",
                 f"final_examples={final}, per_group_cap={cap}, "
                 f"groups={manifest.get('groups_covered')}, "
                 f"total_tokens={manifest.get('token_stats', {}).get('total_tokens')}")
    if cap == 10 and final != 10778:
        findings.add(WARN, "data", "paper example count",
                     f"cap=10 produced {final}, not the paper's 10,778. The upstream release "
                     f"may have changed, or a filter differs. Not fatal; note it in the writeup.")
    elif cap == 10:
        findings.add(OK, "data", "paper example count", "10,778 examples, matching the paper exactly")
    drops = manifest.get("drop_counters") or {}
    for key in ("drop_slime_contract_mismatch",):
        if drops.get(key):
            findings.add(FAIL, "data", "chat-template contract",
                         f"{key}={drops[key]}; slime's qwen3_5 mask would reject these. "
                         f"The dataset must be rebuilt.")
    if not drops.get("drop_slime_contract_mismatch"):
        findings.add(OK, "data", "chat-template contract", "0 contract mismatches")
    rewritten = manifest.get("rewritten_turn_fraction")
    if isinstance(rewritten, (int, float)):
        findings.add(INFO, "data", "json renormalization",
                     f"{rewritten:.1%} of assistant turns renormalized (expected ~0.57-0.58)")


def check_config(findings: Findings, config: dict | None) -> None:
    if not config:
        findings.add(WARN, "config", "config captured",
                     "no run config supplied; cannot verify the load-bearing flags")
        return
    lm = config.get("loss_mask_type")
    if lm != "qwen3_5":
        findings.add(FAIL, "config", "loss mask",
                     f"loss_mask_type={lm!r}, must be 'qwen3_5'. Anything else mis-segments the "
                     f"Qwen3.5 template and trains on terminal output -- results are invalid.")
    else:
        findings.add(OK, "config", "loss mask", "qwen3_5")

    cc, gdn = config.get("compute_cap"), config.get("gdn_backend")
    if str(cc) == "8.0" and gdn != "fla":
        findings.add(FAIL, "config", "GDN backend",
                     f"compute_cap 8.0 (A100) with gdn_backend={gdn!r}; FlashQLA needs SM90+")
    elif gdn:
        findings.add(OK, "config", "GDN backend", f"{gdn} on compute_cap {cc}")

    tp, pp, cp, dp = (config.get(k) for k in ("tp", "pp", "cp", "dp"))
    gpus = config.get("total_gpus")
    if all(isinstance(v, int) for v in (tp, pp, cp, dp, gpus)):
        if tp * pp * cp * dp != gpus:
            findings.add(FAIL, "config", "parallelism",
                         f"TP{tp}*PP{pp}*CP{cp}*DP{dp}={tp*pp*cp*dp} != {gpus} GPUs")
        else:
            findings.add(OK, "config", "parallelism", f"TP{tp}/PP{pp}/CP{cp}/DP{dp} over {gpus} GPUs")
        seq = config.get("max_seq_len")
        mtpg = config.get("max_tokens_per_gpu")
        if isinstance(seq, int) and isinstance(mtpg, int) and mtpg * cp < seq:
            findings.add(FAIL, "config", "sequence placement",
                         f"max_tokens_per_gpu*CP={mtpg*cp} < max_seq_len={seq}: the longest "
                         f"sequence cannot be placed")


def find_lr_restart(steps: list[dict], *, factor: float = 1.5) -> dict | None:
    """The first material rise in the learning rate *after* it began to decay.

    Warmup rises, so the peak is found first and only what follows it is checked. A
    cosine (or linear, or constant) schedule is non-increasing from its peak onward; a
    rise there means the schedule was rebuilt part-way through the run. That is what
    verl's `resume_mode: auto` does when the same RUN_NAME is relaunched with a
    different `total_epochs` -- `total_training_steps` is re-derived and the curve
    restarted at the resumed step. Measured: step 42 at 3.0e-07 (the min_lr floor),
    step 43 at 2.35e-06.

    `factor` is 1.5 so bf16 logging jitter and a mid-curve plateau stay quiet; the
    observed jump was 7.8x. Only detectable when both launches are in the scraped log,
    which is why the launcher passes the appended `run.log` rather than one run's stdout.
    """
    curve = [
        s for s in steps
        if isinstance(s.get("lr"), float) and math.isfinite(s["lr"]) and s["lr"] > 0
    ]
    if len(curve) < 3:
        return None
    peak = max(range(len(curve)), key=lambda i: curve[i]["lr"])
    for prev, cur in zip(curve[peak:], curve[peak + 1:]):
        if cur["lr"] > prev["lr"] * factor:
            return {
                "prev_step": prev.get("step"), "prev_lr": prev["lr"],
                "step": cur.get("step"), "lr": cur["lr"],
                "factor": cur["lr"] / prev["lr"],
                "peak_step": curve[peak].get("step"), "peak_lr": curve[peak]["lr"],
            }
    return None


def check_training(findings: Findings, training: dict, manifest: dict | None, config: dict | None) -> None:
    steps = training.get("steps") or []
    if training.get("steps_unnumbered"):
        findings.add(INFO, "training", "loss curve",
                     f"{len(training['steps_unnumbered'])} scraped loss line(s) carried no step "
                     f"number and are not on the ordered curve")
    if training.get("step_numbers_repeated"):
        findings.add(INFO, "training", "loss curve", training.get("note", ""))
    if training.get("warnings"):
        findings.add(FAIL, "training", "numerical stability",
                     f"{len(training['warnings'])} log lines mention NaN near loss/grad, e.g. "
                     f"{training['warnings'][0][:120]!r}")
    if not steps:
        findings.add(WARN, "training", "loss curve",
                     "no loss values scraped from logs; the curve could not be checked. "
                     "Point --run-dir at the directory holding the trainer stdout.")
        return
    losses = [s["loss"] for s in steps if math.isfinite(s.get("loss", float("nan")))]
    if not losses:
        findings.add(FAIL, "training", "loss curve", "all scraped loss values were non-finite")
        return
    first, last = losses[0], losses[-1]
    findings.add(INFO, "training", "loss",
                 f"{len(losses)} points, first={first:.4f}, last={last:.4f}, min={min(losses):.4f}")
    if last >= first:
        findings.add(WARN, "training", "loss decreased",
                     f"final loss {last:.4f} >= initial {first:.4f}. With only ~82 steps this can "
                     f"happen from warmup alone, but check LR and the loss mask before accepting it.")
    else:
        findings.add(OK, "training", "loss decreased", f"{first:.4f} -> {last:.4f}")

    gnorms = [s["grad_norm"] for s in steps if isinstance(s.get("grad_norm"), float)]
    if gnorms:
        peak = max(gnorms)
        if peak > 100:
            findings.add(WARN, "training", "grad norm",
                         f"peak grad_norm={peak:.1f} (>100); suspect a bad batch or LR")
        else:
            findings.add(OK, "training", "grad norm", f"peak {peak:.2f}")

    restart = find_lr_restart(steps)
    if restart:
        findings.add(WARN, "training", "lr schedule",
                     f"learning rate rose {restart['factor']:.1f}x after the schedule had "
                     f"started decaying: step {restart['prev_step']} lr {restart['prev_lr']:.3e} "
                     f"-> step {restart['step']} lr {restart['lr']:.3e} (peak "
                     f"{restart['peak_lr']:.3e} at step {restart['peak_step']}). A cosine does "
                     f"not do that. verl's resume_mode=auto rebuilds the schedule over a "
                     f"re-derived total_training_steps, so relaunching the same RUN_NAME with "
                     f"different epochs resumes part-way back up the curve: the steps after "
                     f"this point are a second half-anneal, not a continuation. The launcher "
                     f"gate that refuses this is scripts/resume_guard.py.")
    elif any(isinstance(s.get("lr"), float) for s in steps):
        findings.add(OK, "training", "lr schedule", "monotone after the warmup peak")

    # expected step count from data size and batch size
    if manifest and config:
        n = manifest.get("train_examples")
        gbs = config.get("global_batch_size")
        epochs = config.get("num_epoch", 1)
        if isinstance(n, int) and isinstance(gbs, int) and gbs > 0:
            expected = (n // gbs) * max(1, epochs)
            observed = max((s.get("step", 0) for s in steps), default=0)
            if observed and abs(observed - expected) > max(3, 0.15 * expected):
                findings.add(WARN, "training", "step count",
                             f"observed max step {observed}, expected ~{expected} "
                             f"({n} examples / GBS {gbs} x {epochs} epoch)")
            else:
                findings.add(OK, "training", "step count", f"~{observed} steps (expected ~{expected})")


def check_vs_base(findings: Findings, evals: dict[str, dict | None], tolerance: float = 2.0) -> None:
    """Compare the fine-tuned model to the SAME base model measured on the SAME harness.

    This is the only comparison available for models the paper never published, and
    it is a better comparison than the paper's numbers even for 27B, because it
    cancels out every harness difference.
    """
    base = evals.get("base")
    cand = next((v for k, v in evals.items() if k not in ("base",) and "ref" not in k.lower()), None)
    if not base:
        findings.add(WARN, "eval", "base comparison",
                     "no base-model eval supplied, so 'did SFT help?' cannot be answered from "
                     "measurement. Run 06_eval.py on the un-finetuned checkpoint with --label base.")
        return
    if not cand:
        findings.add(WARN, "eval", "base comparison", "no candidate eval to compare against base")
        return
    for name, b in (base.get("benchmarks") or {}).items():
        c = (cand.get("benchmarks") or {}).get(name)
        if not c or b.get("status") != "scored" or c.get("status") != "scored":
            continue
        bm, cm = b.get("pass_rate_mean"), c.get("pass_rate_mean")
        if not isinstance(bm, (int, float)) or not isinstance(cm, (int, float)):
            continue
        delta = cm - bm
        # Noise band: combine the two stds rather than using a flat threshold.
        noise = max(tolerance, (b.get("pass_rate_std") or 0) + (c.get("pass_rate_std") or 0))
        if delta < -noise:
            findings.add(FAIL, "eval", f"{name} vs measured base",
                         f"{cm} vs base {bm} ({delta:+.2f}), worse than the {noise:.1f}-point noise "
                         f"band. Fine-tuning made this model worse on its own harness.")
        elif delta < 0:
            findings.add(WARN, "eval", f"{name} vs measured base",
                         f"{cm} vs base {bm} ({delta:+.2f}) -- inside noise, but no gain")
        else:
            findings.add(OK, "eval", f"{name} vs measured base",
                         f"{cm} vs base {bm} ({delta:+.2f})")


def scored_benchmarks(evals: dict[str, dict | None]) -> list[str]:
    """Every label/benchmark pair that actually produced a number."""
    out = []
    for label, data in evals.items():
        for name, b in ((data or {}).get("benchmarks") or {}).items():
            if b.get("status") == "scored":
                out.append(f"{label}/{name}")
    return out


def check_benchmark_coverage(findings: Findings, evals: dict[str, dict | None],
                             offline: dict | None) -> None:
    """FAIL when nothing was benchmarked at all.

    This exists because the earlier version only WARNed, and a WARN does not
    block. Worse, 20_run_all.sh always PASSES --eval paths whether or not those
    files exist, so `if not evals` was never even true -- a run whose eval stage
    never happened produced a report with no numbers and a green in_range. That is
    the single worst output this tool can emit: it converts "we could not measure
    the checkpoint" into "the checkpoint is fine".

    The offline eval does NOT lift the FAIL. It is teacher-forced agreement with
    recorded trajectories; it cannot yield a pass rate, and RL on top of an
    unmeasured checkpoint is exactly what this gate is for. On a pod that cannot
    run containers the point is moot anyway -- RL needs the same sandbox eval
    does, so nothing is being blocked that could have run.

    That last sentence is why this finding is in POSTTRAIN_EXEMPT_FAILS: it is true of
    RL and false of DPO, which needs no sandbox. So this FAIL still turns in_range off
    -- the checkpoint really is unmeasured and the report must say so -- but it does not
    claim the checkpoint is unsound, and it does not block the one post-SFT stage a
    container-less pod can actually run.
    """
    scored = scored_benchmarks(evals)
    if scored:
        findings.add(OK, "eval", "benchmark coverage",
                     f"{len(scored)} scored benchmark(s): {', '.join(scored[:6])}")
        return
    supplied = [k for k, v in evals.items() if v]
    detail = (
        "NO benchmark produced a score, so this checkpoint is unmeasured. "
        + (f"Eval files were supplied ({', '.join(supplied)}) but contain no scored "
           f"benchmark -- check 06_eval.py's own output for why. "
           if supplied else
           "No eval results file exists at all; the eval stage did not run. ")
        + ("The container-free offline eval IS present, which is the right fallback "
           "-- but it measures agreement with recorded trajectories, not task "
           "success, so it does not clear this gate. "
           if offline else
           "Not even the container-free fallback ran: `python "
           "scripts/06b_eval_offline.py --model-path <ckpt> --holdout "
           "<pretokenized_holdout.parquet> --out <dir>`. ")
        + "Two things this does NOT mean: it does not mean the checkpoint is unsound "
          "(nothing here inspected the weights), and it does not block DPO, which needs "
          "no container -- see POSTTRAIN_EXEMPT_FAILS and `checkpoint_trustworthy` in "
          "verdict.json. It does mean this checkpoint must be reported as NOT EVALUATED "
          "on agentic benchmarks. The two prerequisites are a container runtime "
          "(`bash scripts/00b_setup_sandbox.sh --diagnose`) and sglang "
          "(`INSTALL_ROLLOUT=1 bash scripts/01b_setup_env_verl.sh`); fix whichever is "
          "missing rather than letting a green verdict imply a measurement that does "
          "not exist."
    )
    findings.add(FAIL, "eval", "benchmark coverage", detail)


def check_offline_eval(findings: Findings, offline: dict | None) -> None:
    """Sanity-check the container-free metrics. Weak signal, but real failure modes."""
    if not offline:
        return
    scoring = offline.get("scoring") or {}
    loss, top1 = scoring.get("loss"), scoring.get("top1_accuracy")
    findings.add(INFO, "offline", "held-out loss",
                 f"loss={loss} ppl={scoring.get('perplexity')} top1={top1} over "
                 f"{scoring.get('supervised_tokens')} supervised tokens in "
                 f"{scoring.get('rows')} rows")
    if isinstance(loss, (int, float)) and not math.isfinite(loss):
        findings.add(FAIL, "offline", "held-out loss",
                     "held-out loss is not finite; the checkpoint is broken")

    delta = offline.get("delta_vs_base") or {}
    if isinstance(delta.get("loss"), (int, float)):
        if delta["loss"] >= 0:
            findings.add(FAIL, "offline", "loss vs base",
                         f"held-out NLL did not improve over the base model "
                         f"({delta['loss']:+.4f}). On the fine-tune's own training "
                         f"distribution that means it did not take: suspect the loss "
                         f"mask, the LR, or a train/serve template mismatch.")
        else:
            findings.add(OK, "offline", "loss vs base",
                         f"held-out NLL improved by {-delta['loss']:.4f} over base")
    elif scoring:
        findings.add(WARN, "offline", "loss vs base",
                     "no --base-model was scored, so the held-out loss is an absolute "
                     "number with nothing to compare it against. Re-run 06b with "
                     "--base-model <the un-finetuned checkpoint>.")

    actions = offline.get("actions") or {}
    if actions.get("available") is False:
        findings.add(WARN, "offline", "action protocol",
                     f"not probed: {actions.get('reason')}")
        return
    if not actions:
        return
    attempted = actions.get("actions_attempted") or 0
    parse = actions.get("parsed_rate")
    trunc = actions.get("truncated_rate") or 0
    findings.add(INFO, "offline", "action agreement",
                 f"parse={parse} commands_exact={actions.get('commands_exact_rate')} "
                 f"first_keystrokes={actions.get('first_keystrokes_exact_rate')} "
                 f"over {attempted} turns")
    # Terminus-2 discards any turn it cannot parse, so a low parse rate is a hard
    # defect whatever the agreement numbers say -- but only once truncation is ruled
    # out, otherwise the finding is about --gen-tokens rather than the model.
    if isinstance(parse, (int, float)) and attempted >= 20:
        if trunc > 0.2:
            findings.add(WARN, "offline", "action protocol",
                         f"{trunc:.0%} of probes hit the generation cap "
                         f"({actions.get('gen_tokens')} tokens), so parse={parse} is a "
                         f"floor, not a measurement. Raise --gen-tokens and re-run.")
        elif parse < 0.9:
            findings.add(FAIL, "offline", "action protocol",
                         f"only {parse:.0%} of generated turns parse as an RST action "
                         f"object, and truncation does not explain it. Terminus-2 drops "
                         f"every unparseable turn, so this checkpoint would waste most "
                         f"of its rollout budget. {actions.get('note')}")
        else:
            findings.add(OK, "offline", "action protocol",
                         f"{parse:.0%} of generated turns parse as an RST action")


def check_eval(findings: Findings, label: str, data: dict | None, is_reference: bool,
               model_key: str | None = None) -> None:
    comparable = has_paper_reference(model_key)
    if not data:
        findings.add(WARN, "eval", f"{label} results",
                     f"no eval results for '{label}'")
        return
    for name, b in (data.get("benchmarks") or {}).items():
        if b.get("status") != "scored":
            findings.add(INFO, "eval", f"{label}/{name}", f"unscorable: {b.get('reason')}")
            continue
        mean, std = b.get("pass_rate_mean"), b.get("pass_rate_std")
        infra = b.get("infra_failure_rate", 0.0)
        findings.add(INFO, "eval", f"{label}/{name}",
                     f"{mean} +- {std} % over {b.get('runs')} runs "
                     f"({b.get('trials_scorable')}/{b.get('trials_total')} scorable)")
        if infra >= 20:
            findings.add(FAIL, "eval", f"{label}/{name} infra rate",
                         f"{infra}% of trials failed for infrastructure reasons. The score is not "
                         f"trustworthy -- fix the sandbox and re-run. Top reasons: "
                         f"{list((b.get('infra_reasons') or {}).items())[:3]}")
        elif infra >= 10:
            findings.add(WARN, "eval", f"{label}/{name} infra rate",
                         f"{infra}% infra failures; interpret with caution. Top reasons: "
                         f"{list((b.get('infra_reasons') or {}).items())[:3]}")
        else:
            findings.add(OK, "eval", f"{label}/{name} infra rate", f"{infra}%")

        # Agent-budget failures are the opposite case: they ARE in the denominator
        # (see rst_common/harbor.py), so they cannot inflate the score -- but a large
        # share of them says the wall clock, not the policy, is what the number
        # measures, and that is a fact about the setup the reader needs.
        budget_rate = b.get("agent_budget_rate")
        if isinstance(budget_rate, (int, float)):
            excl = b.get("pass_rate_mean_budget_excluded")
            delta = (f"; pass rate over non-budget trials only would be {excl}"
                     if isinstance(excl, (int, float)) and isinstance(mean, (int, float))
                     and abs(excl - mean) >= 1 else "")
            reasons = list((b.get("agent_budget_reasons") or {}).items())[:3]
            level = WARN if budget_rate >= 40 else INFO
            findings.add(level, "eval", f"{label}/{name} agent budget",
                         f"{budget_rate}% of trials ended by spending the agent's budget "
                         f"(counted as reward 0, never excluded){delta}. Top: {reasons}"
                         + (". Above 40% the score is dominated by the time limit: check "
                            "--task-timeout and --gen-tokens before reading it as capability."
                            if level == WARN else ""))

        runs = b.get("runs", 0) or 0
        if runs < 3:
            findings.add(WARN, "eval", f"{label}/{name} runs",
                         f"only {runs} run(s); the paper reports mean+-std over 3. "
                         f"A single run on ~100 tasks has several points of noise.")
        if isinstance(std, (int, float)) and std > 8:
            findings.add(WARN, "eval", f"{label}/{name} variance",
                         f"std={std} is high; check for flaky tasks or a saturating timeout")
        elif runs >= 2 and std == 0:
            sampling = ((data.get("protocol") or {}).get("sampling") or {})
            findings.add(WARN, "eval", f"{label}/{name} variance",
                         f"std=0 over {runs} runs. Agentic rollouts essentially never tie "
                         f"exactly, so the likeliest explanations are that the runs were not "
                         f"independent or that decoding is deterministic -- in which case "
                         f"'mean+-std over 3 runs' is one run reported three times and the "
                         f"+-0 must not be presented as a confidence interval. "
                         f"Recorded sampling control: {sampling.get('control') or 'not recorded'}.")

        if not comparable:
            findings.add(INFO, "eval", f"{label}/{name} comparability",
                         f"model_key={model_key!r} has no published reference; the paper only "
                         f"reports Qwen3.5-27B and 122B-A10B. Regression-vs-base and "
                         f"reference-reproduction checks are skipped, not passed.")
            continue
        base = PAPER["base"].get(name)
        if is_reference:
            target = REF_TARGET.get(name)
            if target and isinstance(mean, (int, float)):
                if abs(mean - target) > 6:
                    findings.add(FAIL, "eval", f"reference/{name} harness validity",
                                 f"the released SFT checkpoint scored {mean}, but the paper reports "
                                 f"{target}. That gap indicts THE HARNESS, not the checkpoint. "
                                 f"Do not interpret your own numbers until this is resolved.")
                else:
                    findings.add(OK, "eval", f"reference/{name} harness validity",
                                 f"reference reproduces {mean} vs paper {target}")
        elif base and isinstance(mean, (int, float)):
            if mean < base - 2:
                findings.add(FAIL, "eval", f"{label}/{name} regression",
                             f"{mean}% is below the base model's {base}%. SFT made the model worse; "
                             f"suspect the loss mask, LR, or a train/serve template mismatch.")
            elif mean < base:
                findings.add(WARN, "eval", f"{label}/{name} no gain",
                             f"{mean}% vs base {base}% -- within noise but no improvement")
            else:
                findings.add(OK, "eval", f"{label}/{name} vs base",
                             f"{mean}% vs base {base}% (+{mean-base:.2f})")


# -------------------------------------------------------------------- render

def render(args, config, manifest, training, evals, findings: Findings,
           offline: dict | None = None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L: list[str] = []
    A = L.append

    model_key = args.model_key or (config or {}).get("model_key") or "qwen3.5-27b"
    A(f"# RST SFT experiment report — `{model_key}`\n")
    A(f"_Generated {now} by `scripts/14_make_report.py`._\n")

    verdict = findings.worst()
    banner = {
        FAIL: "**VERDICT: FAIL — at least one check invalidates the result. See Findings.**",
        WARN: "**VERDICT: WARN — results usable but with caveats. See Findings.**",
        OK: "**VERDICT: OK — all mechanical checks passed.**",
        INFO: "**VERDICT: INCONCLUSIVE — too little evidence was supplied to check.**",
    }[verdict]
    A(banner + "\n")
    A(f"Checks: {findings.count(FAIL)} FAIL, {findings.count(WARN)} WARN, {findings.count(OK)} OK.\n")

    # ---- headline table
    A("## Results\n")
    A("Pass rate %, mean ± std over independent runs. Two kinds of failure are counted")
    A("differently, and the difference is load-bearing:\n")
    A("- **Infrastructure** (Docker, registry, DNS, disk) — the trial was never measured, so")
    A("  it is excluded from the denominator and reported as its own rate.")
    A("- **Agent budget** (wall clock spent, command refused as too long) — the *policy*")
    A("  failed, so it scores 0 and stays in the denominator. Excluding these would inflate")
    A("  the score most on the hardest tasks, which is where the whole claim lives.\n")
    cols = ["tb-hard", "tb2", "lhtb"]
    A("| model | " + " | ".join(cols) + " |")
    A("|---|" + "---|" * len(cols))
    if not has_paper_reference(model_key):
        A(f"> The paper reports numbers only for Qwen3.5-27B and 122B-A10B. This run is")
        A(f"> `{model_key}`, so the paper rows below are context, **not a target**, and no")
        A(f"> regression-vs-base verdict is claimed.\n")
    for key, label in (("base", "Qwen3.5-27B base *(paper)*"),
                       ("sft_r1", "paper SFT round 1"),
                       ("sft_r3", "paper SFT round 3"),
                       ("rl", "paper RL")):
        A(f"| {label} | " + " | ".join(f"{PAPER[key].get(c, '—')}" for c in cols) + " |")
    for label, data in evals.items():
        cells = []
        for c in cols:
            b = ((data or {}).get("benchmarks") or {}).get(c)
            if not b:
                cells.append("not run")
            elif b.get("status") != "scored":
                cells.append("unscorable")
            else:
                cells.append(f"**{b['pass_rate_mean']} ± {b['pass_rate_std']}**")
        A(f"| **this run: {label}** | " + " | ".join(cells) + " |")
    A("")
    A("> LHTB is not scorable with this harness: upstream withholds its verifiers")
    A("> (0/46 tasks ship `tests/`). A blank or `unscorable` cell there is correct;")
    A("> a number would be fabricated.\n")

    if not scored_benchmarks(evals):
        A("### ⚠️ This checkpoint was NOT benchmarked\n")
        A("Every cell above is `not run` or `unscorable`. **There is no pass rate for this")
        A("checkpoint.** Say exactly that when reporting it. Do not describe it as")
        A("promising, working, or good: none of those words are supported by anything")
        A("measured here.\n")
        A("The usual cause is that agentic eval needs to build a task Dockerfile and drive")
        A("a tmux session inside it, and this machine could not. `bash")
        A("scripts/00b_setup_sandbox.sh --diagnose` names the specific reason and the")
        A("single change that fixes it; `BACKENDS.md` lists the backends that run the")
        A("container somewhere else, which need no local container privilege at all.\n")

    # ---- offline (container-free) eval
    if offline:
        s = offline.get("scoring") or {}
        a = offline.get("actions") or {}
        d = offline.get("delta_vs_base") or {}
        A("## Offline eval (container-free) — a weaker signal, reported as such\n")
        A("From `scripts/06b_eval_offline.py`. Teacher-forced scoring against held-out")
        A("expert trajectories plus greedy action-protocol agreement. **This is not a")
        A("benchmark and must never be quoted as one**: it cannot observe whether the")
        A("agent recovers from a wrong command, which is most of the difficulty.\n")
        A("| metric | value | reads as |")
        A("|---|---|---|")
        A(f"| held-out loss | {s.get('loss', '—')} | lower is better; "
          f"ppl {s.get('perplexity', '—')} |")
        A(f"| next-token top-1 | {s.get('top1_accuracy', '—')} | over "
          f"{s.get('supervised_tokens', '—')} supervised tokens only |")
        if d:
            A(f"| Δ loss vs base | {d.get('loss', '—'):+} | {d.get('reading', '')} |"
              if isinstance(d.get("loss"), (int, float)) else
              f"| Δ loss vs base | — | {d.get('reading', '')} |")
        if a.get("available") is False:
            A(f"| action protocol | not probed | {a.get('reason', '')} |")
        elif a:
            A(f"| action parse rate | {a.get('parsed_rate', '—')} | Terminus-2 discards "
              f"any turn it cannot parse |")
            A(f"| commands exact | {a.get('commands_exact_rate', '—')} | same keystroke "
              f"list as the expert |")
            A(f"| first keystrokes | {a.get('first_keystrokes_exact_rate', '—')} | same "
              f"opening move |")
            A(f"| truncated | {a.get('truncated_rate', '—')} | hit the "
              f"{a.get('gen_tokens', '—')}-token cap; inflates every miss above |")
        A("")
        if a.get("note"):
            A(f"> {a['note']}\n")

    # ---- findings
    A("## Findings\n")
    A("| level | area | check | detail |")
    A("|---|---|---|---|")
    order = {FAIL: 0, WARN: 1, OK: 2, INFO: 3}
    for level, area, check, detail in sorted(findings.rows, key=lambda r: order.get(r[0], 9)):
        mark = {FAIL: "🔴 FAIL", WARN: "🟡 WARN", OK: "🟢 OK", INFO: "ℹ️ INFO"}[level]
        A(f"| {mark} | {area} | {check} | {detail.replace('|', '\\|')} |")
    A("")

    # ---- analysis placeholder
    A("## Analysis\n")
    if verdict in (FAIL, WARN):
        A("<!-- OPERATOR: this section is yours. For every 🔴 FAIL and 🟡 WARN above, work out")
        A("     the actual cause and write it here. Rules:")
        A("       - Name the evidence (log line, file, command output). No speculation")
        A("         presented as fact; if you are guessing, write \"hypothesis:\".")
        A("       - Separate a correctness problem from an infrastructure problem, and say")
        A("         which one you are claiming.")
        A("       - If a check is a false positive, say why the threshold is wrong here.")
        A("       - State explicitly whether the headline numbers are trustworthy. -->\n")
        A("_(to be completed by the operator)_\n")
    else:
        A("All mechanical checks passed. Note anything qualitative worth recording anyway.\n")
        A("_(optional)_\n")

    # ---- reproducibility
    A("## Configuration\n")
    if config:
        A("```json")
        A(json.dumps(config, indent=2, sort_keys=True))
        A("```\n")
    else:
        A("_No run config was captured — `20_run_all.sh` writes one to `run_config.json`._\n")

    A("## Data provenance\n")
    if manifest:
        ts = manifest.get("token_stats") or {}
        A(f"- source: `{manifest.get('source_dataset')}`")
        A(f"- gate: {manifest.get('trajectories_total')} trajectories -> "
          f"{manifest.get('eligible_after_gate')} eligible over {manifest.get('eligible_groups')} groups")
        A(f"- per-group cap {manifest.get('per_group_cap')} -> {manifest.get('selected')} selected -> "
          f"**{manifest.get('final_examples')} final** "
          f"({manifest.get('train_examples')} train / {manifest.get('holdout_examples')} holdout)")
        A(f"- tokens: total {ts.get('total_tokens')}, p50 {ts.get('p50')}, p99 {ts.get('p99')}, max {ts.get('max')}")
        A(f"- model mix: {manifest.get('model_mix')}")
        A(f"- drops: {manifest.get('drop_counters')}\n")
    else:
        A("_No data manifest supplied._\n")

    A("## Training\n")
    steps = training.get("steps") or []
    if steps:
        losses = [s["loss"] for s in steps if math.isfinite(s.get("loss", float("nan")))]
        A(f"- scraped {len(steps)} log records from {training.get('log_files')}")
        if losses:
            A(f"- loss: first {losses[0]:.4f} -> last {losses[-1]:.4f} (min {min(losses):.4f})")
        gn = [s["grad_norm"] for s in steps if isinstance(s.get("grad_norm"), float)]
        if gn:
            A(f"- grad norm: peak {max(gn):.2f}, final {gn[-1]:.2f}")
        A("")
        A("| step | loss | grad_norm | lr |")
        A("|---|---|---|---|")
        keep = steps[:: max(1, len(steps) // 25)]
        for s in keep:
            A(f"| {s.get('step','—')} | {s.get('loss','—')} | {s.get('grad_norm','—')} | {s.get('lr','—')} |")
        A("")
    else:
        A("_No loss curve could be scraped. Log format is not a contract; if slime changed its")
        A("stdout format, read the wandb run instead and paste the curve summary here._\n")

    A("## Eval detail\n")
    for label, data in evals.items():
        A(f"### {label}\n")
        if not data:
            A("_missing_\n"); continue
        proto = data.get("protocol") or {}
        A(f"- endpoint `{proto.get('endpoint')}`, agent `{proto.get('agent')}`, "
          f"sandbox `{proto.get('sandbox')}`, runs {proto.get('runs')}, "
          f"concurrency {proto.get('n_concurrent')}")
        sampling = proto.get("sampling") or {}
        if sampling:
            A(f"- sampling: {sampling.get('control') or sampling}")
        for name, b in (data.get("benchmarks") or {}).items():
            if b.get("status") != "scored":
                A(f"- **{name}**: unscorable — {b.get('reason')}")
                continue
            # .get, not [...]: this tool's contract is that a report is produced even
            # when the run went wrong, so a results.json missing a key must degrade
            # to a gap in one line rather than take the whole report down.
            A(f"- **{name}**: {b.get('pass_rate_mean')} ± {b.get('pass_rate_std')} % "
              f"(per-run {b.get('per_run_pass_rate')}), pass@{b.get('runs')} = "
              f"{b.get('pass_at_k')}%, infra {b.get('infra_failure_rate')}%, "
              f"agent budget {b.get('agent_budget_rate')}%")
            never = b.get("tasks_never_solved") or []
            if never:
                A(f"  - never solved in any run ({len(never)}): "
                  f"`{', '.join(never[:12])}`{' …' if len(never) > 12 else ''}")
            if b.get("infra_reasons"):
                A(f"  - infra reasons (excluded from denominator): {b['infra_reasons']}")
            if b.get("agent_budget_reasons"):
                A(f"  - agent-budget reasons (scored 0, in denominator): "
                  f"{b['agent_budget_reasons']}; pass rate excluding them would be "
                  f"{b.get('pass_rate_mean_budget_excluded')}%")
        A("")

    A("## How to reproduce\n")
    A("```bash")
    A("git clone https://github.com/k1ssloo/RST-Train && cd RST-Train")
    A("export BASE_FOLDER=/shared/rst")
    A("bash scripts/20_run_all.sh          # preflight -> env -> data -> train -> eval -> report")
    A("```\n")
    A("Caveats that apply to every number above:\n")
    A("- Trajectories are **reward-verified, not exact-environment-replay verified** "
      "(in a 500-sample check only 46 instructions mapped exactly to a public task).")
    A("- `sft_r3` in the paper is *three cumulative synthesis rounds*; a single SFT pass "
      "should be compared against `sft_r1`.")
    A("- LHTB cannot be scored locally (verifiers withheld upstream).")
    return "\n".join(L) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-key", default=None,
                   help="key in configs/models.json; decides whether paper numbers apply")
    p.add_argument("--run-dir", type=Path, default=None, help="training run dir (for logs)")
    p.add_argument("--train-log", type=Path, action="append", default=[], metavar="PATH",
                   help="repeatable. The trainer's stdout, when it is not inside "
                        "--run-dir -- which it is not under verl/FSDP, where the run dir "
                        "holds only global_step_*/. 20_run_all.sh passes its own "
                        "logs/run.log.")
    p.add_argument("--train-stage", default="train", metavar="NAME",
                   help="which `=== STAGE <name>` section of a 20_run_all.sh log holds "
                        "the trainer being reported on (train, rl, dpo). One appended log "
                        "holds all of them, and DPO's loss is log 2 by construction.")
    p.add_argument("--run-config", type=Path, default=None, help="run_config.json from 20_run_all.sh")
    p.add_argument("--data-manifest", type=Path, default=None)
    p.add_argument("--eval", action="append", default=[], metavar="LABEL=PATH",
                   help="repeatable, e.g. --eval mine=.../results.json --eval reference=...")
    p.add_argument("--offline-eval", type=Path, default=None,
                   help="offline_results.json from 06b_eval_offline.py -- the "
                        "container-free fallback. Reported as a weaker signal; it "
                        "never substitutes for a benchmark score.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--verdict-json", type=Path, default=None,
                   help="also write a small machine-readable verdict for orchestration")
    args = p.parse_args()

    config = read_json(args.run_config)
    manifest = read_json(args.data_manifest)
    training = parse_training_log(args.run_dir, args.train_log, stage=args.train_stage)

    evals: dict[str, dict | None] = {}
    for item in args.eval:
        if "=" not in item:
            print(f"ignoring --eval {item!r} (want LABEL=PATH)")
            continue
        label, path = item.split("=", 1)
        evals[label] = read_json(Path(path))

    offline = read_json(args.offline_eval)

    findings = Findings()
    check_config(findings, config)
    check_data(findings, manifest)
    check_training(findings, training, manifest, config)
    model_key = args.model_key or (config or {}).get("model_key")
    for label, data in evals.items():
        check_eval(findings, label, data, is_reference=("ref" in label.lower()), model_key=model_key)
    check_vs_base(findings, evals)
    check_benchmark_coverage(findings, evals, offline)
    check_offline_eval(findings, offline)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(args, config, manifest, training, evals, findings, offline),
                        encoding="utf-8")

    # "In range" is deliberately narrow: no FAIL findings at all. WARNs are for a
    # human to weigh; FAILs mean a number is either wrong or untrustworthy, and
    # nothing downstream (least of all RL) should be built on top of one.
    fails = [r for r in findings.rows if r[0] == FAIL]
    in_range = not fails
    # See POSTTRAIN_EXEMPT_FAILS. Same rows, different question: "is this a sound
    # checkpoint?" rather than "is every number in range?". The container-free DPO
    # stage gates on this one so that an unmeasurable pod is not also an untrainable one.
    blocking = [r for r in fails if (r[1], r[2]) not in POSTTRAIN_EXEMPT_FAILS]
    checkpoint_trustworthy = not blocking
    if args.verdict_json:
        args.verdict_json.parent.mkdir(parents=True, exist_ok=True)
        args.verdict_json.write_text(json.dumps({
            "verdict": findings.worst(),
            "in_range": in_range,
            "checkpoint_trustworthy": checkpoint_trustworthy,
            "n_fail": len(fails),
            "n_blocking_fail": len(blocking),
            "n_warn": findings.count(WARN),
            "model_key": model_key,
            "fail_reasons": [f"[{a}] {c}: {d}" for _, a, c, d in fails],
            "blocking_fail_reasons": [f"[{a}] {c}: {d}" for _, a, c, d in blocking],
            "exempt_fail_reasons": [f"[{a}] {c}: {d}" for _, a, c, d in fails
                                    if (a, c) in POSTTRAIN_EXEMPT_FAILS],
            "report": str(args.out),
        }, indent=2) + "\n", encoding="utf-8")

    print(f"verdict={findings.worst()} in_range={in_range} "
          f"checkpoint_trustworthy={checkpoint_trustworthy}  FAIL={findings.count(FAIL)} "
          f"WARN={findings.count(WARN)} OK={findings.count(OK)}")
    if not in_range and checkpoint_trustworthy:
        print("  note: every FAIL is about benchmark COVERAGE, not the checkpoint. The "
              "container-free DPO stage still runs; report both checkpoints as NOT "
              "agentically evaluated.")
    for level, area, check, detail in findings.rows:
        if level in (FAIL, WARN):
            print(f"  {level:4s} [{area}] {check}: {detail[:150]}")
    print(f"wrote {args.out}")
    # non-zero on FAIL so the orchestrator can react
    return 2 if findings.count(FAIL) else 0


if __name__ == "__main__":
    raise SystemExit(main())
