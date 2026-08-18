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
from datetime import datetime, timezone
from pathlib import Path

PAPER = {
    "base":   {"tb2": 41.20, "tb-hard": 22.67, "lhtb": 18.10},
    "sft_r1": {"tb2": 42.32, "tb-hard": 23.00, "lhtb": 21.32},
    "sft_r3": {"tb2": 47.94, "tb-hard": 28.33, "lhtb": 22.44},
    "rl":     {"tb2": 49.44, "tb-hard": 32.00, "lhtb": 22.07},
}
REF_TARGET = {"tb2": 47.94, "tb-hard": 28.33}   # released SFT ckpt should land near here

FAIL, WARN, OK, INFO = "FAIL", "WARN", "OK", "INFO"


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


def parse_training_log(run_dir: Path) -> dict:
    """Best-effort scrape of slime/Megatron stdout for loss / grad-norm / lr.

    Log format is not a contract, so everything here is optional and the report
    says so rather than pretending a missing field is a zero.
    """
    out: dict = {"log_files": [], "steps": [], "warnings": []}
    if not run_dir.is_dir():
        return out
    candidates = sorted(
        [*run_dir.glob("*.log"), *run_dir.glob("logs/*.log"), *run_dir.glob("**/run.log")],
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
    )[-4:]
    pat_loss = re.compile(r"\b(?:lm[ _-]?loss|loss)\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
    pat_step = re.compile(r"\b(?:iteration|step)\s*[:=]?\s*(\d+)")
    pat_gnorm = re.compile(r"\bgrad[ _-]?norm\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
    pat_lr = re.compile(r"\blearning[ _-]?rate\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
    for path in candidates:
        out["log_files"].append(str(path))
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
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


def check_training(findings: Findings, training: dict, manifest: dict | None, config: dict | None) -> None:
    steps = training.get("steps") or []
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


def check_eval(findings: Findings, label: str, data: dict | None, is_reference: bool) -> None:
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

        if b.get("runs", 0) < 3:
            findings.add(WARN, "eval", f"{label}/{name} runs",
                         f"only {b.get('runs')} run(s); the paper reports mean+-std over 3. "
                         f"A single run on ~100 tasks has several points of noise.")
        if isinstance(std, (int, float)) and std > 8:
            findings.add(WARN, "eval", f"{label}/{name} variance",
                         f"std={std} is high; check for flaky tasks or a saturating timeout")

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

def render(args, config, manifest, training, evals, findings: Findings) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L: list[str] = []
    A = L.append

    A(f"# RST -> Qwen3.5-27B experiment report\n")
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
    A("Pass rate %, mean ± std over independent runs. Infrastructure failures are excluded")
    A("from the denominator and reported separately — they are not model errors.\n")
    cols = ["tb-hard", "tb2", "lhtb"]
    A("| model | " + " | ".join(cols) + " |")
    A("|---|" + "---|" * len(cols))
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
        for name, b in (data.get("benchmarks") or {}).items():
            if b.get("status") != "scored":
                A(f"- **{name}**: unscorable — {b.get('reason')}")
                continue
            A(f"- **{name}**: {b['pass_rate_mean']} ± {b['pass_rate_std']} % "
              f"(per-run {b['per_run_pass_rate']}), pass@{b['runs']} = {b.get('pass_at_k')}%, "
              f"infra {b['infra_failure_rate']}%")
            never = b.get("tasks_never_solved") or []
            if never:
                A(f"  - never solved in any run ({len(never)}): "
                  f"`{', '.join(never[:12])}`{' …' if len(never) > 12 else ''}")
            if b.get("infra_reasons"):
                A(f"  - infra reasons: {b['infra_reasons']}")
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
    p.add_argument("--run-dir", type=Path, default=None, help="training run dir (for logs)")
    p.add_argument("--run-config", type=Path, default=None, help="run_config.json from 20_run_all.sh")
    p.add_argument("--data-manifest", type=Path, default=None)
    p.add_argument("--eval", action="append", default=[], metavar="LABEL=PATH",
                   help="repeatable, e.g. --eval mine=.../results.json --eval reference=...")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    config = read_json(args.run_config)
    manifest = read_json(args.data_manifest)
    training = parse_training_log(args.run_dir) if args.run_dir else {"steps": [], "log_files": [], "warnings": []}

    evals: dict[str, dict | None] = {}
    for item in args.eval:
        if "=" not in item:
            print(f"ignoring --eval {item!r} (want LABEL=PATH)")
            continue
        label, path = item.split("=", 1)
        evals[label] = read_json(Path(path))

    findings = Findings()
    check_config(findings, config)
    check_data(findings, manifest)
    check_training(findings, training, manifest, config)
    if not evals:
        findings.add(WARN, "eval", "any results", "no --eval supplied; nothing was benchmarked")
    for label, data in evals.items():
        check_eval(findings, label, data, is_reference=("ref" in label.lower()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(args, config, manifest, training, evals, findings), encoding="utf-8")

    print(f"verdict={findings.worst()}  FAIL={findings.count(FAIL)} WARN={findings.count(WARN)} "
          f"OK={findings.count(OK)}")
    for level, area, check, detail in findings.rows:
        if level in (FAIL, WARN):
            print(f"  {level:4s} [{area}] {check}: {detail[:150]}")
    print(f"wrote {args.out}")
    # non-zero on FAIL so the orchestrator can react
    return 2 if findings.count(FAIL) else 0


if __name__ == "__main__":
    raise SystemExit(main())
