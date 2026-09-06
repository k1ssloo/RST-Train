#!/usr/bin/env python3
"""Curate verifier-passed trajectories before they become SFT data.

    python scripts/03g_curate_sft.py \
        --parquet data/sft-v1-cap10/rst_sft_train.parquet \
        --parquet data/openthoughts-agent-v1/ota_sft_train.parquet \
        --out-dir data/curated-v1                      # + RST_JUDGE_* in the env, optionally

WHY A SECOND GATE AFTER THE VERIFIER
    Every trajectory reaching this script already passed its task's own tests; that is
    the hard gate and nothing here overrides it. But reward 1 is necessary, not
    sufficient, as a training target: an episode that re-ran the same failing command
    nine times, declared the task complete twice and then finished on the tenth try
    is a *success* that teaches thrashing. The paper's pipeline kept everything with
    reward 1; this is the knob it did not have.

TWO STAGES, DELIBERATELY UNEQUAL
    1. Deterministic features over every row -- repeated commands, error-laden
       observations, harness complaints, false completion claims, turn count relative
       to the task's siblings. Milliseconds per trajectory, no network, and they decide
       the clear cases: `clear_keep` / `clear_drop`. Thresholds are constants below
       and are written into the manifest, so a band is a checkable claim.
    2. An LLM judge (`rst_common/judge.py`) on the `borderline` band ONLY, plus a small
       random audit sample of `clear_keep` for calibration. Cached, budgeted, off the
       training path. With no `RST_JUDGE_BASE_URL` in the environment the judge is a
       `NullJudge`: the borderline falls back to `--borderline-default` and the
       manifest says so. The judge never sees the clear bands -- that is the whole
       speed argument: on the 10,778-row RST set the borderline is a few hundred rows,
       and the judge cost is a few hundred cached calls, not ten thousand.

WHAT IT NEVER DOES
    Edit a message. This script selects rows; the canonical form, the mask and the
    tokenizer contract are untouched, so the curated parquet is byte-for-byte a subset
    of its input and `15_export_pretokenized.py` consumes it unchanged.

OUTPUT
    <out-dir>/curation.parquet          every input row: features, band, judge verdict, decision
    <out-dir>/<stem>_curated.parquet    per input parquet: the kept rows, same columns
    <out-dir>/manifest.json             counts per band and decision, thresholds, judge stats,
                                        judge-vs-heuristic agreement on the audit sample
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rst_common.judge import JudgeRequest, judge_from_env  # noqa: E402
from siblings import load_script  # noqa: E402

# One action parser for both dialects -- the RST/OpenThoughts/Nemotron JSON contract
# and TMax's native <tool_call> form -- shared with the offline eval rather than
# re-implemented, so "what did this turn do" means the same thing in both places.
_probe = load_script("06b_eval_offline")
action_of = _probe.action_of
keystrokes_of = _probe.keystrokes_of
is_complete = _probe.task_complete_of

# ------------------------------------------------------------------ features

# Observations that say the previous command failed. Deliberately generic: the point
# is a *ratio* over the whole episode, not a verdict on any one line.
ERROR_PATTERNS = re.compile(
    r"command not found|No such file or directory|Traceback \(most recent call last\)|"
    r"Permission denied|[Ss]yntax error|SyntaxError|ModuleNotFoundError|E: Unable to locate|"
    r"\bfatal:|\b[Ee]rror:|FAILED|Segmentation fault|Killed\b",
)
HARNESS_COMPLAINT = re.compile(r"\APrevious response had (warnings|parsing errors)")


def parse_action(content: str) -> dict[str, Any] | None:
    return action_of(content)


def completion_retractions(claims: list[bool]) -> int:
    """How often the agent said "done" and then kept working.

    Terminus-2 answers a completion claim with "are you sure?", and a clean success
    ends with the claim repeated -- so two consecutive claims are the NORMAL ending,
    not a flip-flop. What marks thrashing is a claim followed by a turn that is not
    one: the agent declared victory, was shown it had not won, and carried on.
    """
    return sum(1 for a, b in zip(claims, claims[1:]) if a and not b)


def trajectory_features(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Cheap, deterministic signals about HOW a success was reached."""
    assistant = [m["content"] for m in messages if m["role"] == "assistant"]
    observations = [m["content"] for m in messages[1:] if m["role"] == "user"]
    actions = [parse_action(c) for c in assistant]

    all_keys: list[str] = []
    batches: list[tuple[str, ...]] = []
    empty_turns = 0
    for action in actions:
        keys = [k.strip() for k in keystrokes_of(action)]
        if not keys:
            empty_turns += 1
        all_keys.extend(k for k in keys if k)
        batches.append(tuple(keys))
    claims = [is_complete(a) for a in actions]
    repeated = len(all_keys) - len(set(all_keys))
    run, best_run = 1, 1
    for prev, cur in zip(batches, batches[1:]):
        run = run + 1 if cur and cur == prev else 1
        best_run = max(best_run, run)

    n_obs = len(observations)
    return {
        "n_turns": len(assistant),
        "n_obs": n_obs,
        "n_commands": len(all_keys),
        "n_empty_command_turns": empty_turns,
        "unparseable_turns": sum(a is None for a in actions),
        "repeat_ratio": round(repeated / len(all_keys), 4) if all_keys else 0.0,
        "max_consecutive_repeat": best_run,
        "error_obs_ratio": round(sum(bool(ERROR_PATTERNS.search(o)) for o in observations)
                                 / n_obs, 4) if n_obs else 0.0,
        "complaint_ratio": round(sum(bool(HARNESS_COMPLAINT.match(o)) for o in observations)
                                 / n_obs, 4) if n_obs else 0.0,
        "task_complete_count": sum(claims),
        "completion_retractions": completion_retractions(claims),
        "final_complete": claims[-1] if claims else False,
    }


# ------------------------------------------------------------------ banding

THRESHOLDS: dict[str, Any] = {
    # clear_drop: the episode is mostly noise even though it ended in reward 1
    "drop_repeat_ratio": 0.50, "drop_repeat_min_commands": 6,
    "drop_error_obs_ratio": 0.60, "drop_error_min_obs": 5,
    "drop_completion_retractions": 2,
    "drop_max_consecutive_repeat": 4,
    # borderline: something is off, but a human (or the judge) could go either way
    "warn_repeat_ratio": 0.25,
    "warn_error_obs_ratio": 0.35,
    "warn_completion_retractions": 1,
    "warn_complaint_ratio": 0.30,
    "warn_turn_z": 2.0, "turn_z_min_group": 3,
    "warn_max_consecutive_repeat": 3,
}


def band_of(f: dict[str, Any], turn_z: float | None, t: dict[str, Any] = THRESHOLDS
            ) -> tuple[str, list[str]]:
    """`clear_drop` | `borderline` | `clear_keep`, with the reasons that put it there."""
    reasons: list[str] = []
    if f["unparseable_turns"]:
        reasons.append("unparseable_turn")
    if f["n_commands"] >= t["drop_repeat_min_commands"] and f["repeat_ratio"] >= t["drop_repeat_ratio"]:
        reasons.append("repeat_ratio")
    if f["n_obs"] >= t["drop_error_min_obs"] and f["error_obs_ratio"] >= t["drop_error_obs_ratio"]:
        reasons.append("error_obs_ratio")
    if f["completion_retractions"] >= t["drop_completion_retractions"]:
        reasons.append("repeated_false_completion")
    if f["max_consecutive_repeat"] >= t["drop_max_consecutive_repeat"]:
        reasons.append("consecutive_repeat")
    if reasons:
        return "clear_drop", reasons

    if f["repeat_ratio"] >= t["warn_repeat_ratio"]:
        reasons.append("repeat_ratio")
    if f["error_obs_ratio"] >= t["warn_error_obs_ratio"]:
        reasons.append("error_obs_ratio")
    if f["completion_retractions"] >= t["warn_completion_retractions"]:
        reasons.append("completion_claimed_then_continued")
    if f["complaint_ratio"] >= t["warn_complaint_ratio"]:
        reasons.append("harness_complaints")
    if f["max_consecutive_repeat"] >= t["warn_max_consecutive_repeat"]:
        reasons.append("consecutive_repeat")
    if turn_z is not None and turn_z >= t["warn_turn_z"]:
        reasons.append("far_longer_than_siblings")
    # `final_complete` is recorded but not banded on: the verifier passed, and some
    # harnesses end an episode on the first claim while others require a repeat.
    return ("borderline" if reasons else "clear_keep"), reasons


def turn_z_scores(rows: list[dict[str, Any]], min_group: int) -> dict[str, float | None]:
    """z-score of each trajectory's turn count within its task group; None if too few."""
    by_group: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_group[row["task_group_id"]].append(row["features"]["n_turns"])
    out: dict[str, float | None] = {}
    for row in rows:
        turns = by_group[row["task_group_id"]]
        if len(turns) < min_group:
            out[row["trajectory_id"]] = None
            continue
        mean = statistics.fmean(turns)
        sd = statistics.pstdev(turns)
        out[row["trajectory_id"]] = (row["features"]["n_turns"] - mean) / sd if sd else 0.0
    return out


# ------------------------------------------------------------------ the judge

JUDGE_SYSTEM = (
    "You review transcripts of an AI agent solving a Linux terminal task. The task's own "
    "tests PASSED, so correctness is not in question. Judge whether the transcript is a "
    "good demonstration to train another agent on: efficient, purposeful, recovering "
    "cleanly from errors, not thrashing or repeating itself, not claiming completion "
    "before it is done. Answer with a JSON object only."
)
JUDGE_KEYS = ("keep", "efficiency", "teaches_bad_habit", "reason")
JUDGE_FORMAT = (
    '{"keep": true|false, "efficiency": 1-5, "teaches_bad_habit": true|false, '
    '"reason": "<one sentence>"}'
)


def condense(messages: list[dict[str, str]], *, instruction_chars: int = 3000,
             turn_chars: int = 220, obs_chars: int = 300, keep_turns: int = 6) -> str:
    """A judge-sized view of an episode: instruction, then per-turn plan/commands/result.

    Long episodes keep the first and last `keep_turns` turns and say how many were
    elided; the judge is asked about the shape of the episode, not its every byte.
    """
    lines = ["TASK:", messages[0]["content"][:instruction_chars], "", "EPISODE:"]
    turns: list[str] = []
    observation_after: dict[int, str] = {}
    assistant_index = -1
    for message in messages[1:]:
        if message["role"] == "assistant":
            assistant_index += 1
            action = parse_action(message["content"])
            plan = str((action or {}).get("plan") or (action or {}).get("analysis") or "")[:turn_chars]
            cmds = " ; ".join(k.replace("\n", "\\n") for k in keystrokes_of(action))[:turn_chars]
            flag = " [claims task complete]" if is_complete(action) else ""
            turns.append(f"T{assistant_index + 1}: plan: {plan}\n     cmds: {cmds or '(none)'}{flag}")
        else:
            observation_after[assistant_index] = message["content"][:obs_chars].replace("\n", " ⏎ ")
    for i, text in enumerate(turns):
        obs = observation_after.get(i)
        turns[i] = text + (f"\n     result: {obs}" if obs else "")
    if len(turns) > 2 * keep_turns:
        elided = len(turns) - 2 * keep_turns
        turns = turns[:keep_turns] + [f"... {elided} turns elided ..."] + turns[-keep_turns:]
    lines.extend(turns)
    return "\n".join(lines)


def judge_request(row: dict[str, Any]) -> JudgeRequest:
    f = row["features"]
    user = (
        condense(row["messages"])
        + f"\n\nSIGNALS: turns={f['n_turns']} repeat_ratio={f['repeat_ratio']} "
          f"error_obs_ratio={f['error_obs_ratio']} completion_retractions={f['completion_retractions']} "
          f"heuristic_reasons={','.join(row['reasons']) or 'none'}"
        + f"\n\nRespond with exactly: {JUDGE_FORMAT}"
    )
    return JudgeRequest(item_id=row["trajectory_id"], system=JUDGE_SYSTEM, user=user,
                        required_keys=JUDGE_KEYS)


def judge_says_keep(data: dict[str, Any]) -> bool:
    keep = data.get("keep")
    if isinstance(keep, str):
        keep = keep.strip().lower() == "true"
    return bool(keep)


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, action="append", required=True,
                    help="a canonical `messages` parquet (any 03* builder's output); repeatable")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--borderline-default", choices=("keep", "drop"), default="keep",
                    help="what a borderline row becomes when the judge is disabled, out of "
                         "budget or unparseable. keep: the verifier said yes and nothing "
                         "contradicted it. drop: quality over quantity.")
    ap.add_argument("--audit-sample", type=int, default=40,
                    help="clear_keep rows also shown to the judge, for calibration only")
    ap.add_argument("--max-judge-items", type=int, default=0,
                    help="cap on borderline rows sent to the judge (0 = all)")
    ap.add_argument("--seed", type=int, default=1228)
    args = ap.parse_args()

    import pandas as pd

    rows: list[dict[str, Any]] = []
    frames: dict[Path, Any] = {}
    for path in args.parquet:
        if not path.is_file():
            sys.exit(f"missing input: {path}")
        frame = pd.read_parquet(path)
        for column in ("messages", "trajectory_id", "task_group_id"):
            if column not in frame.columns:
                sys.exit(f"{path} has no `{column}` column; this takes the canonical "
                         f"messages parquet, not the pretokenized one")
        frames[path] = frame
        for index, record in enumerate(frame.itertuples(index=False)):
            messages = [dict(m) for m in record.messages]
            rows.append({
                "source": str(path), "index": index,
                "trajectory_id": str(record.trajectory_id),
                "task_group_id": str(record.task_group_id),
                "messages": messages,
                "features": trajectory_features(messages),
            })
        print(f"[in] {path} rows={len(frame)}", flush=True)
    if not rows:
        sys.exit("no rows")

    z = turn_z_scores(rows, THRESHOLDS["turn_z_min_group"])
    bands: Counter = Counter()
    bands_by_source: dict[str, Counter] = defaultdict(Counter)
    reason_counts: Counter = Counter()
    for row in rows:
        row["turn_z"] = z[row["trajectory_id"]]
        row["band"], row["reasons"] = band_of(row["features"], row["turn_z"])
        bands[row["band"]] += 1
        bands_by_source[row["source"]][row["band"]] += 1
        reason_counts.update(f"{row['band']}:{r}" for r in row["reasons"])
    print(f"[bands] {dict(bands)}", flush=True)

    # ---- the judge, on the borderline only -----------------------------------
    judge = judge_from_env()
    borderline = [r for r in rows if r["band"] == "borderline"]
    rng = random.Random(args.seed)
    if args.max_judge_items and len(borderline) > args.max_judge_items:
        borderline = rng.sample(borderline, args.max_judge_items)
    clear_keep = [r for r in rows if r["band"] == "clear_keep"]
    audit = rng.sample(clear_keep, min(args.audit_sample, len(clear_keep))) if judge.enabled else []
    to_judge = borderline + audit
    verdicts = {v.item_id: v for v in judge.judge_many([judge_request(r) for r in to_judge])}
    print(f"[judge] enabled={judge.enabled} borderline_sent={len(borderline)} audit_sent={len(audit)} "
          f"stats={judge.stats()}", flush=True)

    # ---- decisions -----------------------------------------------------------
    decisions: Counter = Counter()
    judge_outcomes: Counter = Counter()
    audit_ids = {r["trajectory_id"] for r in audit}
    for row in rows:
        verdict = verdicts.get(row["trajectory_id"])
        row["judge_ok"] = bool(verdict and verdict.ok)
        row["judge_keep"] = judge_says_keep(verdict.data) if verdict and verdict.ok else None
        row["judge_efficiency"] = (verdict.data.get("efficiency") if verdict and verdict.ok else None)
        row["judge_bad_habit"] = (verdict.data.get("teaches_bad_habit") if verdict and verdict.ok else None)
        row["judge_reason"] = (str(verdict.data.get("reason", ""))[:500] if verdict and verdict.ok
                               else (verdict.error if verdict else None))
        row["judge_cached"] = bool(verdict and verdict.cached)
        if verdict is not None:
            judge_outcomes["ok" if verdict.ok else f"failed:{verdict.error}"] += 1
        if row["band"] == "clear_keep":
            decision, by = "keep", "heuristics"
        elif row["band"] == "clear_drop":
            decision, by = "drop", "heuristics"
        elif row["judge_keep"] is not None:
            decision, by = ("keep" if row["judge_keep"] else "drop"), "judge"
        else:
            decision, by = args.borderline_default, "borderline_default"
        row["decision"], row["decided_by"] = decision, by
        decisions[f"{decision}:{by}"] += 1

    # Calibration: how often would the judge have dropped a row the heuristics called
    # clearly fine? A high number means the thresholds are too loose (or the judge too
    # strict); either way it is the number to look at before trusting the split.
    audited = [r for r in rows if r["trajectory_id"] in audit_ids and r["judge_keep"] is not None]
    calibration = {
        "audit_rows_judged": len(audited),
        "judge_would_drop_clear_keep": sum(1 for r in audited if r["judge_keep"] is False),
        "judge_would_drop_fraction": (round(sum(1 for r in audited if r["judge_keep"] is False)
                                            / len(audited), 4) if audited else None),
        "efficiency_mean_clear_keep": (round(statistics.fmean(
            float(r["judge_efficiency"]) for r in audited
            if isinstance(r["judge_efficiency"], (int, float))), 3)
            if any(isinstance(r["judge_efficiency"], (int, float)) for r in audited) else None),
    }

    # ---- write -----------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(rows[0]["features"])
    curation = pd.DataFrame([{
        "source": r["source"], "trajectory_id": r["trajectory_id"],
        "task_group_id": r["task_group_id"],
        **{k: r["features"][k] for k in feature_names},
        "turn_z": r["turn_z"], "band": r["band"], "reasons": ",".join(r["reasons"]),
        "judge_ok": r["judge_ok"], "judge_keep": r["judge_keep"],
        "judge_efficiency": r["judge_efficiency"], "judge_bad_habit": r["judge_bad_habit"],
        "judge_reason": r["judge_reason"], "judge_cached": r["judge_cached"],
        "audit_sample": r["trajectory_id"] in audit_ids,
        "decision": r["decision"], "decided_by": r["decided_by"],
    } for r in rows])
    curation_path = args.out_dir / "curation.parquet"
    curation.to_parquet(curation_path, index=False)

    outputs: dict[str, dict[str, int | str]] = {}
    for path, frame in frames.items():
        keep_index = sorted(r["index"] for r in rows
                            if r["source"] == str(path) and r["decision"] == "keep")
        out_path = args.out_dir / f"{path.stem}_curated.parquet"
        frame.iloc[keep_index].reset_index(drop=True).to_parquet(out_path, index=False)
        outputs[str(path)] = {"rows_in": int(len(frame)), "rows_kept": len(keep_index),
                              "curated_parquet": str(out_path)}
        print(f"[write] {out_path} kept={len(keep_index)}/{len(frame)}", flush=True)

    manifest = {
        "inputs": outputs,
        "rows_total": len(rows),
        "bands": dict(sorted(bands.items())),
        "bands_by_source": {k: dict(sorted(v.items())) for k, v in sorted(bands_by_source.items())},
        "band_reasons": dict(sorted(reason_counts.items())),
        "decisions": dict(sorted(decisions.items())),
        "kept_total": sum(1 for r in rows if r["decision"] == "keep"),
        "dropped_total": sum(1 for r in rows if r["decision"] == "drop"),
        "borderline_default": args.borderline_default,
        "thresholds": THRESHOLDS,
        "judge": judge.stats(),
        "judge_outcomes": dict(sorted(judge_outcomes.items())),
        "judge_items": {"borderline_sent": len(borderline), "audit_sent": len(audit),
                        "max_judge_items": args.max_judge_items},
        "calibration": calibration,
        "seed": args.seed,
        "curation_parquet": str(curation_path),
        "note": "This selects rows; no message was edited. The verifier remains the hard "
                "gate; this is the soft one, and its every decision is in curation.parquet.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("bands", "decisions", "judge", "calibration")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
