#!/usr/bin/env bash
# One recursive-self-improvement round: serve the checkpoint, roll out, keep what the
# verifier passed, curate it, and hand back a parquet the next SFT reads.
#
#   export BASE_FOLDER=/shared/rst MODEL_KEY=qwen3.5-9b
#   export RST_DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock   # or RST_HARBOR_ENV=daytona
#   bash scripts/21_rsi_round.sh --ckpt $BASE_FOLDER/out-hf-full --tasks $BASE_FOLDER/rl-sweet/tasks
#
# WHAT THIS IS, AND WHAT IT IS NOT
#   It is rejection-sampling self-improvement, also called expert iteration: sample the
#   current policy on tasks whose verifier can score it, keep the successes, train on
#   them, repeat. It is NOT on-policy RL -- there is no advantage, no importance ratio,
#   no gradient from a failure. GRPO (12_run_grpo.sh) is that, and it is strictly more
#   informative per rollout. This exists because it needs no trainer-side rollout
#   plumbing at all: one sglang server, one Harbor loop, one parquet. On a pod where
#   GRPO's colocated actor cannot be stood up but containers CAN run, this is the only
#   path that closes the loop, and its output is ordinary SFT data.
#
# THE FOUR STAGES
#   1. 06_eval.py --export-trajectories       serve + roll out + score + KEEP the ATIF
#   2. 03h_build_rollout_sft.py               verifier-passed rollouts -> messages parquet
#   3. 03g_curate_sft.py                      thrashing/repetition gate, judge on the band
#   4. 15_export_pretokenized.py              input_ids + loss_mask, ready for 30_run_sft_verl.sh
#
#   Stage 1 is the expensive one and the only one that needs a sandbox. Stages 2-4 are
#   CPU and re-runnable, so a round can be re-curated with different thresholds -- or a
#   different judge -- without re-rolling anything.
#
# WHY THE VERIFIER IS NOT THE WHOLE GATE
#   Reward 1 says the task was solved, not that the transcript is worth imitating. The
#   RST release itself is 63% malformed assistant output; our own policy's successes
#   carry their own habits -- thrashing, repeated commands, premature completion claims.
#   Stage 3 is the filter, and its LLM half runs ONLY on the ambiguous band (measured on
#   the existing corpora: 18% of rows), so the judge cost scales with the borderline and
#   not with the round. Without RST_JUDGE_BASE_URL it degrades to heuristics and says so.
#
# THE TRAP THIS SCRIPT EXISTS TO AVOID
#   Training round N+1 on round N's successes narrows the policy toward what it already
#   does. Two defences, both on by default:
#     * --mix-ratio: the round's data is MIXED with the seed corpus rather than replacing
#       it, so the next SFT still sees the distribution the first one was trained on.
#     * the pass rate per round is written to rsi_round.json. A round whose pass rate did
#       not move, or whose kept-row count collapsed, is a round that taught nothing --
#       read it before launching the next one.
set -uo pipefail

# --help before anything else: asking what a script does must not require its
# environment to be set up first.
for arg in "$@"; do
  case "$arg" in -h|--help) sed -n '2,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;; esac
done

: "${BASE_FOLDER:?set BASE_FOLDER}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
# shellcheck source=lib_env.sh
source "$REPO_DIR/scripts/lib_env.sh"
rst_bootstrap_python || exit 2
rst_enter_env "${ENV_NAME:-rstverl}" || exit 2

MODEL_KEY="${MODEL_KEY:-qwen3.5-9b}"
ROUND="${ROUND:-1}"
CKPT=""
TASKS="${RSI_TASKS:-$BASE_FOLDER/rl-sweet/tasks}"
OUT="${RSI_OUT:-$BASE_FOLDER/rsi/round$ROUND}"
SEED_DATA="${SEED_DATA:-$BASE_FOLDER/sft-v1-cap10/rst_sft_train.parquet}"
MIX_RATIO="${MIX_RATIO:-1.0}"          # rounds of seed data per round of new data; 0 = new only
MAX_TASKS="${MAX_TASKS:-0}"
RUNS="${RUNS:-1}"                       # samples per task; >1 is what makes rejection sampling work
CONCURRENCY="${CONCURRENCY:-8}"
TASK_TIMEOUT="${TASK_TIMEOUT:-1800}"
TEMPERATURE="${TEMPERATURE:-1.0}"       # NOT greedy: identical samples cannot be rejection-sampled
BORDERLINE="${BORDERLINE:-keep}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --round) ROUND="$2"; OUT="${RSI_OUT:-$BASE_FOLDER/rsi/round$2}"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --max-tasks) MAX_TASKS="$2"; shift 2 ;;
    --seed-data) SEED_DATA="$2"; shift 2 ;;
    --mix-ratio) MIX_RATIO="$2"; shift 2 ;;
    --skip-rollout) SKIP_ROLLOUT=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
: "${CKPT:?give --ckpt (the checkpoint to sample from; 08_prepare_eval_ckpt.sh makes one servable)}"
[[ -d "$TASKS" ]] || { echo "no task dirs at $TASKS -- run 10_build_rl_taskset.py --materialize" >&2; exit 2; }

eval "$(python scripts/model_registry.py --key "$MODEL_KEY" --shell 2>/dev/null || true)"
SERVE_TP="${SERVE_TP:-1}"
SERVE_CONTEXT_LENGTH="${SERVE_CONTEXT_LENGTH:-65536}"
mkdir -p "$OUT"

# Some checkpoints need an explicit serving template or train and serve stop lining up:
# Qwen3.5-0.8B defaults thinking OFF, so its generation prompt already closes the think
# block while our training targets open with "\n</think>\n\n". Serving it with a
# thinking-on template realigns them. The registry says which models need this, and
# 20_run_all.sh does the same thing for eval -- a rollout that skipped it would harvest
# data from a policy that was being served wrong.
SERVE_TEMPLATE_ARG=()
if [[ -n "${SERVE_CHAT_TEMPLATE_REPO:-}" ]]; then
  tmpl="$BASE_FOLDER/chat_templates/$(basename "$SERVE_CHAT_TEMPLATE_REPO").jinja"
  mkdir -p "$(dirname "$tmpl")"
  [[ -f "$tmpl" ]] || curl -sSL --fail \
    "https://huggingface.co/${SERVE_CHAT_TEMPLATE_REPO}/resolve/main/chat_template.jinja" -o "$tmpl" \
    || { echo "could not fetch the serving template for $SERVE_CHAT_TEMPLATE_REPO." >&2
         echo "Refusing to roll out: this model needs it, and without it the policy is" >&2
         echo "served with a prompt shape it was not trained on." >&2; exit 2; }
  SERVE_TEMPLATE_ARG=(--chat-template "$tmpl")
  echo "    template : overridden from $SERVE_CHAT_TEMPLATE_REPO"
fi

echo "=== RSI round $ROUND"
echo "    policy   : $CKPT"
echo "    tasks    : $TASKS"
echo "    sampling : $RUNS run(s) x temperature $TEMPERATURE"
echo "    out      : $OUT"

# ---------------------------------------------------------------- 1. roll out
# --export-trajectories is the whole point: without it Harbor keeps the harness's
# rendering of each turn instead of the model's completion, and stage 2 refuses the
# lot (BUG.md BUG-19). It implies --keep-jobs.
if [[ "${SKIP_ROLLOUT:-0}" != "1" ]]; then
  python scripts/06_eval.py \
    --model-path "$CKPT" --tp "$SERVE_TP" --served-name "rsi-r$ROUND" \
    "${SERVE_TEMPLATE_ARG[@]}" \
    --context-length "$SERVE_CONTEXT_LENGTH" \
    --label "rsi-round$ROUND" --benchmarks tb2 \
    --tb2-tasks "$TASKS" \
    --runs "$RUNS" --temperature "$TEMPERATURE" \
    --n-concurrent "$CONCURRENCY" --task-timeout "$TASK_TIMEOUT" \
    --max-tasks "$MAX_TASKS" \
    --export-trajectories \
    --out "$OUT/rollouts" || { echo "=== FAILED: rollout stage"; exit 1; }
else
  echo "--skip-rollout: reusing $OUT/rollouts/jobs"
fi
[[ -d "$OUT/rollouts/jobs" ]] || { echo "no job dirs under $OUT/rollouts/jobs" >&2; exit 1; }

# ------------------------------------------------------- 2. rollouts -> parquet
python scripts/03h_build_rollout_sft.py \
  --jobs-dir "$OUT/rollouts/jobs" \
  --tokenizer "$CKPT" \
  --out-dir "$OUT/sft" \
  --min-reward 1.0 || { echo "=== FAILED: rollout->sft stage"; exit 1; }

ROWS=$(python - "$OUT/sft/manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("train_examples", 0))
PY
)
if [[ "${ROWS:-0}" -eq 0 ]]; then
  echo "=== round $ROUND produced NO verifier-passed rollouts."
  echo "    That is a result, not a failure: this policy solved nothing in this pool."
  echo "    Read $OUT/rollouts/results.json (pass rate, infra vs budget failures) before"
  echo "    concluding anything about the checkpoint, and do NOT train on an empty round."
  exit 3
fi
echo "=== round $ROUND: $ROWS verifier-passed rows before curation"

# ------------------------------------------------------------------ 3. curate
python scripts/03g_curate_sft.py \
  --parquet "$OUT/sft/rollout_sft_train.parquet" \
  --out-dir "$OUT/curated" \
  --borderline-default "$BORDERLINE" || { echo "=== FAILED: curation stage"; exit 1; }
CURATED="$OUT/curated/rollout_sft_train_curated.parquet"

# ----------------------------------------------------- 4. mix + pre-tokenize
# Self-training on its own output alone is how a policy narrows. The seed corpus is
# concatenated back in at --mix-ratio, and the manifest records the mix, so "what was
# this checkpoint trained on" stays answerable one round later.
MIXED="$OUT/train_mixed.parquet"
python - "$CURATED" "$SEED_DATA" "$MIX_RATIO" "$MIXED" "$OUT/mix.json" <<'PY'
import json, sys
import pandas as pd

curated, seed_path, ratio, out, report = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
new = pd.read_parquet(curated)
frames, note = [new], {"round_rows": len(new), "seed_rows": 0, "mix_ratio": ratio}
if ratio > 0:
    try:
        seed = pd.read_parquet(seed_path)
    except (OSError, ValueError) as exc:
        note["seed_error"] = f"{type(exc).__name__}: {exc}"
        seed = None
    if seed is not None:
        want = int(len(new) * ratio)
        # Deterministic and reproducible: a fixed seed, and the whole corpus when the
        # round is large enough to ask for it.
        seed = seed if want >= len(seed) else seed.sample(n=want, random_state=1228)
        shared = [c for c in new.columns if c in seed.columns]
        frames = [new[shared], seed[shared]]
        note["seed_rows"] = len(seed)
        note["columns_kept"] = shared
mixed = pd.concat(frames, ignore_index=True)
mixed.to_parquet(out, index=False)
note["total_rows"] = len(mixed)
note["output"] = out
json.dump(note, open(report, "w"), indent=2, sort_keys=True)
print(f"[mix] round={note['round_rows']} + seed={note['seed_rows']} = {len(mixed)} -> {out}")
PY

python scripts/15_export_pretokenized.py \
  --parquet "$MIXED" --tokenizer "$CKPT" \
  --out "$OUT/pretokenized_train.parquet" || { echo "=== FAILED: pretokenize stage"; exit 1; }

# ------------------------------------------------------------------- report
python - "$OUT" "$ROUND" "$CKPT" <<'PY'
import json, sys
from pathlib import Path

out, round_no, ckpt = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


def read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


results = read(out / "rollouts" / "results.json")
tb2 = (results.get("benchmarks") or {}).get("tb2") or {}
sft, curated, mix, pretok = (read(out / "sft" / "manifest.json"),
                            read(out / "curated" / "manifest.json"),
                            read(out / "mix.json"),
                            read(out / "pretokenized_train_manifest.json"))
summary = {
    "round": round_no,
    "policy": ckpt,
    "rollout": {"pass_rate_mean": tb2.get("pass_rate_mean"),
                "trials_total": tb2.get("trials_total"),
                "trials_scorable": tb2.get("trials_scorable"),
                "infra_failures": tb2.get("infra_failures"),
                "agent_budget_failures": tb2.get("agent_budget_failures"),
                "trajectory_export": (results.get("protocol") or {}).get("trajectory_export")},
    "harvest": {"reconstructed": sft.get("reconstructed"),
                "by_reward": sft.get("reconstructed_by_reward"),
                "sft_rows": sft.get("train_examples")},
    "curation": {"bands": curated.get("bands"), "decisions": curated.get("decisions"),
                 "judge": curated.get("judge"), "calibration": curated.get("calibration")},
    "mix": mix,
    "pretokenized": {"rows": pretok.get("rows_out"),
                     "trained_fraction": pretok.get("trained_fraction")},
    "next": (f"DATA_DIR={out} RUN_NAME=<model>-rsi-r{round_no} "
             f"bash scripts/30_run_sft_verl.sh  "
             f"(the launcher reads $DATA_DIR/pretokenized_train.parquet, which this round "
             f"wrote, and re-checks it against the parquet itself rather than a manifest)"),
    "read_this_before_the_next_round": (
        "A round whose pass_rate_mean did not move, or whose sft_rows collapsed, taught "
        "nothing -- rejection sampling can only reinforce what the policy already does "
        "sometimes. If the pool is saturated (pass rate high) or hopeless (near zero), "
        "change the task tier rather than running another round."),
}
(out / "rsi_round.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo
echo "=== RSI round $ROUND complete"
echo "    summary : $OUT/rsi_round.json"
echo "    data    : $OUT/pretokenized_train.parquet"
echo "    next    : DATA_DIR=$OUT RUN_NAME=${MODEL_KEY}-rsi-r$ROUND bash scripts/30_run_sft_verl.sh"
