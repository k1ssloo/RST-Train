#!/usr/bin/env bash
# End-to-end: preflight -> env -> data -> train -> export -> eval -> report.
#
#   export BASE_FOLDER=/shared/rst MASTER_ADDR=<head-ip> HOSTFILE=$BASE_FOLDER/hostfile
#   export RST_DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
#   bash scripts/20_run_all.sh 2>&1 | tee $BASE_FOLDER/run_all.log
#
# Resumable: every stage writes a marker under $BASE_FOLDER/.stage/ and is skipped
# if already done. Delete a marker to force a re-run. Set SKIP_STAGES to skip by
# name, e.g. SKIP_STAGES="env download" .
#
# The report is generated even when a stage fails, because a report explaining a
# failure is more useful than no report. The exit code still reflects the failure.
set -uo pipefail

: "${BASE_FOLDER:?set BASE_FOLDER}"
SLIME_DIR="${SLIME_DIR:-$BASE_FOLDER/slime}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_KEY="${MODEL_KEY:-qwen3.5-27b}"      # python scripts/model_registry.py --list
RUN_NAME="${RUN_NAME:-${MODEL_KEY}-rst-sft-v1}"
DATA_DIR="${DATA_DIR:-$BASE_FOLDER/sft-v1-cap10}"
PER_GROUP="${PER_GROUP:-10}"
EVAL_RUNS="${EVAL_RUNS:-3}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-tb-hard,tb2}"
EVAL_REFERENCE="${EVAL_REFERENCE:-1}"   # also score the released SFT ckpt to validate the harness
EVAL_BASE="${EVAL_BASE:-1}"             # also score the UN-finetuned base model on the same harness
RUN_RL="${RUN_RL:-0}"                   # 1 = continue into agentic GRPO if SFT lands in range
STAGE_DIR="$BASE_FOLDER/.stage"
mkdir -p "$STAGE_DIR" "$BASE_FOLDER/logs"
SKIP_STAGES="${SKIP_STAGES:-}"
FAILED_STAGE=""

stage() {  # stage <name> <command...>
  local name="$1"; shift
  if [[ " $SKIP_STAGES " == *" $name "* ]]; then echo "=== SKIP $name (SKIP_STAGES)"; return 0; fi
  if [[ -f "$STAGE_DIR/$name.done" ]]; then echo "=== SKIP $name (already done)"; return 0; fi
  echo "=== STAGE $name  $(date -Is)"
  if "$@"; then
    date -Is > "$STAGE_DIR/$name.done"; echo "=== DONE $name"; return 0
  fi
  echo "=== FAILED $name" >&2
  [[ -z "$FAILED_STAGE" ]] && FAILED_STAGE="$name"
  return 1
}

cd "$REPO_DIR"

# Resolve the model once, up front: this validates the parallelism arithmetic and
# exits here if the config is impossible, before any GPU time is spent.
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ "${MEM_CLASS:-auto}" == "auto" ]]; then
  if [[ -n "$GPU_MEM_MIB" && "$GPU_MEM_MIB" -gt 70000 ]]; then MEM_CLASS=80GB; else MEM_CLASS=40GB; fi
fi
TOTAL_GPUS=$(( ${ACTOR_NUM_NODES:-4} * ${ACTOR_NUM_GPUS_PER_NODE:-8} ))
eval "$(python scripts/model_registry.py --key "$MODEL_KEY" --mem-class "$MEM_CLASS" \
          --gpus "$TOTAL_GPUS" --gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE:-8}" \
          --max-seq-len "${MAX_SEQ_LEN:-32768}" --shell)"
export MODEL_KEY MEM_CLASS
echo "=== model $MODEL_KEY (${PARAMS_B}B) TP$TP/PP$PP/CP$CP/DP$DP/EP$EP @ $MEM_CLASS_USED"
echo "=== est ${EST_EPOCH_MINUTES} min/epoch, vision=$HAS_VISION moe=$IS_MOE"

# ---------------------------------------------------------------- 1. preflight
stage preflight bash scripts/00_preflight.sh ${HOSTFILE:+--hostfile "$HOSTFILE"}

# ------------------------------------------------------------------- 2. env
stage env bash scripts/01_setup_env.sh

# --------------------------------------------------------------- 3. download
stage download env DOWNLOAD_REFERENCE="$EVAL_REFERENCE" MODEL_KEY="$MODEL_KEY" bash scripts/02_download.sh

# ------------------------------------------------------------------ 4. data
build_data() {
  python scripts/03_build_sft_data.py \
    --traj-root "$BASE_FOLDER/rst-trajectories" \
    --tokenizer "$BASE_FOLDER/$MODEL_DIR_NAME" \
    --out-dir "$DATA_DIR" --per-group "$PER_GROUP" --max-seq-len "${MAX_SEQ_LEN:-32768}" \
    --workers "${DATA_WORKERS:-16}" || return 1
  # Hard gate: a nonzero contract failure or user-turn leakage means the training
  # target is wrong, and no amount of training will fix it.
  python scripts/03b_validate_sft_data.py \
    --parquet "$DATA_DIR/rst_sft_train.parquet" \
    --tokenizer "$BASE_FOLDER/$MODEL_DIR_NAME" --sample 300 --show 0 \
    | tee "$BASE_FOLDER/logs/validate_data.log"
  grep -q "contract failures : 0" "$BASE_FOLDER/logs/validate_data.log" \
    && grep -q "user-turn leakage : 0" "$BASE_FOLDER/logs/validate_data.log" \
    || { echo "DATA GATE FAILED: mask contract or leakage check nonzero" >&2; return 1; }
}
stage data build_data

# ------------------------------------------------------------- 5. convert ckpt
stage convert env MODEL_KEY="$MODEL_KEY" bash scripts/04_convert_ckpt.sh to_dist

# ----------------------------------------------------------------- 6. train
stage train env DATA_DIR="$DATA_DIR" RUN_NAME="$RUN_NAME" MODEL_KEY="$MODEL_KEY" \
  MEM_CLASS="$MEM_CLASS" bash scripts/05_run_sft.sh

# ---------------------------------------------------------------- 7. export
export_ckpt() {
  local latest
  latest=$(find "$BASE_FOLDER/$RUN_NAME" -maxdepth 1 -type d -name 'iter_*' | sort | tail -1)
  [[ -n "$latest" ]] || { echo "no iter_* checkpoint under $BASE_FOLDER/$RUN_NAME" >&2; return 1; }
  echo "exporting $latest"
  MODEL_KEY="$MODEL_KEY" bash scripts/04_convert_ckpt.sh to_hf "$latest" "$BASE_FOLDER/out-hf" || return 1
  if [[ "$HAS_VISION" == "1" ]]; then
    # The text-only round trip drops model.visual.* / mtp.*; without restoring them
    # the checkpoint will not load as a ConditionalGeneration model.
    python scripts/07_restore_vision.py \
      --trained "$BASE_FOLDER/out-hf" --original "$BASE_FOLDER/$MODEL_DIR_NAME" \
      --out "$BASE_FOLDER/out-hf-full"
  else
    echo "model has no vision tower; using the converted checkpoint directly"
    ln -sfn "$BASE_FOLDER/out-hf" "$BASE_FOLDER/out-hf-full"
  fi
}
stage export export_ckpt

# ------------------------------------------------------------------ 8. eval
# Eval needs a container runtime; SFT did not. Rootless podman is the primary path
# on clusters without Docker permission -- it serves the Docker API Harbor speaks.
if ! source scripts/00b_setup_sandbox.sh; then
  echo "=== EVAL SKIPPED: no container runtime. SFT results stand, but they are"
  echo "    UNMEASURED. Do not describe the checkpoint as good; say eval was blocked."
  SKIP_STAGES="$SKIP_STAGES eval_candidate eval_reference eval_base"
fi
export EVAL_TP="${EVAL_TP:-$SERVE_TP}"
export TB_HARD_TASKS="${TB_HARD_TASKS:-$BASE_FOLDER/terminal-bench-hard/tasks}"
export TB2_TASKS="${TB2_TASKS:-$BASE_FOLDER/terminal-bench-2}"
export LHTB_TASKS="${LHTB_TASKS:-$BASE_FOLDER/long-horizon-terminal-bench/tasks}"

# Some checkpoints need an explicit serving template: Qwen3.5-0.8B defaults
# thinking OFF, so its generation prompt closes the think block while our training
# target opens with "\n</think>\n\n". Serving with a thinking-on template realigns
# train and serve. The registry says which models need this.
SERVE_TEMPLATE_ARG=()
if [[ -n "${SERVE_CHAT_TEMPLATE_REPO:-}" ]]; then
  tmpl="$BASE_FOLDER/chat_templates/$(basename "$SERVE_CHAT_TEMPLATE_REPO").jinja"
  mkdir -p "$(dirname "$tmpl")"
  [[ -f "$tmpl" ]] || curl -sSL --fail \
    "https://huggingface.co/${SERVE_CHAT_TEMPLATE_REPO}/resolve/main/chat_template.jinja" -o "$tmpl"
  SERVE_TEMPLATE_ARG=(--chat-template "$tmpl")
  echo "serving with overridden chat template from $SERVE_CHAT_TEMPLATE_REPO"
fi

eval_candidate() {
  python scripts/06_eval.py --model-path "$BASE_FOLDER/out-hf-full" \
    "${SERVE_TEMPLATE_ARG[@]}" \
    --tp "${EVAL_TP:-$SERVE_TP}" --served-name rst-sft --label sft \
    --benchmarks "$EVAL_BENCHMARKS" --runs "$EVAL_RUNS" \
    --n-concurrent "${EVAL_CONCURRENCY:-8}" \
    --out "$BASE_FOLDER/eval/mine"
}
stage eval_candidate eval_candidate

# The base model on the SAME harness. For 4B/9B/35B this is the ONLY thing that can
# answer "did fine-tuning help?", since the paper published no numbers for them. Even
# for 27B it is the better comparison, because it cancels harness differences.
eval_base() {
  [[ "$EVAL_BASE" == "1" ]] || { echo "base eval disabled"; return 0; }
  python scripts/06_eval.py --model-path "$BASE_FOLDER/$MODEL_DIR_NAME" \
    "${SERVE_TEMPLATE_ARG[@]}" \
    --tp "${EVAL_TP:-$SERVE_TP}" --served-name rst-base --label base \
    --benchmarks "$EVAL_BENCHMARKS" --runs "$EVAL_RUNS" \
    --n-concurrent "${EVAL_CONCURRENCY:-8}" \
    --out "$BASE_FOLDER/eval/base"
}
stage eval_base eval_base

# Scoring the authors' released checkpoint through the SAME harness is what makes
# our own number interpretable. If the reference does not land near the paper's
# 47.94 / 28.33, the harness is wrong -- not the training.
eval_reference() {
  [[ "$EVAL_REFERENCE" == "1" ]] || { echo "reference eval disabled"; return 0; }
  [[ -n "${REFERENCE_CHECKPOINT:-}" ]] || {
    echo "no published reference checkpoint for $MODEL_KEY; skipping (the paper only"
    echo "reports Qwen3.5-27B and 122B-A10B). Validate the harness once with 27B."
    return 0; }
  [[ -d "$BASE_FOLDER/ref-Qwen3.5-27B-SFT" ]] || { echo "reference ckpt absent; skipping"; return 0; }
  python scripts/06_eval.py --model-path "$BASE_FOLDER/ref-Qwen3.5-27B-SFT" \
    --tp "${EVAL_TP:-$SERVE_TP}" --served-name rst-ref --label reference \
    --benchmarks "$EVAL_BENCHMARKS" --runs "$EVAL_RUNS" \
    --n-concurrent "${EVAL_CONCURRENCY:-8}" \
    --out "$BASE_FOLDER/eval/reference"
}
stage eval_reference eval_reference

# ----------------------------------------------------------------- 9. report
# Capture the config the run actually used, so the report checks facts rather than
# intentions.
cat > "$BASE_FOLDER/run_config.json" <<EOF_CFG
{
  "run_name": "$RUN_NAME",
  "model_key": "$MODEL_KEY",
  "hf_repo": "$HF_REPO",
  "params_b": ${PARAMS_B},
  "data_dir": "$DATA_DIR",
  "per_group": ${PER_GROUP},
  "max_seq_len": ${MAX_SEQ_LEN:-32768},
  "global_batch_size": ${GLOBAL_BATCH_SIZE:-128},
  "num_epoch": ${NUM_EPOCH:-1},
  "lr": "${LR:-3e-6}",
  "min_lr": "${MIN_LR:-3e-7}",
  "loss_mask_type": "${LOSS_MASK_TYPE}",
  "compute_cap": "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)",
  "gpu_mem_mib": "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)",
  "gdn_backend": "${GDN_BACKEND:-fla}",
  "tp": ${TP}, "pp": ${PP}, "cp": ${CP}, "dp": ${DP}, "ep": ${EP},
  "total_gpus": $(( ${ACTOR_NUM_NODES:-4} * ${ACTOR_NUM_GPUS_PER_NODE:-8} )),
  "max_tokens_per_gpu": ${MAX_TOKENS_PER_GPU},
  "eval_runs": ${EVAL_RUNS},
  "eval_benchmarks": "$EVAL_BENCHMARKS",
  "failed_stage": "${FAILED_STAGE:-none}"
}
EOF_CFG

python scripts/14_make_report.py \
  --model-key "$MODEL_KEY" \
  --run-dir "$BASE_FOLDER/$RUN_NAME" \
  --run-config "$BASE_FOLDER/run_config.json" \
  --data-manifest "$DATA_DIR/manifest.json" \
  --eval "mine=$BASE_FOLDER/eval/mine/results.json" \
  --eval "base=$BASE_FOLDER/eval/base/results.json" \
  --eval "reference=$BASE_FOLDER/eval/reference/results.json" \
  --out "$BASE_FOLDER/REPORT.md" \
  --verdict-json "$BASE_FOLDER/verdict.json"
REPORT_RC=$?

# ---------------------------------------------------- 10. RL (gated on the SFT verdict)
# This encodes the operating rule mechanically instead of leaving it to judgement:
# results in range -> continue; out of range -> stop and wait for a fix. "In range"
# means the report produced ZERO FAIL findings. WARNs do not block, because a WARN is
# a caveat for a human to weigh; a FAIL means a number is wrong or untrustworthy, and
# spending days of sandbox time on top of one is worse than waiting.
if [[ "$RUN_RL" == "1" ]]; then
  IN_RANGE=$(python - "$BASE_FOLDER/verdict.json" <<'EOF_PY'
import json, sys
try:
    print("1" if json.load(open(sys.argv[1]))["in_range"] else "0")
except Exception:
    print("0")
EOF_PY
)
  if [[ "$FIRST_BATCH" != "1" ]]; then
    echo "=== RL SKIPPED: $MODEL_KEY is not in the authorized first batch (27B and 9B only)."
    echo "    Run its SFT, confirm the report is in range, then ask before adding it."
  elif [[ "$IN_RANGE" != "1" ]]; then
    echo "=== RL BLOCKED: the SFT report has FAIL findings, so RL would build on an"
    echo "    untrustworthy checkpoint. Fix the FAILs, re-run the report, then retry."
    python -c "import json;d=json.load(open('$BASE_FOLDER/verdict.json'));[print('      -',r) for r in d.get('fail_reasons',[])]" 2>/dev/null || true
  else
    echo "=== SFT in range -> continuing into agentic GRPO"
    stage rl env MODEL_KEY="$MODEL_KEY" MEM_CLASS="$MEM_CLASS" \
      INIT_CKPT="$BASE_FOLDER/$RUN_NAME" RUN_NAME="${MODEL_KEY}-rst-grpo-v1" \
      TASKSET="${TASKSET:-$BASE_FOLDER/rl-sweet}" bash scripts/12_run_grpo.sh
    # Re-evaluate the RL checkpoint and regenerate the report with both stages.
    rl_export_eval() {
      local latest
      latest=$(find "$BASE_FOLDER/${MODEL_KEY}-rst-grpo-v1" -maxdepth 1 -type d -name 'iter_*' | sort | tail -1)
      [[ -n "$latest" ]] || { echo "no RL checkpoint found" >&2; return 1; }
      MODEL_KEY="$MODEL_KEY" bash scripts/04_convert_ckpt.sh to_hf "$latest" "$BASE_FOLDER/out-hf-rl" || return 1
      if [[ "$HAS_VISION" == "1" ]]; then
        python scripts/07_restore_vision.py --trained "$BASE_FOLDER/out-hf-rl" \
          --original "$BASE_FOLDER/$MODEL_DIR_NAME" --out "$BASE_FOLDER/out-hf-rl-full" || return 1
      else
        ln -sfn "$BASE_FOLDER/out-hf-rl" "$BASE_FOLDER/out-hf-rl-full"
      fi
      python scripts/06_eval.py --model-path "$BASE_FOLDER/out-hf-rl-full" \
        "${SERVE_TEMPLATE_ARG[@]}" \
        --tp "${EVAL_TP:-$SERVE_TP}" --served-name rst-rl --label rl \
        --benchmarks "$EVAL_BENCHMARKS" --runs "$EVAL_RUNS" \
        --n-concurrent "${EVAL_CONCURRENCY:-8}" --out "$BASE_FOLDER/eval/rl"
    }
    stage rl_eval rl_export_eval
    python scripts/14_make_report.py \
      --model-key "$MODEL_KEY" \
      --run-dir "$BASE_FOLDER/${MODEL_KEY}-rst-grpo-v1" \
      --run-config "$BASE_FOLDER/run_config.json" \
      --data-manifest "$DATA_DIR/manifest.json" \
      --eval "mine=$BASE_FOLDER/eval/rl/results.json" \
      --eval "sft=$BASE_FOLDER/eval/mine/results.json" \
      --eval "base=$BASE_FOLDER/eval/base/results.json" \
      --eval "reference=$BASE_FOLDER/eval/reference/results.json" \
      --out "$BASE_FOLDER/REPORT_RL.md" \
      --verdict-json "$BASE_FOLDER/verdict_rl.json"
    echo "rl report : $BASE_FOLDER/REPORT_RL.md"
  fi
fi

echo
echo "================================================================"
echo "report : $BASE_FOLDER/REPORT.md"
echo "verdict: $BASE_FOLDER/verdict.json"
[[ -n "$FAILED_STAGE" ]] && echo "FAILED STAGE: $FAILED_STAGE (see report + logs)"
echo "report generator exit: $REPORT_RC  (2 = at least one FAIL finding)"
echo "================================================================"

[[ -n "$FAILED_STAGE" ]] && exit 1
exit $REPORT_RC
