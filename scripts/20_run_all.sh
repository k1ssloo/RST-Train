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
BACKEND="${BACKEND:-verl}"                 # verl (primary) | slime (needs a cuDNN swap on A100)
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
# The two backends need DIFFERENT environments and they are not compatible in one
# env: the slime recipe builds apex + TransformerEngine + Megatron (the part that
# needs a cuDNN swap on A100), while verl needs none of that.
case "$BACKEND" in
  verl)  stage env bash scripts/01b_setup_env_verl.sh ;;
  slime) stage env bash scripts/01_setup_env.sh ;;
  *) echo "unknown BACKEND=$BACKEND (want verl|slime)" >&2; exit 2 ;;
esac

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
# Megatron needs an HF -> torch_dist conversion. verl/FSDP loads the HF checkpoint
# directly, so this whole stage (and its ~120GB RAM requirement, and the open
# question about whether the text-only spec tolerates the ViT/MTP tensors) simply
# does not exist on the verl path.
if [[ "$BACKEND" == "slime" ]]; then
  stage convert env MODEL_KEY="$MODEL_KEY" bash scripts/04_convert_ckpt.sh to_dist
else
  echo "=== SKIP convert (verl/FSDP loads the HF checkpoint directly)"
fi

# ----------------------------------------------------------------- 6. train
if [[ "$BACKEND" == "slime" ]]; then
  stage train env DATA_DIR="$DATA_DIR" RUN_NAME="$RUN_NAME" MODEL_KEY="$MODEL_KEY" \
    MEM_CLASS="$MEM_CLASS" bash scripts/05_run_sft.sh
else
  stage train env DATA_DIR="$DATA_DIR" RUN_NAME="$RUN_NAME" MODEL_KEY="$MODEL_KEY" \
    MEM_CLASS="$MEM_CLASS" NNODES="${ACTOR_NUM_NODES:-4}" NGPUS="${ACTOR_NUM_GPUS_PER_NODE:-8}" \
    bash scripts/30_run_sft_verl.sh
fi

# ---------------------------------------------------------------- 7. export
export_ckpt() {
  local latest
  latest=$(find "$BASE_FOLDER/$RUN_NAME" -maxdepth 1 -type d -name 'iter_*' | sort | tail -1)
  [[ -n "$latest" ]] || { echo "no iter_* checkpoint under $BASE_FOLDER/$RUN_NAME" >&2; return 1; }
  echo "exporting $latest"
  if [[ "$BACKEND" == "slime" ]]; then
    MODEL_KEY="$MODEL_KEY" bash scripts/04_convert_ckpt.sh to_hf "$latest" "$BASE_FOLDER/out-hf" || return 1
  else
    # verl's checkpointer can already emit HF format (save_contents includes
    # hf_model). Find it rather than assuming a layout.
    hf=$(find "$latest" -maxdepth 2 -type d -name 'huggingface' | head -1)
    [[ -n "$hf" ]] || hf=$(find "$latest" -maxdepth 2 -name 'config.json' -printf '%h\n' | head -1)
    [[ -n "$hf" ]] || { echo "no HF-format dir under $latest; check verl checkpoint.save_contents" >&2; return 1; }
    rm -rf "$BASE_FOLDER/out-hf"; cp -r "$hf" "$BASE_FOLDER/out-hf"
  fi
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
# Eval needs somewhere to run task containers; SFT did not. 00b picks that place:
# rootless podman locally when this machine may mount(2), otherwise an off-machine
# backend (daytona/e2b/modal, a remote daemon, or k8s sibling pods). Only if there
# is no place at all does agentic eval become impossible.
SANDBOX_OK=1
if ! source scripts/00b_setup_sandbox.sh; then
  SANDBOX_OK=0
  echo "=== AGENTIC EVAL BLOCKED: nowhere to run task containers."
  echo "    SFT results stand. The benchmark numbers do not exist, and must be"
  echo "    reported as not-run -- never as a checkpoint that 'looks good'."
  echo "    Falling back to the container-free eval, which is a WEAKER signal but"
  echo "    is not nothing: held-out loss, token accuracy, next-action agreement,"
  echo "    tool-call parse rate. See scripts/06b_eval_offline.py."
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

# The container-free fallback. Runs whenever agentic eval could not, so that a
# checkpoint is never simply unmeasured. Deliberately NOT a substitute: it scores
# the model against recorded trajectories, which cannot tell you whether the agent
# would have solved the task.
eval_offline() {
  # Prefer the messages-shaped holdout, not the pretokenized one. Both give the
  # same loss (the export's mask is re-derived by importing qwen3_5_mask, verified
  # identical supervised-token counts), but only the messages shape carries the
  # assistant text the action probe needs -- and next-action agreement is the more
  # informative half. --base-model doubles the load time and is worth it: the
  # base-vs-tuned delta is the only offline number that answers "did SFT do
  # anything at all", and 14_make_report.py WARNs when it is missing.
  local args=(--model-path "$BASE_FOLDER/out-hf-full"
              --base-model "$BASE_FOLDER/$MODEL_DIR_NAME"
              --out "$BASE_FOLDER/eval/offline")
  if [[ -f "$DATA_DIR/rst_sft_holdout.parquet" ]]; then
    args+=(--holdout "$DATA_DIR/rst_sft_holdout.parquet"
           --tokenizer "$BASE_FOLDER/$MODEL_DIR_NAME")
  else
    args+=(--holdout "$DATA_DIR/pretokenized_holdout.parquet")
  fi
  python scripts/06b_eval_offline.py "${args[@]}"
}
if [[ "$SANDBOX_OK" == "0" || "${FORCE_OFFLINE_EVAL:-0}" == "1" ]]; then
  stage eval_offline eval_offline
fi

# ----------------------------------------------------------------- 9. report
# Capture the config the run actually used, so the report checks facts rather than
# intentions.
cat > "$BASE_FOLDER/run_config.json" <<EOF_CFG
{
  "run_name": "$RUN_NAME",
  "backend": "$BACKEND",
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
  --offline-eval "$BASE_FOLDER/eval/offline/offline_results.json" \
  --out "$BASE_FOLDER/REPORT.md" \
  --verdict-json "$BASE_FOLDER/verdict.json"
REPORT_RC=$?

# ------------------------------------ 10. post-SFT training (gated on the SFT verdict)
# This encodes the operating rule mechanically instead of leaving it to judgement:
# results in range -> continue; out of range -> stop and wait for a fix. "In range"
# means the report produced ZERO FAIL findings. WARNs do not block, because a WARN is
# a caveat for a human to weigh; a FAIL means a number is wrong or untrustworthy, and
# spending days of sandbox time on top of one is worse than waiting.
#
# Two paths continue from the SFT checkpoint, so both read the same verdict:
#   10a  agentic GRPO -- on-policy, and every rollout needs a task container
#   10b  DPO          -- off-policy on logged trajectories, needs no container at all
IN_RANGE=$(python - "$BASE_FOLDER/verdict.json" <<'EOF_PY'
import json, sys
try:
    print("1" if json.load(open(sys.argv[1]))["in_range"] else "0")
except Exception:
    print("0")
EOF_PY
)

# ------------------------------------------------------------- 10a. agentic GRPO
if [[ "$RUN_RL" == "1" ]]; then
  if [[ "$FIRST_BATCH" != "1" ]]; then
    echo "=== RL SKIPPED: $MODEL_KEY is not in the authorized first batch (27B and 9B only)."
    echo "    Run its SFT, confirm the report is in range, then ask before adding it."
  elif [[ "$SANDBOX_OK" == "0" ]]; then
    # Not a soft warning: every rollout in 12_run_grpo.sh builds a task image and
    # drives tmux inside it, so with nowhere to run containers there is nothing for
    # GRPO to be on-policy about. Starting it anyway would burn the queue slot to
    # reach the same conclusion an hour later.
    echo "=== RL BLOCKED: nowhere to run task containers, and every GRPO rollout needs"
    echo "    one. Not fixable from inside the training job -- see BACKENDS.md for the"
    echo "    one-flag ops ask and the off-machine backends."
    echo "    Falling through to 10b (DPO), which needs no container."
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
      --offline-eval "$BASE_FOLDER/eval/offline/offline_results.json" \
      --out "$BASE_FOLDER/REPORT_RL.md" \
      --verdict-json "$BASE_FOLDER/verdict_rl.json"
    echo "rl report : $BASE_FOLDER/REPORT_RL.md"
  fi
fi

# ------------------------------------- 10b. DPO, the container-free fallback for RL
# See DPO_PLAN.md. RUN_DPO=auto fires this exactly when GRPO was asked for and cannot
# run, i.e. there is nowhere to run task containers. That is a fallback, not a second
# opinion: with a working sandbox the same GPU hours are better spent on-policy, and
# DPO over other policies' logged trajectories cannot discover a strategy no logged
# trajectory used. RUN_DPO=1 forces it anyway, RUN_DPO=0 disables it.
RUN_DPO="${RUN_DPO:-auto}"
if [[ "$RUN_DPO" == "auto" ]]; then
  if [[ "$RUN_RL" == "1" && "$SANDBOX_OK" == "0" ]]; then RUN_DPO=1; else RUN_DPO=0; fi
fi
DPO_TRAJ_ROOT="${DPO_TRAJ_ROOT:-$BASE_FOLDER/rst-trajectories}"
if [[ "$RUN_DPO" == "1" ]]; then
  if [[ "$IN_RANGE" != "1" ]]; then
    # DPO starts FROM this checkpoint and scores divergence from it, so a checkpoint
    # the report does not trust makes every implicit reward meaningless -- and unlike
    # a bad eval number, that failure is invisible in the loss curve.
    echo "=== DPO BLOCKED: the SFT report has FAIL findings, and DPO would use that"
    echo "    checkpoint as both policy and frozen reference. Fix the FAILs first."
    python -c "import json;d=json.load(open('$BASE_FOLDER/verdict.json'));[print('      -',r) for r in d.get('fail_reasons',[])]" 2>/dev/null || true
  elif [[ ! -d "$DPO_TRAJ_ROOT" && ! -f "${DPO_PAIRS_DIR:-$BASE_FOLDER/dpo-v1}/dpo_train.parquet" ]]; then
    echo "=== DPO SKIPPED: no trajectories at $DPO_TRAJ_ROOT and no prebuilt pairs at"
    echo "    ${DPO_PAIRS_DIR:-$BASE_FOLDER/dpo-v1}. scripts/02_download.sh fetches the release; or set"
    echo "    DPO_TRAJ_ROOT / DPO_PAIRS_DIR."
  else
    echo "=== SFT in range, no sandbox -> DPO on logged trajectories (this is NOT GRPO)"
    # NNODES=1 by default: 19_train_dpo.py is FSDP2-only and torchrun here is local.
    # For 27B on 8x80GB the launcher's own note applies (fp32 masters + Adam is 444.8
    # GB sharded, so 8 is tight and 16 is comfortable) -- set DPO_NNODES and
    # MASTER_ADDR and run this script on each node if 8 OOMs.
    stage dpo env MODEL_KEY="$MODEL_KEY" MEM_CLASS="$MEM_CLASS" \
      NNODES="${DPO_NNODES:-1}" NGPUS="${DPO_NGPUS:-${ACTOR_NUM_GPUS_PER_NODE:-8}}" \
      TRAJ_ROOT="$DPO_TRAJ_ROOT" PAIRS_DIR="${DPO_PAIRS_DIR:-$BASE_FOLDER/dpo-v1}" \
      POLICY="$BASE_FOLDER/out-hf-full" OUT_DIR="$BASE_FOLDER/out-dpo" \
      MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}" \
      bash scripts/33_run_dpo.sh
    # The only eval available without a container. --base-model is the SFT checkpoint,
    # not the base model, so the delta attributed to DPO is DPO's alone.
    dpo_eval_offline() {
      local args=(--model-path "$BASE_FOLDER/out-dpo/hf"
                  --base-model "$BASE_FOLDER/out-hf-full"
                  --out "$BASE_FOLDER/eval/dpo-offline")
      if [[ -f "$DATA_DIR/rst_sft_holdout.parquet" ]]; then
        args+=(--holdout "$DATA_DIR/rst_sft_holdout.parquet"
               --tokenizer "$BASE_FOLDER/$MODEL_DIR_NAME")
      else
        args+=(--holdout "$DATA_DIR/pretokenized_holdout.parquet")
      fi
      python scripts/06b_eval_offline.py "${args[@]}"
    }
    stage dpo_eval_offline dpo_eval_offline
    # Not folded into REPORT.md: 14_make_report.py checks SFT findings against one
    # offline eval, and quietly replacing that eval with a DPO one would relabel the
    # SFT report. These two files are the DPO evidence, and DPO_PLAN.md says how to
    # read them -- in particular that holdout_reward_accuracy is likelihood ranking
    # and 0.5 means no preference, so it is not comparable with a benchmark pass rate.
    echo "dpo summary : $BASE_FOLDER/out-dpo/dpo_training_summary.json"
    echo "dpo offline : $BASE_FOLDER/eval/dpo-offline/offline_results.json"
    echo "dpo ckpt    : $BASE_FOLDER/out-dpo/hf  (NOT agentically evaluated -- no sandbox)"
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
