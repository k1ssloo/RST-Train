#!/usr/bin/env bash
# End-to-end: preflight -> env -> data -> train -> export -> eval -> report -> DPO.
#
# DPO runs by default once the SFT report is in range (RUN_DPO=0 to skip); agentic
# GRPO is opt-in with RUN_RL=1, because it is the only stage that needs a sandbox.
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
RUN_RL="${RUN_RL:-0}"                   # 1 = also run agentic GRPO (needs a sandbox)
                                        # RUN_DPO defaults to auto=on; see section 10b
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

# shellcheck source=lib_env.sh
source "$REPO_DIR/scripts/lib_env.sh"
# The registry call below and 00_preflight.sh are stdlib-only python, but they still
# need the interpreter to be called `python`, which on Ubuntu it is not.
rst_bootstrap_python || exit 2

# Resolve the model once, up front: this validates the parallelism arithmetic and
# exits here if the config is impossible, before any GPU time is spent.
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ "${MEM_CLASS:-auto}" == "auto" ]]; then
  if [[ -n "$GPU_MEM_MIB" && "$GPU_MEM_MIB" -gt 70000 ]]; then MEM_CLASS=80GB; else MEM_CLASS=40GB; fi
fi

# ---- how many GPUs are we actually planning for? -----------------------------
# TOTAL_GPUS is DERIVED from ACTOR_NUM_NODES x ACTOR_NUM_GPUS_PER_NODE, whose defaults
# are 4 and 8. MODEL_KEY does not influence it, and the registry validates the
# arithmetic rather than the intent -- it will happily place 0.8B on 32 GPUs (CP 2, DP
# 16, exit 0). So a smoke test on one node, run without setting these, used to plan a
# 4-node/32-GPU job and die in NCCL rendezvous minutes later with nothing pointing at
# the cause. Reconcile the three sources of truth here instead: the two env vars, the
# hostfile, and the GPUs this host can see.
VISIBLE_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
VISIBLE_GPUS="${VISIBLE_GPUS:-0}"
NODES_WANT="${ACTOR_NUM_NODES:-4}"
GPN_WANT="${ACTOR_NUM_GPUS_PER_NODE:-8}"
if [[ -n "${HOSTFILE:-}" && -f "${HOSTFILE:-}" ]]; then
  HOSTFILE_NODES=$(grep -cve '^[[:space:]]*$' "$HOSTFILE" || true)
  if [[ -z "${ACTOR_NUM_NODES:-}" ]] && (( HOSTFILE_NODES > 0 )); then
    NODES_WANT="$HOSTFILE_NODES"
    echo "=== ACTOR_NUM_NODES unset; taking $NODES_WANT node(s) from $HOSTFILE"
  elif (( HOSTFILE_NODES != NODES_WANT )); then
    echo "=== WARNING: ACTOR_NUM_NODES=$NODES_WANT but $HOSTFILE lists $HOSTFILE_NODES."
    echo "    Using $NODES_WANT. One of the two is wrong; say which in the report."
  fi
fi
if (( VISIBLE_GPUS > 0 && GPN_WANT > VISIBLE_GPUS )); then
  cat >&2 <<EOF

=== GPU SHAPE MISMATCH -- refusing to plan a job this host cannot run.
    ACTOR_NUM_GPUS_PER_NODE = $GPN_WANT   $([[ -z "${ACTOR_NUM_GPUS_PER_NODE:-}" ]] && echo "(unset -> default 8)" || echo "(you set this)")
    ACTOR_NUM_NODES         = $NODES_WANT   $([[ -z "${ACTOR_NUM_NODES:-}" ]] && echo "(unset -> default 4)" || echo "(you set this)")
    visible on this host    = $VISIBLE_GPUS
    => this run would plan $(( NODES_WANT * GPN_WANT )) GPUs.

    These two variables are the ONLY thing that sets the job shape. MODEL_KEY does
    not, and the registry checks the arithmetic, not whether the cluster exists -- so
    without this check a 2-GPU smoke test silently becomes a 32-GPU plan.

    Single node, all of its GPUs:
      export ACTOR_NUM_NODES=1 ACTOR_NUM_GPUS_PER_NODE=$VISIBLE_GPUS
    The 0.8B smoke test on 2 GPUs:
      export ACTOR_NUM_NODES=1 ACTOR_NUM_GPUS_PER_NODE=2
    Full 4x8 cluster (the default) -- then run this on the head node with a hostfile:
      export ACTOR_NUM_NODES=4 ACTOR_NUM_GPUS_PER_NODE=8 HOSTFILE=\$BASE_FOLDER/hostfile

    If this host is a login node with no GPUs, the check does not fire; it fired
    because it can see $VISIBLE_GPUS.
EOF
  exit 2
fi
export ACTOR_NUM_NODES="$NODES_WANT" ACTOR_NUM_GPUS_PER_NODE="$GPN_WANT"
TOTAL_GPUS=$(( NODES_WANT * GPN_WANT ))
echo "=== planning for $NODES_WANT node(s) x $GPN_WANT GPU(s) = $TOTAL_GPUS GPUs (this host sees $VISIBLE_GPUS)"

# Capture the registry output before eval'ing it: on failure `eval ""` leaves every
# downstream variable unset, and `set -u` then reports it as "MODEL_DIR_NAME: unbound
# variable" -- which reads like a launcher bug rather than a rejected config.
if ! REGISTRY_SHELL=$(python scripts/model_registry.py --key "$MODEL_KEY" --mem-class "$MEM_CLASS" \
          --gpus "$TOTAL_GPUS" --gpus-per-node "$GPN_WANT" \
          --max-seq-len "${MAX_SEQ_LEN:-32768}" --shell); then
  echo "=== the model registry rejected this configuration (its reason is above)." >&2
  echo "    MODEL_KEY=$MODEL_KEY gpus=$TOTAL_GPUS gpus-per-node=$GPN_WANT" >&2
  echo "    mem-class=$MEM_CLASS max-seq-len=${MAX_SEQ_LEN:-32768}" >&2
  echo "    Its asserts are load-bearing (tp*pp*cp*dp == gpus, and" >&2
  echo "    max_tokens_per_gpu*CP >= max_seq_len). Change the GPU count or fix the row" >&2
  echo "    in configs/models.json -- do not relax the assert." >&2
  echo "    python scripts/model_registry.py --list" >&2
  exit 2
fi
eval "$REGISTRY_SHELL"
export MODEL_KEY MEM_CLASS
echo "=== model $MODEL_KEY (${PARAMS_B}B) TP$TP/PP$PP/CP$CP/DP$DP/EP$EP @ $MEM_CLASS_USED"
echo "=== est ${EST_EPOCH_MINUTES} min/epoch, vision=$HAS_VISION moe=$IS_MOE"

# ---------------------------------------------------------------- 1. preflight
stage preflight bash scripts/00_preflight.sh ${HOSTFILE:+--hostfile "$HOSTFILE"}

# ------------------------------------------------------------------- 2. env
# The two backends need DIFFERENT environments and they are not compatible in one
# env: the slime recipe builds apex + TransformerEngine + Megatron (the part that
# needs a cuDNN swap on A100), while verl needs none of that.
#
# INSTALL_ROLLOUT is forwarded on purpose: the default chain below evaluates three
# checkpoints, and 06_eval.py's only serving path is `python -m sglang.launch_server`.
# An env built without sglang can train and cannot measure.
case "$BACKEND" in
  verl)  ENV_NAME="${ENV_NAME:-rstverl}"
         stage env env INSTALL_ROLLOUT="${INSTALL_ROLLOUT:-1}" ENV_NAME="$ENV_NAME" \
           bash scripts/01b_setup_env_verl.sh ;;
  slime) ENV_NAME="${ENV_NAME:-slime}"
         stage env env ENV_NAME="$ENV_NAME" bash scripts/01_setup_env.sh ;;
  *) echo "unknown BACKEND=$BACKEND (want verl|slime)" >&2; exit 2 ;;
esac
# Exported so every child launcher resolves the same env name; without it 04/05/33 fall
# back to their own default, which is the wrong one on the other backend.
export ENV_NAME

# ---- 2b. ENTER the env -------------------------------------------------------
# Not optional, and not something the operator should have to remember. The setup
# script above activated the env inside its own process; this shell is the parent and
# is unaffected. Without this, `stage data` runs 03_build_sft_data.py under the system
# interpreter and dies on `import pandas` two stages from the cause.
#
# rst_enter_env is a no-op when we are already in a usable env, so it is also correct
# under SKIP_STAGES="env" and inside a prebaked image.
rst_enter_env "$ENV_NAME" || exit 2

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
# The two backends name their checkpoints differently and reach HF format by
# different routes, and getting it wrong is NOT a loud failure:
#   * Megatron/slime writes `iter_<n>/`; verl+FSDP writes `global_step_<n>/`. A glob
#     for one finds nothing under the other -- which at least fails here.
#   * Worse, verl writes a `global_step_<n>/huggingface/` directory containing
#     config.json + the tokenizer and NO WEIGHTS unless `checkpoint.save_contents`
#     included `hf_model`. Copying it blind produces a directory that looks like a
#     checkpoint, passes every path check, and only falls over much later inside
#     sglang. So weights are required explicitly, and if they are absent we merge
#     the FSDP shards with verl's own tool instead of guessing.
latest_step_dir() {  # latest_step_dir <root> <glob> -> absolute path, or empty
  local root="$1" glob="$2" name
  [[ -d "$root" ]] || return 0
  # sort -V, not sort: global_step_9 must not outrank global_step_10.
  name=$(find "$root" -maxdepth 1 -type d -name "$glob" -printf '%f\n' 2>/dev/null | sort -V | tail -1)
  [[ -n "$name" ]] && echo "$root/$name"
}

has_hf_weights() {  # has_hf_weights <dir> -- config.json alone is not a checkpoint
  local d="$1"
  [[ -f "$d/config.json" ]] || return 1
  compgen -G "$d/*.safetensors" > /dev/null || compgen -G "$d/pytorch_model*.bin" > /dev/null
}

# Files eval needs that a shard merge does not necessarily write.
HF_SIDECAR_FILES=(tokenizer.json tokenizer_config.json tokenizer.model vocab.json
                  merges.txt added_tokens.json special_tokens_map.json
                  generation_config.json chat_template.jinja preprocessor_config.json)

export_ckpt() {
  local latest ckpt_hf="" cand merge_src f
  if [[ "$BACKEND" == "slime" ]]; then
    latest=$(latest_step_dir "$BASE_FOLDER/$RUN_NAME" 'iter_*')
    [[ -n "$latest" ]] || { echo "no iter_* checkpoint under $BASE_FOLDER/$RUN_NAME" >&2; return 1; }
    echo "exporting $latest (megatron -> hf)"
    MODEL_KEY="$MODEL_KEY" bash scripts/04_convert_ckpt.sh to_hf "$latest" "$BASE_FOLDER/out-hf" || return 1
  else
    latest=$(latest_step_dir "$BASE_FOLDER/$RUN_NAME" 'global_step_*')
    # Fall back to the Megatron name so a mixed-backend $BASE_FOLDER still exports.
    [[ -n "$latest" ]] || latest=$(latest_step_dir "$BASE_FOLDER/$RUN_NAME" 'iter_*')
    [[ -n "$latest" ]] || {
      echo "no global_step_*/iter_* checkpoint under $BASE_FOLDER/$RUN_NAME." >&2
      echo "verl writes trainer.default_local_dir/global_step_<n>/; if training ran but" >&2
      echo "nothing is there, trainer.save_freq never divided the step count." >&2
      return 1; }
    echo "exporting $latest (verl/fsdp -> hf)"
    for cand in "$latest/huggingface" "$latest" "$latest/actor/huggingface"; do
      has_hf_weights "$cand" && { ckpt_hf="$cand"; break; }
    done
    rm -rf "$BASE_FOLDER/out-hf"
    if [[ -n "$ckpt_hf" ]]; then
      echo "  found HF weights in $ckpt_hf"
      cp -r "$ckpt_hf" "$BASE_FOLDER/out-hf" || return 1
    else
      # Sharded-only checkpoint. This is the DEFAULT verl layout, not an error.
      merge_src="$latest"; [[ -d "$latest/actor" ]] && merge_src="$latest/actor"
      echo "  no HF weights under $latest -> merging FSDP shards from $merge_src"
      if ! python -m verl.model_merger merge --backend fsdp \
             --local_dir "$merge_src" --target_dir "$BASE_FOLDER/out-hf"; then
        echo "verl.model_merger failed. Two ways forward:" >&2
        echo "  1) re-run training with SAVE_HF_MODEL=1 (30_run_sft_verl.sh) so the" >&2
        echo "     checkpointer writes HF weights directly, then re-run this stage;" >&2
        echo "  2) merge by hand: python -m verl.model_merger merge --backend fsdp" >&2
        echo "     --local_dir $merge_src --target_dir $BASE_FOLDER/out-hf" >&2
        echo "  (older verl spells it 'python -m verl.model_merger --backend fsdp" >&2
        echo "   --local_dir ... --target_dir ...' with no 'merge' subcommand.)" >&2
        return 1
      fi
    fi
    # The merger writes weights + config; the tokenizer it does not always carry.
    # 06_eval.py serves this directory to sglang, which needs both.
    for f in "${HF_SIDECAR_FILES[@]}"; do
      [[ -e "$BASE_FOLDER/out-hf/$f" ]] && continue
      if [[ -e "$latest/huggingface/$f" ]]; then
        cp "$latest/huggingface/$f" "$BASE_FOLDER/out-hf/"
      elif [[ -e "$BASE_FOLDER/$MODEL_DIR_NAME/$f" ]]; then
        cp "$BASE_FOLDER/$MODEL_DIR_NAME/$f" "$BASE_FOLDER/out-hf/"
      fi
    done
  fi
  has_hf_weights "$BASE_FOLDER/out-hf" || {
    echo "$BASE_FOLDER/out-hf has no config.json + weight shards after export." >&2
    echo "Refusing to hand a weightless directory to eval -- it would fail inside" >&2
    echo "sglang with a message about the model, not about the export." >&2
    ls -la "$BASE_FOLDER/out-hf" >&2 2>/dev/null
    return 1; }
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
fi

# The second prerequisite, and the one that used to be invisible: 06_eval.py serves
# the checkpoint with `python -m sglang.launch_server` and has no other serving path.
# 01b_setup_env_verl.sh installs sglang unless INSTALL_ROLLOUT=0, and the install is
# best-effort (it is rolled back if it replaces the driver-matched torch). Check for it
# HERE rather than discovering it at 06_eval.py:311, where a missing module surfaces as
# "the server never came up" and the real reason sits in $out/sglang.log.
SGLANG_OK=1
if ! rst_has_modules sglang; then
  SGLANG_OK=0
  echo "=== AGENTIC EVAL BLOCKED: sglang is not installed in $(command -v python)."
  echo "    06_eval.py can only serve a checkpoint through sglang.launch_server, so the"
  echo "    benchmarks cannot run. Nothing about SFT or DPO is affected."
  echo "    To fix it and re-run just the eval stages:"
  echo "      INSTALL_ROLLOUT=1 bash scripts/01b_setup_env_verl.sh"
  echo "      rm -f $STAGE_DIR/eval_*.done && bash scripts/20_run_all.sh"
  echo "    If sglang cannot be built for this torch, say so in the report and keep the"
  echo "    benchmark-coverage FAIL. Do not report the offline eval as a pass rate."
fi

# One flag for "can we run agentic benchmarks at all". Both prerequisites are external
# to the training code, and neither is a reason to skip the container-free eval.
AGENTIC_EVAL_OK=1
if [[ "$SANDBOX_OK" == "0" || "$SGLANG_OK" == "0" ]]; then
  AGENTIC_EVAL_OK=0
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
  # Same path 02_download.sh writes: ref-<basename of the registry's repo id>.
  local ref_dir="$BASE_FOLDER/ref-$(basename "$REFERENCE_CHECKPOINT")"
  [[ -d "$ref_dir" ]] || { echo "reference ckpt absent ($ref_dir); skipping"; return 0; }
  python scripts/06_eval.py --model-path "$ref_dir" \
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
if [[ "$AGENTIC_EVAL_OK" == "0" || "${FORCE_OFFLINE_EVAL:-0}" == "1" ]]; then
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
# Two signals, because the two paths do not have the same prerequisites:
#   in_range               zero FAIL findings of any kind. GRPO needs this.
#   checkpoint_trustworthy zero FAILs that impugn the CHECKPOINT. Differs from in_range
#                          only for "benchmark coverage", i.e. "we had nowhere to run
#                          containers, or no sglang to serve with". DPO needs neither,
#                          so gating it on in_range meant a container-less pod could
#                          produce an SFT checkpoint and then nothing -- see
#                          POSTTRAIN_EXEMPT_FAILS in 14_make_report.py.
read -r IN_RANGE CKPT_OK <<<"$(python - "$BASE_FOLDER/verdict.json" <<'EOF_PY'
import json
import sys

try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("0 0")
else:
    # checkpoint_trustworthy is absent in verdicts written by an older report
    # generator; fall back to in_range, which is what it used to mean.
    ir = bool(d.get("in_range"))
    print(f"{int(ir)} {int(bool(d.get('checkpoint_trustworthy', ir)))}")
EOF_PY
)"
IN_RANGE="${IN_RANGE:-0}"; CKPT_OK="${CKPT_OK:-0}"

# ------------------------------------------------------------- 10a. agentic GRPO
if [[ "$RUN_RL" == "1" ]]; then
  if [[ "$BACKEND" != "slime" ]]; then
    # 12_run_grpo.sh is a Megatron/slime launcher: it sources a slime model spec and
    # passes --ref-load a torch_dist checkpoint. On the verl path stage `convert`
    # was skipped by design, so that checkpoint does not exist and never will.
    # verl_backend/harbor_agent_loop.py is the rollout half of a verl GRPO path, but
    # there is no launcher for it yet -- so this is a missing feature, not a
    # misconfiguration, and it must say so instead of failing 40 minutes in.
    echo "=== RL SKIPPED: BACKEND=$BACKEND, and GRPO is implemented only for slime."
    echo "    Either run the whole pipeline with BACKEND=slime, or stop after DPO."
    echo "    (verl_backend/harbor_agent_loop.py exists; a verl PPO/GRPO launcher does not.)"
  elif [[ "$FIRST_BATCH" != "1" ]]; then
    echo "=== RL SKIPPED: $MODEL_KEY is not in the authorized first batch (27B and 9B only)."
    echo "    Run its SFT, confirm the report is in range, then ask before adding it."
  elif [[ "$AGENTIC_EVAL_OK" == "0" ]]; then
    # Not a soft warning: every rollout in 12_run_grpo.sh builds a task image and
    # drives tmux inside it, so with nowhere to run containers there is nothing for
    # GRPO to be on-policy about. Starting it anyway would burn the queue slot to
    # reach the same conclusion an hour later. It also serves the policy under SGLang,
    # so a missing sglang blocks it for the same reason it blocks eval.
    if [[ "$SANDBOX_OK" == "0" ]]; then
      echo "=== RL BLOCKED: nowhere to run task containers, and every GRPO rollout needs"
      echo "    one. Not fixable from inside the training job -- see BACKENDS.md for the"
      echo "    one-flag ops ask and the off-machine backends."
    else
      echo "=== RL BLOCKED: sglang is not installed, and the rollout serves the policy"
      echo "    through it. INSTALL_ROLLOUT=1 bash scripts/01b_setup_env_verl.sh"
    fi
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
      # GRPO here is Megatron/slime only (12_run_grpo.sh), so iter_* is the right
      # name -- but see the BACKEND guard above: this branch is unreachable on verl.
      latest=$(latest_step_dir "$BASE_FOLDER/${MODEL_KEY}-rst-grpo-v1" 'iter_*')
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

# ----------------------------------------------- 10b. DPO on logged trajectories
# See DPO_PLAN.md. This is the DEFAULT post-SFT training stage: RUN_DPO=auto runs it
# whenever the SFT verdict is in range, because it needs no container, no network and
# no privilege, so it is the one continuation that cannot be blocked by the sandbox
# question. GRPO (10a) stays explicit opt-in -- it is the stronger, on-policy result
# and it is also the one that needs days of sandbox time.
#
# Running both is fine and they do not interfere: 10a trains from $RUN_NAME into
# out-hf-rl, 10b trains from out-hf-full into out-dpo. RUN_DPO=0 disables it.
RUN_DPO="${RUN_DPO:-auto}"
if [[ "$RUN_DPO" == "auto" ]]; then RUN_DPO=1; fi
DPO_TRAJ_ROOT="${DPO_TRAJ_ROOT:-$BASE_FOLDER/rst-trajectories}"
# dpo-v2 is the adopted build (--per-side 14 -> 2,673 pairs) and the one published on
# the Hub. dpo-v1 was the --per-side 5 build (1,330 pairs); do not default to it.
DPO_PAIRS_DIR="${DPO_PAIRS_DIR:-$BASE_FOLDER/dpo-v2}"
if [[ "$RUN_DPO" == "1" ]]; then
  if [[ "$CKPT_OK" != "1" ]]; then
    # DPO starts FROM this checkpoint and scores divergence from it, so a checkpoint
    # the report does not trust makes every implicit reward meaningless -- and unlike
    # a bad eval number, that failure is invisible in the loss curve.
    echo "=== DPO BLOCKED: the SFT report has FAIL findings about the checkpoint itself,"
    echo "    and DPO would use it as both policy and frozen reference. Fix these first:"
    python -c "import json;d=json.load(open('$BASE_FOLDER/verdict.json'));[print('      -',r) for r in d.get('blocking_fail_reasons',d.get('fail_reasons',[]))]" 2>/dev/null || true
  elif [[ ! -d "$DPO_TRAJ_ROOT" && ! -f "$DPO_PAIRS_DIR/dpo_train.parquet" \
          && "${DPO_FETCH_HF:-1}" != "1" ]]; then
    # Only reachable with the published-pairs fetch switched off: 33_run_dpo.sh pulls
    # them from the Hub otherwise, which needs neither the 23 GB release nor a rebuild.
    echo "=== DPO SKIPPED: DPO_FETCH_HF=0, no trajectories at $DPO_TRAJ_ROOT, and no"
    echo "    prebuilt pairs at $DPO_PAIRS_DIR. Allow the fetch, run"
    echo "    scripts/02_download.sh, or point DPO_PAIRS_DIR at a prebuilt copy."
  else
    if [[ "$IN_RANGE" != "1" ]]; then
      # The only FAILs left are the exempt ones, i.e. "this checkpoint was never
      # measured on tb-hard/tb2". Proceed, but say so out loud: the report is still a
      # FAIL and the DPO checkpoint inherits "not agentically evaluated" from it.
      echo "=== SFT verdict is out of range, but every remaining FAIL is about benchmark"
      echo "    COVERAGE, not about the checkpoint -- and DPO needs no container or"
      echo "    server. Continuing. These are the FAILs being carried forward:"
      python -c "import json;d=json.load(open('$BASE_FOLDER/verdict.json'));[print('      -',r) for r in d.get('exempt_fail_reasons',[])]" 2>/dev/null || true
      echo "    Both checkpoints must be reported as NOT agentically evaluated."
    fi
    echo "=== DPO on logged trajectories (off-policy; this is NOT GRPO)"
    # NNODES=1 by default: 19_train_dpo.py is FSDP2-only and torchrun here is local.
    # For 27B on 8x80GB the launcher's own note applies (fp32 masters + Adam is 444.8
    # GB sharded, so 8 is tight and 16 is comfortable) -- set DPO_NNODES and
    # MASTER_ADDR and run this script on each node if 8 OOMs.
    stage dpo env MODEL_KEY="$MODEL_KEY" MEM_CLASS="$MEM_CLASS" \
      NNODES="${DPO_NNODES:-1}" NGPUS="${DPO_NGPUS:-${ACTOR_NUM_GPUS_PER_NODE:-8}}" \
      TRAJ_ROOT="$DPO_TRAJ_ROOT" PAIRS_DIR="$DPO_PAIRS_DIR" \
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
