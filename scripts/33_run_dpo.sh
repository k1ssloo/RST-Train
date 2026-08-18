#!/usr/bin/env bash
# DPO on the logged RST trajectories — the container-free fallback for the RL stage.
#
#   export BASE_FOLDER=/shared/rst
#   export MODEL_KEY=qwen3.5-27b
#   bash scripts/33_run_dpo.sh
#
# WHEN TO USE THIS
#   When GRPO cannot run. Every rollout in scripts/12_run_grpo.sh builds a task
#   image and drives tmux inside it; on a pod whose AppArmor profile denies
#   mount(2) that is impossible, and if no off-machine backend is reachable either
#   (BACKENDS.md) then on-policy RL is blocked outright. This path needs no
#   container, no network and no privilege: the trajectory release already contains
#   successes and failures on the same tasks, scored by those tasks' own verifiers.
#
#   It is also a defensible warm-up when GRPO *is* available, but it is NOT GRPO.
#   DPO reweights behaviour already present in the data, and that data came from
#   other policies, so it can sharpen modes the model already has and cannot
#   discover a strategy no logged trajectory used. The summary this writes says so
#   in its own first field; keep that wording in the report.
#
# THREE STAGES, EACH SKIPPED IF ITS OUTPUT EXISTS
#   17_build_dpo_data.py    trajectories -> tokenized preference pairs   (CPU, ~25 min)
#   18_dpo_ref_logprobs.py  frozen reference logprobs                    (GPU, sharded)
#   19_train_dpo.py         the DPO step itself                          (GPU, FSDP2)
#
# COST SHAPE, 27.8B on 8x80GB, pairs at ~3k supervised tokens per side:
#   reference pass  one forward per side, no optimizer state -> ~55.6 GB weights
#                   plus a chunked LM head; shard it one process per GPU and it is
#                   roughly (pairs / 8) * 2 forwards.
#   training        fp32 masters + Adam = 444.8 GB sharded over the world, plus one
#                   side's activations at a time (see 19's split-backward note).
#                   8 GPUs is tight; 16 is comfortable.
set -uo pipefail

: "${BASE_FOLDER:?set BASE_FOLDER}"
MODEL_KEY="${MODEL_KEY:-qwen3.5-27b}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ "${MEM_CLASS:-auto}" == "auto" ]]; then
  if [[ -n "$GPU_MEM_MIB" && "$GPU_MEM_MIB" -gt 70000 ]]; then MEM_CLASS=80GB; else MEM_CLASS=40GB; fi
fi
NNODES="${NNODES:-1}"; NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NGPUS="${NGPUS:-1}"; (( NGPUS > 0 )) || NGPUS=1

# Resolve the model, but do not let the registry's parallelism arithmetic gate this
# path. 19_train_dpo.py is FSDP2-only -- no TP, PP or CP -- so a GPU count the
# registry rejects for SFT (tp*pp*cp must divide the world) is perfectly runnable
# here, and refusing to start would be a false negative. All this needs from the
# registry is the checkpoint directory name.
if REGISTRY=$(python scripts/model_registry.py --key "$MODEL_KEY" --mem-class "$MEM_CLASS" \
                --gpus "$(( NNODES * NGPUS ))" --gpus-per-node "$NGPUS" \
                --max-seq-len "${MAX_SEQ_LEN:-32768}" --shell 2>/dev/null); then
  eval "$REGISTRY"
else
  MODEL_DIR_NAME=$(python - "$MODEL_KEY" <<'EOF_PY'
import json, sys
models = json.load(open("configs/models.json"))["models"]
key = sys.argv[1]
if key not in models:
    sys.exit(f"unknown MODEL_KEY {key!r}; python scripts/model_registry.py --list")
print(models[key].get("model_dir_name") or models[key]["hf_repo"].split("/")[-1])
EOF_PY
) || exit 1
  echo "note: the registry could not place $MODEL_KEY on $(( NNODES * NGPUS )) GPUs with its"
  echo "      TP/PP/CP plan. DPO does not use any of those, so continuing with"
  echo "      MODEL_DIR_NAME=$MODEL_DIR_NAME. (SFT on this GPU count would still be blocked.)"
  MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
fi

PAIRS_DIR="${PAIRS_DIR:-$BASE_FOLDER/dpo-v1}"
REF_DIR="${REF_DIR:-$PAIRS_DIR/ref}"
TRAJ_ROOT="${TRAJ_ROOT:-$BASE_FOLDER/rst-trajectories}"
TOKENIZER="${TOKENIZER:-$BASE_FOLDER/$MODEL_DIR_NAME}"
# The policy AND the reference. They are the same checkpoint by construction: DPO
# starts from the SFT model and measures divergence from it.
POLICY="${POLICY:-$BASE_FOLDER/out-hf-full}"
OUT_DIR="${OUT_DIR:-$BASE_FOLDER/out-dpo}"
DPO_SEQ_LEN="${DPO_SEQ_LEN:-${MAX_SEQ_LEN:-32768}}"
PER_SIDE="${PER_SIDE:-14}"
# fp32 masters unless overridden. At lr 5e-7 a bf16 master rounds most updates to
# zero while a few small-magnitude weights still move, which produces a moving loss
# curve on a model that barely trained. 19 warns; this sets the safe default.
PARAM_DTYPE="${PARAM_DTYPE:-fp32}"
# Score the reference in the dtype the policy's forward will use, so 19's step-0
# calibration lands on log 2 exactly instead of leaving a ~1e-3 residual to explain.
REF_DTYPE="${REF_DTYPE:-$( (( NNODES * NGPUS > 1 )) && echo bf16 || echo "$PARAM_DTYPE" )}"
# ONE chunk size for both stages. The logprob is a sum of per-chunk terms, so a
# different chunk size regroups that sum and moves it by ~1e-8 per token -- harmless
# in itself, but it turns the step-0 identity from exact into approximate and puts
# random signs on margins that should be zero.
LOGIT_CHUNK="${DPO_LOGIT_CHUNK:-512}"

[[ -d "$POLICY" ]] || { echo "no checkpoint at POLICY=$POLICY. Run the SFT stage first, or point POLICY at an HF dir." >&2; exit 1; }
# Before the reference stage, not after: its shards redirect straight into logs/, and
# a missing directory would kill every shard on the redirect with nothing to read.
mkdir -p "$BASE_FOLDER/logs" "$OUT_DIR"

# ------------------------------------------------------------------ 1. pairs
if [[ ! -f "$PAIRS_DIR/dpo_train.parquet" ]]; then
  echo "=== building preference pairs -> $PAIRS_DIR"
  [[ -d "$TRAJ_ROOT" ]] || { echo "no trajectories at TRAJ_ROOT=$TRAJ_ROOT (scripts/02_download.sh fetches them)" >&2; exit 1; }
  python scripts/17_build_dpo_data.py \
    --traj-root "$TRAJ_ROOT" \
    --tokenizer "$TOKENIZER" \
    --out-dir "$PAIRS_DIR" \
    --per-side "$PER_SIDE" \
    --max-seq-len "$DPO_SEQ_LEN" \
    --cache "$PAIRS_DIR/reconstructed.jsonl.gz" \
    ${DPO_BUILD_ARGS:-} || exit 1
fi

# Refuse-to-train gate, read off the parquet itself rather than its manifest: the
# file may have been copied or rebuilt while a sidecar went stale, and the tensors
# are the thing being trained on.
python - "$PAIRS_DIR" "$DPO_SEQ_LEN" <<'EOF_PY' || exit 1
import sys
from pathlib import Path

import pandas as pd

root, max_len = Path(sys.argv[1]), int(sys.argv[2])
train = pd.read_parquet(root / "dpo_train.parquet")
need = ["pair_id", "chosen_input_ids", "chosen_loss_mask", "rejected_input_ids",
        "rejected_loss_mask", "common_prefix_tokens"]
for col in need:
    if col not in train.columns:
        sys.exit(f"REFUSING TO TRAIN: {root}/dpo_train.parquet has no {col!r} column. This "
                 f"must come from scripts/17_build_dpo_data.py.")

bad = []
for side in ("chosen", "rejected"):
    ids, mask = train[f"{side}_input_ids"], train[f"{side}_loss_mask"]
    if int((ids.map(len) != mask.map(len)).sum()):
        bad.append(f"{side}: len(input_ids) != len(loss_mask) on some rows; the mask does "
                   f"not line up with the tokens, so every supervised position is wrong")
    if int((mask.map(sum) == 0).sum()):
        bad.append(f"{side}: rows with zero supervised tokens. Their logprob sum is empty, "
                   f"so the pair contributes no preference signal")
    if int((ids.map(len) > max_len).sum()):
        bad.append(f"{side}: rows longer than {max_len:,}. 18 and 19 would skip them at "
                   f"different points; rebuild with --max-seq-len {max_len}")
if int((train.common_prefix_tokens <= 0).sum()):
    bad.append("pairs with no shared prefix: the two sides do not answer the same prompt, "
               "which is the one assumption DPO cannot do without")
identical = int((train.chosen_input_ids.map(tuple) == train.rejected_input_ids.map(tuple)).sum())
if identical:
    bad.append(f"{identical} pairs whose two sides are token-identical; their gradient is "
               f"exactly zero and they dilute the batch")
if train.pair_id.duplicated().any():
    bad.append("duplicate pair_id values: the same pair would be trained on more than once "
               "and the reference-logprob join would be ambiguous")
if bad:
    sys.exit("REFUSING TO TRAIN:\n  - " + "\n  - ".join(bad))

trained = train.chosen_loss_mask.map(sum).sum() + train.rejected_loss_mask.map(sum).sum()
print(f"pairs: train={len(train):,} supervised_tokens={int(trained):,} "
      f"chosen_p50={int(train.chosen_loss_mask.map(sum).median()):,} "
      f"max_len={int(max(train.chosen_input_ids.map(len).max(), train.rejected_input_ids.map(len).max())):,}")
holdout = root / "dpo_holdout.parquet"
if holdout.is_file():
    print(f"holdout: {len(pd.read_parquet(holdout)):,} pairs (disjoint task groups)")
else:
    print("WARNING: no dpo_holdout.parquet -- there will be no held-out reward accuracy, "
          "so the only evidence the run did anything is its own training loss")
EOF_PY

# ------------------------------------------------------- 2. reference logprobs
# One process per GPU, no communication: each shard writes its own parquet and 19
# reads them together. Resumable, so an interrupted pass is re-run cheaply.
if ! compgen -G "$REF_DIR/ref_logps*.parquet" > /dev/null; then
  echo "=== scoring reference logprobs ($REF_DTYPE) -> $REF_DIR"
  mkdir -p "$REF_DIR"
  pids=()
  for (( i = 0; i < NGPUS; i++ )); do
    CUDA_VISIBLE_DEVICES="$i" python scripts/18_dpo_ref_logprobs.py \
      --pairs "$PAIRS_DIR" --model-path "$POLICY" --out "$REF_DIR" \
      --dtype "$REF_DTYPE" --max-seq-len "$DPO_SEQ_LEN" \
      --logit-chunk "$LOGIT_CHUNK" \
      --shard "$i" --num-shards "$NGPUS" \
      > "$BASE_FOLDER/logs/dpo_ref_shard$i.log" 2>&1 &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  if (( rc != 0 )); then
    echo "a reference shard failed; see $BASE_FOLDER/logs/dpo_ref_shard*.log" >&2
    echo "18 is resumable -- fix the cause and re-run this script, already-scored rows are kept" >&2
    exit 1
  fi
fi

# ------------------------------------------------------------------ 3. train
# --length-normalize is ON by default here, deliberately. With summed logprobs the
# gradient norm runs ~1e3 against --max-grad-norm 1.0, so the clip becomes the whole
# step-size schedule and --lr stops meaning what it usually means (19 measures this
# and warns). Per-token normalization also removes the length term the objective
# would otherwise reward. Set DPO_LENGTH_NORM=0 to train the summed objective, and
# say which one you used in the report.
LENGTH_NORM_ARG=(--length-normalize)
[[ "${DPO_LENGTH_NORM:-1}" == "0" ]] && LENGTH_NORM_ARG=()

export WANDB_MODE="${WANDB_KEY:+online}"; export WANDB_MODE="${WANDB_MODE:-offline}"

torchrun \
  --nnodes "$NNODES" --nproc_per_node "$NGPUS" \
  --node_rank "${NODE_RANK:-0}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29501}" \
  scripts/19_train_dpo.py \
  --pairs "$PAIRS_DIR" \
  --ref-logps "$REF_DIR" \
  --model-path "$POLICY" \
  --out "$OUT_DIR" \
  --beta "${DPO_BETA:-0.1}" \
  --lr "${DPO_LR:-5e-7}" \
  --grad-accum "${DPO_GRAD_ACCUM:-4}" \
  --epochs "${DPO_EPOCHS:-1}" \
  --max-seq-len "$DPO_SEQ_LEN" \
  --param-dtype "$PARAM_DTYPE" \
  --logit-chunk "$LOGIT_CHUNK" \
  "${LENGTH_NORM_ARG[@]}" \
  "$@"
RC=$?

if (( RC != 0 )); then
  cat >&2 <<EOF

DPO failed (exit $RC). The gates in 19_train_dpo.py fail loudly on purpose:
  GATE 1 fingerprint  -> POLICY is not the checkpoint the reference was scored on.
                         Delete $REF_DIR and re-run, or point POLICY back.
  GATE 2 coverage     -> the reference pass did not finish. Re-run this script; 18
                         resumes from the partial parquet.
  GATE 3 calibration  -> something changed between the reference pass and training
                         (mask, tokenizer, dtype). Fix the cause; do not raise the
                         tolerance to get past it.
EOF
  exit "$RC"
fi

cat <<EOF

Done. $OUT_DIR/dpo_training_summary.json is the artifact to read, and to quote from:
  * gates.step0_loss must be ~0.693147 (log 2). That is the proof the frozen
    reference and the policy at initialization were the same model. With
    gates.dtype_match true it is exact.
  * metrics.holdout_reward_accuracy is LIKELIHOOD RANKING on held-out task groups:
    how often the model assigns higher likelihood to a trajectory that passed the
    verifier than to one that failed the same task. 0.5 is no preference. It is not
    a pass rate and must never be compared with terminal-bench numbers.
  * optimization.clip_active_fraction says whether --lr or --max-grad-norm set the
    real step size. Quote whichever one did.
  * warnings[] is empty on a clean run. If it is not, it belongs in the report.

The checkpoint is $OUT_DIR/hf. To evaluate it without a sandbox, use the SFT
holdout -- 06b takes an SFT/pretokenized parquet, not a pairs file -- and pass the
SFT model as --base-model so the delta is DPO's and not the fine-tune's:
  python scripts/06b_eval_offline.py --model-path $OUT_DIR/hf \\
    --base-model $POLICY \\
    --holdout ${DATA_DIR:-$BASE_FOLDER/sft-v1-cap10}/pretokenized_holdout.parquet \\
    --out $BASE_FOLDER/eval/dpo-offline
Agentic eval still needs a container; DPO removes that requirement from TRAINING
only. A DPO checkpoint with no agentic eval is an untested checkpoint, and the
report has to say that.
EOF
