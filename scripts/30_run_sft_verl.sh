#!/usr/bin/env bash
# SFT on verl's FSDP engine — the non-Megatron path.
#
#   export BASE_FOLDER=/shared/rst
#   export MODEL_KEY=qwen3.5-9b
#   bash scripts/30_run_sft_verl.sh
#
# WHEN TO USE THIS INSTEAD OF scripts/05_run_sft.sh (slime + Megatron)
#   - you cannot get the Megatron/TE/apex stack to build
#   - Megatron context parallelism turns out to be wrong on the gated-delta-net
#     layers (see PLAN.md §5; torchtitan lists CP for this architecture as TODO,
#     which is independent corroboration that it is not settled)
#   - you want one framework for both SFT and RL without Megatron
#
# WHAT YOU GIVE UP
#   No context parallelism. verl's FSDP engine shards parameters, not the
#   sequence, so one GPU must hold a whole 32K sequence's activations. That is
#   affordable ONLY with a fused/chunked cross-entropy, because this model's vocab
#   is 248,320: materialized logits for a 32K sequence are 16.3 GiB in bf16 and
#   ~32.6 GiB if the loss upcasts to fp32 — larger than the activations. Hence
#   Liger below; it has a qwen3_5 kernel.
#
# ROUGH BUDGET, 27.8B on 32x80GB, seq 32K, fused CE:
#   sharded params+grads+Adam  444.8 GB / 32  = 13.9 GB/GPU
#   activations (full recompute, 64 layers)   = 21.5 GB/GPU
#   working set                               ~  2   GB/GPU
#                                             ------------
#                                             ~ 37   GB/GPU   -> fits 80GB
#   On 40GB cards this does not fit; use optimizer CPU offload and/or drop to
#   16K sequences, and say so in the report because it changes what the run means.
set -ex

: "${BASE_FOLDER:?set BASE_FOLDER}"
MODEL_KEY="${MODEL_KEY:-qwen3.5-9b}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [[ "${MEM_CLASS:-auto}" == "auto" ]]; then
  if (( GPU_MEM_MIB > 70000 )); then MEM_CLASS=80GB; else MEM_CLASS=40GB; fi
fi
NNODES="${NNODES:-4}"; NGPUS="${NGPUS:-8}"
eval "$(python scripts/model_registry.py --key "$MODEL_KEY" --mem-class "$MEM_CLASS" \
          --gpus "$(( NNODES * NGPUS ))" --gpus-per-node "$NGPUS" \
          --max-seq-len "${MAX_SEQ_LEN:-32768}" --shell)"

DATA_DIR="${DATA_DIR:-$BASE_FOLDER/sft-v1-cap10}"
PRETOK="$DATA_DIR/pretokenized_train.parquet"
RUN_NAME="${RUN_NAME:-${MODEL_KEY}-rst-sft-verl}"

# The pre-tokenized file is what makes this path safe. verl's own multi-turn
# dataset tokenizes message-by-message, which for Qwen3.5 produces one empty
# <think> block per assistant turn instead of one per conversation -- measured at
# 100% of rows. We bypass it entirely.
if [[ ! -f "$PRETOK" ]]; then
  echo "building $PRETOK"
  python scripts/15_export_pretokenized.py \
    --parquet "$DATA_DIR/rst_sft_train.parquet" \
    --tokenizer "$BASE_FOLDER/$MODEL_DIR_NAME" \
    --out "$PRETOK" --max-seq-len "${MAX_SEQ_LEN:-32768}"
fi

# Sanity gate: refuse to train if the export dropped rows for a contract failure.
python - "$DATA_DIR" <<'EOF_PY'
import json, sys
from pathlib import Path
m = Path(sys.argv[1]) / "pretokenized_train_manifest.json"
d = json.loads(m.read_text())
bad = d["dropped"]["contract"] + d["dropped"]["error"]
print(f"pretokenized: rows={d['rows_out']} trained_fraction={d['trained_fraction']}")
if bad:
    sys.exit(f"REFUSING TO TRAIN: {bad} rows failed the chat-template contract. "
             f"The mask is not trustworthy; fix the export first.")
if not (0.25 <= d["trained_fraction"] <= 0.45):
    sys.exit(f"REFUSING TO TRAIN: trained_fraction={d['trained_fraction']} is outside the "
             f"0.25-0.45 band measured for this dataset. A mask bug is the likely cause.")
EOF_PY

export WANDB_MODE="${WANDB_KEY:+online}"; export WANDB_MODE="${WANDB_MODE:-offline}"
[[ -n "${WANDB_KEY:-}" ]] && export WANDB_API_KEY="$WANDB_KEY"
export WANDB_DIR="${BASE_FOLDER}/wandb"; mkdir -p "$WANDB_DIR"

# FSDP shards params; TP is used only if the registry asked for it. `ulysses` is
# verl's sequence-parallel knob and is the closest analogue to Megatron CP if you
# do need to shard the sequence -- it is NOT enabled here because it has not been
# validated on the gated-delta-net layers either.
torchrun \
  --nnodes "$NNODES" --nproc_per_node "$NGPUS" \
  --node_rank "${NODE_RANK:-0}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29500}" \
  -m verl.trainer.sft_trainer \
  --config-name sft_trainer_engine \
  data.train_files="$PRETOK" \
  data.custom_cls.path="$REPO_DIR/verl_backend/rst_sft_dataset.py" \
  data.custom_cls.name=RSTPretokenizedSFTDataset \
  data.pad_mode=no_padding \
  data.use_dynamic_bsz=True \
  data.max_length="${MAX_SEQ_LEN:-32768}" \
  data.max_token_len_per_gpu="$MAX_TOKENS_PER_GPU" \
  data.train_batch_size="$GLOBAL_BATCH_SIZE" \
  data.truncation=error \
  model.path="$BASE_FOLDER/$MODEL_DIR_NAME" \
  model.use_liger=True \
  model.enable_gradient_checkpointing=True \
  engine.strategy=fsdp2 \
  engine.tensor_parallel_size="$TP" \
  optim.lr="$LR" \
  optim.lr_scheduler_type=cosine \
  optim.min_lr_ratio=0.1 \
  optim.warmup_steps_ratio="$LR_WARMUP_FRACTION" \
  optim.weight_decay=0.1 \
  optim.betas="[0.9,0.98]" \
  trainer.total_epochs="$NUM_EPOCH" \
  trainer.project_name="${WANDB_PROJECT:-rst-qwen35-verl}" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$BASE_FOLDER/$RUN_NAME" \
  trainer.logger="['console','wandb']" \
  trainer.save_freq=20 \
  "$@"

cat <<EOF

Done. Notes for the report:
  * model.use_liger=True is load-bearing here, not an optimization: without a
    fused cross-entropy the 248,320-row logits dominate memory at 32K.
  * engine.strategy=fsdp2 shards parameters only. If you enabled ulysses sequence
    parallelism to fit longer sequences, SAY SO -- it has not been validated on the
    gated-delta-net layers, same open question as Megatron CP.
  * Convert to HF and restore the vision tower before evaluating:
      python scripts/07_restore_vision.py --trained <hf_out> \\
        --original $BASE_FOLDER/$MODEL_DIR_NAME --out <hf_out>-full
    then scripts/06_eval.py as usual. The eval path is backend-independent.
EOF
