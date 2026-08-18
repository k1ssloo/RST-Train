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

# Enter the training env before the first `python`. 20_run_all.sh already did it and
# rst_enter_env then just confirms it (one find_spec, no re-activation), but this
# script is also documented as a standalone entry point, and torchrun launching an
# interpreter without verl in it fails deep inside the worker with a traceback that
# looks like a verl bug.
# shellcheck source=lib_env.sh
source "$REPO_DIR/scripts/lib_env.sh"
rst_bootstrap_python || exit 2
rst_enter_env "${ENV_NAME:-rstverl}" || exit 2

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

# Sanity gate. Verified against the PARQUET ITSELF, not a sidecar manifest: the file
# may have been downloaded (HF ships its manifest under a different name) or copied,
# and a manifest can be stale while the data is not. Checking the actual tensors is
# both stronger and provenance-independent.
python - "$PRETOK" "${MAX_SEQ_LEN:-32768}" <<'EOF_PY'
import sys
import pandas as pd

path, max_len = sys.argv[1], int(sys.argv[2])
df = pd.read_parquet(path)
for col in ("input_ids", "loss_mask"):
    if col not in df.columns:
        sys.exit(f"REFUSING TO TRAIN: {path} has no {col!r} column. This must be PRE-TOKENIZED "
                 f"data from scripts/15_export_pretokenized.py, not a `messages` parquet.")

n_tok = df.input_ids.map(len)
n_msk = df.loss_mask.map(len)
misaligned = int((n_tok != n_msk).sum())
empty = int((df.loss_mask.map(sum) == 0).sum())
too_long = int((n_tok > max_len).sum())
total = int(n_tok.sum())
trained = int(df.loss_mask.map(sum).sum())
frac = trained / max(1, total)

print(f"pretokenized: rows={len(df)} tokens={total:,} trained={trained:,} ({frac:.2%}) "
      f"max_len={int(n_tok.max())}")

if misaligned:
    sys.exit(f"REFUSING TO TRAIN: {misaligned} rows have len(input_ids) != len(loss_mask). "
             f"The mask does not line up with the tokens; every masked position would be "
             f"wrong. Re-export.")
if empty:
    sys.exit(f"REFUSING TO TRAIN: {empty} rows have zero trained tokens. They contribute no "
             f"gradient and, with per-token loss normalization, skew the average. Re-export.")
if too_long:
    sys.exit(f"REFUSING TO TRAIN: {too_long} rows exceed max_seq_len={max_len}. Re-export with "
             f"--max-seq-len {max_len} so the drop is counted, or raise the limit.")
# 32.42% measured for cap10. A mask bug typically lands far outside this band: ~100%
# means nothing is masked (training on the harness prompt and terminal output), ~0%
# means everything is.
# This band is tighter than the 0.15-0.55 one in 16_smoke_forward_backward.py on purpose:
# here the fraction covers every row at full length, there it covers 4 rows truncated to
# --seq-len, which has both sampling spread and a truncation bias toward the untrained
# preamble. Tight where the measurement is stable, loose where it is not.
if not (0.25 <= frac <= 0.45):
    sys.exit(f"REFUSING TO TRAIN: trained fraction {frac:.2%} is outside the 0.25-0.45 band "
             f"measured for this dataset. Near 100% means the mask is absent (you would train "
             f"on terminal output); near 0% means it masks everything. Investigate before "
             f"spending GPU time.")
EOF_PY


export WANDB_MODE="${WANDB_KEY:+online}"; export WANDB_MODE="${WANDB_MODE:-offline}"
[[ -n "${WANDB_KEY:-}" ]] && export WANDB_API_KEY="$WANDB_KEY"
export WANDB_DIR="${BASE_FOLDER}/wandb"; mkdir -p "$WANDB_DIR"

# By default verl's FSDP checkpointer writes sharded weights only: each
# global_step_<n>/ gets model_world_size_*_rank_*.pt plus a huggingface/ directory
# holding config.json and the tokenizer and NO weights. 20_run_all.sh handles that
# by merging the shards afterwards (`python -m verl.model_merger merge --backend
# fsdp`), which costs a second full read of the checkpoint.
#
# SAVE_HF_MODEL=1 asks the checkpointer to write HF weights directly instead. It is
# opt-in rather than the default because the config key for save_contents has moved
# between verl versions, and an unknown key makes hydra abort the launch. If it
# aborts with "Could not override 'trainer.checkpoint.save_contents'", drop the flag
# (the merge path works regardless) or point SAVE_HF_MODEL_KEY at the right path.
CKPT_ARGS=()
if [[ "${SAVE_HF_MODEL:-0}" == "1" ]]; then
  CKPT_ARGS+=("${SAVE_HF_MODEL_KEY:-trainer.checkpoint.save_contents}=['model','optimizer','extra','hf_model']")
  echo "SAVE_HF_MODEL=1 -> ${CKPT_ARGS[0]}"
fi

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
  "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}" \
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
