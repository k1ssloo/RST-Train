#!/usr/bin/env bash
# Qwen3.5-27B SFT on 4 nodes x 8 A100 (32 GPUs) with slime + Megatron.
#
#   export BASE_FOLDER=/shared/rst
#   export SLIME_DIR=$BASE_FOLDER/slime
#   export MASTER_ADDR=<head node IP>
#   export HOSTFILE=$BASE_FOLDER/hostfile      # one IP per line, head first
#   export MODEL_KEY=qwen3.5-27b               # see: python scripts/model_registry.py --list
#   export MEM_CLASS=80GB                      # or 40GB / 40GB-alt  (or auto)
#   export WANDB_KEY=...                       # omit for offline mode
#   bash scripts/05_run_sft.sh
#
# Paper reference (arXiv:2608.05466v3, Appendix F): 64 GPUs, SLIME on Ray,
# TP4/PP2/CP2, Adam + cosine decay + optimizer CPU offload, flash attention,
# 1 epoch over 10,778 examples, global batch size 128, max context 262,145,
# LR 3e-6 -> 3e-7. Appendix F states the launcher "provides the same training
# path for four-node runs", which is what this script is.
#
# TWO DELIBERATE DEVIATIONS FROM THE PAPER, both justified by measurement:
#  1. --max-seq-len 32768, not 262145. Measured on the actual release: the SFT
#     sequence length is p50 8.0K / p90 17.0K / p99 28.2K / max 32.3K after the
#     32K cap (see data/sft-v1/manifest.json). 262,145 is a launcher ceiling, not
#     a data property; budgeting for it would waste most of the activation memory.
#  2. 32 GPUs, so DP is halved vs the paper. Global batch size is preserved at
#     128 through gradient accumulation, so the optimization trajectory matches.
set -ex

pkill -9 sglang || true; sleep 2
ray stop --force || true; pkill -9 ray || true; pkill -9 python || true; sleep 2

: "${BASE_FOLDER:?set BASE_FOLDER}"
: "${MASTER_ADDR:?set MASTER_ADDR}"
SLIME_DIR="${SLIME_DIR:-$BASE_FOLDER/slime}"
DATA_DIR="${DATA_DIR:-$BASE_FOLDER/sft-v1-cap10}"
MODEL_KEY="${MODEL_KEY:-qwen3.5-27b}"
RUN_NAME="${RUN_NAME:-${MODEL_KEY}-rst-sft-v1}"

export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
SOCKET_IFNAME="${SOCKET_IFNAME:-eth0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
NUM_EPOCH="${NUM_EPOCH:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
LR="${LR:-3e-6}"
MIN_LR="${MIN_LR:-3e-7}"

# ---- hardware-derived switches ----------------------------------------------
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$(( NVLINK_COUNT > 0 ? 1 : 0 ))

# A100 = SM80: FlashQLA needs SM90+, so the gated-delta-net layers must use fla.
if [[ "$COMPUTE_CAP" == "8.0" ]]; then GDN_BACKEND=fla; else GDN_BACKEND="${GDN_BACKEND:-fla}"; fi

if [[ "${MEM_CLASS:-auto}" == "auto" ]]; then
  if   (( GPU_MEM_MIB > 70000 )); then MEM_CLASS=80GB
  elif (( HAS_NVLINK == 1 ));     then MEM_CLASS=40GB
  else                                 MEM_CLASS=40GB-alt; fi
fi

# All parallelism comes from configs/models.json via the registry, which validates
# tp*pp*cp*dp == gpus, tp inside one node, and max_tokens_per_gpu*cp >= max_seq_len.
# It exits non-zero on an impossible config, so we fail here rather than 40 minutes in.
TOTAL_GPUS=$(( ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE ))
eval "$(python scripts/model_registry.py --key "$MODEL_KEY" --mem-class "$MEM_CLASS" \
          --gpus "$TOTAL_GPUS" --gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" \
          --max-seq-len "$MAX_SEQ_LEN" --shell)"

echo "=== model=$MODEL_KEY (${PARAMS_B}B, ${N_LAYERS} layers, moe=$IS_MOE, vision=$HAS_VISION)"
echo "=== cc=$COMPUTE_CAP mem=${GPU_MEM_MIB}MiB nvlink=$HAS_NVLINK class=$MEM_CLASS_USED gdn=$GDN_BACKEND"
echo "=== TP=$TP PP=$PP CP=$CP DP=$DP EP=$EP tokens/gpu=$MAX_TOKENS_PER_GPU seq=$MAX_SEQ_LEN"
echo "=== est. ${EST_EPOCH_MINUTES} min/epoch"

source "${SLIME_DIR}/scripts/models/${SLIME_SPEC}"

CKPT_ARGS=(
   --hf-checkpoint "${BASE_FOLDER}/${MODEL_DIR_NAME}"
   --ref-load      "${BASE_FOLDER}/${MODEL_DIR_NAME}_torch_dist/"
   --load          "${BASE_FOLDER}/${RUN_NAME}/"
   --save          "${BASE_FOLDER}/${RUN_NAME}/"
   --save-interval 20
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${DATA_DIR}/rst_sft_train.parquet"
   --input-key messages
   --rollout-shuffle
   --num-epoch "${NUM_EPOCH}"
   --rollout-batch-size "${GLOBAL_BATCH_SIZE}"
   --global-batch-size  "${GLOBAL_BATCH_SIZE}"

   --loss-type sft_loss
   # THE critical flag: masks everything except assistant content. Default is
   # "qwen", which mis-segments the Qwen3.5 template -> you would train on the
   # terminal output and the harness prompt. Verified locally against
   # slime/utils/mask_utils.py: 32.98% of tokens trained, 0 user-turn leakage.
   --loss-mask-type "${LOSS_MASK_TYPE}"
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

# The last PP stage also carries the 248320-row lm_head (~1.27B params at
# hidden 5120), so models with an uneven split give it fewer layers.
PP_SPLIT_ARGS=()
if [[ "${PP}" -gt 1 && "${DECODER_LAST_PP_LAYERS}" -gt 0 ]]; then
  PP_SPLIT_ARGS=(--decoder-last-pipeline-num-layers "${DECODER_LAST_PP_LAYERS}")
fi
MOE_ARGS=()
if [[ "${IS_MOE}" == "1" && -n "${MOE_FLAGS}" ]]; then
  # shellcheck disable=SC2206
  MOE_ARGS=(${MOE_FLAGS})
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TP}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP}"
   --context-parallel-size "${CP}"
   # Expert parallelism is 1 for dense models; the registry sets EP for MoE.
   --expert-model-parallel-size "${EP}"
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --qwen-gdn-backend "${GDN_BACKEND}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style cosine
   --min-lr "${MIN_LR}"
   --lr-warmup-fraction 0.03
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-distributed-optimizer
   # 27.8B Adam states in fp32 are ~334GB; offloading them is what makes this
   # fit. Confirm host RAM headroom with scripts/00_preflight.sh first.
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

if [[ -n "${WANDB_KEY:-}" ]]; then
  WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT:-rst-qwen35-27b}"
              --wandb-group "${RUN_NAME}" --wandb-key "${WANDB_KEY}")
else
  WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT:-rst-qwen35-27b}"
              --wandb-group "${RUN_NAME}")
  export WANDB_MODE=offline
  export WANDB_DIR="${BASE_FOLDER}/wandb"
  mkdir -p "$WANDB_DIR"
  echo "WANDB_KEY unset -> offline mode in $WANDB_DIR (sync later: wandb sync $WANDB_DIR/offline-*)"
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ---- Ray cluster ------------------------------------------------------------
export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

if [[ -n "${HOSTFILE:-}" ]]; then
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]] && continue
    ssh "${SSH_USER:-root}@${WORKER_IP}" \
      "pkill -9 sglang; ray stop --force; pkill -9 python; \
       ray start --address=${MASTER_ADDR}:6379 --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} \
       --node-ip-address ${WORKER_IP} --disable-usage-stats" &
  done
  wait
fi

# Ethernet-only clusters: NCCL will otherwise hunt for RDMA and hang.
IB_ENV=""
if [[ ! -d /sys/class/infiniband ]] || [[ -z "$(ls -A /sys/class/infiniband 2>/dev/null)" ]]; then
  IB_ENV='"NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "'"${SOCKET_IFNAME}"'",'
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    ${IB_ENV}
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "${BASE_FOLDER}/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${PP_SPLIT_ARGS[@]}" \
   "${MOE_ARGS[@]}" \
   "${MISC_ARGS[@]}"
