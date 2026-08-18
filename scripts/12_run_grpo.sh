#!/usr/bin/env bash
# Agentic GRPO for Qwen3.5-27B on RST terminal tasks, 4 nodes x 8 A100.
#
#   export BASE_FOLDER=/shared/rst SLIME_DIR=$BASE_FOLDER/slime
#   export MASTER_ADDR=<head-ip> HOSTFILE=$BASE_FOLDER/hostfile
#   export RST_DOCKER_HOST=unix:///run/user/1000/docker.sock   # dedicated daemon
#   export ADAPTER_PUBLIC_HOST=<head-ip>
#   export INIT_CKPT=$BASE_FOLDER/qwen35-27b-rst-sft-v1        # or the base model
#   bash scripts/12_run_grpo.sh
#
# Derived from slime's own scripts/run-qwen3.5-27B.sh (which already defaults to
# 4 nodes x 8 GPUs) with the math/DAPO rollout swapped for the Harbor/Terminus-2
# agentic rollout in rl/generate.py.
#
# WHY GRPO AND NOT THE PAPER'S PPO
#   The paper (§5.5) used agentic PPO with a critic warm-loaded from an earlier
#   terminal-agent critic. A 27.8B critic adds ~55.6GB of bf16 params plus ~334GB
#   of fp32 Adam state -- roughly doubling optimizer memory on a cluster that is
#   already half the paper's size. GRPO drops the critic and estimates the
#   baseline from the group, and slime ships this exact GRPO config for this exact
#   model. The paper's other RL settings are preserved verbatim below:
#   KL penalty off, entropy bonus off, eps-clip 0.2.
set -ex

pkill -9 sglang || true; sleep 3
ray stop --force || true; pkill -9 ray || true; pkill -9 python || true; sleep 3

: "${BASE_FOLDER:?set BASE_FOLDER}"
: "${MASTER_ADDR:?set MASTER_ADDR}"
: "${RST_DOCKER_HOST:?set RST_DOCKER_HOST to a dedicated (non-default) docker socket}"
: "${ADAPTER_PUBLIC_HOST:=$MASTER_ADDR}"
SLIME_DIR="${SLIME_DIR:-$BASE_FOLDER/slime}"
TASKSET="${TASKSET:-$BASE_FOLDER/rl-sweet}"
MODEL_KEY="${MODEL_KEY:-qwen3.5-27b}"
INIT_CKPT="${INIT_CKPT:-$BASE_FOLDER/${MODEL_KEY}-rst-sft-v1}"
RUN_NAME="${RUN_NAME:-${MODEL_KEY}-rst-grpo-v1}"

export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
SOCKET_IFNAME="${SOCKET_IFNAME:-eth0}"

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$(( NVLINK_COUNT > 0 ? 1 : 0 ))
[[ "$COMPUTE_CAP" == "8.0" ]] && GDN_BACKEND=fla || GDN_BACKEND="${GDN_BACKEND:-fla}"
if [[ "${MEM_CLASS:-auto}" == "auto" ]]; then
  if (( GPU_MEM_MIB > 70000 )); then MEM_CLASS=80GB; else MEM_CLASS=40GB; fi
fi

# RL parallelism comes from the registry's rl_* rows (colocated rollout engine, so
# tighter than SFT) and is validated there.
TOTAL_GPUS=$(( ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE ))
eval "$(python scripts/model_registry.py --key "$MODEL_KEY" --phase rl --mem-class "$MEM_CLASS" \
          --gpus "$TOTAL_GPUS" --gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" \
          --max-seq-len "${RL_MAX_SEQ_LEN:-32768}" --shell)"
MAX_RESP="${ROLLOUT_MAX_RESPONSE_LEN}"
echo "=== model=$MODEL_KEY (${PARAMS_B}B) cc=$COMPUTE_CAP mem=${GPU_MEM_MIB}MiB gdn=$GDN_BACKEND"
echo "=== TP=$TP PP=$PP CP=$CP DP=$DP EP=$EP mtpg=$MAX_TOKENS_PER_GPU resp=$MAX_RESP"

# ---- rollout capacity: sandboxes are the bottleneck, not the GPUs ------------
CPU_CORES=$(nproc)
RAM_GB=$(free -g | awk 'NR==2{print $2}')
# Each Terminus-2 rollout = 1+ containers + a tmux session. Budget ~2 cores and
# ~4GB per concurrent sandbox, and never exceed half the cores (training needs
# them for the CPU-offloaded optimizer).
MAX_SANDBOXES="${RST_MAX_SANDBOXES:-$(( CPU_CORES / 4 < RAM_GB / 8 ? CPU_CORES / 4 : RAM_GB / 8 ))}"
(( MAX_SANDBOXES < 1 )) && MAX_SANDBOXES=1
echo "=== cores=$CPU_CORES ram=${RAM_GB}G -> RST_MAX_SANDBOXES=$MAX_SANDBOXES per node"

source "${SLIME_DIR}/scripts/models/${SLIME_SPEC}"

CKPT_ARGS=(
   --hf-checkpoint "${BASE_FOLDER}/${MODEL_DIR_NAME}"
   --ref-load      "${INIT_CKPT}_torch_dist/"
   --load          "${BASE_FOLDER}/${RUN_NAME}/"
   --save          "${BASE_FOLDER}/${RUN_NAME}/"
   --save-interval 10
)

ROLLOUT_ARGS=(
   --custom-generate-function-path rl.generate.generate
   --prompt-data "${TASKSET}/rl_tasks.jsonl"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle

   # group size 8 = 8 rollouts of the same task -> the GRPO baseline. Do not set
   # this to 1; without a group there is no advantage signal at all.
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --num-rollout 200
   --global-batch-size 64

   --rollout-max-response-len "${MAX_RESP}"
   --rollout-max-context-len 65536
   --rollout-temperature 1.0
   --balance-data
   --save-debug-rollout-data "${BASE_FOLDER}/${RUN_NAME}/rollout_dumps/rollout_{rollout_id}.pt"
)

# Paper §5.5 verbatim: KL penalty and entropy bonus disabled, PPO clip eps=0.2.
GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
)

PP_SPLIT_ARGS=()
if [[ "${PP}" -gt 1 && "${DECODER_LAST_PP_LAYERS}" -gt 0 ]]; then
  PP_SPLIT_ARGS=(--decoder-last-pipeline-num-layers "${DECODER_LAST_PP_LAYERS}")
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TP}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP}"
   --context-parallel-size "${CP}"
   --expert-model-parallel-size "${EP}"
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --calculate-per-token-loss
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --qwen-gdn-backend "${GDN_BACKEND}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static 0.65
   # Hybrid linear-attention models need the mamba-aware scheduler.
   --sglang-mamba-scheduler-strategy extra_buffer
   # NOTE: slime's shipped script enables EAGLE speculative decoding. It relies on
   # the MTP head, which a text-only Megatron round trip drops. Leave it OFF until
   # correctness is established, then re-enable purely for rollout throughput.
   # --sglang-speculative-algorithm EAGLE
   # --sglang-speculative-num-steps 3
   # --sglang-speculative-eagle-topk 1
   # --sglang-speculative-num-draft-tokens 4
)

if [[ -n "${WANDB_KEY:-}" ]]; then
  WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT:-rst-qwen35-27b-rl}"
              --wandb-group "${RUN_NAME}" --wandb-key "${WANDB_KEY}")
else
  WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT:-rst-qwen35-27b-rl}" --wandb-group "${RUN_NAME}")
  export WANDB_MODE=offline WANDB_DIR="${BASE_FOLDER}/wandb"; mkdir -p "$WANDB_DIR"
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

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
    "PYTHONPATH": "${BASE_FOLDER}/Megatron-LM/:${PWD}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "ADAPTER_PUBLIC_HOST": "${ADAPTER_PUBLIC_HOST}",
    "ADAPTER_PORT": "${ADAPTER_PORT:-18101}",
    "RST_DOCKER_HOST": "${RST_DOCKER_HOST}",
    "RST_MAX_SANDBOXES": "${MAX_SANDBOXES}",
    "RST_AGENT": "terminus-2",
    "RST_AGENT_TIMEOUT_SEC": "${RST_AGENT_TIMEOUT_SEC:-1800}",
    "RST_JOBS_ROOT": "${RST_JOBS_ROOT:-/tmp/rst-rl-jobs}",
    "RST_SERVED_MODEL": "${RST_SERVED_MODEL:-hosted_vllm/rst-policy}"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${PP_SPLIT_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
