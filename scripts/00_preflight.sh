#!/usr/bin/env bash
# Preflight: detect the facts that decide the training config, then print the
# recommended parallelism row. Run on EVERY node; run the shared-FS check from
# the head node only.
#
#   bash scripts/00_preflight.sh            # local node report
#   bash scripts/00_preflight.sh --hostfile hostfile   # fan out over all nodes
set -uo pipefail

HOSTFILE=""
[[ "${1:-}" == "--hostfile" ]] && HOSTFILE="${2:-}"

report_local() {
  echo "================ NODE $(hostname) ================"

  echo "--- GPU ---"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version \
               --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    # MiB -> round to nearest 10GB bucket
    GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    echo "GPU_COUNT=${GPU_COUNT}"
    echo "GPU_MEM_MIB=${GPU_MEM_MIB}"
    echo "COMPUTE_CAP=${CC}"
    if   (( GPU_MEM_MIB > 70000 )); then echo "MEM_CLASS=80GB"
    elif (( GPU_MEM_MIB > 30000 )); then echo "MEM_CLASS=40GB"
    else                                 echo "MEM_CLASS=UNSUPPORTED"; fi
    case "$CC" in
      8.0) echo "GDN_BACKEND=fla    # A100/SM80: FlashQLA needs SM90+, must use fla" ;;
      9.0|10.*|12.*) echo "GDN_BACKEND=flashqla  # SM90+: flashqla available" ;;
      *)   echo "GDN_BACKEND=fla    # unknown cc, default to fla" ;;
    esac
    echo "FP8_CAPABLE=$( [[ "$CC" == 8.0 ]] && echo no || echo yes )"
  else
    echo "nvidia-smi MISSING"
  fi

  echo "--- NVLink (intra-node TP domain) ---"
  NV=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
  echo "NVLINK_LINK_COUNT=${NV}"
  if (( NV > 0 )); then echo "HAS_NVLINK=1  # TP up to 8 within a node is fine"
  else echo "HAS_NVLINK=0  # PCIe only: keep TP<=4 and expect lower MFU"; fi
  nvidia-smi topo -m 2>/dev/null | head -12

  echo "--- InfiniBand / RDMA (inter-node PP+DP domain) ---"
  if [[ -d /sys/class/infiniband ]] && [[ -n "$(ls -A /sys/class/infiniband 2>/dev/null)" ]]; then
    echo "HAS_IB=1"
    for d in /sys/class/infiniband/*; do
      echo "  $(basename "$d") rate=$(cat "$d"/ports/1/rate 2>/dev/null) state=$(cat "$d"/ports/1/state 2>/dev/null)"
    done
  else
    echo "HAS_IB=0  # Ethernet only: set NCCL_IB_DISABLE=1 and pin SOCKET_IFNAME"
  fi
  echo "--- network interfaces ---"
  ip -o -4 addr show 2>/dev/null | awk '{print "  "$2" "$4}'

  echo "--- CPU / RAM (Adam CPU offload needs headroom) ---"
  echo "CPU_CORES=$(nproc)"
  free -g | awk 'NR<=2{print "  "$0}'
  echo "  NOTE: optimizer CPU offload for 27.8B needs >=350GB host RAM per node at DP1;"
  echo "        scale by 1/DP. Verify before enabling --optimizer-cpu-offload."

  echo "--- disk ---"
  df -h / /tmp "${BASE_FOLDER:-/root}" 2>/dev/null | sort -u

  echo "--- docker (needed only for RL rollout sandboxes) ---"
  if command -v docker >/dev/null; then
    docker info --format 'DOCKER_OK server={{.ServerVersion}} root={{.DockerRootDir}}' 2>&1 | head -2
  else
    echo "docker MISSING"
  fi

  echo "--- egress (HF / wandb) ---"
  for url in https://huggingface.co https://api.wandb.ai; do
    code=$(curl -sS -o /dev/null -m 10 -w '%{http_code}' "$url" 2>/dev/null || echo FAIL)
    echo "  $url -> $code"
  done
}

if [[ -n "$HOSTFILE" ]]; then
  while read -r host _; do
    [[ -z "$host" ]] && continue
    ssh -o StrictHostKeyChecking=no "$host" "bash -s" < "$0"
  done < "$HOSTFILE"
  echo
  echo "================ SHARED FS CHECK ================"
  probe="${BASE_FOLDER:-/root}/.sharedfs_probe_$$"
  echo probe > "$probe" 2>/dev/null || { echo "cannot write ${BASE_FOLDER:-/root}"; exit 1; }
  ok=1
  while read -r host _; do
    [[ -z "$host" ]] && continue
    if ssh -o StrictHostKeyChecking=no "$host" "test -f $probe" 2>/dev/null; then
      echo "  $host: SEES probe -> shared"
    else
      echo "  $host: does NOT see probe -> local-only"; ok=0
    fi
  done < "$HOSTFILE"
  rm -f "$probe"
  echo "SHARED_FS=$ok"
else
  report_local
fi

cat <<'EOF'

================ CONFIG DECISION TABLE (Qwen3.5-27B SFT, 32 GPUs, seq<=32768) ==
Pick the row matching MEM_CLASS. TP must stay inside one node (TP<=8, and TP<=4
without NVLink). CP is what makes a 32K sequence fit: per-GPU tokens = seq/CP.

  MEM_CLASS  TP  PP  CP  DP  --max-tokens-per-gpu  optimizer-cpu-offload
  80GB        4   2   2   2   16384                 yes
  40GB        8   2   2   1    8192                 yes  (requires NVLink)
  40GB-alt    4   2   4   1    8192                 yes  (no NVLink; = slime default)

TP*PP*CP*DP must equal 32. Fallback ladder if a config OOMs or CP misbehaves on
the gated-delta-net layers:
  1. halve --max-tokens-per-gpu
  2. raise CP (2 -> 4), lowering DP
  3. raise PP (2 -> 4)
  4. last resort: --max-seq-len 16384 (drops ~10% of examples, biased toward the
     long/hard trajectories you most want -- prefer 1-3 first)
EOF
