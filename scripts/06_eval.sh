#!/usr/bin/env bash
# Evaluate a checkpoint the way the paper does: a local OpenAI-compatible SGLang
# endpoint + Harbor + Terminus-2, with Docker as the sandbox backend.
#
#   export BASE_FOLDER=/shared/rst
#   bash scripts/06_eval.sh /shared/rst/out-hf-full 4
#
# Paper Appendix F: "Evaluation uses a local OpenAI-compatible SGLang endpoint
# together with Harbor and Daytona. Each run starts the model server, verifies
# readiness with a chat-completions request, generates a Harbor configuration
# with trajectory recording, and launches sandbox evaluation. ... Three
# evaluation lanes run on separate GPU nodes."
# We substitute Docker for Daytona (you have Docker on the training nodes).
#
# Reference numbers to beat (Table 3/4, pass rate %, mean over 3 runs):
#            TB2          TB-Hard      LHTB
#   base     41.20+-1.72  22.67+-2.52  18.10+-0.89
#   SFT R3   47.94+-2.34  28.33+-0.58  22.44+-0.40
#   RL       49.44        32.00        22.07
set -ex
: "${BASE_FOLDER:?set BASE_FOLDER}"
MODEL_PATH="${1:?usage: $0 <hf_model_dir> [tp]}"
TP="${2:-4}"
PORT="${PORT:-30000}"
SERVED_NAME="${SERVED_NAME:-Qwen3.5-27B-RST-SFT}"
CONCURRENCY="${CONCURRENCY:-8}"
N_ATTEMPTS="${N_ATTEMPTS:-3}"

mkdir -p "$BASE_FOLDER/eval/logs"

# ---- 1. serve ---------------------------------------------------------------
# Hybrid linear-attention models need the mamba-aware scheduler. Prefix caching
# is disabled: it is unreliable for hybrid KV-cache layouts.
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --tp "$TP" \
  --port "$PORT" \
  --host 127.0.0.1 \
  --mem-fraction-static 0.85 \
  --context-length 65536 \
  --disable-radix-cache \
  --mamba-scheduler-strategy extra_buffer \
  > "$BASE_FOLDER/eval/logs/sglang.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# ---- 2. readiness probe (a real chat-completions call, not just /health) -----
for i in $(seq 1 120); do
  if curl -sS -m 10 "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}],\"max_tokens\":8}" \
      | grep -q choices; then
    echo "server ready after ${i}0s"; break
  fi
  sleep 10
  [[ $i -eq 120 ]] && { echo "server never became ready" >&2; tail -50 "$BASE_FOLDER/eval/logs/sglang.log"; exit 1; }
done

# ---- 3. Harbor eval ---------------------------------------------------------
# Terminus-2 is the harness the trajectories were generated with; using anything
# else changes the prompt format and invalidates comparison to the paper.
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="dummy"

# Harbor 0.21.0 takes `--path <SINGLE_TASK_DIR>`, not a benchmark directory
# (verified against terminalevo/runner/harbor.py, which is pinned to 0.21.0 and
# validated against real runs). So iterate over task dirs. GNU parallel-free:
# bounded by a simple job-slot loop.
run_lane () {
  local lane="$1" tasks_dir="$2"
  local stamp; stamp="$(date +%Y%m%d-%H%M%S)"
  local n=0
  for task_dir in "$tasks_dir"/*/; do
    [[ -f "$task_dir/instruction.md" ]] || continue
    local tid; tid="$(basename "$task_dir")"
    harbor run \
      --path "$task_dir" \
      --agent terminus-2 \
      --model "hosted_vllm/${SERVED_NAME}" \
      --env docker \
      --n-attempts "$N_ATTEMPTS" \
      --n-concurrent 1 \
      --max-retries 0 \
      --jobs-dir "$BASE_FOLDER/eval/${lane}" \
      --job-name "$(printf '%s' "${lane}-${stamp}-${tid}" | tr -c 'A-Za-z0-9_.-' '-')" \
      --quiet >> "$BASE_FOLDER/eval/logs/${lane}.log" 2>&1 &
    n=$((n+1))
    if (( n % CONCURRENCY == 0 )); then wait; fi
  done
  wait
  echo "$lane: dispatched $n tasks"
}

# Terminal-Bench-Hard ships in the collection (114 task dirs).
run_lane tb-hard "$BASE_FOLDER/terminal-bench-hard/tasks"

# Terminal-Bench 2 and Long-Horizon Terminal Bench are upstream Harbor task sets;
# point these at your local copies once available.
if [[ -d "${TB2_TASKS:-}" ]];  then run_lane tb2  "$TB2_TASKS";  else echo "SKIP tb2: set TB2_TASKS"; fi
if [[ -d "${LHTB_TASKS:-}" ]]; then run_lane lhtb "$LHTB_TASKS"; else echo "SKIP lhtb: set LHTB_TASKS"; fi

# ---- 4. score ---------------------------------------------------------------
python - "$BASE_FOLDER/eval" <<'PY'
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
for lane in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
    rewards = []
    for result in lane.rglob("result.json"):
        try:
            data = json.loads(result.read_text())
        except Exception:
            continue
        value = (data.get("verifier_result") or {}).get("rewards", {}).get("reward")
        if isinstance(value, (int, float)):
            rewards.append(float(value))
    if rewards:
        rate = 100.0 * statistics.mean(r >= 1.0 for r in rewards)
        print(f"{lane.name}: n={len(rewards)} pass_rate={rate:.2f}%")
    else:
        print(f"{lane.name}: no scored trials found")
PY
