#!/usr/bin/env bash
# Download model weights + datasets, verifying shard checksums.
#   export BASE_FOLDER=/shared/rst
#   bash scripts/02_download.sh
set -ex
: "${BASE_FOLDER:?set BASE_FOLDER}"
mkdir -p "$BASE_FOLDER"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
pip install -q "huggingface_hub[hf_transfer]" hf_transfer

# ---- base model: whichever the run needs (see configs/models.json) ----------
MODEL_KEY="${MODEL_KEY:-qwen3.5-27b}"
eval "$(python "$(dirname "${BASH_SOURCE[0]}")/model_registry.py" --key "$MODEL_KEY" --shell)"
echo "base model: $HF_REPO (${PARAMS_B}B, ${bf16_note:-bf16})"
hf download "$HF_REPO" --local-dir "$BASE_FOLDER/$MODEL_DIR_NAME"

# Extra models to stage in the same pass, e.g. EXTRA_MODEL_KEYS="qwen3.5-0.8b qwen3.5-4b"
for k in ${EXTRA_MODEL_KEYS:-}; do
  repo=$(python "$(dirname "${BASH_SOURCE[0]}")/model_registry.py" --key "$k" --json | python -c "import json,sys;print(json.load(sys.stdin)['HF_REPO'])")
  hf download "$repo" --local-dir "$BASE_FOLDER/$(basename "$repo")"
done

# ---- reference checkpoints: for eval comparison, NOT for training -----------
# The authors' own SFT/RL results. Use them as an upper-bound sanity check on
# your eval harness before trusting your own numbers.
if [[ "${DOWNLOAD_REFERENCE:-0}" == "1" ]]; then
  hf download Zhongzhi1228/Qwen3.5-27B-SFT --local-dir "$BASE_FOLDER/ref-Qwen3.5-27B-SFT"
  hf download Zhongzhi1228/Qwen3.5-27B-RL  --local-dir "$BASE_FOLDER/ref-Qwen3.5-27B-RL"
fi

# ---- trajectories: the SFT source (22.4 GiB, 66 tars) -----------------------
hf download Zhongzhi1228/Recursive-Task-Synthesis-Trajectories --repo-type dataset \
  --local-dir "$BASE_FOLDER/rst-trajectories" \
  --include "data/*.tar" "metadata/*"

# ---- tasks: needed for RL rollouts + Terminal-Bench-Hard eval ---------------
hf download Zhongzhi1228/Recursive-Task-Synthesis --repo-type dataset \
  --local-dir "$BASE_FOLDER/rst-tasks" --include "data/*.tar" "metadata/*"
hf download Zhongzhi1228/Terminal-Bench-Hard --repo-type dataset \
  --local-dir "$BASE_FOLDER/terminal-bench-hard"        # 100 tasks, verifiers ship

# ---- benchmark task sets for evaluation ------------------------------------
# Terminal-Bench 2: 89 task dirs at the repo root, each with instruction.md +
# task.toml + environment/ + solution/ + tests/.
if [[ ! -d "$BASE_FOLDER/terminal-bench-2/.git" ]]; then
  git clone --depth 1 https://github.com/harbor-framework/terminal-bench-2.git \
    "$BASE_FOLDER/terminal-bench-2"
fi

# Long-Horizon Terminal-Bench: 46 tasks. NOTE its verifiers are WITHHELD upstream
# (0/46 ship tests/; the card says grading uses "hidden, rebuild-from-artifact
# verifiers"). We fetch it for inspection, but scripts/06_eval.py correctly reports
# it as `unscorable` rather than inventing a number.
hf download IntelligenceLab/Long-Horizon-Terminal-Bench --repo-type dataset \
  --local-dir "$BASE_FOLDER/long-horizon-terminal-bench" || \
  echo "WARN: LHTB download failed; it is unscorable anyway"

# ---- verify every shard against the published manifest ----------------------
python - "$BASE_FOLDER" <<'PY'
import hashlib, json, sys
from pathlib import Path
base = Path(sys.argv[1])
bad = 0
for repo in ("rst-trajectories", "rst-tasks"):
    manifest = base / repo / "metadata" / "shard_manifest.jsonl"
    if not manifest.exists():
        print(f"{repo}: NO MANIFEST"); continue
    for line in manifest.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        path = base / repo / row["shard"]
        if not path.exists():
            print(f"MISSING {path}"); bad += 1; continue
        if path.stat().st_size != row["size_bytes"]:
            print(f"SIZE MISMATCH {path}"); bad += 1; continue
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != row["sha256"]:
            print(f"SHA MISMATCH {path}"); bad += 1
    print(f"{repo}: verified")
print("BAD SHARDS:", bad)
sys.exit(1 if bad else 0)
PY
echo "DOWNLOAD + VERIFY OK"
