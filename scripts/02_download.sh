#!/usr/bin/env bash
# Download model weights + datasets, verifying shard checksums.
#   export BASE_FOLDER=/shared/rst
#   bash scripts/02_download.sh
set -ex
: "${BASE_FOLDER:?set BASE_FOLDER}"
mkdir -p "$BASE_FOLDER"

# The pip install and the `hf` CLI below have to land in the TRAINING env; installed into
# the system interpreter they are invisible to every later stage. No-op when the caller
# already entered it. See scripts/lib_env.sh.
# shellcheck source=lib_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_env.sh"
rst_bootstrap_python || exit 2
rst_enter_env "${ENV_NAME:-rstverl}" || exit 2

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

# ---- reference checkpoint: for eval comparison, NOT for training ------------
# The authors' released SFT checkpoint, scored through OUR harness, is what makes
# our own number interpretable (20_run_all.sh::eval_reference).
#
# Gated on REFERENCE_CHECKPOINT as well as on DOWNLOAD_REFERENCE, because only
# qwen3.5-27b has one (configs/models.json). 20_run_all.sh passes
# DOWNLOAD_REFERENCE="$EVAL_REFERENCE", which defaults to 1 -- so a 9B or 4B run
# used to pull ~110 GB of 27B reference weights that eval_reference() then skipped,
# on the same filesystem as the 400 GB budget.
if [[ "${DOWNLOAD_REFERENCE:-0}" == "1" && -n "${REFERENCE_CHECKPOINT:-}" ]]; then
  hf download "$REFERENCE_CHECKPOINT" \
    --local-dir "$BASE_FOLDER/ref-$(basename "$REFERENCE_CHECKPOINT")"
  # The RL reference is a further ~56 GB and nothing in this pipeline reads it: the
  # RL row in rst_common/paper.py is a published number, not a scored checkpoint.
  # Opt in explicitly if you want to score it by hand.
  if [[ "${DOWNLOAD_REFERENCE_RL:-0}" == "1" ]]; then
    hf download Zhongzhi1228/Qwen3.5-27B-RL --local-dir "$BASE_FOLDER/ref-Qwen3.5-27B-RL"
  fi
elif [[ "${DOWNLOAD_REFERENCE:-0}" == "1" ]]; then
  echo "no published reference checkpoint for MODEL_KEY=$MODEL_KEY; skipping the" \
       "reference download (the paper only reports Qwen3.5-27B and 122B-A10B)."
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
