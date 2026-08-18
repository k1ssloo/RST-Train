#!/usr/bin/env bash
# Build the conda environment for the PRIMARY backend: verl + FSDP.
#
#   export BASE_FOLDER=/shared/rst
#   bash scripts/01b_setup_env_verl.sh 2>&1 | tee $BASE_FOLDER/logs/setup_env_verl.log
#
# Use scripts/01_setup_env.sh instead ONLY for the slime + Megatron path. That one
# builds apex and TransformerEngine, which is exactly what needs a cuDNN swap on a
# shared A100 cluster -- the reason verl is primary here. Nothing below builds apex,
# TransformerEngine, or Megatron.
#
# THIS SCRIPT ADAPTS TO THE MACHINE ON PURPOSE.
# The slime recipe pins every version because its FA2/TE/Megatron combination is
# genuinely fragile. The verl/FSDP stack is not: it is torch + transformers + a few
# kernels. Pinning a torch wheel to a CUDA build the local driver cannot run is a
# self-inflicted failure, so here we DETECT the driver and compute capability and
# choose accordingly, then verify by importing. Every choice is printed so the
# report can state what was actually installed.
set -euo pipefail

: "${BASE_FOLDER:?set BASE_FOLDER}"
ENV_NAME="${ENV_NAME:-rstverl}"
mkdir -p "$BASE_FOLDER/logs"

# rst_write_env_stub: records how to re-enter this env, because the `micromamba
# activate` below only affects THIS process and every launcher runs as a child.
# shellcheck source=lib_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_env.sh"

# ---- 0. what are we actually on? --------------------------------------------
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi missing; cannot size the CUDA wheel. Aborting." >&2; exit 1
fi
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=${DRIVER%%.*}
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
CC_NODOT=${COMPUTE_CAP/./}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
PY_VER="${PY_VER:-3.12}"

echo "=== detected: $GPU_NAME | driver $DRIVER | compute_cap $COMPUTE_CAP | ${GPU_MEM_MIB}MiB"

# Pick the newest CUDA wheel the DRIVER can actually run. A too-new wheel fails at
# `torch.cuda.init()` with a cryptic error; a too-old one just leaves performance on
# the table. So bias slightly old and verify.
#   driver >= 570 -> cu128 ;  >= 550 -> cu126 ;  >= 525 -> cu121 ;  else abort
if   (( DRIVER_MAJOR >= 570 )); then TORCH_CUDA=cu128
elif (( DRIVER_MAJOR >= 550 )); then TORCH_CUDA=cu126
elif (( DRIVER_MAJOR >= 525 )); then TORCH_CUDA=cu121
else
  echo "driver $DRIVER is too old for any supported torch CUDA build (need >= 525)." >&2
  exit 1
fi
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
echo "=== chose torch CUDA build: $TORCH_CUDA  (driver major $DRIVER_MAJOR)"

# A100 is sm80: no FP8, and FlashAttention-3 is Hopper-only. sm90+ can use more.
if [[ "$COMPUTE_CAP" == "8.0" ]]; then
  echo "=== sm80 (A100): bf16 only, FlashAttention-2 only, GDN via fla/Triton or the"
  echo "    pure-PyTorch fallback. FP8 and FlashQLA are NOT available and must not be"
  echo "    enabled anywhere."
fi

# ---- 1. conda ----------------------------------------------------------------
if ! command -v micromamba >/dev/null 2>&1; then
  yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
  export PS1=tmp
  # shellcheck disable=SC1090
  source ~/.bashrc
fi
[[ -f ~/.condarc ]] && sed -i '/^\s*-\s*nodefaults\s*$/d' ~/.condarc
eval "$(micromamba shell hook --shell bash)"
micromamba create -n "$ENV_NAME" "python=$PY_VER" pip -y -c conda-forge
micromamba activate "$ENV_NAME"

# ---- 2. torch, matched to the driver ----------------------------------------
# No version pin: take the current stable for the chosen CUDA build, then verify.
# Pinning here is what breaks on an unfamiliar cluster.
pip install --index-url "$TORCH_INDEX" torch torchvision
python - <<'PY'
import torch, sys
print(f"[verify] torch {torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("torch cannot see the GPU. Wrong CUDA build for this driver -- rerun with "
             "TORCH_CUDA overridden one step older.")
cap = torch.cuda.get_device_capability()
print(f"[verify] device capability {cap}  name={torch.cuda.get_device_name(0)}")
if cap[0] < 8:
    sys.exit("this pipeline assumes bf16; pre-Ampere GPUs are not supported")
PY

# ---- 3. the model stack ------------------------------------------------------
# transformers must be new enough to KNOW qwen3_5; that is a hard floor, not taste.
#
# Every install below carries --extra-index-url for the chosen CUDA build, because
# packages that depend on torch (liger-kernel does) otherwise pull the DEFAULT PyPI
# torch and silently replace the driver-matched build. This bit me while validating
# this script: `pip install liger-kernel` swapped a cu128 torch for PyPI's cu130 one
# without a word. On a box whose driver cannot run the newer build that becomes
# "torch cannot see the GPU" much later, looking entirely unrelated.
TORCH_BEFORE=$(python -c "import torch;print(torch.__version__)")
pip install --extra-index-url "$TORCH_INDEX" \
            "transformers>=5.15.0" "tokenizers>=0.22" accelerate datasets \
            pyarrow pandas hf_transfer "huggingface_hub[hf_xet]" wandb jinja2

# Liger fused cross-entropy. NOT optional on this model: vocab is 248,320, so
# materialized logits for a 32K sequence are ~16.3 GiB bf16 (~32.6 GiB if the loss
# upcasts). Liger ships qwen3_5.py / qwen3_5_moe.py.
pip install --extra-index-url "$TORCH_INDEX" "liger-kernel>=0.6"

# Guard: if anything above replaced torch, stop now instead of debugging it later.
python - "$TORCH_BEFORE" "$TORCH_INDEX" <<'GUARD'
import sys, torch
before, index, after = sys.argv[1], sys.argv[2], torch.__version__
print(f"[verify] torch before={before} after={after}")
if before != after:
    sys.exit(
        f"a dependency replaced torch ({before} -> {after}). The build must stay "
        f"matched to this driver. Repair with:\n"
        f"  pip install --force-reinstall --no-deps --index-url {index} torch=={before}"
    )
GUARD
python - <<'PY'
import importlib.util as u, sys
missing = [m for m in ("liger_kernel.transformers.model.qwen3_5",) if u.find_spec(m) is None]
print("[verify] liger qwen3_5 kernel:", "MISSING" if missing else "present")
if missing:
    sys.exit("liger-kernel lacks a qwen3_5 kernel. Upgrade it; without a fused CE this "
             "model's 248,320-row logits will not fit at 32K.")
PY

# ---- 4. attention / linear-attention kernels (optional, sm-dependent) -------
# transformers' qwen3_5 declares
#   @use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
# so `fla` is a SPEED choice, not a requirement -- there is a pure-PyTorch path.
# We therefore try and continue on failure rather than failing the whole setup.
pip install flash-linear-attention || echo "[warn] fla not installed; using the pure-PyTorch GDN fallback (slower, correct)"
if [[ "$COMPUTE_CAP" == "8.0" || "$COMPUTE_CAP" == "9.0" ]]; then
  pip install flash-attn --no-build-isolation || \
    echo "[warn] flash-attn build failed; transformers will fall back to sdpa. Note it in the report."
else
  echo "[info] skipping flash-attn on compute_cap $COMPUTE_CAP"
fi
# NEVER install FlashQLA here: it requires sm90+, and on A100 it fails or misbehaves.

# ---- 5. verl -----------------------------------------------------------------
# Installed with --no-deps so it cannot re-resolve and replace the torch build we
# just matched to the driver. Its own deps are the ones we installed above plus ray.
pip install ray "omegaconf" "hydra-core" "tensordict" "codetiming" "pylatexenc"
if [[ -n "${VERL_GIT_REF:-}" ]]; then
  pip install --no-deps "git+https://github.com/verl-project/verl.git@${VERL_GIT_REF}"
else
  pip install --no-deps verl
fi
python - <<'PY'
import sys
try:
    import verl
    print("[verify] verl", getattr(verl, "__version__", "(no __version__)"))
except Exception as e:
    sys.exit(f"verl import failed: {e}")
for mod in ("verl.trainer.sft_trainer",):
    try:
        __import__(mod); print(f"[verify] {mod} importable")
    except Exception as e:
        print(f"[warn] {mod} not importable yet: {e}")
PY

# ---- 6. rollout engine (eval serving + RL) ----------------------------------
# ON by default, and that is a deliberate change. It used to default to 0 on the
# reasoning that SFT does not need it -- true, but 06_eval.py's ONLY serving path is
# `python -m sglang.launch_server`, and 20_run_all.sh's default chain evaluates three
# checkpoints (candidate, base, reference). So the old default shipped an environment
# that could train and could not measure, and the failure surfaced as "the server
# never came up" with the real cause buried in $out/sglang.log.
#
# INSTALL_ROLLOUT=0 skips it: correct for a train-only node, and 20_run_all.sh then
# skips the agentic evals loudly instead of failing inside them.
#
# The risk that motivated the old default is real, so it is handled rather than
# avoided: sglang pulls a torch of its own choosing, and training must win that
# argument. If it swaps the driver-matched build, roll sglang back out.
if [[ "${INSTALL_ROLLOUT:-1}" == "1" ]]; then
  TORCH_BEFORE_SGL=$(python -c "import torch;print(torch.__version__)")
  pip install "sglang[all]" --extra-index-url "$TORCH_INDEX" || \
    echo "[warn] sglang install failed; 06_eval.py cannot serve, so 20_run_all.sh will
       skip the agentic benchmarks and the report will FAIL on benchmark coverage"
  TORCH_AFTER_SGL=$(python -c "import torch;print(torch.__version__)" 2>/dev/null || echo BROKEN)
  echo "[verify] torch after sglang: $TORCH_AFTER_SGL (before: $TORCH_BEFORE_SGL)"
  if [[ "$TORCH_AFTER_SGL" != "$TORCH_BEFORE_SGL" ]]; then
    echo "[warn] sglang replaced torch ($TORCH_BEFORE_SGL -> $TORCH_AFTER_SGL)."
    echo "       Training outranks eval: restoring the driver-matched build and"
    echo "       removing sglang. Agentic eval will be skipped and reported as not-run."
    echo "       To get it back, find an sglang built for torch $TORCH_BEFORE_SGL."
    pip uninstall -y sglang sgl-kernel >/dev/null 2>&1 || true
    pip install --force-reinstall --no-deps --index-url "$TORCH_INDEX" "torch==$TORCH_BEFORE_SGL"
    python - <<'PY'
import sys

import torch

print("[verify] torch restored:", torch.__version__, "cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("torch no longer sees the GPU after the sglang rollback. Rebuild this env "
             "with INSTALL_ROLLOUT=0 before doing anything else.")
PY
  fi
else
  echo "[info] INSTALL_ROLLOUT=0: no sglang. 06_eval.py cannot serve a checkpoint, so"
  echo "       20_run_all.sh will skip the agentic benchmarks and record them as not-run."
fi

# ---- 7. Harbor (eval + RL rollout harness) ----------------------------------
# Needs python >= 3.12, which is why PY_VER defaults to 3.12.
pip install "harbor==0.21.0" || echo "[warn] harbor install failed; eval and RL cannot run"

# ---- 8. final report --------------------------------------------------------
# Also written to $BASE_FOLDER/env_summary.json so the run report can state what was
# actually installed instead of what was intended -- in particular whether flash-attn
# and sglang made it, both of which change what the numbers mean.
python - "$BASE_FOLDER/env_summary.json" <<'PY'
import json, sys
info = {}
import torch
info["torch"] = torch.__version__
info["torch_cuda"] = torch.version.cuda
info["cuda_available"] = torch.cuda.is_available()
info["compute_capability"] = list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
import transformers
info["transformers"] = transformers.__version__
try:
    from transformers.models.qwen3_5 import configuration_qwen3_5  # noqa: F401
    info["qwen3_5_supported"] = True
except Exception:
    info["qwen3_5_supported"] = False
for name, mod in (("liger", "liger_kernel"), ("fla", "fla"), ("flash_attn", "flash_attn"),
                  ("verl", "verl"), ("harbor", "harbor"), ("sglang", "sglang")):
    try:
        m = __import__(mod); info[name] = getattr(m, "__version__", "present")
    except Exception:
        info[name] = None
print("\n=== ENVIRONMENT SUMMARY (paste into the report) ===")
print(json.dumps(info, indent=2))
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(info, fh, indent=2, sort_keys=True)
    fh.write("\n")
if not info["qwen3_5_supported"]:
    sys.exit("transformers does not know qwen3_5. Upgrade transformers; nothing else will work.")
PY

# ---- 9. record how to get back in -------------------------------------------
# The `micromamba activate` at the top of this script applied to THIS process only.
# Everything downstream runs as a child of some launcher, so hand them the answer
# instead of a printed instruction nobody executes.
rst_write_env_stub "$ENV_NAME" "$BASE_FOLDER/env-$ENV_NAME.sh"

cat <<EOF

ENV READY.

  In a shell:   micromamba activate $ENV_NAME
  In a script:  source $BASE_FOLDER/env-$ENV_NAME.sh

scripts/20_run_all.sh, 30_run_sft_verl.sh and 33_run_dpo.sh source that file
themselves and refuse to start if it does not give them a working env, so you do not
have to remember either form.

Deliberately NOT installed, and do not add them:
  * apex, TransformerEngine, Megatron-LM  -- the slime path only, and the reason it
    needs a cuDNN swap on A100
  * FlashQLA                              -- requires sm90+; A100 is sm80
  * anything FP8                          -- A100 has no FP8 tensor cores

sglang (eval serving + RL rollout) is installed unless INSTALL_ROLLOUT=0. Whether it
actually landed is in $BASE_FOLDER/env_summary.json under "sglang" -- null there means
the agentic benchmarks cannot run and the report will say so.
EOF
