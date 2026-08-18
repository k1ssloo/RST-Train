#!/usr/bin/env bash
# Build the `slime` conda environment on an A100 (SM80) node.
#
# This is slime's upstream build_conda.sh with the A100-specific deltas applied.
# Run it on ONE node if $BASE_FOLDER is on a shared FS, otherwise on every node.
#
#   export BASE_FOLDER=/shared/rst        # checkpoints + envs live here
#   export SLIME_DIR=$BASE_FOLDER/slime
#   bash scripts/01_setup_env.sh 2>&1 | tee $BASE_FOLDER/logs/setup_env.log
#
# WHY WE DON'T JUST RUN UPSTREAM build_conda.sh VERBATIM:
#   1. FlashQLA (`pip install git+https://github.com/QwenLM/FlashQLA.git`) is the
#      optional GDN backend and REQUIRES SM90+. On A100 it either fails to build
#      or fails at runtime. We skip it and use `--qwen-gdn-backend fla` instead.
#   2. Upstream hardcodes BASE_DIR=/root. We parameterize it.
#   3. Upstream targets cu129 + torch 2.11. That is fine on A100 (sm_80 is in the
#      wheel's arch list); we keep the pins EXACTLY, because the FA2/TE/Megatron
#      combination is version-sensitive. Do not "upgrade" these.
#   4. No FP8 anywhere: A100 has no FP8 tensor cores. Stay bf16; never use
#      tools/convert_hf_to_fp8.py.
set -ex

: "${BASE_FOLDER:?set BASE_FOLDER (shared dir for checkpoints/envs)}"
export BASE_DIR="${BASE_DIR:-$BASE_FOLDER}"
export SLIME_DIR="${SLIME_DIR:-$BASE_DIR/slime}"
ENV_NAME="${ENV_NAME:-slime}"
mkdir -p "$BASE_DIR" "$BASE_FOLDER/logs"

# rst_write_env_stub: the `micromamba activate "$ENV_NAME"` below affects THIS process only,
# and every launcher runs as a child of something. See scripts/lib_env.sh.
# shellcheck source=lib_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_env.sh"

# ---- pinned versions: keep in sync with slime/docker/Dockerfile --------------
export SGLANG_VERSION="v0.5.15.post1"
export SGLANG_COMMIT="0b3bb0cbe31873994c9f989fddfe2f87ca839fdd"
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
export PATCH_VERSION="v0.5.15.post1"
export TMS_COMMIT="8d30c59ca12a68d9deccbc9c6599076a1218cbc5"

# ---- 0. conda (micromamba) --------------------------------------------------
if ! command -v micromamba >/dev/null; then
  yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
  export PS1=tmp
  mkdir -p "$HOME/.cargo/" && touch "$HOME/.cargo/env"
  # shellcheck disable=SC1090
  source ~/.bashrc
fi
# micromamba writes the meta-tag `nodefaults` as a real channel; it then tries to
# fetch it as an anaconda.org repo and times out. Strip it.
[[ -f ~/.condarc ]] && sed -i '/^\s*-\s*nodefaults\s*$/d' ~/.condarc

eval "$(micromamba shell hook --shell bash)"
micromamba create -n "$ENV_NAME" python=3.12 pip -c conda-forge -y
micromamba activate "$ENV_NAME"
export CUDA_HOME="$CONDA_PREFIX"

micromamba install -n "$ENV_NAME" cuda=12.9.1 cuda-nvtx=12.9.79 cuda-nvtx-dev=12.9.79 nccl \
  -c nvidia/label/cuda-12.9.1 -c nvidia -c conda-forge -y
micromamba install -n "$ENV_NAME" -c conda-forge cudnn rust -y
pip install cuda-python==12.9

# ---- 1. sglang (rollout engine + eval server) -------------------------------
cd "$BASE_DIR"
[[ -d sglang ]] || git clone https://github.com/sgl-project/sglang.git
cd sglang && git checkout "${SGLANG_COMMIT}"
pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps sglang-kernel==0.4.4 sgl-deep-gemm==0.1.4 \
  --index-url https://docs.sglang.ai/whl/cu129/
# repair the cu13 spill that sglang's deps drag in
pip uninstall -y nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver \
  nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink \
  nvidia-nvshmem-cu13 nvidia-nvtx nvidia-cutlass-dsl-libs-cu13 || true
pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12==9.16.0.29 nvidia-cufft-cu12 nvidia-cufile-cu12 nvidia-curand-cu12 \
  nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 \
  --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple
pip install cmake ninja

# ---- 2. attention + linear-attention kernels --------------------------------
# FA2 2.8.3 (sm80-compatible) is what the validated TE 2.16 CP stack wants.
pip uninstall -y flash-attn-4 flash_attn_4 || true
pip install --no-deps \
  "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=3d0c8e60f820321eedd7166e79c33cb816263d8be6e35c3f5ba8fe2df6fea697"
# Gated-DeltaNet kernels for the 48 linear_attention layers. Triton-based, works
# on SM80. THIS is the A100 GDN path (`--qwen-gdn-backend fla`).
pip install flash-linear-attention==0.4.2
# >>> A100 DELTA: upstream installs FlashQLA here. SKIPPED -- requires SM90+. <<<
pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

pip install --no-build-isolation "transformer_engine[pytorch]==2.16.1"
NVCC_APPEND_FLAGS="--threads 4" pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
pip install -v "git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT}" \
  --no-cache-dir --force-reinstall --no-build-isolation
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# ---- 3. Megatron-LM ---------------------------------------------------------
cd "$BASE_DIR"
[[ -d Megatron-LM ]] || git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
cd Megatron-LM && git checkout "${MEGATRON_COMMIT}" && pip install -e . --no-build-isolation

# ---- 4. slime ---------------------------------------------------------------
[[ -d "$SLIME_DIR" ]] || git clone https://github.com/THUDM/slime.git "$SLIME_DIR"
cd "$SLIME_DIR"
pip install -r requirements.txt
pip install -e . --no-deps
cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat" && pip install . --no-build-isolation

pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy==1.26.4" "scipy==1.17.1"
pip install "kernels<0.15.0"

# ---- 5. patches (same order as the Dockerfile) ------------------------------
patch_dir="$SLIME_DIR/docker/patch/${PATCH_VERSION}"
[[ -d "$patch_dir" ]] || { echo "missing patch dir $patch_dir" >&2; exit 1; }
cd "$BASE_DIR/sglang"
for p in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch; do
  [[ -f "$patch_dir/$p" ]] || continue
  if git apply --check "$patch_dir/$p"; then git apply "$patch_dir/$p"
  elif git apply --reverse --check "$patch_dir/$p"; then echo "$p already applied"
  else echo "$p does not apply cleanly" >&2; exit 1; fi
done
cd "$BASE_DIR/Megatron-LM"
if git apply --reverse --check "$patch_dir/megatron.patch"; then
  echo "megatron.patch already applied"
else
  git update-index --refresh || true
  git apply "$patch_dir/megatron.patch" --3way
  git grep -n '^<<<<<<< ' -- . && { echo "conflicts in megatron.patch" >&2; exit 1; }
fi

# ---- 6. verify --------------------------------------------------------------
python - <<'PY'
import torch, torchaudio, torchvision, sglang
assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torchaudio.__version__ == "2.11.0+cu129"
assert torchvision.__version__ == "0.26.0+cu129"
assert hasattr(torch.ops.torchvision, "nms")
cc = torch.cuda.get_device_capability()
print("torch", torch.__version__, "cuda", torch.version.cuda, "compute_cap", cc)
if cc[0] == 8:
    print("A100/SM80 detected -> ALWAYS pass --qwen-gdn-backend fla, never flashqla, never FP8")
import flash_attn, fla
print("flash_attn", flash_attn.__version__, "fla ok")
import transformer_engine, megatron.core as mcore
print("TE", transformer_engine.__version__, "megatron.core ok")
PY
pip install wandb

# Record how to re-enter this env; the activate above was process-local.
rst_write_env_stub "$ENV_NAME" "$BASE_FOLDER/env-$ENV_NAME.sh"
echo "ENV READY. env name: $ENV_NAME"
echo "  In a shell:   micromamba activate $ENV_NAME"
echo "  In a script:  source $BASE_FOLDER/env-$ENV_NAME.sh   (the launchers do this)"
