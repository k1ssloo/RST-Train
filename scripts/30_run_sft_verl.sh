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
#   ~32.6 GiB if the loss upcasts to fp32 — larger than the activations.
#
#   On THIS path the fused CE comes from verl's own fused kernels
#   (model.use_fused_kernels), NOT from model.use_liger — see the long comment at
#   the fused-kernel gate below. Getting that wrong is a ~78 GB/GPU mistake that
#   presents as a parallelism problem.
#
# ROUGH BUDGET, 27.8B on 32x80GB, seq 32K, fused CE, ULYSSES_SP=1:
#   sharded params+grads+Adam  444.8 GB / 32  = 13.9 GB/GPU
#   activations (full recompute, 64 layers)   = 21.5 GB/GPU
#   working set                               ~  2   GB/GPU
#                                             ------------
#                                             ~ 37   GB/GPU   -> fits 80GB
#
#   ULYSSES_SP=N divides ONLY the activation line, and only because the registry divides
#   data.max_token_len_per_gpu by N (`--ulysses-sp`). Per-GPU activation memory is
#   max_token_len_per_gpu tokens at every N; SP's whole contribution is letting that budget
#   drop below max_seq_len, which sp=1 cannot do because bin packing cannot split one
#   sample. Launch with N>1 and an undivided budget and you get the 21.5 GB line unchanged
#   plus every SP cost -- the gate at the ULYSSES_SP block refuses that combination.
#   That budget also assumes the gradient stays SHARDED across micro-batches. verl's
#   own path does not: it skips FSDP gradient sync on non-final micro-batches, and FSDP2
#   answers by keeping a full unsharded fp32 gradient (27.78e9 x 4 B = 103.5 GiB/GPU)
#   until the final backward. verl_backend/fsdp2_grad_accum.py removes that term; the
#   gate below reports it, and "[rst-fsdp2]" in the log is the proof it was applied.
#
#   That budget assumes the CE is fused. Unfused, add ~30 GB of logits on top and
#   the same run reports ~78 GB/GPU of activations beside ~2 GB of sharded params.
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
# --ulysses-sp is not decoration: the registry divides max_tokens_per_gpu by it, because
# the unit that holds a whole sequence under Ulysses is the SP GROUP, not one GPU. Omit it
# and you launch with engine.ulysses_sequence_parallel_size=8 while still asking each GPU
# for a whole 32K sequence's activations -- see the ULYSSES_SP block further down.
eval "$(python scripts/model_registry.py --key "$MODEL_KEY" --mem-class "$MEM_CLASS" \
          --backend verl --ulysses-sp "${ULYSSES_SP:-1}" \
          --gpus "$(( NNODES * NGPUS ))" --gpus-per-node "$NGPUS" \
          --max-seq-len "${MAX_SEQ_LEN:-32768}" --shell)"

# ---- multi-node rendezvous gate ---------------------------------------------
# torchrun's defaults below are MASTER_ADDR=127.0.0.1 NODE_RANK=0, which is right for
# one node and catastrophic for four: every node then forms its OWN world_size=8 group,
# FSDP shards 8 ways instead of 32, and the per-GPU static footprint quadruples. That
# failure does not announce itself -- training starts and OOMs in backward, and the
# peak does not move when you cut the token budget, because the static term dominates.
# Fail here instead.
if (( NNODES > 1 )); then
  if [[ -z "${MASTER_ADDR:-}" || "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
    echo "FATAL: NNODES=$NNODES but MASTER_ADDR=${MASTER_ADDR:-<unset>}." >&2
    echo "       Export MASTER_ADDR=<rank-0 host> on every node, or set NNODES=1." >&2
    exit 2
  fi
  if [[ -z "${NODE_RANK:-}" ]]; then
    echo "FATAL: NNODES=$NNODES but NODE_RANK is unset, so every node would claim rank 0." >&2
    echo "       Export a distinct NODE_RANK (0..$((NNODES - 1))) per node." >&2
    exit 2
  fi

  # ---- interconnect note (a throughput fact, not a correctness one) -----------
  # 05_run_sft.sh and 12_run_grpo.sh already detect this; the PRIMARY path did not, which
  # is how a run can be network-bound with nothing in its log saying so. FSDP2 all-gathers
  # every parameter of every layer on every micro-batch, so on this model that is 64
  # unshards of ~766 MB (bf16) per forward and again per recompute. Over NVLink+IB that is
  # background noise; over TCP sockets it is the run. Reported, never fatal: a slow run
  # that finishes still beats a launcher that refuses to start one.
  if [[ ! -d /sys/class/infiniband ]] || [[ -z "$(ls -A /sys/class/infiniband 2>/dev/null)" ]]; then
    echo "NOTE: no InfiniBand/RoCE device under /sys/class/infiniband, so NCCL will fall back" >&2
    echo "      to TCP over ${NCCL_SOCKET_IFNAME:-the default interface} across $NNODES nodes." >&2
    echo "      FSDP2 moves every parameter across that fabric on every micro-batch, and" >&2
    echo "      verl_backend/fsdp2_grad_accum.py adds one reduce-scatter per micro-batch on" >&2
    echo "      top. If the step time is unusable, the shape to try is HSDP -- FSDP_SIZE=$NGPUS" >&2
    echo "      keeps all-gathers inside a node and leaves only one gradient all-reduce" >&2
    echo "      between nodes per step -- which needs OFFLOAD_OPTIM=1 to fit the smaller" >&2
    echo "      shard degree. Record whichever you used in the report: it changes tokens/s" >&2
    echo "      by an order of magnitude and therefore what the epoch estimate means." >&2
  fi
fi

# ---- static footprint gate ---------------------------------------------------
# Under engine.strategy=fsdp2 with verl 0.9.0 defaults (engine/fsdp.yaml: model_dtype
# fp32, dtype bfloat16, no offload) every parameter costs a FIXED 16 B per GPU divided
# by the shard degree: fp32 master 4 + fp32 grad 4 + Adam exp_avg 4 + exp_avg_sq 4.
# No token-budget knob touches it. Note 16 B/param over 8 ranks is exactly 2 B/param,
# i.e. numerically identical to the whole bf16 model -- if a run reports a static term
# that looks like "the bf16 model size", the shard degree is 8.
OFFLOAD_ARGS=()
if [[ "${OFFLOAD_OPTIM:-0}" == "1" ]]; then
  OFFLOAD_ARGS+=("engine.offload_policy=true")
fi
python - "$PARAMS_B" "$(( NNODES * NGPUS ))" "${FSDP_SIZE:--1}" "$GPU_MEM_MIB" \
        "${OFFLOAD_OPTIM:-0}" "${FORCE_STATIC:-0}" <<'EOF_PY' || exit 2
import sys

params_b, world, fsdp_size, card_mib, offload, force = (
    float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    sys.argv[5] == "1", sys.argv[6] == "1",
)
shard = world if (fsdp_size < 0 or fsdp_size >= world) else fsdp_size
gib = params_b * 1e9 * 16 / shard / (1 << 30)
card = card_mib / 1024
print(f"[gate] shard degree {shard} (world {world}, fsdp_size {fsdp_size}): "
      f"params+grads+Adam = {gib:.1f} GiB/GPU of a {card:.0f} GiB card, "
      f"before any activation")
if offload:
    # Only true of `engine.offload_policy=true`, which is FSDP2's CPUOffloadPolicy and moves
    # the sharded PARAMS, GRADS and optimizer state to host memory. It is NOT true of
    # `engine.optimizer_offload=True`, a different key that moves the Adam state alone and
    # leaves fp32 master + fp32 grad (8 B/param / shard) resident. The two are one word
    # apart and OFFLOAD_OPTIM's name suggests the weaker one, so spell out which is which --
    # a launcher that reports "GPU static ~0" while the run holds 6.5 GiB before its first
    # activation has spent the operator's next hour on the wrong hypothesis.
    print(f"[gate] OFFLOAD_OPTIM=1 -> engine.offload_policy=true (FSDP2 CPUOffloadPolicy): "
          f"params + grads + Adam all live on the host, GPU static ~0.")
    print(f"       If you instead pass engine.optimizer_offload=True by hand, only Adam "
          f"moves and {params_b * 1e9 * 8 / shard / (1 << 30):.1f} GiB/GPU of fp32 master + "
          f"fp32 grad stays resident.")
    raise SystemExit(0)
if gib > 0.55 * card and not force:
    print(f"FATAL: {gib:.1f} GiB static is {100 * gib / card:.0f}% of the card. The activations for a",
          file=sys.stderr)
    print("       whole sequence will not fit beside it, and no data-length change can help.",
          file=sys.stderr)
    print("       Fix one of: more nodes in ONE process group; FSDP_SIZE=-1; OFFLOAD_OPTIM=1", file=sys.stderr)
    print("       (engine.offload_policy=true, CPU optimizer+param offload, slower but fits).", file=sys.stderr)
    print("       Set FORCE_STATIC=1 to launch anyway and report the OOM as expected.", file=sys.stderr)
    raise SystemExit(2)
EOF_PY

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
# data.pad_mode=no_padding makes verl align the mask with a CYCLIC roll over the packed
# micro-batch (workers/utils/losses.py::sft_loss), so a supervised token 0 trains the
# PREVIOUS document's last hidden state to predict this document's first token. Free to
# check, and invisible in the loss curve if it is ever wrong.
leading = int((df.loss_mask.map(lambda m: int(m[0]) if len(m) else 0) != 0).sum())
if leading:
    sys.exit(f"REFUSING TO TRAIN: {leading} rows have loss_mask[0] == 1. Under "
             f"pad_mode=no_padding that leaks supervision across the packed document "
             f"boundary. Re-export with scripts/15_export_pretokenized.py.")
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


# ---- unsharded-gradient-accumulation gate ------------------------------------
# The OOM that survives a CORRECT shard degree. verl skips FSDP gradient sync on every
# non-final micro-batch (FSDPEngine._gradient_sync_context), and FSDP2 answers that by
# upcasting each parameter's gradient to reduce_dtype (fp32 by default beside bf16
# params) and keeping it UNSHARDED until the final backward -- params x 4 B per GPU,
# 103.5 GiB at 27.78B. It needs >= 2 micro-batches per step to trigger, it is
# proportional to parameters rather than tokens, and it does not shrink with more GPUs
# or with engine.optimizer_offload. verl_backend/fsdp2_grad_accum.py removes it; this
# gate reports whether the run would have needed it and refuses if it is switched off.
python verl_backend/fsdp2_grad_accum.py \
  --lengths "$PRETOK" \
  --params-b "$PARAMS_B" \
  --world-size "$(( NNODES * NGPUS ))" \
  --ulysses-sp "${ULYSSES_SP:-1}" \
  --max-token-len-per-gpu "$MAX_TOKENS_PER_GPU" \
  --train-batch-size "$GLOBAL_BATCH_SIZE" \
  --shard-degree "$(( ${FSDP_SIZE:--1} < 0 ? NNODES * NGPUS : FSDP_SIZE ))" \
  --card-gib "$(python -c "print($GPU_MEM_MIB/1024)")" || exit 2

# ---- the fused cross-entropy switch, and why it is NOT model.use_liger -------
# model.use_liger=True does NOT give you a fused cross-entropy on this path. verl's
# FSDP engine applies Liger with fused_linear_cross_entropy HARDCODED to False:
#
#   verl/workers/engine/fsdp/transformer_impl.py
#     # Apply Liger kernel; disable fused_linear_cross_entropy (conflicts with
#     # verl's forward patching)
#     _apply_liger_kernel_to_instance(model=module, fused_linear_cross_entropy=False,
#                                     swiglu=True)
#
# so use_liger buys swiglu + rms_norm and nothing else. The log line that tells you
# this is happening — and it is the one to grep for in a suspicious run — is
#
#   Applying Liger kernels to model instance with model type: qwen3_5 with kwargs:
#   {'fused_linear_cross_entropy': False, 'swiglu': True}
#
# With the CE unfused, verl's forward materializes the whole [tokens, 248320] logit
# tensor: 16.3 GiB in bf16 at 32K, ~30 GiB once the loss upcasts to fp32. On an 80 GB
# card that shows up as ~78 GB of "activations" sitting next to ~2 GB of sharded
# parameters, which reads like a parallelism misconfiguration and is not one.
#
# The switch that actually fuses it is verl's own:
#
#   model.use_fused_kernels=True
#   model.fused_kernel_options.impl_backend=torch|triton
#
# It replaces the model's forward with verl/models/transformers/qwen3_5.py's
# forward_with_torch_backend (FusedLinearForPPO, chunked over the sequence) or
# forward_with_triton_backend (a triton linear_cross_entropy). Both return log_probs
# directly, which is exactly what verl.workers.utils.losses.sft_loss consumes — the
# loss function is byte-identical either way, only the memory differs. Default
# `torch`: triton is the faster kernel but has not been exercised on SM80 here.
#
# FUSED_KERNELS=0 disables the whole thing for a verl old enough not to have the
# config keys (hydra aborts on an unknown override). Only do that at a sequence
# length where unfused logits fit, and say so in the report.
FUSED_KERNELS="${FUSED_KERNELS:-1}"
FUSED_KERNEL_BACKEND="${FUSED_KERNEL_BACKEND:-torch}"
FUSED_ARGS=()
if [[ "$FUSED_KERNELS" == "1" ]]; then
  case "$FUSED_KERNEL_BACKEND" in
    torch|triton) ;;
    *) echo "FUSED_KERNEL_BACKEND must be 'torch' or 'triton', got '$FUSED_KERNEL_BACKEND'" >&2
       exit 2 ;;
  esac
  # Fail here rather than 32 processes deep. Two distinct failures are possible: a
  # verl without these config keys (hydra aborts at launch), and a verl without a
  # qwen3_5 fused forward (it would silently fall back to dense_common's, which
  # does not know this architecture's packed-sequence arguments).
  python - <<'EOF_PY' || exit 2
import importlib.util as u
import pathlib
import sys

spec = u.find_spec("verl")
if spec is None or not spec.origin:
    sys.exit("verl is not importable; the env is wrong, not the flags.")
cfg = pathlib.Path(spec.origin).parent / "trainer" / "config" / "model" / "hf_model.yaml"
if not cfg.is_file():
    sys.exit(f"REFUSING TO TRAIN: {cfg} is missing, so the fused-kernel config keys "
             f"cannot be checked. Set FUSED_KERNELS=0 only if unfused logits fit at "
             f"your sequence length.")
text = cfg.read_text(encoding="utf-8")
missing = [key for key in ("use_fused_kernels", "impl_backend") if key not in text]
if missing:
    sys.exit(f"REFUSING TO TRAIN: this verl's model config has no {missing} — hydra "
             f"would abort on the override. Upgrade verl, or set FUSED_KERNELS=0 and "
             f"drop to a sequence length where a [tokens, 248320] fp32 logit tensor "
             f"fits (~30 GB at 32K).")
try:
    found = u.find_spec("verl.models.transformers.qwen3_5") is not None
except ImportError as exc:  # a parent package that will not import is the same problem
    sys.exit(f"REFUSING TO TRAIN: verl.models.transformers does not import ({exc}); the "
             f"fused-kernel forward cannot be resolved. Fix the env before the launch.")
if not found:
    sys.exit("REFUSING TO TRAIN: this verl has no verl/models/transformers/qwen3_5.py, "
             "so use_fused_kernels would patch in dense_common's forward, which does "
             "not take this architecture's cu_seqlens arguments. Upgrade verl.")
print("[gate] verl fused kernels available (model.use_fused_kernels + qwen3_5 forward)")
EOF_PY
  FUSED_ARGS+=(model.use_fused_kernels=True
               "model.fused_kernel_options.impl_backend=$FUSED_KERNEL_BACKEND")
else
  echo "WARNING: FUSED_KERNELS=0 — the cross-entropy will materialize full logits." >&2
fi

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
# between verl versions: in 0.9.0 it is top-level `checkpoint.save_contents`, in
# older layouts it lived under `trainer.`. The override gate below names the key it
# rejected, and SAVE_HF_MODEL_KEY points it somewhere else without editing this file.
CKPT_ARGS=()
if [[ "${SAVE_HF_MODEL:-0}" == "1" ]]; then
  CKPT_ARGS+=("${SAVE_HF_MODEL_KEY:-checkpoint.save_contents}=['model','optimizer','extra','hf_model']")
  echo "SAVE_HF_MODEL=1 -> ${CKPT_ARGS[0]}"
fi

# ---- packed-varlen correctness gate (runs even at ULYSSES_SP=1) --------------
# data.pad_mode=no_padding packs several conversations into one sequence and relies on
# cu_seqlens to keep the recurrence from running across the boundaries. verl passes
# cu_seqlens into the gated-delta-net rule, but decides whether the rule accepts it
# with inspect.signature (verl/models/transformers/qwen3_5.py:41). The pure-torch
# fallback in transformers carries an unused **kwargs, so that check is a PERMANENT
# false positive: without FLA's real kernel, cu_seqlens is silently dropped and
# documents bleed into each other. That is a correctness failure, not a slowdown, and
# it is invisible in the loss curve.
#
# The PyPI wheel `flash-linear-attention` ships only fla/layers and fla/models -- no
# fla/ops -- so `pip install flash-linear-attention` is NOT enough. Install from git:
#   pip install --no-deps "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention"
python - "${ULYSSES_SP:-1}" "$BASE_FOLDER/$MODEL_DIR_NAME" "${ALLOW_UNSAFE_PACKING:-0}" "$*" \
        <<'EOF_PY' || exit 2
import inspect
import json
import sys
from pathlib import Path

sp, model_dir, allow = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"
problems: list[str] = []

# Not bypassable by ALLOW_UNSAFE_PACKING: this one does not degrade quality, it aborts
# the run. model.use_remove_padding defaults to True (verl hf_model.yaml:40) and verl
# then sets attn_implementation=flash_attention_2, so a missing flash_attn kills the job
# at model load, after every rank has read the weights.
passthrough = sys.argv[4] if len(sys.argv) > 4 else ""
try:
    if "use_remove_padding=False" in passthrough.replace("=false", "=False"):
        raise RuntimeError("skipped: caller passed model.use_remove_padding=False")
    import flash_attn  # noqa: F401
except RuntimeError as exc:
    print(f"[gate] flash_attn check {exc}")
except Exception as exc:  # noqa: BLE001
    print(f"FATAL: flash_attn not importable ({type(exc).__name__}: {exc}).", file=sys.stderr)
    print("       verl sets attn_implementation=flash_attention_2 for use_remove_padding=True.",
          file=sys.stderr)
    print("       Install it (pip install flash-attn --no-build-isolation) or pass",
          file=sys.stderr)
    print("       model.use_remove_padding=False to this launcher (disables Ulysses SP).",
          file=sys.stderr)
    raise SystemExit(2) from None

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_rule
except Exception as exc:  # noqa: BLE001
    fla_rule = None
    problems.append(f"cannot import fla.ops.gated_delta_rule ({type(exc).__name__}: {exc})")
else:
    params = inspect.signature(fla_rule).parameters
    for name in ("cu_seqlens", "cp_context"):
        if name not in params:
            problems.append(f"fla chunk_gated_delta_rule has no explicit `{name}` parameter")

try:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
except Exception as exc:  # noqa: BLE001
    problems.append(f"cannot import transformers qwen3_5 modeling ({type(exc).__name__}: {exc})")
else:
    import transformers

    src = inspect.getsource(mq.Qwen3_5GatedDeltaNet.__init__)
    if "self.chunk_gated_delta_rule" not in src:
        problems.append(
            f"transformers {transformers.__version__} does not set "
            f"self.chunk_gated_delta_rule in Qwen3_5GatedDeltaNet.__init__; verl 0.9.0's "
            f"qwen3_5 patch reads that attribute (removed in 5.15.0 in favour of "
            f"kernels' _kernel_funcs). Pin transformers >=5.11,<5.15."
        )
    # Only meaningful on the layouts that keep a module-level symbol (5.11-5.14): there
    # `chunk_gated_delta_rule or torch_chunk_gated_delta_rule` silently picks the torch
    # fallback when FLA is missing. On 5.15 the name is gone and the check above covers it.
    if "chunk_gated_delta_rule" in vars(mq) and vars(mq)["chunk_gated_delta_rule"] is None:
        problems.append(
            "transformers' qwen3_5 module resolved chunk_gated_delta_rule to None, so the "
            "layer will use the pure-torch fallback that ignores cu_seqlens"
        )

if sp > 1:
    for mod in ("fla.ops.cp.comm", "fla.ops.cp.context"):
        try:
            __import__(mod)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"ULYSSES_SP={sp} needs {mod} ({type(exc).__name__}: {exc})")
    cfg_path = model_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        cfg = cfg.get("text_config", cfg)
        heads = int(cfg.get("num_attention_heads", 0))
        if heads and heads % sp:
            problems.append(f"num_attention_heads={heads} is not divisible by ULYSSES_SP={sp}")
    print(f"[gate] ULYSSES_SP={sp}: every packed micro-batch's total length must also be a "
          f"multiple of {sp}, or verl raises ValueError at qwen3_5.py:215 mid-step.")

if not problems:
    print("[gate] gated-delta-net varlen path is real (FLA ops present, cu_seqlens honored)"
          + (f", CP APIs present for SP={sp}" if sp > 1 else ""))
    raise SystemExit(0)

for p in problems:
    print(f"{'WARNING' if allow else 'FATAL'}: {p}", file=sys.stderr)
if allow:
    print("ALLOW_UNSAFE_PACKING=1 -- launching anyway. Packed documents may share a "
          "recurrent state; say so in the report.", file=sys.stderr)
    raise SystemExit(0)
print("Fix: pip install --no-deps 'flash-linear-attention @ "
      "git+https://github.com/fla-org/flash-linear-attention' and pin "
      "transformers>=5.11,<5.15. Or set data.pad_mode=padding (slower, no packing), "
      "or ALLOW_UNSAFE_PACKING=1 to accept the contamination knowingly.", file=sys.stderr)
raise SystemExit(2)
EOF_PY

# `ulysses` is verl's sequence-parallel knob and the closest analogue to Megatron CP
# if you do need to shard the sequence. verl 0.9.0 DOES implement it for this
# architecture -- cross-rank recurrent-state passing through FLA's cp_context plus a
# causal-conv1d halo exchange (verl/models/transformers/qwen3_5.py:75-226,
# monkey_patch.py:497-548) -- but only when the preconditions checked just above hold.
# It also requires model.use_remove_padding=True (default) and one packed micro-batch
# per rank. There is no TP on this path at all -- verl 0.9.0's engine/fsdp.yaml has no
# tensor_parallel_size key, which is why model_registry.py --backend verl reports TP=1
# and folds the row's CP into MAX_TOKENS_PER_GPU instead of pretending either exists.
SP_ARGS=()
if [[ "${ULYSSES_SP:-1}" != "1" ]]; then
  SP_ARGS+=("engine.ulysses_sequence_parallel_size=${ULYSSES_SP}")
  echo "ULYSSES_SP=${ULYSSES_SP} -> data.max_token_len_per_gpu=${MAX_TOKENS_PER_GPU}" >&2
  echo "         (the registry divided the budget by ${ULYSSES_SP}; group budget" >&2
  echo "         $(( MAX_TOKENS_PER_GPU * ULYSSES_SP )) >= max_seq_len ${MAX_SEQ_LEN:-32768})." >&2
  echo "         Activations per GPU are ~1/${ULYSSES_SP} of the SP=1 figure. The static" >&2
  echo "         params+grads+Adam footprint is unchanged: SP is the fix for long" >&2
  echo "         sequences, not for a static-footprint OOM." >&2
  # The trap this gate exists for. verl computes max_token_len = max_token_len_per_gpu *
  # sp_size, so if the budget was NOT divided, every GPU still holds max_seq_len tokens and
  # SP buys exactly nothing while costing: an all-to-all per full-attention layer, FLA
  # cp_context recurrent-state passing plus a conv1d halo exchange per gated-delta-net
  # layer, and -- because this config has a vision_config, so verl pads but does NOT slice
  # input_ids (the slice happens after embed_tokens) -- an inputs_embeds tensor SP times
  # larger, twice over, since verl adds `0.0 * image_embeds.mean()` to keep the ViT in the
  # graph. At 27B/SP=8 that last term alone is ~5 GiB/GPU of pure waste.
  #
  # The criterion below is exact, not a heuristic. Per-GPU activation memory is
  # max_token_len_per_gpu tokens at EVERY sp, so the one thing SP buys is the ability to put
  # that budget BELOW max_seq_len -- which sp=1 can never do, because bin packing cannot
  # split a single sample. Once max_token_len_per_gpu >= max_seq_len, sp=1 gives the same
  # packing and the same per-GPU footprint for none of the cost: SP is strictly dominated.
  if (( MAX_TOKENS_PER_GPU >= ${MAX_SEQ_LEN:-32768} )); then
    echo "FATAL: ULYSSES_SP=${ULYSSES_SP} with data.max_token_len_per_gpu=${MAX_TOKENS_PER_GPU} >=" >&2
    echo "       max_seq_len=${MAX_SEQ_LEN:-32768}. Each GPU would hold ${MAX_TOKENS_PER_GPU} tokens of" >&2
    echo "       activations -- exactly what it holds at ULYSSES_SP=1 -- so sequence" >&2
    echo "       parallelism buys no memory here and costs the SP collectives, the" >&2
    echo "       ${ULYSSES_SP}x unsliced inputs_embeds, and an unvalidated GDN context-parallel path." >&2
    echo "       Fix: let the registry size the budget (do not set MAX_TOKENS_PER_GPU by" >&2
    echo "       hand), or set ULYSSES_SP=1. ALLOW_OVERSIZED_SP_BUDGET=1 launches anyway." >&2
    [[ "${ALLOW_OVERSIZED_SP_BUDGET:-0}" == "1" ]] || exit 2
  fi
fi

# One array, used twice: validated below, then handed to torchrun. Writing the
# overrides twice is how a launcher ends up validating flags it does not pass.
VERL_ARGS=(
  data.train_files="$PRETOK"
  data.custom_cls.path="$REPO_DIR/verl_backend/rst_sft_dataset.py"
  data.custom_cls.name=RSTPretokenizedSFTDataset
  data.pad_mode=no_padding
  data.use_dynamic_bsz=True
  data.max_length="${MAX_SEQ_LEN:-32768}"
  data.max_token_len_per_gpu="$MAX_TOKENS_PER_GPU"
  data.train_batch_size="$GLOBAL_BATCH_SIZE"
  data.truncation=error
  model.path="$BASE_FOLDER/$MODEL_DIR_NAME"
  # model.use_liger=True stays: swiglu and rms_norm are still real savings. It is just
  # not what fuses the cross-entropy here -- FUSED_ARGS is.
  model.use_liger=True
  "${FUSED_ARGS[@]+"${FUSED_ARGS[@]}"}"
  model.enable_gradient_checkpointing=True
  engine.strategy=fsdp2
  engine.fsdp_size="${FSDP_SIZE:--1}"
  "${OFFLOAD_ARGS[@]+"${OFFLOAD_ARGS[@]}"}"
  "${SP_ARGS[@]+"${SP_ARGS[@]}"}"
  optim.lr="$LR"
  optim.lr_scheduler_type=cosine
  optim.min_lr_ratio=0.1
  optim.lr_warmup_steps_ratio="$LR_WARMUP_FRACTION"
  optim.weight_decay=0.1
  optim.betas="[0.9,0.98]"
  trainer.total_epochs="$NUM_EPOCH"
  trainer.project_name="${WANDB_PROJECT:-rst-qwen35-verl}"
  trainer.experiment_name="$RUN_NAME"
  trainer.default_local_dir="$BASE_FOLDER/$RUN_NAME"
  trainer.logger="['console','wandb']"
  trainer.save_freq=20
  "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"
  "$@"
)

# ---- do these overrides exist in THIS verl? ---------------------------------
# hydra rejects an unknown override with "Could not override 'x.y'" -- and it does so
# in every rank after torchrun has already started them and each has read the model
# from disk. verl renames config keys between versions (measured on 0.9.0:
# `engine.tensor_parallel_size` does not exist for FSDP at all,
# `optim.warmup_steps_ratio` is spelled `optim.lr_warmup_steps_ratio`, and
# `checkpoint.save_contents` has moved out from under `trainer.`), so a launcher that
# hardcodes names is a version-drift trap. Compose the real config with the real
# overrides here, name every rejected one, and stop before the launch.
python - "${VERL_ARGS[@]}" <<'EOF_PY' || exit 2
import sys

try:
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
except ImportError as exc:
    sys.exit(f"hydra is not importable ({exc}); verl cannot start either. Fix the env.")

# Flags (a leading '-') are passed through to hydra/torchrun untouched: --config-name
# and friends are not overrides and composing them here would be wrong.
overrides = [a for a in sys.argv[1:] if not a.startswith("-")]
passthrough = [a for a in sys.argv[1:] if a.startswith("-")]


def compose_ok(items: list[str]) -> str:
    GlobalHydra.instance().clear()
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        try:
            compose(config_name="sft_trainer_engine", overrides=items)
        except Exception as exc:  # hydra raises several distinct types here
            return str(exc).splitlines()[0]
    return ""


whole = compose_ok(overrides)
if whole:
    # Bisect to the individual offenders so the message names all of them at once
    # rather than one per failed launch.
    bad = [(item, why) for item in overrides if (why := compose_ok([item]))]
    if not bad:  # the overrides are individually fine but conflict as a set
        sys.exit(f"REFUSING TO TRAIN: verl rejected this override set as a whole: {whole}")
    lines = "\n".join(f"    {item}\n      {why}" for item, why in bad)
    sys.exit(f"REFUSING TO TRAIN: {len(bad)} override(s) do not exist in the installed "
             f"verl:\n{lines}\n"
             f"  This is version drift, not a typo in your command. Check the key against\n"
             f"  verl/trainer/config/{{engine,optim,model,data}}/*.yaml in the installed\n"
             f"  package and fix scripts/30_run_sft_verl.sh -- do not delete the flag if it\n"
             f"  is load-bearing (the fused-kernel and gradient-checkpointing ones are).")
print(f"[gate] {len(overrides)} hydra overrides all exist in this verl"
      + (f" ({len(passthrough)} flag(s) passed through unchecked)" if passthrough else ""))
EOF_PY

# ---- resume schedule gate ---------------------------------------------------
# trainer.resume_mode defaults to auto, so a global_step_* already sitting in
# trainer.default_local_dir is loaded and step counting continues -- while
# total_training_steps is re-derived from total_epochs and the cosine rebuilt over the
# new total. Relaunching this RUN_NAME with different epochs therefore resumes part-way
# back UP the curve (measured on the 4B tmax run: step 42 ended at lr 3.0e-07, the
# min_lr floor, and step 43 of the relaunch started at 2.35e-06). verl warns about none
# of it. scripts/resume_guard.py diffs the schedule knobs against the ones recorded
# beside the checkpoints and refuses a resume that is not one.
# Node rank 0 only: the record is a single file and every node runs this script.
if [[ "${NODE_RANK:-0}" == "0" ]]; then
  python "$REPO_DIR/scripts/resume_guard.py" \
    --run-dir "$BASE_FOLDER/$RUN_NAME" \
    --world-size "$(( NNODES * NGPUS ))" \
    -- "${VERL_ARGS[@]}" || exit 2
fi

torchrun \
  --nnodes "$NNODES" --nproc_per_node "$NGPUS" \
  --node_rank "${NODE_RANK:-0}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29500}" \
  -m verl.trainer.sft_trainer \
  --config-name sft_trainer_engine \
  "${VERL_ARGS[@]}"

cat <<EOF

Done. Notes for the report:
  * The fused cross-entropy is load-bearing, not an optimization: without it the
    248,320-row logits dominate memory at 32K. On this path it comes from
    model.use_fused_kernels (backend: ${FUSED_KERNEL_BACKEND:-torch}), NOT from
    model.use_liger -- verl's FSDP engine disables Liger's FLCE on purpose. Confirm
    it in the log: "Using Torch backend for fused kernels in ..." must appear, and
    "Skipping monkey patch ... use_fused_kernels is False" must NOT.
  * "[rst-fsdp2]" must appear too. That line is verl_backend/fsdp2_grad_accum.py
    reporting that FSDP2 now reduce-scatters every micro-batch. Without it, any step
    that splits into two or more micro-batches retains a full UNSHARDED fp32 gradient
    ($(python -c "print(f'{$PARAMS_B * 1e9 * 4 / 2**30:.0f}')") GiB/GPU here) and the
    run dies inside loss.backward() with a large "allocated by PyTorch" figure and a
    small requested size. It is applied from the dataset module, so it is present on
    any launcher that uses data.custom_cls.path=verl_backend/rst_sft_dataset.py.
  * engine.strategy=fsdp2 shards parameters only. If you enabled ulysses sequence
    parallelism to fit longer sequences, SAY SO -- it has not been validated on the
    gated-delta-net layers, same open question as Megatron CP.
  * Convert to HF and restore the vision tower before evaluating:
      python scripts/07_restore_vision.py --trained <hf_out> \\
        --original $BASE_FOLDER/$MODEL_DIR_NAME --out <hf_out>-full
    then scripts/06_eval.py as usual. The eval path is backend-independent.
EOF
