#!/usr/bin/env bash
# DPO on the logged RST trajectories — the container-free fallback for the RL stage.
#
#   export BASE_FOLDER=/shared/rst
#   export MODEL_KEY=qwen3.5-27b
#   bash scripts/33_run_dpo.sh
#
# WHEN TO USE THIS
#   When GRPO cannot run. Every rollout in scripts/12_run_grpo.sh builds a task
#   image and drives tmux inside it; on a pod whose AppArmor profile denies
#   mount(2) that is impossible, and if no off-machine backend is reachable either
#   (BACKENDS.md) then on-policy RL is blocked outright. This path needs no
#   container, no network and no privilege: the trajectory release already contains
#   successes and failures on the same tasks, scored by those tasks' own verifiers.
#
#   It is also a defensible warm-up when GRPO *is* available, but it is NOT GRPO.
#   DPO reweights behaviour already present in the data, and that data came from
#   other policies, so it can sharpen modes the model already has and cannot
#   discover a strategy no logged trajectory used. The summary this writes says so
#   in its own first field; keep that wording in the report.
#
# THREE STAGES, ALL RE-ENTRANT
#   17_build_dpo_data.py    trajectories -> tokenized preference pairs   (CPU, ~25 min)
#   18_dpo_ref_logprobs.py  frozen reference logprobs                    (GPU, sharded)
#   19_train_dpo.py         the DPO step itself                          (GPU, FSDP2)
#   Stage 1 is skipped when its parquet is already there. Stage 2 is ALWAYS entered
#   and resumes from whatever it already scored -- a shard file exists minutes into a
#   pass that takes hours, so "output exists" was never the same question as "output
#   is complete". Stage 3 is not resumable; it starts from POLICY each time.
#
# COST SHAPE, 27.8B on 8x80GB, pairs at ~3k supervised tokens per side:
#   reference pass  one forward per side, no optimizer state -> ~55.6 GB weights
#                   plus a chunked LM head; shard it one process per GPU and it is
#                   roughly (pairs / 8) * 2 forwards.
#   training        fp32 masters + Adam = 444.8 GB sharded over the world, plus one
#                   side's activations at a time (see 19's split-backward note).
#                   8 GPUs is tight; 16 is comfortable.
set -uo pipefail

: "${BASE_FOLDER:?set BASE_FOLDER}"
MODEL_KEY="${MODEL_KEY:-qwen3.5-27b}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

# Enter the training env before the first `python`. DPO_PLAN.md tells the operator to
# run this script directly, so it cannot assume 20_run_all.sh already did it. If the
# env is already active this is one find_spec and a printed path.
# shellcheck source=lib_env.sh
source "$REPO_DIR/scripts/lib_env.sh"
rst_bootstrap_python || exit 2
rst_enter_env "${ENV_NAME:-rstverl}" || exit 2

GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ "${MEM_CLASS:-auto}" == "auto" ]]; then
  if [[ -n "$GPU_MEM_MIB" && "$GPU_MEM_MIB" -gt 70000 ]]; then MEM_CLASS=80GB; else MEM_CLASS=40GB; fi
fi
NNODES="${NNODES:-1}"; NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NGPUS="${NGPUS:-1}"; (( NGPUS > 0 )) || NGPUS=1

# Resolve the model, but do not let the registry's parallelism arithmetic gate this
# path. 19_train_dpo.py is FSDP2-only -- no TP, PP or CP -- so a GPU count the
# registry rejects for SFT (tp*pp*cp must divide the world) is perfectly runnable
# here, and refusing to start would be a false negative. All this needs from the
# registry is the checkpoint directory name.
#
# --backend verl is the honest shape for an FSDP2 trainer, so the fallback below now
# fires only for a genuinely unknown key rather than for every odd GPU count.
if REGISTRY=$(python scripts/model_registry.py --key "$MODEL_KEY" --mem-class "$MEM_CLASS" \
                --backend verl \
                --gpus "$(( NNODES * NGPUS ))" --gpus-per-node "$NGPUS" \
                --max-seq-len "${MAX_SEQ_LEN:-32768}" --shell 2>/dev/null); then
  eval "$REGISTRY"
else
  MODEL_DIR_NAME=$(python - "$MODEL_KEY" <<'EOF_PY'
import json, sys
models = json.load(open("configs/models.json"))["models"]
key = sys.argv[1]
if key not in models:
    sys.exit(f"unknown MODEL_KEY {key!r}; python scripts/model_registry.py --list")
print(models[key].get("model_dir_name") or models[key]["hf_repo"].split("/")[-1])
EOF_PY
) || exit 1
  echo "note: the registry could not place $MODEL_KEY on $(( NNODES * NGPUS )) GPUs with its"
  echo "      TP/PP/CP plan. DPO does not use any of those, so continuing with"
  echo "      MODEL_DIR_NAME=$MODEL_DIR_NAME. (SFT on this GPU count would still be blocked.)"
  MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
fi

PAIRS_DIR="${PAIRS_DIR:-$BASE_FOLDER/dpo-v2}"
REF_DIR="${REF_DIR:-$PAIRS_DIR/ref}"
TRAJ_ROOT="${TRAJ_ROOT:-$BASE_FOLDER/rst-trajectories}"
TOKENIZER="${TOKENIZER:-$BASE_FOLDER/$MODEL_DIR_NAME}"
# The policy AND the reference. They are the same checkpoint by construction: DPO
# starts from the SFT model and measures divergence from it.
POLICY="${POLICY:-$BASE_FOLDER/out-hf-full}"
OUT_DIR="${OUT_DIR:-$BASE_FOLDER/out-dpo}"
DPO_SEQ_LEN="${DPO_SEQ_LEN:-${MAX_SEQ_LEN:-32768}}"
PER_SIDE="${PER_SIDE:-14}"
# fp32 masters unless overridden. At lr 5e-7 a bf16 master rounds most updates to
# zero while a few small-magnitude weights still move, which produces a moving loss
# curve on a model that barely trained. 19 warns; this sets the safe default.
PARAM_DTYPE="${PARAM_DTYPE:-fp32}"
# Score the reference in the dtype the policy's forward will use, so 19's step-0
# calibration lands on log 2 exactly instead of leaving a ~1e-3 residual to explain.
REF_DTYPE="${REF_DTYPE:-$( (( NNODES * NGPUS > 1 )) && echo bf16 || echo "$PARAM_DTYPE" )}"
# ONE chunk size for both stages. The logprob is a sum of per-chunk terms, so a
# different chunk size regroups that sum and moves it by ~1e-8 per token -- harmless
# in itself, but it turns the step-0 identity from exact into approximate and puts
# random signs on margins that should be zero.
LOGIT_CHUNK="${DPO_LOGIT_CHUNK:-512}"

[[ -d "$POLICY" ]] || { echo "no checkpoint at POLICY=$POLICY. Run the SFT stage first, or point POLICY at an HF dir." >&2; exit 1; }
# Before the reference stage, not after: its shards redirect straight into logs/, and
# a missing directory would kill every shard on the redirect with nothing to read.
mkdir -p "$BASE_FOLDER/logs" "$OUT_DIR"

# A parquet whose path exists is not yet a parquet that finished being written, and
# on a shared filesystem the two are seconds apart. Probe the footer instead of the
# inode.
wait_readable_parquet() {  # <path> <budget_sec>
  local path="$1" budget="$2" waited=0
  while (( waited < budget )); do
    if python - "$path" > /dev/null 2>&1 <<'EOF_PY'
import sys

import pyarrow.parquet as pq

pq.ParquetFile(sys.argv[1]).metadata.num_rows
EOF_PY
    then
      return 0
    fi
    sleep 10; waited=$(( waited + 10 ))
  done
  return 1
}

# ------------------------------------------------------------------ 1. pairs
# Only node 0 produces the pairs. $PAIRS_DIR is on the shared filesystem, so with
# NNODES>1 every node would otherwise fetch the same 48 MB onto the same three paths
# concurrently, or -- much worse -- one node would start reading dpo_train.parquet
# while another is still writing it, and a truncated parquet either raises here or
# (if the footer happens to be there) silently trains on a prefix of the data.
PAIRS_WAIT_SEC="${PAIRS_WAIT_SEC:-3600}"
if [[ "${NODE_RANK:-0}" != "0" ]]; then
  echo "=== node ${NODE_RANK:-0}: waiting for node 0 to publish $PAIRS_DIR/dpo_train.parquet"
  if ! wait_readable_parquet "$PAIRS_DIR/dpo_train.parquet" "$PAIRS_WAIT_SEC"; then
    echo "waited ${PAIRS_WAIT_SEC}s and $PAIRS_DIR/dpo_train.parquet is still not a" >&2
    echo "readable parquet. Node 0 either has not been started, failed in stage 1 (check" >&2
    echo "its console), or does not share $PAIRS_DIR with this node. Raise PAIRS_WAIT_SEC" >&2
    echo "if the local rebuild from $TRAJ_ROOT is genuinely still running." >&2
    exit 1
  fi
fi

# Three sources, tried in cost order.
#   a) already on disk          -> nothing to do
#   b) the published pairs      -> 48 MB, seconds, and it is the exact set every number
#                                  in DPO_PLAN.md refers to
#   c) rebuild from the release -> 23 GB of trajectories and ~25 min of CPU to produce
#                                  the same thing (seed 1228, deterministic)
# (b) also means a pod that never downloaded the trajectory release can still run this
# stage. Set DPO_FETCH_HF=0 to force the local rebuild.
DPO_PAIRS_REPO="${DPO_PAIRS_REPO:-NiuNiu0110/RST-DPO-Qwen3.5-27B}"
if [[ ! -f "$PAIRS_DIR/dpo_train.parquet" && "${DPO_FETCH_HF:-1}" == "1" ]]; then
  echo "=== fetching published preference pairs from $DPO_PAIRS_REPO -> $PAIRS_DIR"
  python - "$DPO_PAIRS_REPO" "$PAIRS_DIR" <<'EOF_PY'
import shutil
import sys
from pathlib import Path

repo, out = sys.argv[1], Path(sys.argv[2])
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("huggingface_hub not installed")

# Download everything BEFORE writing anything into place: a half-fetched pairs dir
# (train but no holdout) would look complete to the next stage and silently cost the
# only held-out measurement this path has.
want = {"data/v2/train.parquet": "dpo_train.parquet",
        "data/v2/holdout.parquet": "dpo_holdout.parquet",
        "manifest.json": "manifest.json"}
staged = {}
for remote, local in want.items():
    staged[local] = hf_hub_download(repo, remote, repo_type="dataset")
out.mkdir(parents=True, exist_ok=True)
for local, src in staged.items():
    shutil.copyfile(src, out / local)
print(f"fetched {len(staged)} files into {out}")
EOF_PY
  if [[ ! -f "$PAIRS_DIR/dpo_train.parquet" ]]; then
    echo "    fetch did not produce pairs (offline pod, or no such repo). Falling back to"
    echo "    building them locally from $TRAJ_ROOT."
  fi
fi

if [[ ! -f "$PAIRS_DIR/dpo_train.parquet" ]]; then
  echo "=== building preference pairs -> $PAIRS_DIR"
  if [[ ! -d "$TRAJ_ROOT" ]]; then
    {
      echo "no pairs and nothing to build them from."
      echo "  local pairs   : $PAIRS_DIR/dpo_train.parquet  (absent)"
      if [[ "${DPO_FETCH_HF:-1}" == "1" ]]; then
        echo "  published set : $DPO_PAIRS_REPO  (fetch attempted and failed -- offline?)"
      else
        echo "  published set : not tried (DPO_FETCH_HF=0). Unset it to fetch 48 MB and skip the build."
      fi
      echo "  trajectories  : $TRAJ_ROOT  (absent; scripts/02_download.sh fetches the 23 GB release)"
      echo "Make ONE of those three available; any of them produces the same 2,673 pairs."
    } >&2
    exit 1
  fi
  python scripts/17_build_dpo_data.py \
    --traj-root "$TRAJ_ROOT" \
    --tokenizer "$TOKENIZER" \
    --out-dir "$PAIRS_DIR" \
    --per-side "$PER_SIDE" \
    --max-seq-len "$DPO_SEQ_LEN" \
    --cache "$PAIRS_DIR/reconstructed.jsonl.gz" \
    ${DPO_BUILD_ARGS:-} || exit 1
fi

# Refuse-to-train gate, read off the parquet itself rather than its manifest: the
# file may have been copied or rebuilt while a sidecar went stale, and the tensors
# are the thing being trained on.
python - "$PAIRS_DIR" "$DPO_SEQ_LEN" <<'EOF_PY' || exit 1
import sys
from pathlib import Path

import pandas as pd

root, max_len = Path(sys.argv[1]), int(sys.argv[2])
train = pd.read_parquet(root / "dpo_train.parquet")
need = ["pair_id", "chosen_input_ids", "chosen_loss_mask", "rejected_input_ids",
        "rejected_loss_mask", "common_prefix_tokens"]
for col in need:
    if col not in train.columns:
        sys.exit(f"REFUSING TO TRAIN: {root}/dpo_train.parquet has no {col!r} column. This "
                 f"must come from scripts/17_build_dpo_data.py.")

bad = []
for side in ("chosen", "rejected"):
    ids, mask = train[f"{side}_input_ids"], train[f"{side}_loss_mask"]
    if int((ids.map(len) != mask.map(len)).sum()):
        bad.append(f"{side}: len(input_ids) != len(loss_mask) on some rows; the mask does "
                   f"not line up with the tokens, so every supervised position is wrong")
    if int((mask.map(sum) == 0).sum()):
        bad.append(f"{side}: rows with zero supervised tokens. Their logprob sum is empty, "
                   f"so the pair contributes no preference signal")
    if int((ids.map(len) > max_len).sum()):
        bad.append(f"{side}: rows longer than {max_len:,}. 18 and 19 would skip them at "
                   f"different points; rebuild with --max-seq-len {max_len}")
if int((train.common_prefix_tokens <= 0).sum()):
    bad.append("pairs with no shared prefix: the two sides do not answer the same prompt, "
               "which is the one assumption DPO cannot do without")
identical = int((train.chosen_input_ids.map(tuple) == train.rejected_input_ids.map(tuple)).sum())
if identical:
    bad.append(f"{identical} pairs whose two sides are token-identical; their gradient is "
               f"exactly zero and they dilute the batch")
if train.pair_id.duplicated().any():
    bad.append("duplicate pair_id values: the same pair would be trained on more than once "
               "and the reference-logprob join would be ambiguous")
if bad:
    sys.exit("REFUSING TO TRAIN:\n  - " + "\n  - ".join(bad))

trained = train.chosen_loss_mask.map(sum).sum() + train.rejected_loss_mask.map(sum).sum()
print(f"pairs: train={len(train):,} supervised_tokens={int(trained):,} "
      f"chosen_p50={int(train.chosen_loss_mask.map(sum).median()):,} "
      f"max_len={int(max(train.chosen_input_ids.map(len).max(), train.rejected_input_ids.map(len).max())):,}")
holdout = root / "dpo_holdout.parquet"
if holdout.is_file():
    print(f"holdout: {len(pd.read_parquet(holdout)):,} pairs (disjoint task groups)")
else:
    print("WARNING: no dpo_holdout.parquet -- there will be no held-out reward accuracy, "
          "so the only evidence the run did anything is its own training loss")
EOF_PY

# ------------------------------------------------------- 2. reference logprobs
# One process per GPU, no communication: each shard writes its own parquet and 19
# reads them together.
#
# THIS STAGE ALWAYS RUNS. It used to be wrapped in
#   if ! compgen -G "$REF_DIR/ref_logps*.parquet"
# which skipped all of it the moment ONE shard file existed -- and 18 flushes every
# 50 pairs, so a shard parquet appears minutes into a pass that takes hours. An
# interrupted reference pass therefore left exactly the state that made the re-run
# skip scoring, after which 19 failed GATE 2 (coverage) even though this script's own
# error message had just told the operator that "18 is resumable -- re-run this
# script". 18 is idempotent by design: it reads the partial parquet, REFUSES to
# resume across a changed checkpoint / dtype / chunk (its provenance sidecar), and
# scores only the rows that are missing. Re-entering it costs a directory listing when
# there is nothing to do; skipping it wrongly costs a whole reference pass.
echo "=== scoring reference logprobs ($REF_DTYPE) -> $REF_DIR"
mkdir -p "$REF_DIR"

# Shard IDs are GLOBAL to the job, not per node. With --shard $i --num-shards $NGPUS
# on every node, all NNODES nodes take the same NGPUS slices of the same frame and
# write them to the same ref_logps_shard$i.parquet on the shared filesystem: the
# pairs in slices NGPUS..NNODES*NGPUS-1 are never scored by anyone, and the files
# that do get written are several processes overwriting one path. The offset makes
# each (node, GPU) pair own exactly one slice.
TOTAL_SHARDS=$(( NNODES * NGPUS ))
SHARD_BASE=$(( ${NODE_RANK:-0} * NGPUS ))
pids=()
shard_ids=()
for (( i = 0; i < NGPUS; i++ )); do
  shard=$(( SHARD_BASE + i ))
  CUDA_VISIBLE_DEVICES="$i" python scripts/18_dpo_ref_logprobs.py \
    --pairs "$PAIRS_DIR" --model-path "$POLICY" --out "$REF_DIR" \
    --dtype "$REF_DTYPE" --max-seq-len "$DPO_SEQ_LEN" \
    --logit-chunk "$LOGIT_CHUNK" \
    --shard "$shard" --num-shards "$TOTAL_SHARDS" \
    > "$BASE_FOLDER/logs/dpo_ref_shard$shard.log" 2>&1 &
  pids+=("$!")
  shard_ids+=("$shard")
done
# Keep the shard id beside the pid and the exit STATUS beside both. `wait "$pid" ||
# rc=1` threw away which of the sixteen failed and how, which on 2026-08-20 left the
# operator with a glob of logs that all ended in [done] and no way to tell which
# process had returned non-zero.
failed=()
for i in "${!pids[@]}"; do
  code=0
  wait "${pids[$i]}" || code=$?
  (( code == 0 )) || failed+=("${shard_ids[$i]}=$code")
done
if (( ${#failed[@]} > 0 )); then
  echo "${#failed[@]} of ${#pids[@]} reference shards on node ${NODE_RANK:-0} failed:" >&2
  rst_explain_shard_failures "$BASE_FOLDER/logs/dpo_ref_shard" "${failed[@]}"
  echo "18 is resumable -- fix the cause and re-run this script, already-scored rows are kept" >&2
  exit 1
fi

# Multi-node: this node's eight shards are done, the other nodes' twenty-four may not
# be. Waiting here rather than at torchrun means a missing node is reported as a
# missing node, instead of as 19's coverage gate failing on rank 0 after the whole
# world has already loaded a 27.8 B model.
if (( TOTAL_SHARDS > NGPUS )); then
  echo "=== waiting for the other nodes' shards ($TOTAL_SHARDS total)"
  for (( shard = 0; shard < TOTAL_SHARDS; shard++ )); do
    (( shard >= SHARD_BASE && shard < SHARD_BASE + NGPUS )) && continue
    if ! wait_readable_parquet "$REF_DIR/ref_logps_shard$shard.parquet" "${REF_WAIT_SEC:-7200}"; then
      echo "shard $shard never appeared at $REF_DIR/ref_logps_shard$shard.parquet." >&2
      echo "Node $(( shard / NGPUS )) is not running this script, failed early (check its" >&2
      echo "own $BASE_FOLDER/logs/dpo_ref_shard*.log), or was launched with a different" >&2
      echo "NGPUS/NNODES so its shard numbering does not match this node's." >&2
      exit 1
    fi
  done
fi

# Coverage, checked HERE and again by 19's GATE 2. Not redundant: this check can name
# the shard that is short and the stale file that has to go, because it still knows
# NNODES, NGPUS and the shard numbering; by the time 19 runs, all it can see is a set
# of parquets that do not cover the pairs.
python - "$PAIRS_DIR" "$REF_DIR" "$DPO_SEQ_LEN" "$TOTAL_SHARDS" <<'EOF_PY' || exit 1
import json
import sys
from pathlib import Path

import pandas as pd

pairs_dir, ref_dir = Path(sys.argv[1]), Path(sys.argv[2])
max_len, total_shards = int(sys.argv[3]), int(sys.argv[4])

want: set[str] = set()
for name in ("dpo_train.parquet", "dpo_holdout.parquet"):
    path = pairs_dir / name
    if not path.is_file():
        continue
    frame = pd.read_parquet(path, columns=["pair_id", "chosen_n_tokens", "rejected_n_tokens"])
    keep = frame[(frame.chosen_n_tokens <= max_len) & (frame.rejected_n_tokens <= max_len)]
    want |= set(keep.pair_id)

shards = sorted(ref_dir.glob("ref_logps*.parquet"))
if not shards:
    sys.exit(f"REFUSING TO TRAIN: no ref_logps*.parquet under {ref_dir} even though every "
             f"shard process exited 0. Check {ref_dir}/../../logs/dpo_ref_shard*.log.")

# Stale shards from a previous run with a different GPU count are the failure mode
# that a plain row count cannot see: ref_logps_shard5.parquet left over from an
# 8-GPU pass holds pairs that a 4-GPU pass assigns to other shards, so the union
# looks complete while some pairs carry logprobs from a different slicing (and, if
# the checkpoint was re-exported in between, from different weights).
stale, fingerprints = [], {}
for shard in shards:
    manifest = shard.with_name(shard.stem + "_manifest.json")
    if not manifest.is_file():
        stale.append(f"{shard.name}: no manifest, so its shard count and checkpoint are unknown")
        continue
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    if int(meta.get("num_shards", -1)) != total_shards:
        stale.append(f"{shard.name}: num_shards={meta.get('num_shards')} but this job has "
                     f"{total_shards}")
    fingerprints.setdefault(meta.get("checkpoint_fingerprint"), []).append(shard.name)
if len(fingerprints) > 1:
    lines = "\n".join(f"    {fp!s:.16}...: {', '.join(names)}"
                      for fp, names in sorted(fingerprints.items(), key=lambda kv: str(kv[0])))
    sys.exit(f"REFUSING TO TRAIN: these shards were scored against DIFFERENT checkpoints, so "
             f"the reference is not one model:\n{lines}\n  Delete {ref_dir} and rescore.")
if stale:
    sys.exit("REFUSING TO TRAIN: leftover reference shards from an incompatible run:\n  - "
             + "\n  - ".join(stale)
             + f"\n  Delete {ref_dir} and rescore, or restore the NNODES/NGPUS that wrote them.")

got: set[str] = set()
overlap = 0
for shard in shards:
    ids = set(pd.read_parquet(shard, columns=["pair_id"]).pair_id)
    overlap += len(got & ids)
    got |= ids
if overlap:
    sys.exit(f"REFUSING TO TRAIN: {overlap} pair_ids appear in more than one shard. The "
             f"slices are supposed to be disjoint, so two processes scored the same pairs "
             f"-- delete {ref_dir} and rescore.")

missing = want - got
print(f"reference coverage: {len(got):,}/{len(want):,} pairs over {len(shards)} shards")
if missing:
    sys.exit(f"REFUSING TO TRAIN: {len(missing):,} pairs have no reference logprob "
             f"(e.g. {sorted(missing)[:3]}). Every shard exited 0, so the likely cause is a "
             f"--max-seq-len or --pairs mismatch between this script and the shards. Re-run "
             f"this script; 18 keeps what it already scored.")
extra = got - want
if extra:
    print(f"note: {len(extra):,} scored pairs are not in the current pairs set (rebuilt data, "
          f"or a narrower --max-seq-len now). 19 joins on pair_id, so they are ignored.")
EOF_PY

# ------------------------------------------------------------------ 3. train
# --length-normalize is ON by default here, deliberately. With summed logprobs the
# gradient norm runs ~1e3 against --max-grad-norm 1.0, so the clip becomes the whole
# step-size schedule and --lr stops meaning what it usually means (19 measures this
# and warns). Per-token normalization also removes the length term the objective
# would otherwise reward. Set DPO_LENGTH_NORM=0 to train the summed objective, and
# say which one you used in the report.
LENGTH_NORM_ARG=(--length-normalize)
[[ "${DPO_LENGTH_NORM:-1}" == "0" ]] && LENGTH_NORM_ARG=()

# 19_train_dpo.py does not import wandb at all -- it writes dpo_training_summary.json,
# and THAT is the evidence for this stage. These two lines only keep a wandb-enabled
# library in the env from silently opening a run; they are not how you get metrics.
export WANDB_MODE="${WANDB_KEY:+online}"; export WANDB_MODE="${WANDB_MODE:-offline}"
[[ -n "${WANDB_KEY:-}" ]] && export WANDB_API_KEY="$WANDB_KEY"

# Tee, so the failure block below can READ what happened instead of only knowing that
# something did. An NCCL watchdog timeout is ~500 lines of C++ frames whose one
# diagnostic detail (the per-rank NumelIn) is three screens above the exit status; with
# the output only streaming to the console there is nothing left to classify.
# `pipefail` is already set, and tee exits 0, so RC below is still torchrun's status.
TRAIN_LOG="$BASE_FOLDER/logs/dpo_train.log"
torchrun \
  --nnodes "$NNODES" --nproc_per_node "$NGPUS" \
  --node_rank "${NODE_RANK:-0}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29501}" \
  scripts/19_train_dpo.py \
  --pairs "$PAIRS_DIR" \
  --ref-logps "$REF_DIR" \
  --model-path "$POLICY" \
  --out "$OUT_DIR" \
  --beta "${DPO_BETA:-0.1}" \
  --lr "${DPO_LR:-5e-7}" \
  --grad-accum "${DPO_GRAD_ACCUM:-4}" \
  --epochs "${DPO_EPOCHS:-1}" \
  --max-seq-len "$DPO_SEQ_LEN" \
  --param-dtype "$PARAM_DTYPE" \
  --logit-chunk "$LOGIT_CHUNK" \
  "${LENGTH_NORM_ARG[@]}" \
  "$@" 2>&1 | tee "$TRAIN_LOG"
RC=$?

if (( RC != 0 )); then
  cat >&2 <<EOF

DPO failed (exit $RC). The gates in 19_train_dpo.py fail loudly on purpose:
  GATE 1 fingerprint  -> POLICY is not the checkpoint the reference was scored on.
                         Delete $REF_DIR and re-run, or point POLICY back.
  GATE 2 coverage     -> the reference pass did not finish. Re-run this script; 18
                         resumes from the partial parquet.
  GATE 3 calibration  -> something changed between the reference pass and training
                         (mask, tokenizer, dtype). Fix the cause; do not raise the
                         tolerance to get past it.
EOF
  # ... and when it is none of those, say so rather than leaving three wrong hypotheses
  # as the last word. This prints nothing unless the trainer actually timed out in NCCL.
  rst_explain_nccl_timeout "$TRAIN_LOG"
  exit "$RC"
fi

cat <<EOF

Done. $OUT_DIR/dpo_training_summary.json is the artifact to read, and to quote from:
  * gates.step0_loss must be ~0.693147 (log 2). That is the proof the frozen
    reference and the policy at initialization were the same model. With
    gates.dtype_match true it is exact.
  * metrics.holdout_reward_accuracy is LIKELIHOOD RANKING on held-out task groups:
    how often the model assigns higher likelihood to a trajectory that passed the
    verifier than to one that failed the same task. 0.5 is no preference. It is not
    a pass rate and must never be compared with terminal-bench numbers.
  * optimization.clip_active_fraction says whether --lr or --max-grad-norm set the
    real step size. Quote whichever one did.
  * warnings[] is empty on a clean run. If it is not, it belongs in the report.

The checkpoint is $OUT_DIR/hf. To evaluate it without a sandbox, use the SFT
holdout -- 06b takes an SFT/pretokenized parquet, not a pairs file -- and pass the
SFT model as --base-model so the delta is DPO's and not the fine-tune's:
  python scripts/06b_eval_offline.py --model-path $OUT_DIR/hf \\
    --base-model $POLICY \\
    --holdout ${DATA_DIR:-$BASE_FOLDER/sft-v1-cap10}/pretokenized_holdout.parquet \\
    --out $BASE_FOLDER/eval/dpo-offline
Agentic eval still needs a container; DPO removes that requirement from TRAINING
only. A DPO checkpoint with no agentic eval is an untested checkpoint, and the
report has to say that.
EOF
