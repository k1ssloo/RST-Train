#!/usr/bin/env bash
# Turn a verl FSDP checkpoint into a checkpoint eval can actually serve.
#
#   # from the Hub (what a finished cluster run leaves behind)
#   bash scripts/08_prepare_eval_ckpt.sh --from-hub khazic/rst-qwen3.5-4b-sft --step 82
#
#   # from a local training dir
#   bash scripts/08_prepare_eval_ckpt.sh --ckpt $BASE_FOLDER/qwen3.5-4b-rst-sft-v1/global_step_82
#
# WHY THIS EXISTS SEPARATELY FROM 20_run_all.sh
#   20_run_all.sh's export stage does the same merge, but only for a checkpoint that is
#   still sitting where training wrote it. A run that finished, got uploaded, and is now
#   being evaluated somewhere else -- a different node, a different pod, after the scratch
#   dir was reclaimed -- has no such path, and the manual sequence is four commands with
#   two failure modes that produce a checkpoint which loads and evaluates as if it were
#   trained. This script is that sequence with the checks attached.
#
# WHAT verl LEAVES YOU, AND WHY IT IS NOT LOADABLE
#   By default verl's FSDP checkpointer writes SHARDS:
#     global_step_82/model_world_size_8_rank_{0..7}.pt   the weights, one slice each
#     global_step_82/optim_world_size_8_rank_{0..7}.pt   optimizer state (not needed here)
#     global_step_82/huggingface/                        config.json + tokenizer, NO weights
#   Pointing sglang at that directory fails inside sglang with a message about the model.
#   `verl.model_merger` reassembles the shards; everything after it here is checking that
#   what came out is the model that was trained.
#
# THE TWO CHECKS THAT MATTER
#   1. EVERY shard present. A merge over 7 of 8 shards does not fail -- it produces a model
#      with a slice of every tensor left at its initialization. `fsdp_config.json` records
#      the world size, so the count is knowable before merging rather than after evaluating.
#   2. The text weights actually MOVED. 07_restore_vision.py refuses when text tensors would
#      have to come from the base checkpoint, but it cannot tell a merge that silently
#      reproduced the base from one that worked. This script diffs the merged text stack
#      against the base and fails if it is identical -- at lr 3e-6 over 82 steps the change
#      is small (1e-3-ish relative), and small is very different from zero.
set -euo pipefail

: "${BASE_FOLDER:?set BASE_FOLDER}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
# shellcheck source=lib_env.sh
source "$REPO_DIR/scripts/lib_env.sh"
rst_bootstrap_python || exit 2
rst_enter_env "${ENV_NAME:-rstverl}" || exit 2

MODEL_KEY="${MODEL_KEY:-qwen3.5-4b}"
CKPT=""; FROM_HUB=""; STEP=""; OUT=""; REVISION="main"; SKIP_GENERATE=0
while (($#)); do
  case "$1" in
    --ckpt)       CKPT="$2"; shift 2 ;;
    --from-hub)   FROM_HUB="$2"; shift 2 ;;
    --step)       STEP="$2"; shift 2 ;;
    --revision)   REVISION="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    --model-key)  MODEL_KEY="$2"; shift 2 ;;
    --skip-generate) SKIP_GENERATE=1; shift ;;
    -h|--help)    sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

eval "$(python scripts/model_registry.py --key "$MODEL_KEY" --mem-class 80GB \
          --backend verl --gpus 8 --gpus-per-node 8 --max-seq-len 32768 --shell)"
BASE_MODEL="$BASE_FOLDER/$MODEL_DIR_NAME"
OUT="${OUT:-$BASE_FOLDER/eval-ckpt/$MODEL_KEY-step${STEP:-local}}"

# ---- 1. get the checkpoint ---------------------------------------------------
# Only ckpt/global_step_<n>/ and only the model shards: optim_* is 2x the weights and
# nothing downstream reads it. allow_patterns beats a full snapshot_download by a wide
# margin on a repo that also holds four earlier steps.
if [[ -n "$FROM_HUB" ]]; then
  [[ -n "$STEP" ]] || { echo "--from-hub needs --step (e.g. --step 82)" >&2; exit 2; }
  CKPT="$BASE_FOLDER/hub-ckpt/$(basename "$FROM_HUB")/global_step_$STEP"
  echo "==> downloading $FROM_HUB ckpt/global_step_$STEP -> $CKPT"
  python - "$FROM_HUB" "$STEP" "$CKPT" "$REVISION" <<'EOF_PY' || exit 2
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo, step, dest, revision = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
prefix = f"ckpt/global_step_{step}"
local = snapshot_download(
    repo_id=repo, revision=revision,
    allow_patterns=[f"{prefix}/model_world_size_*", f"{prefix}/fsdp_config.json",
                    f"{prefix}/huggingface/*"],
)
src = Path(local) / prefix
if not src.is_dir():
    sys.exit(f"{repo} has no {prefix}/ (check --step against the repo's file list)")
dest.parent.mkdir(parents=True, exist_ok=True)
if dest.is_symlink() or dest.exists():
    shutil.rmtree(dest, ignore_errors=True) if dest.is_dir() else dest.unlink()
# copy, not symlink: the merger reads these repeatedly and the HF cache is often on a
# different (smaller, faster-to-evict) filesystem than BASE_FOLDER.
shutil.copytree(src, dest, symlinks=False)
print(f"[hub] {sum(1 for _ in dest.rglob('*') if _.is_file())} files -> {dest}")
EOF_PY
fi
[[ -n "$CKPT" ]] || { echo "pass --ckpt <dir> or --from-hub <repo> --step <n>" >&2; exit 2; }
[[ -d "$CKPT" ]] || { echo "no such directory: $CKPT" >&2; exit 2; }

# ---- 2. is every shard here? -------------------------------------------------
# The failure this prevents is silent. verl.model_merger loads whichever
# model_world_size_N_rank_*.pt files it finds; a missing rank leaves that rank's slice of
# EVERY tensor at its initialization value, and the result loads, serves and evaluates.
python - "$CKPT" <<'EOF_PY' || exit 2
import json
import re
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])
shards = sorted(ckpt.glob("model_world_size_*_rank_*.pt"))
if not shards:
    if (ckpt / "config.json").is_file() and (
        list(ckpt.glob("*.safetensors")) or list(ckpt.glob("pytorch_model*.bin"))):
        print("[shards] already an HF checkpoint (weights present); nothing to merge")
        raise SystemExit(0)
    sys.exit(f"{ckpt} holds neither model_world_size_*_rank_*.pt shards nor HF weights.\n"
             f"  A verl FSDP checkpoint dir looks like: model_world_size_8_rank_0.pt ...\n"
             f"  If you pointed at the parent, append /global_step_<n>.")

worlds = {int(m.group(1)) for s in shards if (m := re.search(r"world_size_(\d+)_rank", s.name))}
if len(worlds) != 1:
    sys.exit(f"shards from more than one world size in {ckpt}: {sorted(worlds)}. Two runs "
             f"wrote into the same directory; merging them mixes unrelated weights.")
world = worlds.pop()

declared = None
cfg = ckpt / "fsdp_config.json"
if cfg.is_file():
    declared = json.loads(cfg.read_text()).get("world_size")
    if declared is not None and int(declared) != world:
        sys.exit(f"fsdp_config.json says world_size={declared} but the shard names say "
                 f"{world}. Trust neither; find out which ranks actually wrote.")

ranks = {int(m.group(1)) for s in shards if (m := re.search(r"_rank_(\d+)\.pt$", s.name))}
missing = sorted(set(range(world)) - ranks)
if missing:
    sys.exit(f"REFUSING TO MERGE: {len(missing)} of {world} shards are missing "
             f"(ranks {missing}). The merge would NOT fail -- it would leave those ranks' "
             f"slice of every tensor at its initialization value, and the checkpoint would "
             f"load, serve and evaluate as if it were whole. Re-download, or copy the "
             f"missing ranks from the node that wrote them.")
sizes = {s.stat().st_size for s in shards}
print(f"[shards] {world}/{world} present, {min(sizes) / 2**30:.2f}-{max(sizes) / 2**30:.2f} GiB each")
if min(sizes) * 2 < max(sizes):
    print(f"[shards] WARNING: shard sizes differ by more than 2x. FSDP shards are within a "
          f"few percent of each other; a truncated upload looks exactly like this.")
EOF_PY

# ---- 3. merge ----------------------------------------------------------------
MERGED="$OUT.merged"
if [[ -f "$MERGED/config.json" ]] && compgen -G "$MERGED/*.safetensors" > /dev/null; then
  echo "==> reusing existing merge at $MERGED"
elif compgen -G "$CKPT/*.safetensors" > /dev/null; then
  echo "==> $CKPT is already HF-format; copying"
  rm -rf "$MERGED"; mkdir -p "$MERGED"; cp -r "$CKPT/." "$MERGED/"
else
  echo "==> merging FSDP shards -> $MERGED"
  rm -rf "$MERGED"; mkdir -p "$MERGED"
  # Three attempts, cheapest first. verl renamed this CLI (0.9.x takes a `merge`
  # subcommand, older builds do not), and separately its merger reads the shard layout off
  # a PICKLED DeviceMesh, which an unrelated torch cannot always reconstruct -- see
  # verl_backend/fsdp_merge_compat.py. The fallback replaces one method and calls verl's
  # own merge, so it is not a second implementation of the format.
  if ! python -m verl.model_merger merge --backend fsdp \
         --local_dir "$CKPT" --target_dir "$MERGED"; then
    echo "  'merge' subcommand failed; trying the older CLI without it" >&2
    if ! python -m verl.model_merger --backend fsdp \
           --local_dir "$CKPT" --target_dir "$MERGED"; then
      echo "  both verl CLIs failed; retrying with the pickled-DeviceMesh workaround" >&2
      python verl_backend/fsdp_merge_compat.py \
        --local_dir "$CKPT" --target_dir "$MERGED" || {
          echo "FATAL: the merge failed three ways. The shards are present (the gate above" >&2
          echo "       checked), so this is not a download problem." >&2
          echo "       If the last error mentions _MeshLayout / device_mesh, the checkpoint" >&2
          echo "       was pickled by a torch this one cannot read and the workaround did" >&2
          echo "       not cover it; merge on the cluster that wrote it instead." >&2
          exit 2; }
    fi
  fi
fi

# ---- 4. sidecars -------------------------------------------------------------
# The merger writes weights and config; the tokenizer it does not always carry, and
# sglang needs both. Prefer the checkpoint's own huggingface/ dir over the base model:
# if training changed the tokenizer, the checkpoint's copy is the one it was trained with.
for f in tokenizer.json tokenizer_config.json tokenizer.model vocab.json merges.txt \
         added_tokens.json special_tokens_map.json generation_config.json \
         chat_template.jinja preprocessor_config.json processor_config.json; do
  [[ -e "$MERGED/$f" ]] && continue
  if [[ -e "$CKPT/huggingface/$f" ]]; then cp "$CKPT/huggingface/$f" "$MERGED/"
  elif [[ -e "$BASE_MODEL/$f" ]];   then cp "$BASE_MODEL/$f" "$MERGED/"
  fi
done
[[ -f "$MERGED/tokenizer.json" ]] || {
  echo "FATAL: no tokenizer.json in $MERGED, and neither $CKPT/huggingface/ nor" >&2
  echo "       $BASE_MODEL had one. sglang will fail on the tokenizer, which reads as a" >&2
  echo "       model problem. Download the base model first (scripts/02_download.sh)." >&2
  exit 2; }

# ---- 5. restore the vision tower --------------------------------------------
# The training round trip carries only the text stack. Without model.visual.* the
# checkpoint does not load as a Qwen3_5ForConditionalGeneration at all.
if [[ "$HAS_VISION" == "1" ]]; then
  [[ -d "$BASE_MODEL" ]] || {
    echo "FATAL: $MODEL_KEY has a vision tower, so model.visual.* must be spliced back in" >&2
    echo "       from the original weights, and $BASE_MODEL is not there." >&2
    echo "       Run: MODEL_KEY=$MODEL_KEY bash scripts/02_download.sh" >&2
    exit 2; }
  echo "==> restoring model.visual.* / mtp.* from $BASE_MODEL"
  python scripts/07_restore_vision.py --trained "$MERGED" --original "$BASE_MODEL" --out "$OUT"
else
  echo "==> no vision tower for $MODEL_KEY; using the merge directly"
  rm -rf "$OUT"; mkdir -p "$(dirname "$OUT")"; cp -r "$MERGED" "$OUT"
fi

# ---- 6. did the text weights actually move? ---------------------------------
# 07_restore_vision.py refuses when text tensors would have to be TAKEN from the base. It
# cannot detect a merge that produced base-equal values on its own. This can: sample the
# text stack and diff. Expect small and nonzero -- 82 steps at lr 3e-6 moves weights by
# ~1e-3 relative, and both "0" and "1" would mean something is wrong.
python - "$OUT" "$BASE_MODEL" "$MODEL_KEY" <<'EOF_PY' || exit 2
import sys
from pathlib import Path

import torch
from safetensors import safe_open

trained, base, key = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]


def index(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                out[name] = shard
    return out


tmap, bmap = index(trained), index(base)
if not tmap:
    sys.exit(f"no safetensors in {trained}")

vision = [k for k in tmap if k.startswith(("model.visual.", "visual.", "mtp."))]
text = [k for k in tmap if k not in set(vision) and k in bmap and "weight" in k]
if not text:
    sys.exit(f"no comparable text tensors between {trained} and {base}; the two exporters "
             f"name things differently and no downstream check would notice.")


def load(name: str, mapping: dict[str, Path]) -> torch.Tensor:
    with safe_open(mapping[name], framework="pt") as handle:
        return handle.get_tensor(name).float()


# Sample 1-D and multi-dim tensors SEPARATELY, because only one of the two can carry the
# alarm and a naive stride can miss the matrices entirely. Spread each across the depth: a
# merge that dropped an FSDP unit changes some layers and not others.
def spread(names: list[str], count: int) -> list[str]:
    return names[:: max(1, len(names) // count)][:count]


def shape_of(name: str, mapping: dict[str, Path]) -> tuple[int, ...]:
    with safe_open(mapping[name], framework="pt") as handle:
        return tuple(handle.get_slice(name).get_shape())


flat = [k for k in text if len(shape_of(k, tmap)) < 2]
matrices = [k for k in text if len(shape_of(k, tmap)) >= 2]
sample = spread(matrices, 16) + spread(flat, 8)
if not matrices:
    sys.exit(f"no multi-dimensional text weights found in {trained}; without a projection "
             f"matrix to compare, this check cannot distinguish training from rounding.")

moved, still_flat, still_matrix = [], [], []
for name in sample:
    t, b = load(name, tmap), load(name, bmap)
    if t.shape != b.shape:
        continue
    denom = b.norm().item() or 1.0
    rel = (t - b).norm().item() / denom
    if rel != 0.0:
        moved.append((name, rel))
    elif t.dim() < 2:
        still_flat.append(name)
    else:
        still_matrix.append(name)

print(f"[diff] {len(moved)}/{len(sample)} sampled text tensors differ from base "
      f"({len(matrices)} matrices / {len(flat)} 1-D available)")
if moved:
    rels = sorted(r for _, r in moved)
    print(f"[diff] relative L2 change: min {rels[0]:.2e}  median {rels[len(rels) // 2]:.2e}  "
          f"max {rels[-1]:.2e}")

# A 1-D norm going bit-identical is EXPECTED and not evidence of anything. Training keeps
# fp32 masters while the merge saves bf16, and bf16 has 8 mantissa bits: at magnitude ~0.33
# the spacing is ~2e-3, so the ~6e-5 that 82 steps at lr 3e-6 moves a norm weight rounds
# straight back to the base value. Measured on this very checkpoint:
#   layers.26.input_layernorm.weight  fp32 |delta| max 5.77e-05  -> identical after bf16
#   layers.26.mlp.down_proj.weight    fp32 |delta| max 9.24e-05  -> NOT identical after bf16
# The matrices are the informative ones: a [d, d] projection has enough elements far from a
# bf16 tie boundary that some always move.
if still_flat:
    print(f"[diff] {len(still_flat)} 1-D tensor(s) identical to base, e.g. "
          f"{still_flat[:3]} -- expected: an fp32 update of ~1e-5 is below bf16 resolution "
          f"at these magnitudes, so it rounds away when the merge saves bf16. Not a merge "
          f"fault; the fp32 master weights did move.")

if not moved:
    sys.exit("REFUSING: every sampled text tensor is bit-identical to the base model. The "
             "merge produced the base weights, so eval would measure the base model under "
             "the trained model's name. Check that --ckpt is the step you meant and that "
             "the merge read all shards.")
if still_matrix:
    sys.exit(f"REFUSING: {len(still_matrix)} multi-dimensional text weight(s) are "
             f"bit-identical to the base while others moved, e.g. {still_matrix[:4]}. Unlike "
             f"a 1-D norm, a projection matrix cannot round back to the base wholesale, so "
             f"this is a partial merge (an FSDP unit's shards missing or unread) rather than "
             f"a precision effect. Re-merge from a complete shard set.")

if vision:
    name = vision[0]
    if name in bmap:
        t, b = load(name, tmap), load(name, bmap)
        same = t.shape == b.shape and torch.equal(t, b)
        verdict = "matches base (expected)" if same else (
            "DIFFERS from base -- unexpected, 07_restore_vision.py copies it verbatim")
        print(f"[diff] vision tower {verdict} ({name})")
print(f"[diff] ok: {key} text stack is trained, {len(vision)} vision/mtp tensors present")
EOF_PY

# ---- 7. does it load and generate? ------------------------------------------
# The last thing that can still be wrong is the thing sglang would report as its own
# problem: a config/tokenizer/architecture mismatch. One real generate settles it.
if [[ "$SKIP_GENERATE" == "0" ]]; then
  echo "==> load + generate smoke test"
  python - "$OUT" <<'EOF_PY' || exit 2
import sys
from pathlib import Path

import torch
import transformers

path = Path(sys.argv[1])
tok = transformers.AutoTokenizer.from_pretrained(str(path))
cls = None
for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
    cls = getattr(transformers, name, None)
    if cls is None:
        continue
    try:
        model = cls.from_pretrained(str(path), dtype=torch.bfloat16,
                                    device_map="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as exc:  # noqa: BLE001 -- probing which auto class fits
        print(f"[smoke] {name}: {type(exc).__name__}: {str(exc)[:160]}")
        cls = None
        continue
    print(f"[smoke] loaded with {name}")
    break
else:
    sys.exit("no transformers auto class could load this checkpoint; the messages above say "
             "why each one refused")

model.eval()
# The training render, not a bare prompt: this is also the only cheap check that the chat
# template survived the merge, and a template mismatch is invisible in a loss number.
messages = [{"role": "user", "content": "Reply with the single word OK."}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                               enable_thinking=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=24, do_sample=False)
reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
print(f"[smoke] prompt tail  : {text[-60:]!r}")
print(f"[smoke] generated    : {reply!r}")
if not reply.strip():
    sys.exit("the model generated nothing. An empty continuation after a correct prompt "
             "usually means the lm_head did not survive the round trip -- check whether "
             "config.json says tie_word_embeddings and whether lm_head.weight exists.")
EOF_PY
fi

# ---- 8. what to run next -----------------------------------------------------
cat <<EOF

=============================================================================
Eval-ready checkpoint: $OUT
=============================================================================

Prerequisites eval needs that this script does NOT provide -- check them now, because
06_eval.py discovers each one late and reports it as a model problem:

  python -c 'import sglang; print("sglang", sglang.__version__)'
  source scripts/00b_setup_sandbox.sh          # a place to run task containers
  ls "\${TB_HARD_TASKS:-$BASE_FOLDER/terminal-bench-hard/tasks}" | head -3
  ls "\${TB2_TASKS:-$BASE_FOLDER/terminal-bench-2}" | head -3

Agentic eval (the real numbers). $MODEL_KEY serves at TP=$SERVE_TP:

  python scripts/06_eval.py --model-path "$OUT" \\
    --tp $SERVE_TP --served-name rst-sft --label sft \\
    --benchmarks tb-hard,tb2 --runs 3 --n-concurrent 8 \\
    --out "$BASE_FOLDER/eval/sft-$MODEL_KEY"

  AND the base model through the SAME harness. $MODEL_KEY has no published paper number
  (registry: paper_reference $( [[ -z "${REFERENCE_CHECKPOINT:-}" ]] && echo null || echo "$REFERENCE_CHECKPOINT" )),
  so base-vs-SFT is the ONLY thing that can answer "did fine-tuning help":

  python scripts/06_eval.py --model-path "$BASE_MODEL" \\
    --tp $SERVE_TP --served-name rst-base --label base \\
    --benchmarks tb-hard,tb2 --runs 3 --n-concurrent 8 \\
    --out "$BASE_FOLDER/eval/base-$MODEL_KEY"

Container-free eval. Weaker, but it is not nothing, and it runs anywhere:

  python scripts/06b_eval_offline.py --model-path "$OUT" \\
    --base-model "$BASE_MODEL" \\
    --holdout "\${DATA_DIR:-$BASE_FOLDER/sft-v1-cap10}/rst_sft_holdout.parquet" \\
    --tokenizer "$BASE_MODEL" \\
    --out "$BASE_FOLDER/eval/offline-$MODEL_KEY"

Then the report:

  python scripts/14_make_report.py --help

Disk: $MERGED is a merge artifact and can be deleted once $OUT exists
($(du -sh "$MERGED" 2>/dev/null | cut -f1) held).
EOF
