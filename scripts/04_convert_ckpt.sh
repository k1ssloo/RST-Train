#!/usr/bin/env bash
# HF -> Megatron torch_dist (before training) and torch_dist -> HF (after).
#
#   export BASE_FOLDER=/shared/rst SLIME_DIR=$BASE_FOLDER/slime
#   bash scripts/04_convert_ckpt.sh to_dist
#   bash scripts/04_convert_ckpt.sh to_hf   <iter_dir> <out_dir>
#
# READ THIS FIRST -- the vision tower.
#   Qwen/Qwen3.5-27B is `Qwen3_5ForConditionalGeneration`: a 64-layer text stack
#   (48 gated-delta-net + 16 full-attention) PLUS a 27-block ViT (`model.visual.*`)
#   PLUS a 1-layer MTP head (`mtp.*`). slime's `slime_plugins.models.qwen3_5` spec
#   is the TEXT path, so conversion carries the text stack only. Our SFT data is
#   pure text, so training the text stack is correct -- but the converted-back
#   checkpoint will be missing the ViT/MTP tensors and will not load as a
#   ConditionalGeneration model. Fix with scripts/07_restore_vision.py, which
#   splices the trained text weights into a copy of the original HF checkpoint
#   and leaves ViT/MTP bit-identical.
#   (If you instead want to train vision too, use the `qwen3_5_vl` spec -- not
#   needed for Terminal-Bench, which is text-only.)
set -ex
: "${BASE_FOLDER:?set BASE_FOLDER}"
SLIME_DIR="${SLIME_DIR:-$BASE_FOLDER/slime}"
MODEL_KEY="${MODEL_KEY:-qwen3.5-27b}"
eval "$(python "$(dirname "${BASH_SOURCE[0]}")/model_registry.py" --key "$MODEL_KEY" --shell 2>/dev/null)"
MODEL_NAME="${MODEL_NAME:-$MODEL_DIR_NAME}"
export PYTHONPATH="${BASE_FOLDER}/Megatron-LM:${PYTHONPATH:-}"

case "${1:-to_dist}" in
to_dist)
  # ~52 GiB in, similar out. Needs ~120GB host RAM and ~60GB free disk.
  # Single-process, CPU-bound; takes roughly 20-40 min. No GPU needed for the
  # conversion itself, but keep one visible so torch initializes cleanly.
  cd "$SLIME_DIR"
  source "$SLIME_DIR/scripts/models/${SLIME_SPEC}"
  python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$BASE_FOLDER/$MODEL_NAME" \
    --save "$BASE_FOLDER/${MODEL_NAME}_torch_dist"
  du -sh "$BASE_FOLDER/${MODEL_NAME}_torch_dist"
  ;;
to_hf)
  ITER_DIR="${2:?usage: $0 to_hf <slime_ckpt_dir> <out_hf_dir>}"
  OUT_DIR="${3:?usage: $0 to_hf <slime_ckpt_dir> <out_hf_dir>}"
  cd "$SLIME_DIR"
  source "$SLIME_DIR/scripts/models/${SLIME_SPEC}"
  python tools/convert_torch_dist_to_hf.py \
    "${MODEL_ARGS[@]}" \
    --load "$ITER_DIR" \
    --output-dir "$OUT_DIR" \
    --origin-hf-dir "$BASE_FOLDER/$MODEL_NAME"
  echo "Now restore ViT/MTP so the checkpoint is loadable by vLLM/SGLang:"
  echo "  python scripts/07_restore_vision.py --trained $OUT_DIR \\"
  echo "      --original $BASE_FOLDER/$MODEL_NAME --out ${OUT_DIR}-full"
  ;;
*) echo "usage: $0 {to_dist|to_hf}" >&2; exit 2 ;;
esac
