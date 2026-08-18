#!/usr/bin/env python3
"""Splice trained text weights into the original HF checkpoint.

Why this exists
---------------
`Qwen/Qwen3.5-27B` is a `Qwen3_5ForConditionalGeneration` checkpoint containing
three tensor families:

  model.language_model.*   64-layer text stack (48 gated-delta-net + 16 full attn)
  model.visual.*           27-block ViT (patch_embed / blocks / merger)
  mtp.*                    1-layer multi-token-prediction head

slime's `slime_plugins.models.qwen3_5` spec is the text path, so a round trip
through Megatron returns only the text stack. Loading that alone as a
ConditionalGeneration model fails on missing keys. This script writes a complete
checkpoint: trained tensors where training touched them, original tensors
(bit-identical) everywhere else.

    python scripts/07_restore_vision.py \
        --trained  /shared/rst/out-hf \
        --original /shared/rst/Qwen3.5-27B \
        --out      /shared/rst/out-hf-full

TWO THINGS THIS IS CAREFUL ABOUT
    Memory. The obvious implementation reads both checkpoints into dicts and then
    writes shards. At 27.8 B that is ~55.6 GB of original plus ~48 GB of trained
    plus the output buffers -- over 110 GB of host RAM, on a machine that may have
    less, and the failure is the OOM killer taking the process with no message. So
    nothing here holds more than one output shard at a time: the plan is built from
    safetensors HEADERS (a few KB per file), and tensors are read only as they are
    written.

    Silence. Falling back to the original tensor is correct for `model.visual.*` and
    `mtp.*`, which training never touches. For anything else it means the trained
    checkpoint is missing a weight the model has, and the base value is being shipped
    in its place -- a checkpoint that loads, runs, evaluates, and is partly untrained.
    That is refused by default; `--allow-original-fallback` lets it through and
    records every affected key in the manifest.

OUTPUT
    <out>/model-*.safetensors + model.safetensors.index.json
    <out>/restore_vision_manifest.json   which tensor came from where, and why
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

COPY_ALWAYS = ("model.visual.", "mtp.")

# safetensors dtype name -> torch dtype. Only used when the two checkpoints disagree
# about a tensor's dtype, which is the case a bf16 Megatron round trip can produce
# against an fp32 original.
TORCH_DTYPE = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8,
    "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64,
}


@dataclass(frozen=True, slots=True)
class Entry:
    """Where a tensor lives and how big it is, read from the file header only."""

    shard: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


def read_header(path: Path) -> dict:
    """The JSON header of a .safetensors file: 8-byte little-endian length, then JSON."""
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise SystemExit(f"{path} is truncated (no safetensors header)")
        (length,) = struct.unpack("<Q", raw)
        return json.loads(handle.read(length))


def shard_files(root: Path) -> list[str]:
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return sorted(set(weight_map.values()))
    if (root / "model.safetensors").exists():
        return ["model.safetensors"]
    raise SystemExit(f"no safetensors index or single file under {root}")


def scan(root: Path) -> dict[str, Entry]:
    """Every tensor in a checkpoint, without reading one byte of tensor data.

    The index's weight_map is not trusted for sizes (it has none) or for
    completeness (a stale index outlives a re-export); the headers of the files
    themselves are the ground truth.
    """
    entries: dict[str, Entry] = {}
    for name in shard_files(root):
        path = root / name
        if not path.is_file():
            raise SystemExit(f"{path} is listed in the index but missing on disk")
        for key, meta in read_header(path).items():
            if key == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            entries[key] = Entry(shard=name, dtype=meta["dtype"],
                                 shape=tuple(meta["shape"]), nbytes=int(end - begin))
    if not entries:
        raise SystemExit(f"{root} contains no tensors")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-size-gb", type=float, default=4.5)
    parser.add_argument("--allow-original-fallback", action="store_true",
                        help="write the checkpoint even though some non-vision tensors "
                             "had to come from the ORIGINAL (missing from the trained "
                             "export, or a different shape). The result is partly "
                             "untrained; every such key is listed in the manifest")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    original = scan(args.original)
    trained = scan(args.trained)
    print(f"[scan] original {len(original):,} tensors "
          f"({sum(e.nbytes for e in original.values()) / 2**30:.1f} GiB) in "
          f"{len({e.shard for e in original.values()})} files")
    print(f"[scan] trained  {len(trained):,} tensors "
          f"({sum(e.nbytes for e in trained.values()) / 2**30:.1f} GiB) in "
          f"{len({e.shard for e in trained.values()})} files")

    # ---- plan: decide every key's source before writing anything -------------
    plan: dict[str, tuple[str, str]] = {}       # key -> (source, reason)
    unexpected_fallback: list[str] = []
    shape_mismatch: list[str] = []
    for key, ref in original.items():
        if any(key.startswith(prefix) for prefix in COPY_ALWAYS):
            plan[key] = ("original", "copy_always")
            continue
        candidate = trained.get(key)
        if candidate is None:
            plan[key] = ("original", "missing_from_trained")
            unexpected_fallback.append(key)
            continue
        if candidate.shape != ref.shape:
            plan[key] = ("original", "shape_mismatch")
            shape_mismatch.append(
                f"{key}: trained={candidate.shape} original={ref.shape}")
            continue
        plan[key] = ("trained", "trained")

    took_trained = sum(1 for source, _ in plan.values() if source == "trained")
    copy_always = sum(1 for _, reason in plan.values() if reason == "copy_always")
    print(f"[plan] from trained: {took_trained:,}   vision/mtp from original: "
          f"{copy_always:,}   unexpected fallbacks: {len(unexpected_fallback):,}   "
          f"shape mismatches: {len(shape_mismatch):,}")

    extra = sorted(set(trained) - set(original))
    if extra:
        print(f"[plan] {len(extra)} tensors in the trained checkpoint are absent from the "
              f"original and are NOT written, e.g. {extra[:5]}")

    if took_trained == 0:
        raise SystemExit(
            f"refusing to write: not one tensor name in {args.trained} matched a "
            f"non-vision tensor in {args.original}. This is a naming mismatch, not a "
            f"training failure -- compare a few keys from each "
            f"model.safetensors.index.json. The output would be a byte-for-byte copy of "
            f"the base model."
        )

    # This is the gate the previous version did not have. Both branches used to fold
    # into one `took_original` counter next to the legitimate vision copies, so an
    # export that dropped half the text stack printed a plausible-looking line and
    # produced a checkpoint that loads and evaluates as if it were trained.
    if (unexpected_fallback or shape_mismatch) and not args.allow_original_fallback:
        detail = []
        if unexpected_fallback:
            detail.append(f"  {len(unexpected_fallback)} tensors are missing from the "
                          f"trained checkpoint, e.g.\n    "
                          + "\n    ".join(unexpected_fallback[:8]))
        if shape_mismatch:
            detail.append(f"  {len(shape_mismatch)} tensors have a different shape, e.g.\n    "
                          + "\n    ".join(shape_mismatch[:8]))
        raise SystemExit(
            "REFUSING TO WRITE: tensors outside model.visual.* / mtp.* would have to come "
            "from the ORIGINAL checkpoint.\n" + "\n".join(detail) + "\n"
            "  Those are text-stack weights training was supposed to produce. Taking the "
            "base value instead yields a checkpoint that loads and runs while being partly "
            "untrained, and no downstream eval can tell.\n"
            "  Likely causes: the conversion (04_convert_ckpt.sh / verl.model_merger) did "
            "not finish; --trained points at a checkpoint step that was still being "
            "written; tied weights (lm_head) named differently by the two exporters.\n"
            "  Fix the export, or pass --allow-original-fallback to accept it deliberately "
            "-- the manifest then lists every affected key and the report has to say so."
        )

    # ---- stream the output: one shard resident at a time --------------------
    limit = int(args.shard_size_gb * (1 << 30))
    groups: list[list[str]] = [[]]
    sizes = [0]
    for key in sorted(plan):
        source, _ = plan[key]
        nbytes = (trained[key] if source == "trained" else original[key]).nbytes
        if sizes[-1] + nbytes > limit and groups[-1]:
            groups.append([])
            sizes.append(0)
        groups[-1].append(key)
        sizes[-1] += nbytes

    roots = {"trained": args.trained, "original": args.original}
    indices = {"trained": trained, "original": original}
    total = len(groups)
    weight_map: dict[str, str] = {}
    written_bytes = 0
    for position, keys in enumerate(groups, 1):
        name = f"model-{position:05d}-of-{total:05d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        with ExitStack() as stack:
            handles = {}
            for key in keys:
                source, _ = plan[key]
                entry = indices[source][key]
                slot = (source, entry.shard)
                if slot not in handles:
                    handles[slot] = stack.enter_context(
                        safe_open(roots[source] / entry.shard, framework="pt"))
                tensor = handles[slot].get_tensor(key)
                want = TORCH_DTYPE.get(original[key].dtype)
                if source == "trained" and want is not None and tensor.dtype != want:
                    tensor = tensor.to(want)
                tensors[key] = tensor
        save_file(tensors, str(args.out / name), metadata={"format": "pt"})
        for key in keys:
            weight_map[key] = name
        written_bytes += sizes[position - 1]
        print(f"  wrote {name}  tensors={len(keys)}  "
              f"{sizes[position - 1] / 2**30:.2f} GiB", flush=True)
        tensors.clear()

    (args.out / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": sum(sizes)}, "weight_map": weight_map},
            indent=2,
        )
        + "\n"
    )

    # carry over every non-weight file (config, tokenizer, chat template, ...)
    copied = []
    for item in sorted(args.original.iterdir()):
        if item.is_file() and not item.name.endswith(".safetensors") \
           and item.name != "model.safetensors.index.json":
            shutil.copy2(item, args.out / item.name)
            copied.append(item.name)
    print(f"  copied {len(copied)} non-weight files: {', '.join(copied[:8])}"
          f"{' ...' if len(copied) > 8 else ''}")

    manifest = {
        "trained": str(args.trained),
        "original": str(args.original),
        "out": str(args.out),
        "tensors_total": len(plan),
        "tensors_from_trained": took_trained,
        "tensors_copy_always": copy_always,
        "tensors_unexpected_fallback": len(unexpected_fallback),
        "tensors_shape_mismatch": len(shape_mismatch),
        "unexpected_fallback_keys": unexpected_fallback,
        "shape_mismatch_keys": shape_mismatch,
        "trained_only_keys_dropped": extra,
        "allow_original_fallback": bool(args.allow_original_fallback),
        "shards": total,
        "bytes": written_bytes,
        "note": "tensors_unexpected_fallback and tensors_shape_mismatch MUST be 0 for this "
                "checkpoint to be fully trained. Non-zero means those weights are the base "
                "model's, and the run's evaluation numbers describe a partly-untrained "
                "model. copy_always (model.visual.*, mtp.*) is expected and is not that.",
    }
    (args.out / "restore_vision_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nDONE -> {args.out}  ({written_bytes / 2**30:.1f} GiB in {total} shards)")
    if unexpected_fallback or shape_mismatch:
        print(f"  WARNING: {len(unexpected_fallback) + len(shape_mismatch)} tensors came "
              f"from the ORIGINAL checkpoint because the trained one lacked them. This "
              f"checkpoint is partly untrained; see restore_vision_manifest.json.")
    print(f"Manifest: {args.out / 'restore_vision_manifest.json'}\nSmoke test:\n"
          f"  python -m sglang.launch_server --model-path {args.out} "
          f"--tp 4 --trust-remote-code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
