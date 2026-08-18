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
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

COPY_ALWAYS = ("model.visual.", "mtp.")


def load_index(root: Path) -> dict[str, str]:
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())["weight_map"]
    single = root / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt") as handle:
            return {key: "model.safetensors" for key in handle.keys()}
    raise SystemExit(f"no safetensors index or single file under {root}")


def read_all(root: Path, weight_map: dict[str, str]) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)
    tensors: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(root / shard, framework="pt") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-size-gb", type=float, default=4.5)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    original = read_all(args.original, load_index(args.original))
    trained = read_all(args.trained, load_index(args.trained))

    merged: dict[str, torch.Tensor] = {}
    took_trained = took_original = mismatched = 0
    for key, ref in original.items():
        if any(key.startswith(prefix) for prefix in COPY_ALWAYS):
            merged[key] = ref
            took_original += 1
            continue
        candidate = trained.get(key)
        if candidate is None:
            merged[key] = ref
            took_original += 1
            continue
        if tuple(candidate.shape) != tuple(ref.shape):
            print(f"  SHAPE MISMATCH {key}: trained={tuple(candidate.shape)} "
                  f"original={tuple(ref.shape)} -> keeping original")
            merged[key] = ref
            mismatched += 1
            continue
        merged[key] = candidate.to(ref.dtype)
        took_trained += 1

    extra = sorted(set(trained) - set(original))
    if extra:
        print(f"  {len(extra)} tensors in trained ckpt absent from original "
              f"(ignored), e.g. {extra[:5]}")

    print(f"from trained: {took_trained}   from original: {took_original}   "
          f"shape-mismatched: {mismatched}   total: {len(merged)}")
    if took_trained == 0:
        raise SystemExit("refusing to write: no trained tensors matched")

    # shard out
    limit = int(args.shard_size_gb * (1 << 30))
    shards: list[dict[str, torch.Tensor]] = [{}]
    sizes = [0]
    for key in sorted(merged):
        tensor = merged[key]
        nbytes = tensor.numel() * tensor.element_size()
        if sizes[-1] + nbytes > limit and shards[-1]:
            shards.append({})
            sizes.append(0)
        shards[-1][key] = tensor
        sizes[-1] += nbytes

    total = len(shards)
    weight_map: dict[str, str] = {}
    for index, shard in enumerate(shards, 1):
        name = f"model-{index:05d}-of-{total:05d}.safetensors"
        save_file(shard, str(args.out / name), metadata={"format": "pt"})
        for key in shard:
            weight_map[key] = name
        print(f"  wrote {name}  tensors={len(shard)}  {sizes[index-1]/2**30:.2f} GiB")

    (args.out / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": sum(sizes)}, "weight_map": weight_map},
            indent=2,
        )
        + "\n"
    )

    # carry over every non-weight file (config, tokenizer, chat template, ...)
    for item in sorted(args.original.iterdir()):
        if item.is_file() and not item.name.endswith(".safetensors") \
           and item.name != "model.safetensors.index.json":
            shutil.copy2(item, args.out / item.name)
            print(f"  copied {item.name}")

    print(f"\nDONE -> {args.out}\nSmoke test:\n"
          f"  python -m sglang.launch_server --model-path {args.out} "
          f"--tp 4 --trust-remote-code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
