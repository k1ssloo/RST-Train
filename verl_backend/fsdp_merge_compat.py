"""Merge a verl FSDP checkpoint whose DeviceMesh was pickled by a DIFFERENT torch.

    python verl_backend/fsdp_merge_compat.py \
        --local_dir  .../global_step_82 \
        --target_dir .../out-hf

THE FAILURE
-----------
`verl.model_merger merge --backend fsdp` reads the shard layout off the first DTensor it
finds:

    verl/model_merger/fsdp_model_merger.py:107
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh              # <-- here

`model_world_size_N_rank_0.pt` is a plain `torch.save` of the sharded state dict, so it
contains **pickled DeviceMesh objects**. Unpickling one under a torch whose internals have
moved gives an object that is structurally incomplete, and every property that walks its
layout raises. Measured on torch 2.13.0+cu130 reading a checkpoint written by the
cluster's torch:

    AttributeError: '_MeshLayout' object has no attribute 'axes'
      device_mesh.mesh   RAISED
      device_mesh.shape  RAISED
      device_mesh.ndim   RAISED
      device_mesh.size   RAISED

verl catches nothing, so the merge dies and the checkpoint stays unloadable. Nothing about
the weights is wrong -- `_local_tensor` and `placements` unpickle fine, and only the mesh's
internal layout is unreadable.

WHY NOT JUST DOWNGRADE TORCH
----------------------------
Because the merge does not need the mesh. `_calculate_shard_configuration` uses exactly
`mesh.shape[-1]` (the FSDP shard count) and `mesh_dim_names`, and both are recoverable:

  * `mesh_dim_names` is a plain tuple attribute set in `__init__`; it survives.
  * the shard count is in the tensor itself -- `ceil(global_shape[d] / local_shape[d])`
    along the sharded dim. That is measured from the data rather than assumed from
    `world_size`, so it stays right for HSDP, where world_size is ddp x fsdp and only
    the fsdp factor is the number of distinct shards.

So this module replaces that one method and calls verl's own merge and save. Everything
that decides what the weights ARE stays verl's code.

WHEN TO USE IT
--------------
`scripts/08_prepare_eval_ckpt.sh` tries `verl.model_merger` first and falls back here, so
on a matching torch nothing changes. Run it directly when merging by hand.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

LOG_PREFIX = "[rst-merge-compat]"


def fsdp_shard_count(weight) -> int:
    """How many distinct FSDP shards this DTensor is split into, from the tensor itself.

    Derived rather than taken from world_size: under HSDP the world is ddp x fsdp and only
    the fsdp factor is the number of files that hold distinct data.
    """
    shard_dims = [placement.dim for placement in weight.placements if placement.is_shard()]
    if not shard_dims:
        raise ValueError(
            f"the pivot tensor has placements {weight.placements} and no Shard among them, "
            f"so there is nothing to count. Pass --shards to say it explicitly."
        )
    if len(shard_dims) > 1:
        raise ValueError(
            f"the pivot tensor is sharded on {len(shard_dims)} dims ({shard_dims}); that is "
            f"FSDP+TP, which verl's fsdp merger raises NotImplementedError on anyway."
        )
    dim = shard_dims[0]
    full = int(weight.shape[dim])
    local = int(weight._local_tensor.shape[dim])
    if local <= 0:
        raise ValueError(f"pivot tensor has a zero-length local shard on dim {dim}")
    # ceil, because torch's Shard splits unevenly rather than padding: the last rank holds
    # the remainder, so full/local is not an integer whenever the dim is not divisible.
    return math.ceil(full / local)


def reconstruct_mesh(names: tuple[str, ...], shards: int, world_size: int):
    """A rank-id array whose SHAPE is the device mesh's shape.

    Split out from `patched_extract` because the contract is subtle enough to get wrong:
    verl reads `mesh.shape[-1]` and never the values, so `np.array([shards])` means "a 1-D
    mesh of ONE rank" rather than "a mesh of `shards` ranks". verl's own non-DTensor
    fallback (`np.array([world_size])`, fsdp_model_merger.py:112) has exactly that bug --
    it merges rank 0 alone and writes a checkpoint 1/world_size of the right size, with no
    error. Nothing downstream notices except a shape comparison against the base.
    """
    import numpy as np

    if shards < 1:
        raise ValueError(f"shard count must be >= 1, got {shards}")
    if names == ("fsdp",):
        mesh = np.arange(shards, dtype=np.int64)
    elif names == ("ddp", "fsdp"):
        if world_size % shards:
            raise ValueError(
                f"mesh_dim_names says {names} but {shards} fsdp shards do not divide "
                f"world_size={world_size}, so the ddp factor is not an integer. Merge on "
                f"the torch that wrote the checkpoint."
            )
        mesh = np.arange(world_size, dtype=np.int64).reshape(world_size // shards, shards)
    else:
        raise ValueError(
            f"unsupported mesh_dim_names {names}; verl's fsdp merger only accepts "
            f"('fsdp',) and ('ddp','fsdp')."
        )
    # Asserted next to the line that establishes it, because getting it wrong does not
    # fail -- it silently merges the wrong number of shards.
    if mesh.shape[-1] != shards:
        raise AssertionError(
            f"reconstructed mesh has shape {mesh.shape}, so verl would merge "
            f"{mesh.shape[-1]} shard(s) instead of {shards}")
    return mesh


def patched_extract(self, state_dict: dict, world_size: int):
    """Replacement for `FSDPModelMerger._extract_device_mesh_info`.

    Same contract: return `(mesh, mesh_dim_names)` where only `mesh.shape` is ever read.
    Tries verl's own path first, so a torch that CAN read the pickled mesh is unaffected.
    """
    import numpy as np
    from torch.distributed.tensor import DTensor

    pivot_key = sorted(state_dict.keys())[0]
    weight = state_dict[pivot_key]

    if not isinstance(weight, DTensor):
        return np.array([world_size], dtype=np.int64), ("fsdp",)

    device_mesh = weight.device_mesh
    names = tuple(device_mesh.mesh_dim_names or ("fsdp",))
    try:
        mesh = device_mesh.mesh
    except Exception as exc:  # noqa: BLE001 -- any layout attribute may be the missing one
        shards = fsdp_shard_count(weight)
        try:
            mesh = reconstruct_mesh(names, shards, world_size)
        except ValueError as why:
            raise ValueError(f"{why} (measured from {pivot_key})") from None
        print(f"{LOG_PREFIX} device_mesh.mesh is unreadable under torch "
              f"{__import__('torch').__version__} ({type(exc).__name__}: {exc}); it was "
              f"pickled by a different torch. Reconstructed a {names} mesh of shape "
              f"{mesh.shape} from {pivot_key}: global {tuple(weight.shape)} / local "
              f"{tuple(weight._local_tensor.shape)} -> {shards} fsdp shard(s) to merge.",
              flush=True)
        return mesh, names

    return mesh, names


def apply() -> bool:
    """Patch verl's FSDP merger. Returns True if the patch was installed."""
    try:
        from verl.model_merger.fsdp_model_merger import FSDPModelMerger
    except ImportError as exc:
        print(f"{LOG_PREFIX} verl's fsdp merger is not importable ({exc})", file=sys.stderr)
        return False
    if not hasattr(FSDPModelMerger, "_extract_device_mesh_info"):
        print(f"{LOG_PREFIX} this verl's FSDPModelMerger has no _extract_device_mesh_info; "
              f"the method was renamed and this patch cannot help. Read "
              f"verl/model_merger/fsdp_model_merger.py before going further.",
              file=sys.stderr)
        return False
    FSDPModelMerger._extract_device_mesh_info = patched_extract
    FSDPModelMerger._rst_mesh_compat = True
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local_dir", required=True, help="dir holding model_world_size_*_rank_*.pt")
    ap.add_argument("--target_dir", required=True, help="where to write the HF checkpoint")
    ap.add_argument("--hf_model_config_path", default="",
                    help="default: <local_dir>/huggingface")
    args = ap.parse_args()

    if not apply():
        return 2

    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    local_dir = Path(args.local_dir)
    config_path = args.hf_model_config_path or str(local_dir / "huggingface")
    if not Path(config_path).is_dir():
        sys.exit(f"no HF config dir at {config_path}. verl writes it as "
                 f"<global_step_N>/huggingface/; pass --hf_model_config_path if it is "
                 f"somewhere else.")

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(local_dir),
        target_dir=args.target_dir,
        hf_model_config_path=config_path,
    )
    Path(args.target_dir).mkdir(parents=True, exist_ok=True)
    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    if hasattr(merger, "cleanup"):
        merger.cleanup()
    print(f"{LOG_PREFIX} merged {args.local_dir} -> {args.target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
