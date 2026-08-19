"""`verl_backend/fsdp_merge_compat.py` -- merging a checkpoint another torch pickled.

WHY IT EXISTS
    `model_world_size_N_rank_0.pt` is a plain `torch.save` of a sharded state dict, so it
    contains pickled `DeviceMesh` objects. Unpickling one under a torch whose internals have
    moved gives a structurally incomplete object, and every property that walks its layout
    raises. Measured on torch 2.13.0+cu130 reading the cluster's 4B checkpoint:

        AttributeError: '_MeshLayout' object has no attribute 'axes'
        device_mesh.mesh / .shape / .ndim / .size  all RAISED

    verl's merger reads the shard layout off exactly that property
    (fsdp_model_merger.py:107), so the merge dies and the checkpoint stays unloadable, while
    the weights themselves -- `_local_tensor` and `placements` -- unpickled perfectly.

THE BUG THIS FILE MOSTLY EXISTS TO PIN
    `mesh` is a rank-id array whose SHAPE is the mesh shape; verl uses `mesh.shape[-1]` and
    never the values. The first version of the workaround returned `np.array([shards])`,
    whose shape is `(1,)`. verl then merged ONE shard of eight and wrote an 8.5-GiB-model's
    worth of weights into 1.1 GiB -- with no error, no warning, and a checkpoint that loads.
    It was caught by the size on disk, not by anything in the code.

    So `reconstruct_mesh` is a separate pure function, and these tests assert the shape
    rather than the values. verl's own non-DTensor fallback has the identical shape bug
    (`np.array([world_size])`, fsdp_model_merger.py:112), which is worth knowing before
    trusting a merge that took that path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, need  # noqa: E402

sys.path.insert(0, str(ROOT / "verl_backend"))
np = need("numpy")
import fsdp_merge_compat as compat  # noqa: E402


class _FakeShard:
    """Enough of a DTensor for `fsdp_shard_count`, which does not isinstance-check."""

    def __init__(self, full: tuple[int, ...], local: tuple[int, ...], placements):
        self.shape = full
        self._local_tensor = type("T", (), {"shape": local})()
        self.placements = placements


class _Shard:
    def __init__(self, dim: int):
        self.dim = dim

    def is_shard(self) -> bool:
        return True


class _Replicate:
    dim = None

    def is_shard(self) -> bool:
        return False


# ---------------------------------------------------------------- reconstruct_mesh


def test_the_mesh_shape_carries_the_shard_count_not_its_value():
    """The regression. `np.array([8])` has shape (1,) and would merge one shard."""
    for shards in (1, 2, 4, 8, 16):
        mesh = compat.reconstruct_mesh(("fsdp",), shards, shards)
        assert mesh.shape[-1] == shards, (
            f"mesh for {shards} shards has shape {mesh.shape}; verl reads shape[-1], so it "
            f"would merge {mesh.shape[-1]} shard(s)"
        )
        assert mesh.ndim == 1


def test_the_hsdp_mesh_is_two_dimensional_with_fsdp_last():
    # world 32 = ddp 4 x fsdp 8. verl reads shape[-1] as the number of distinct shards.
    mesh = compat.reconstruct_mesh(("ddp", "fsdp"), 8, 32)
    assert mesh.shape == (4, 8), mesh.shape
    assert mesh.shape[-1] == 8


def test_an_hsdp_split_that_is_not_integral_is_refused():
    try:
        compat.reconstruct_mesh(("ddp", "fsdp"), 5, 32)
    except ValueError as exc:
        assert "do not divide" in str(exc)
        return
    raise AssertionError("a non-integral ddp factor was accepted")


def test_an_unsupported_mesh_layout_is_refused_rather_than_guessed():
    # verl's _calculate_shard_configuration asserts on exactly these two, so inventing a
    # third here would only move the failure later.
    try:
        compat.reconstruct_mesh(("dp", "tp", "fsdp"), 8, 32)
    except ValueError as exc:
        assert "unsupported mesh_dim_names" in str(exc)
        return
    raise AssertionError("an unknown mesh layout was accepted")


def test_a_zero_shard_count_is_refused():
    try:
        compat.reconstruct_mesh(("fsdp",), 0, 8)
    except ValueError as exc:
        assert ">= 1" in str(exc)
        return
    raise AssertionError("a zero shard count was accepted")


# ---------------------------------------------------------------- fsdp_shard_count


def test_the_shard_count_is_measured_from_the_tensor_not_assumed_from_world_size():
    # The real 4B pivot: lm_head.weight, global (248320, 2560), local (31040, 2560).
    # Measured rather than assumed so it stays right under HSDP, where world_size is
    # ddp x fsdp and only the fsdp factor is the number of files holding distinct data.
    weight = _FakeShard((248320, 2560), (31040, 2560), (_Shard(0),))
    assert compat.fsdp_shard_count(weight) == 8


def test_an_uneven_split_rounds_up():
    # torch's Shard splits unevenly rather than padding: the last rank holds the remainder,
    # so full/local is not an integer and floor would undercount by one whole shard.
    weight = _FakeShard((100, 8), (13, 8), (_Shard(0),))
    assert compat.fsdp_shard_count(weight) == 8  # ceil(100/13) == 8


def test_sharding_on_a_dim_other_than_zero_is_handled():
    weight = _FakeShard((8, 4096), (8, 512), (_Shard(1),))
    assert compat.fsdp_shard_count(weight) == 8


def test_a_replicated_tensor_cannot_give_a_shard_count():
    weight = _FakeShard((8, 8), (8, 8), (_Replicate(),))
    try:
        compat.fsdp_shard_count(weight)
    except ValueError as exc:
        assert "no Shard among them" in str(exc)
        return
    raise AssertionError("a replicated tensor was used to infer a shard count")


def test_two_dimensional_sharding_is_refused_as_fsdp_plus_tp():
    weight = _FakeShard((64, 64), (8, 8), (_Shard(0), _Shard(1)))
    try:
        compat.fsdp_shard_count(weight)
    except ValueError as exc:
        assert "FSDP+TP" in str(exc)
        return
    raise AssertionError("FSDP+TP was accepted, which verl's fsdp merger cannot merge")


# ---------------------------------------------------------------- wiring


def test_the_launcher_falls_back_to_this_module():
    launcher = (ROOT / "scripts" / "08_prepare_eval_ckpt.sh").read_text(encoding="utf-8")
    invocation = "python verl_backend/fsdp_merge_compat.py"
    assert invocation in launcher, (
        "08_prepare_eval_ckpt.sh no longer falls back here, so a checkpoint pickled by "
        "another torch fails the merge with nothing to try next"
    )
    # verl's own CLIs must still be tried FIRST: on a matching torch they are the tested
    # path, and this workaround replaces one of their methods. Compare the INVOCATIONS, not
    # any mention -- the module is named in a comment above the merge block too.
    assert launcher.index("python -m verl.model_merger merge") < launcher.index(invocation), (
        "the workaround is attempted before verl's own merger"
    )


def test_apply_is_a_no_op_without_verl_rather_than_a_crash():
    # The pure helpers above are used by these tests on machines with no verl at all.
    assert compat.apply() in (True, False)


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
