"""The flags in `30_run_sft_verl.sh` that decide whether a 32 K sequence fits at all.

This file exists because of a real 27B run that reported ~78 GB/GPU of activations
next to ~2 GB of sharded parameters and looked like a parallelism bug. It was not:
`model.use_liger=True` was set, but verl's FSDP engine applies Liger with

    _apply_liger_kernel_to_instance(model=module, fused_linear_cross_entropy=False,
                                    swiglu=True)

hardcoded ("conflicts with verl's forward patching"), so the cross-entropy was never
fused and the full [tokens, 248320] logit tensor was materialized. The switch that
does fuse it is verl's own `model.use_fused_kernels` plus
`model.fused_kernel_options.impl_backend`.

These assertions are textual on purpose: the failure mode is someone tidying a flag
out of the launcher, and no unit test can run a 32-GPU job. Nothing here needs torch,
verl, a GPU or a cluster.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT  # noqa: E402

LAUNCHER = ROOT / "scripts" / "30_run_sft_verl.sh"
TEXT = LAUNCHER.read_text(encoding="utf-8")


def torchrun_block() -> str:
    """The `torchrun ... \\` invocation only — not the comments that surround it.

    A flag mentioned in a comment is documentation; a flag inside this block is what
    the job actually receives. The two must not be confused, which is the whole
    lesson of the run that motivated this file.
    """
    lines = TEXT.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("torchrun"))
    block: list[str] = []
    for line in lines[start:]:
        block.append(line)
        if not line.rstrip().endswith("\\"):
            break
    return "\n".join(block)


def verl_args_block() -> str:
    """The `VERL_ARGS=( ... )` array — the single list the job actually receives.

    The launcher builds the overrides once, validates that exact array against the
    installed verl, and then hands the same array to torchrun. So "does the job get
    this flag" is a question about this array plus the one-line check below that
    torchrun is still expanded from it.
    """
    lines = TEXT.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("VERL_ARGS=("))
    block: list[str] = []
    for line in lines[start:]:
        block.append(line)
        if line.strip() == ")":
            break
    return "\n".join(block)


def test_the_validated_array_is_the_one_that_is_launched():
    # Two lists is how a launcher ends up validating flags it does not pass.
    assert '"${VERL_ARGS[@]}"' in torchrun_block(), (
        "torchrun no longer expands VERL_ARGS, so the pre-launch gates are checking "
        "something other than what the job receives"
    )


def test_the_fused_cross_entropy_reaches_the_job_not_just_a_comment():
    assert "FUSED_ARGS" in verl_args_block(), (
        "the fused-kernel overrides are not in VERL_ARGS; without them verl "
        "materializes the full 248,320-wide logits and a 32K sequence does not fit"
    )
    assert "model.use_fused_kernels=True" in TEXT
    assert "model.fused_kernel_options.impl_backend=" in TEXT


def test_liger_is_kept_but_is_not_relied_on_for_the_cross_entropy():
    # use_liger is still worth having (swiglu + rms_norm), so it must stay -- but the
    # launcher must not present it as the thing that fuses the CE.
    assert "model.use_liger=True" in verl_args_block()
    assert "fused_linear_cross_entropy=False" in TEXT, (
        "the launcher no longer records WHY use_liger is not enough here; the next "
        "person to read it will delete the use_fused_kernels flags as redundant"
    )


def test_the_backend_defaults_to_torch_and_rejects_anything_unknown():
    match = re.search(r'FUSED_KERNEL_BACKEND="\$\{FUSED_KERNEL_BACKEND:-(\w+)\}"', TEXT)
    assert match, "FUSED_KERNEL_BACKEND has no default"
    assert match.group(1) == "torch", (
        f"default backend is {match.group(1)!r}; triton's linear_cross_entropy has not "
        f"been exercised on SM80 in this project, so it must not be the default"
    )
    # verl raises on an unknown backend deep inside the workers; catch it in the shell.
    assert re.search(r"torch\|triton\)", TEXT), "unknown backends are not rejected early"


def test_there_is_a_pre_launch_gate_on_the_config_keys_existing():
    # An older verl without these keys makes hydra abort *after* 32 processes have
    # started and the model has been read from disk. The gate must run before torchrun.
    gate_at = TEXT.index("hf_model.yaml")
    torchrun_at = TEXT.index("\ntorchrun")
    assert gate_at < torchrun_at, "the verl capability gate runs after the launch"
    assert "FUSED_KERNELS=0" in TEXT, "no documented escape hatch for an older verl"


def test_the_report_note_tells_the_operator_which_log_line_proves_it():
    # "It ran" is not evidence the CE was fused; one printed line is.
    assert "Using Torch backend for fused kernels" in TEXT


def test_the_static_footprint_is_gated_before_the_launch():
    """The OOM that no data-length change can fix.

    16 B/param (fp32 master + fp32 grad + Adam m + v) divided by the shard degree is a
    constant per GPU. On 8 ranks that is exactly 2 B/param -- numerically identical to
    the whole bf16 model, which is how a 27B run on one node presents as "the model is
    not being sharded". The launcher must compute this and refuse, before torchrun.
    """
    gate_at = TEXT.index("static footprint gate")
    torchrun_at = TEXT.index("\ntorchrun")
    assert gate_at < torchrun_at, "the static-footprint gate runs after the launch"
    assert "16" in TEXT[gate_at:gate_at + 1500], "the gate does not state the bytes/param"
    assert "OFFLOAD_OPTIM" in TEXT and "engine.offload_policy=true" in TEXT, (
        "the gate refuses without offering the one override that actually shrinks the "
        "static term"
    )
    assert "FORCE_STATIC" in TEXT, "no documented escape hatch for a deliberate overrun"


def test_a_multinode_launch_requires_an_explicit_rendezvous():
    """NNODES=4 with MASTER_ADDR=127.0.0.1 is four separate world_size=8 jobs.

    Each shards 8 ways instead of 32, so the static footprint quadruples and the run
    OOMs in backward with a peak that does not respond to the token budget. Nothing in
    the logs says "you are not multi-node", so the launcher has to.
    """
    gate_at = TEXT.index("multi-node rendezvous gate")
    torchrun_at = TEXT.index("\ntorchrun")
    assert gate_at < torchrun_at
    window = TEXT[gate_at:torchrun_at]
    assert "127.0.0.1" in window and "NODE_RANK" in window, (
        "the gate does not reject the single-node defaults when NNODES > 1"
    )


def test_the_shard_degree_is_passed_explicitly_rather_than_inherited():
    # engine.fsdp_size defaults to -1 in verl, but a stale config or a copied command
    # line can set it to 8 and silently cap the shard degree at one node.
    assert "engine.fsdp_size=" in verl_args_block()


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
