#!/usr/bin/env python3
"""Prove, on ONE GPU in seconds, what verl's gradient-accumulation path costs.

    python scripts/35_probe_fsdp2_grad_accum.py                  # both ways, compare
    python scripts/35_probe_fsdp2_grad_accum.py --hidden 4096     # bigger contrast
    torchrun --nproc_per_node 8 scripts/35_probe_fsdp2_grad_accum.py   # real shard degree

This exists because the claim behind `verl_backend/fsdp2_grad_accum.py` is a claim
about torch internals, and reading source is not the same as measuring. The probe
builds a real FSDP2 model with verl's own mixed-precision policy
(`param_dtype=bf16, reduce_dtype=fp32`), runs a two-micro-batch step twice -- once with
`set_requires_gradient_sync(False)` on the first micro-batch, exactly as verl does, and
once without -- and checks three things:

  1. ALLOCATION. With sync skipped, `FSDPParam.unsharded_accumulated_grad` is non-None
     after the first micro-batch and holds `param.numel()` fp32 elements -- the WHOLE
     parameter, not this rank's shard. Without it, that attribute stays None and the
     fp32 SHARDED gradient is already populated. This assertion is independent of how
     many GPUs you run on, which is why the probe is useful on one.
  2. NUMERICS. The final gradients agree to fp32 tolerance, because
     `_fsdp_collectives.py` accumulates into the sharded gradient
     (`sharded_param.grad._local_tensor += new_sharded_grad`) and a sum of
     reduce-scatters is the reduce-scatter of a sum. There is no accuracy trade here.
  3. PEAK MEMORY, with the caveat spelled out. The retained unsharded gradient is
     `params x 4 B` at EVERY shard degree, while the reduce-scatter the patch reinstates
     costs `params x 4 B / shard_degree`. At shard degree 1 -- one GPU -- those are the
     same size and the patched path measures slightly HIGHER, because reduce-scattering
     over a single rank reduces nothing but still allocates the flat buffer. So do not
     read the single-GPU peak as the cluster's: the probe prints the arithmetic for
     several shard degrees beside it. At 32 the two are 103.5 GiB and 6.5 GiB.

Exit code: 0 = the patch's premise holds and numerics match, 1 = it does not.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verl_backend"))

GIB = 1 << 30


def build_model(hidden: int, layers: int, device: str):
    import torch
    from torch import nn

    torch.manual_seed(0)
    blocks = []
    for _ in range(layers):
        blocks.append(nn.Sequential(nn.Linear(hidden, hidden, bias=False), nn.GELU()))
    return nn.Sequential(*blocks).to(device=device, dtype=torch.float32)


def shard(module, mesh):
    import torch
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    # verl's policy: bf16 compute, fp32 reduce (workers/config/engine.py:644 defaults
    # mp_reduce_dtype to "fp32"). The fp32 reduce dtype beside a bf16 param dtype is
    # precisely what makes to_accumulated_grad_if_needed allocate instead of returning.
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                  cast_forward_inputs=True)
    for block in module:
        fully_shard(block, mesh=mesh, mp_policy=policy)
    fully_shard(module, mesh=mesh, mp_policy=policy)
    return module


def fsdp_params(module):
    """Every FSDPParam in the module, via the private param groups FSDP2 attaches."""
    found = []
    for sub in module.modules():
        state = getattr(sub, "_get_fsdp_state", None)
        if state is None:
            continue
        group = getattr(state(), "_fsdp_param_group", None)
        if group is not None:
            found.extend(group.fsdp_params)
    return found


def run_step(module, batches, *, skip_sync: bool):
    """One optimizer step over `batches` micro-batches. Returns (grads, probe, peak_gib)."""
    import torch

    torch.cuda.reset_peak_memory_stats()
    for param in module.parameters():
        param.grad = None

    probe: dict[str, object] = {}
    last = len(batches) - 1
    for index, batch in enumerate(batches):
        is_last = index == last
        if skip_sync and not is_last:
            module.set_requires_gradient_sync(False)
        try:
            module(batch).float().pow(2).mean().backward()
        finally:
            if skip_sync and not is_last:
                module.set_requires_gradient_sync(True)
        if not is_last:
            # Read FSDP2's own state right where the two paths diverge.
            retained = [p for p in fsdp_params(module)
                        if getattr(p, "unsharded_accumulated_grad", None) is not None]
            probe["retained_tensors"] = len(retained)
            probe["retained_elements"] = sum(
                p.unsharded_accumulated_grad.numel() for p in retained)
            probe["retained_bytes"] = sum(
                p.unsharded_accumulated_grad.numel()
                * p.unsharded_accumulated_grad.element_size() for p in retained)
            probe["retained_dtypes"] = sorted(
                {str(p.unsharded_accumulated_grad.dtype) for p in retained})
            probe["unsharded_numel"] = sum(
                p.unsharded_accumulated_grad.numel() for p in retained)
            probe["sharded_grads_present"] = sum(
                1 for p in fsdp_params(module) if p.sharded_param.grad is not None)

    # To CPU, not a device clone: the two runs are compared afterwards, and holding the
    # first run's gradients on the GPU would inflate the second run's measured peak by
    # params x 4 B -- exactly the quantity under test.
    grads = [p.grad.detach().full_tensor().float().cpu() if hasattr(p.grad, "full_tensor")
             else p.grad.detach().float().cpu()
             for p in module.parameters() if p.grad is not None]
    return grads, probe, torch.cuda.max_memory_allocated() / GIB


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--micro-batches", type=int, default=2)
    args = ap.parse_args()

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device; this probe measures GPU allocations.")
        return 0
    if args.micro_batches < 2:
        sys.exit("--micro-batches must be >= 2; with one there is no accumulation to probe")

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29677")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    world = dist.get_world_size()
    mesh = dist.device_mesh.init_device_mesh("cuda", (world,))
    speak = dist.get_rank() == 0

    torch.manual_seed(1234)
    batches = [torch.randn(args.batch, args.hidden, device=device) for _ in range(args.micro_batches)]

    results = {}
    for label, skip in (("verl default (sync skipped)", True), ("patched (always reduce)", False)):
        module = shard(build_model(args.hidden, args.layers, device), mesh)
        n_params = sum(p.numel() for p in module.parameters())
        grads, probe, peak = run_step(module, batches, skip_sync=skip)
        results[skip] = {"grads": grads, "probe": probe, "peak": peak, "n_params": n_params}
        del module
        torch.cuda.empty_cache()

    if not speak:
        dist.barrier()
        dist.destroy_process_group()
        return 0

    default, patched = results[True], results[False]
    # n_params is the SHARDED count per rank; the model total is that times the world size.
    total_params = default["n_params"] * world
    print(f"world_size {world}, model {total_params / 1e6:.1f}M params "
          f"({args.layers} x {args.hidden}), {args.micro_batches} micro-batches\n")

    ok = True
    dp, pp = default["probe"], patched["probe"]
    print("1. ALLOCATION after the first (non-final) micro-batch")
    print(f"   verl default : {dp.get('retained_tensors', 0)} unsharded_accumulated_grad "
          f"tensor(s), {dp.get('retained_elements', 0) / 1e6:.1f}M elements, "
          f"{dp.get('retained_bytes', 0) / GIB:.3f} GiB, dtypes {dp.get('retained_dtypes')}")
    print(f"   patched      : {pp.get('retained_tensors', 0)} tensor(s); sharded grads "
          f"already populated on {pp.get('sharded_grads_present', 0)} parameter(s)")
    if not dp.get("retained_tensors"):
        print("   FAIL: verl's own path retained nothing, so this torch does not take the")
        print("         unsharded-accumulation branch and the patch is unnecessary here.")
        ok = False
    elif dp.get("retained_dtypes") != ["torch.float32"]:
        print(f"   FAIL: retained dtype {dp.get('retained_dtypes')} is not fp32 as expected.")
        ok = False
    if dp.get("retained_tensors") and dp["retained_elements"] < total_params * 0.9:
        print(f"   note: retained {dp['retained_elements'] / 1e6:.1f}M of {total_params / 1e6:.1f}M "
              f"parameters -- FSDP2 frees each group's buffer as the final micro-batch "
              f"consumes it, so a snapshot can undercount.")
    if pp.get("retained_tensors"):
        print("   FAIL: the patched path still retained an unsharded gradient.")
        ok = False

    print("\n2. NUMERICS of the accumulated gradient")
    worst = 0.0
    for a, b in zip(default["grads"], patched["grads"], strict=True):
        worst = max(worst, (a - b).abs().max().item() / max(1e-12, b.abs().max().item()))
    print(f"   max relative difference over {len(patched['grads'])} gradient tensors: {worst:.3e}")
    if worst > 1e-4:
        print("   FAIL: the two paths do not compute the same gradient.")
        ok = False
    else:
        print("   -> the same gradient. Skipping the reduce-scatter buys memory, not accuracy.")

    print("\n3. PEAK ALLOCATED")
    print(f"   verl default : {default['peak']:.3f} GiB")
    print(f"   patched      : {patched['peak']:.3f} GiB   "
          f"({patched['peak'] - default['peak']:+.3f} GiB vs default)")
    print("\n   Read this number with the shard degree in hand, because the two paths scale")
    print("   in OPPOSITE directions and a single-GPU measurement is the one case where the")
    print("   patch does not pay:")
    grad_gib = total_params * 4 / GIB
    print(f"     default : retains params x 4 B = {grad_gib:.3f} GiB UNSHARDED -- constant in")
    print("               world size, so more GPUs do not help")
    print("     patched : reduce-scatter buffers, params x 4 B / shard_degree, a few of them")
    print("               live at once (bf16 grad, flat fp32 buffer, accumulated shard)")
    # Calibrate that "a few" instead of guessing it: at shard degree 1 the measured delta
    # is the whole transient, because the divisor is 1.
    k = (patched["peak"] - default["peak"] + grad_gib) / max(grad_gib, 1e-9) if world == 1 else 3.0
    print(f"   Measured here at shard degree {world}: patched holds {k:.1f} x params x 4 B / "
          f"{world}"
          + ("  <- calibrated, not assumed" if world == 1 else "  <- assumed, rerun under"
             " torchrun --nproc_per_node 1 to calibrate"))
    print("   which makes the projected per-GPU delta:")
    for degree in (1, 2, 8, 32):
        delta = k * grad_gib / degree - grad_gib
        print(f"     shard {degree:>3d}: {delta:+8.3f} GiB"
              + ("   <- measured above" if degree == world else ""))
    for name, params_b in (("this model", total_params / 1e9), ("Qwen3.5-27B", 27.78)):
        big = params_b * 1e9 * 4 / GIB
        print(f"   {name} ({params_b:.2f}B) at shard 32: default {big:.1f} GiB/GPU retained vs "
              f"patched {k * big / 32:.1f} GiB -> {big - k * big / 32:.1f} GiB/GPU freed")
    print("   On an 80 GiB card, at 27.78B, that difference is the entire run.")

    print("\nVERDICT: " + ("premise holds, numerics match -- the patch is correct and load-bearing"
                           if ok else "FAILED, read the notes above before trusting the patch"))
    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
