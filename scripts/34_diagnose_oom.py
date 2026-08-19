#!/usr/bin/env python3
"""Decide, in under a minute, WHY a verl FSDP2 run OOMs -- static footprint or activations.

    # 1. from a log you already have (no cluster time, no GPU)
    python scripts/34_diagnose_oom.py --from-log outputs/sft-27b/train.log --key qwen3.5-27b

    # 2. pure arithmetic: what does each shard degree cost per GPU?
    python scripts/34_diagnose_oom.py --key qwen3.5-27b --observed-peak-gib 77.01

    # 3. runtime truth: is the job REALLY sharding over all the cards you think?
    torchrun --nnodes 4 --nproc_per_node 8 --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR --master_port 29500 \
        scripts/34_diagnose_oom.py --runtime

Why this script exists. An OOM whose peak does not move when you halve
`data.max_token_len_per_gpu` is not an activation problem, so cutting the token budget
(or the dataset) cannot fix it. Under `engine.strategy=fsdp2` with the verl 0.9.0
defaults (`engine/fsdp.yaml`: `model_dtype: fp32`, `dtype: bfloat16`, `param_offload:
false`, `optimizer_offload: false`, `offload_policy: false`) every parameter costs a
FIXED 16 bytes per GPU divided by the shard degree:

    fp32 master 4 + fp32 grad 4 + Adam exp_avg 4 + Adam exp_avg_sq 4 = 16 B/param

so 16 B/param over 8 ranks is exactly 2 B/param -- numerically identical to the whole
bf16 model. If you see a static term that looks like "the bf16 model size", that is not
a coincidence and it is not a sharding bug in FSDP: it is a shard degree of 8.

Shard degree is NOT what `engine.fsdp_size` says. `create_device_mesh()` in
verl/workers/engine/fsdp/utils.py:35 gives a 1-D mesh over `world_size` when
`fsdp_size < 0 or fsdp_size >= world_size`, and a 2-D (ddp x fsdp) mesh otherwise.
So the shard degree is `min(world_size, fsdp_size)`, and `world_size` is a property of
how torchrun was launched, not of the config. Four nodes that never formed one process
group are four world_size=8 jobs that each shard 8 ways.

Exit code: 0 = diagnosed and within budget, 1 = static footprint alone cannot fit,
2 = could not parse / not enough information.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

GIB = 1 << 30
BYTES_PER_PARAM_FP32_ADAM = 16  # fp32 master + fp32 grad + exp_avg + exp_avg_sq


def gib(n: float) -> float:
    return n / GIB


def registry_row(key: str) -> dict:
    import model_registry as reg

    models = reg.load()["models"]
    if key not in models:
        sys.exit(f"unknown model key {key!r}; have {', '.join(sorted(models))}")
    m = models[key]
    return {
        "params_b": float(m["params_b"]),
        "layers": int(m["n_layers"]),
        "min_gpus": int(m.get("min_gpus") or 0),
    }


def static_table(params: float, degrees: tuple[int, ...]) -> list[tuple[int, float, float, float]]:
    """(shard_degree, after_init, after_first_backward, after_first_optim_step) in GiB."""
    rows = []
    for d in degrees:
        after_init = params * 4 / d
        after_bwd = params * 8 / d
        after_step = params * BYTES_PER_PARAM_FP32_ADAM / d
        rows.append((d, gib(after_init), gib(after_bwd), gib(after_step)))
    return rows


def activation_estimate(cfg: dict, tokens: int, fused_ce: bool) -> dict[str, float]:
    """Rough per-GPU activation band with full-recompute gradient checkpointing.

    Deliberately crude: the point is the ORDER of magnitude, so you can tell a 60 GiB
    residual (impossible for these shapes) from a 25 GiB one (entirely normal).
    """
    hidden = int(cfg["hidden_size"])
    layers = int(cfg["num_hidden_layers"])
    vocab = int(cfg["vocab_size"])
    saved_inputs = layers * tokens * hidden * 2  # one bf16 tensor per checkpointed block
    recompute_peak = tokens * hidden * 2 * 24  # one block's internals, bf16, generous
    logits = 0 if fused_ce else tokens * vocab * (2 + 4)  # bf16 logits + fp32 softmax grad
    return {
        "saved_inputs_gib": gib(saved_inputs),
        "recompute_peak_gib": gib(recompute_peak),
        "logits_gib": gib(logits),
        "total_gib": gib(saved_inputs + recompute_peak + logits),
    }


LOG_MEM = re.compile(
    r"(?P<head>[^,\n]+), memory allocated \(GB\): (?P<alloc>[\d.]+), "
    r"memory reserved \(GB\): (?P<res>[\d.]+), device memory used/total \(GB\): "
    r"(?P<used>[\d.]+)/(?P<total>[\d.]+)"
)
LOG_OOM = re.compile(r"GiB.{0,80}?(?:is allocated by PyTorch|of which)", re.S)
LOG_CAP = re.compile(r"has a total capacity of ([\d.]+) GiB")
LOG_INUSE = re.compile(r"Of the allocated memory ([\d.]+) GiB")
LOG_PROC = re.compile(r"Including non-PyTorch memory, this process has ([\d.]+) GiB")


def from_log(path: Path, params: float) -> int:
    text = path.read_text(errors="replace")
    hits = [m.groupdict() for m in LOG_MEM.finditer(text)]
    if not hits:
        print(f"FAIL: no verl `log_gpu_memory_usage` line in {path}.")
        print("      Expected e.g. `After FSDP, memory allocated (GB): 3.47, ...`.")
        print("      verl prints it at workers/engine/fsdp/transformer_impl.py:595 with")
        print("      logger=None, so it goes to stdout -- make sure stdout is captured.")
        return 2

    print(f"parsed {len(hits)} memory lines from {path.name}\n")
    for h in hits[:12]:
        print(f"  {h['head'].strip():<52s} alloc {float(h['alloc']):7.2f} GB   "
              f"reserved {float(h['res']):7.2f} GB   card {float(h['used']):6.1f}/{float(h['total']):.0f} GB")
    if len(hits) > 12:
        print(f"  ... {len(hits) - 12} more")

    after_fsdp = next((h for h in hits if h["head"].strip().startswith("After FSDP")), None)
    if after_fsdp is None:
        print("\nFAIL: no `After FSDP` line -- cannot infer the shard degree.")
        return 2

    # verl divides by 1024**3 but labels the field "GB" (performance.py:48), so it is GiB.
    alloc_bytes = float(after_fsdp["alloc"]) * GIB
    implied = params * 4 / alloc_bytes if alloc_bytes > 0 else 0.0
    print(f"\n`After FSDP` allocated {float(after_fsdp['alloc']):.2f} GiB holds the fp32 master")
    print(f"shard only (no grads, no Adam state yet) = {params / 1e9:.2f}B x 4 B / shard_degree.")
    print(f"  => implied shard degree = {implied:.1f}  (round: {round(implied)})")

    rc = 0
    if round(implied) >= 1:
        d = round(implied)
        after_step = gib(params * BYTES_PER_PARAM_FP32_ADAM / d)
        print(f"  => once Adam state exists this becomes {after_step:.1f} GiB/GPU, before a single")
        print("     activation byte. Anything left over is your activation budget.")
        card_total = float(after_fsdp["total"])
        if card_total > 0 and after_step > 0.55 * card_total:
            print(f"\nVERDICT: static footprint {after_step:.1f} GiB is {100 * after_step / card_total:.0f}% "
                  f"of the {card_total:.0f} GiB card, before any activation.")
            print("         Cutting max_token_len_per_gpu or max_seq_len cannot fix this. Raise the")
            print("         shard degree (one process group over all nodes, engine.fsdp_size=-1) or")
            print("         move the optimizer off the GPU (engine.offload_policy=true).")
            rc = 1

    cap = LOG_CAP.search(text)
    proc = LOG_PROC.search(text)
    if cap and proc:
        print(f"\nOOM line says: card capacity {cap.group(1)} GiB, this process holding "
              f"{proc.group(1)} GiB.")
    return rc


def runtime_probe(fsdp_size: int) -> int:
    import torch
    import torch.distributed as dist

    if not dist.is_available():
        print("FAIL: torch.distributed unavailable")
        return 2
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    host = socket.gethostname()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    hosts: list[object] = [None] * world
    dist.all_gather_object(hosts, host)
    nodes = sorted(set(str(h) for h in hosts))
    shard_degree = world if (fsdp_size < 0 or fsdp_size >= world) else fsdp_size

    if rank == 0:
        free, total = (torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0))
        print(f"WORLD_SIZE      = {world}")
        print(f"distinct nodes  = {len(nodes)}  {nodes}")
        print(f"ranks per node  = {world // max(len(nodes), 1)}")
        print(f"engine.fsdp_size= {fsdp_size}  ->  mesh = "
              f"{'1-D (fsdp,)' if (fsdp_size < 0 or fsdp_size >= world) else f'2-D (ddp={world // fsdp_size}, fsdp={fsdp_size})'}")
        print(f"EFFECTIVE SHARD DEGREE = {shard_degree}")
        print(f"card free/total = {gib(free):.1f}/{gib(total):.1f} GiB "
              f"({gib(total - free):.1f} GiB already held by someone else)")
        if len(nodes) == 1 and world <= 8:
            print("\nWARNING: this is a SINGLE-NODE process group. If you meant to shard over")
            print("         four nodes, torchrun never rendezvoused: check --nnodes, --node_rank")
            print("         and --master_addr on EVERY node (the launcher defaults are")
            print("         NNODES=4 but MASTER_ADDR=127.0.0.1 NODE_RANK=0, which only works")
            print("         for a single-node run).")
    dist.barrier()
    dist.destroy_process_group()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--key", default="qwen3.5-27b", help="model_registry key")
    ap.add_argument("--params-b", type=float, help="override the registry parameter count")
    ap.add_argument("--from-log", type=Path, help="parse a verl training log")
    ap.add_argument("--runtime", action="store_true", help="run under torchrun to read the real mesh")
    ap.add_argument("--fsdp-size", type=int, default=-1, help="engine.fsdp_size as passed to verl")
    ap.add_argument("--config", type=Path, help="HF config.json, enables the activation estimate")
    ap.add_argument("--tokens", type=int, default=16384, help="tokens in the largest micro-batch")
    ap.add_argument("--no-fused-ce", action="store_true", help="model.use_fused_kernels is off")
    ap.add_argument("--observed-peak-gib", type=float, help="peak from the OOM message")
    args = ap.parse_args()

    if args.runtime:
        return runtime_probe(args.fsdp_size)

    if args.params_b is not None:
        params_b, layers, min_gpus = args.params_b, 0, 0
    else:
        row = registry_row(args.key)
        params_b, layers, min_gpus = row["params_b"], row["layers"], row["min_gpus"]
    params = params_b * 1e9

    print(f"model {args.key}: {params_b:.2f}B params, {layers} layers, "
          f"registry min_gpus={min_gpus}")
    print(f"bf16 weights alone: {gib(params * 2):.1f} GiB ({params * 2 / 1e9:.1f} GB)\n")

    if args.from_log:
        return from_log(args.from_log, params)

    print("verl 0.9.0 fsdp2 defaults: engine.model_dtype=fp32 -> fp32 master shard,")
    print("mp_policy(param_dtype=bf16, reduce_dtype=fp32), no offload. 16 B/param total.\n")
    print(f"{'shard':>6s} {'after init':>12s} {'after 1st bwd':>15s} {'after 1st step':>16s}")
    for d, a, b, c in static_table(params, (64, 32, 16, 8, 4)):
        flag = ""
        if abs(c - gib(params * 2)) / gib(params * 2) < 0.02:
            flag = "  <-- identical to the whole bf16 model (16 B/param / 8 = 2 B/param)"
        print(f"{d:>6d} {a:>9.1f} GiB {b:>12.1f} GiB {c:>13.1f} GiB{flag}")

    if args.config and args.config.is_file():
        cfg = json.loads(args.config.read_text())
        cfg = cfg.get("text_config", cfg)
        est = activation_estimate(cfg, args.tokens, not args.no_fused_ce)
        print(f"\nactivation estimate at {args.tokens} tokens/micro-batch, "
              f"fused CE {'off' if args.no_fused_ce else 'on'}:")
        for k, v in est.items():
            print(f"  {k:<20s} {v:7.1f} GiB")

    if args.observed_peak_gib:
        print(f"\nobserved peak {args.observed_peak_gib:.2f} GiB decomposes as:")
        for d, _a, _b, c in static_table(params, (32, 16, 8)):
            print(f"  shard {d:>2d}: static {c:6.1f} GiB  +  residual "
                  f"{args.observed_peak_gib - c:6.1f} GiB of activations/buffers/fragmentation")
        print("\nPick the row whose residual is physically plausible for your token count.")
        print("A residual far larger than layers*tokens*hidden*2 means the shard degree is wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
