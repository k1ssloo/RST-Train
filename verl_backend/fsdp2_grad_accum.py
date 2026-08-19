"""Stop FSDP2 from retaining a FULL, UNSHARDED fp32 gradient across micro-batches.

This is the second, distinct OOM on the verl FSDP2 path. It is not the static
footprint one (`scripts/34_diagnose_oom.py`, fixed by making torchrun actually
rendezvous so the shard degree is 32 instead of 8). This one survives a correct
shard degree, survives `engine.optimizer_offload=True`, and does not move when you
cut `data.max_token_len_per_gpu` or `data.max_length` -- because the term it adds
is proportional to PARAMETERS, not to tokens.

THE MECHANISM
-------------
verl saves gradient collectives on non-final accumulation steps
(`verl/workers/engine/fsdp/transformer_impl.py::FSDPEngine._gradient_sync_context`):

    elif version == 2:
        self.module.set_requires_gradient_sync(False)

torch's FSDP2 reacts to that in `_fsdp_param_group.py:607-611`:

    if not self.reduce_grads:
        if self.reshard_after_backward:
            self.reshard()
        for fsdp_param in self.fsdp_params:
            fsdp_param.to_accumulated_grad_if_needed()
        return                      # <-- no reduce-scatter

and `to_accumulated_grad_if_needed` (`_fsdp_param.py:802-813`) is:

    if (self.reduce_dtype is None
        or self._unsharded_param.grad is None
        or self._unsharded_param.grad.dtype == self.reduce_dtype):
        return
    unsharded_grad = self._unsharded_param.grad
    self._unsharded_param.grad = None
    self.unsharded_accumulated_grad = unsharded_grad.to(self.reduce_dtype)

`unsharded_accumulated_grad` is a WHOLE-PARAMETER tensor -- not a shard -- and it is
held until the final micro-batch's `foreach_reduce` consumes it. verl builds the
mixed-precision policy with `param_dtype=bf16, reduce_dtype=fp32`
(`workers/config/engine.py:644` defaults `mp_reduce_dtype: "fp32"`), so the early
return never fires: bf16 grad != fp32 reduce dtype, every time.

Cost, for Qwen3.5-27B at 27.78e9 parameters: 27.78e9 x 4 B = **103.5 GiB per GPU**,
and it is allocated layer by layer as backward walks the model. On an 80 GiB card the
job therefore dies partway through the FIRST non-final micro-batch's backward, with a
huge `allocated by PyTorch` figure and a tiny requested size -- which is exactly what
`logs/train_rank1.log` shows (75.36 GiB allocated, 552 MiB requested).

It needs >= 2 micro-batches per optimizer step to trigger at all. With one
micro-batch `is_last_micro_batch` is true immediately and this path is never taken,
which is why smaller runs of the same config are fine.

THE FIX, AND WHY IT IS SAFE
---------------------------
Let FSDP2 reduce-scatter on EVERY micro-batch. `_fsdp_collectives.py:744-749` already
accumulates into the existing sharded gradient:

    if to_accumulate_grad:
        fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad

and a sum of reduce-scatters equals the reduce-scatter of a sum, so the resulting
gradient is numerically the same computation, just with the partial sums kept in the
fp32 SHARDED gradient (params x 4 B / shard_degree = 3.24 GiB at shard 32) instead of
an unsharded fp32 buffer (103.5 GiB).

What it costs: one reduce-scatter per micro-batch instead of one per optimizer step.
At 2-4 micro-batches that is 2-4x the gradient communication volume of the step -- the
same volume plain DDP would move -- in exchange for ~100 GiB of memory. There is no
correctness trade here, only bandwidth.

MEASURED, not argued (`scripts/35_probe_fsdp2_grad_accum.py`, single H100, 134 M params,
2 micro-batches, verl's own bf16/fp32 mixed-precision policy):

    unsharded_accumulated_grad after micro-batch 1   default: 8 fp32 tensors, all
                                                     134.2 M elements   patched: none
    max relative gradient difference                 0.000e+00
    peak allocated                                   1.189 -> 2.189 GiB at shard degree 1

Note the direction of that last line, and its condition. The retained tensor is
`params x 4 B` at EVERY shard degree, while the reduce-scatter buffers the patch
reinstates cost `3 x params x 4 B / shard_degree` (the 3 is measured, not assumed).
So the break-even is shard degree 3, and **at shard degree 1 or 2 this patch costs
memory rather than saving it**. It is left unconditionally on because a single-GPU run
is a smoke test where a few hundred MiB is irrelevant, while the runs that matter here
shard 32 ways, where the projection is 103.5 -> 9.7 GiB/GPU, i.e. **93.8 GiB/GPU freed**.
If you ever do need it off for a 1-2 GPU memory measurement, `RST_FSDP2_ALWAYS_REDUCE=0`
is the switch and one micro-batch per step is the precondition.

HOW IT GETS APPLIED
-------------------
`apply()` is called at import time from `rst_sft_dataset.py`, because verl loads that
module through `data.custom_cls.path` with `load_extern_object` in EVERY rank, before
training starts. That makes the patch launcher-independent: any command line that uses
our pre-tokenized dataset gets it, including the cluster's own wrapper scripts.

`RST_FSDP2_ALWAYS_REDUCE=0` restores verl's behaviour. Only do that if you have
independently made the step one single micro-batch, and say so in the report.

The line to grep for in a log is `[rst-fsdp2]`. If it is absent, the patch did not run.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager

ENV_FLAG = "RST_FSDP2_ALWAYS_REDUCE"
LOG_PREFIX = "[rst-fsdp2]"

_STATE: dict[str, object] = {"applied": False, "patched": False, "status": "not applied"}


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "1").strip().lower() not in ("0", "false", "no", "off")


def unsharded_grad_gib(params_b: float, reduce_bytes: int = 4) -> float:
    """GiB of unsharded accumulated gradient verl's default path would retain per GPU.

    Divided by nothing: the whole point is that this tensor is not sharded, so the
    number does not improve with more GPUs.
    """
    return params_b * 1e9 * reduce_bytes / (1 << 30)


def sharded_grad_gib(params_b: float, shard_degree: int, reduce_bytes: int = 4) -> float:
    """What the same gradient costs once every micro-batch reduce-scatters."""
    return params_b * 1e9 * reduce_bytes / max(1, shard_degree) / (1 << 30)


def estimate_micro_batches(
    *,
    sample_lengths: list[int],
    train_batch_size: int,
    world_size: int,
    ulysses_sp: int,
    max_token_len_per_gpu: int,
) -> dict[str, int]:
    """How many micro-batches an optimizer step will be split into.

    Mirrors what verl actually does, which is easy to get wrong from the config alone:

      * `workers/engine/utils.py::prepare_micro_batches` scales the budget by the
        sequence-parallel size -- `max_token_len = max_token_len_per_gpu * sp_size` --
        because an Ulysses group of `sp` ranks jointly holds one micro-batch. So
        raising `ULYSSES_SP` raises the numerator (tokens per group) and the
        denominator (budget per group) equally: **it cannot change this ratio.**
      * the data-parallel size is `world_size // ulysses_sequence_parallel_size`
        (`transformer_impl.py:633`), so that is how many groups split the global batch.
      * `rearrange_micro_batches(..., same_micro_num_in_dp=True)` pads every dp rank to
        the LARGEST count any rank needs, so one unlucky rank with long samples sets the
        number for all of them. `worst_case` below is that rank: the longest
        `samples_per_dp` sequences in the dataset.

    Bin packing cannot do better than `ceil(total_tokens / budget)`, so both figures are
    lower bounds on the real count -- which is the safe direction for a gate.
    """
    if not sample_lengths:
        raise ValueError("sample_lengths is empty; cannot estimate a micro-batch count")
    if ulysses_sp < 1 or world_size < 1 or max_token_len_per_gpu < 1:
        raise ValueError("world_size, ulysses_sp and max_token_len_per_gpu must be >= 1")
    if world_size % ulysses_sp:
        raise ValueError(f"world_size={world_size} is not a multiple of ulysses_sp={ulysses_sp}")

    dp_size = world_size // ulysses_sp
    samples_per_dp = math.ceil(train_batch_size / dp_size)
    budget = max_token_len_per_gpu * ulysses_sp

    ordered = sorted(sample_lengths)
    mean_len = sum(ordered) / len(ordered)
    typical = math.ceil(samples_per_dp * mean_len / budget)
    worst = math.ceil(sum(ordered[-samples_per_dp:]) / budget)

    return {
        "dp_size": dp_size,
        "samples_per_dp": samples_per_dp,
        "budget_per_dp_group": budget,
        "budget_per_gpu": max_token_len_per_gpu,
        "typical_micro_batches": max(1, typical),
        "worst_case_micro_batches": max(1, worst),
        "longest_sample": ordered[-1],
    }


def largest_single_micro_batch_size(
    *,
    sample_lengths: list[int],
    world_size: int,
    ulysses_sp: int,
    max_token_len_per_gpu: int,
) -> int:
    """Largest `train_batch_size` whose WORST-CASE step is still one micro-batch.

    Reported by the gate as the honest cost of the alternative fix, not as a
    recommendation: shrinking the global batch to fit changes the effective batch size
    and therefore the LR schedule, and it only holds for this dataset's length tail.
    Returns 0 when even one sample per dp rank does not fit the budget.
    """
    dp_size = max(1, world_size // max(1, ulysses_sp))
    budget = max_token_len_per_gpu * ulysses_sp
    ordered = sorted(sample_lengths)
    total = 0
    per_dp = 0
    for length in reversed(ordered):
        if total + length > budget:
            break
        total += length
        per_dp += 1
    return per_dp * dp_size


@contextmanager
def _always_reduce(self, *, is_last_micro_batch: bool):  # noqa: ANN001, ARG001
    """Replacement for `FSDPEngine._gradient_sync_context`: never disable grad sync.

    Leaving `reduce_grads` at its default True means `post_backward` takes the
    reduce-scatter branch for every micro-batch and accumulates into the fp32 sharded
    gradient, so `to_accumulated_grad_if_needed` is never reached and no unsharded
    gradient is ever materialized. `is_last_micro_batch` is accepted and ignored.
    """
    yield


def apply(*, verbose: bool = True) -> str:
    """Patch verl's FSDP engine. Idempotent; returns a one-line status.

    Raises only on version drift -- a verl whose `FSDPEngine` has no
    `_gradient_sync_context` is a verl where this file's reasoning may not hold, and
    silently doing nothing there would reintroduce a 100 GiB OOM under a log line
    claiming the fix is in place. A verl that is not importable at all is not an error:
    the pure-python helpers above are used by tests and by the launcher gate.
    """
    if _STATE["applied"]:
        return str(_STATE["status"])

    if not enabled():
        status = (
            f"{ENV_FLAG}=0 -- verl's own gradient-sync skipping is left in place. Every "
            f"non-final micro-batch will retain a full UNSHARDED fp32 gradient; this is "
            f"only survivable if the step is exactly one micro-batch."
        )
        _STATE.update(applied=True, patched=False, status=status)
        if verbose:
            print(f"{LOG_PREFIX} {status}", flush=True)
        return status

    try:
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    except ImportError as exc:
        status = f"verl FSDP engine not importable ({exc}); nothing patched"
        _STATE.update(applied=True, patched=False, status=status)
        return status

    if not hasattr(FSDPEngine, "_gradient_sync_context"):
        raise RuntimeError(
            "verl's FSDPEngine has no _gradient_sync_context, so this patch cannot "
            "neutralize it. Either this verl does not skip gradient sync on non-final "
            "micro-batches (in which case delete this module and its callers), or the "
            "method was renamed. Read workers/engine/fsdp/transformer_impl.py before "
            f"setting {ENV_FLAG}=0 -- the unpatched path costs a full unsharded fp32 "
            "gradient per GPU."
        )

    FSDPEngine._gradient_sync_context = _always_reduce
    FSDPEngine._rst_always_reduce = True
    status = (
        "FSDPEngine._gradient_sync_context neutralized: FSDP2 now reduce-scatters every "
        "micro-batch, so no unsharded fp32 gradient is retained. Costs one extra "
        "reduce-scatter per micro-batch; saves params x 4 B (103.5 GiB at 27.78B)."
    )
    _STATE.update(applied=True, patched=True, status=status)
    if verbose:
        print(f"{LOG_PREFIX} {status}", flush=True)
    return status


def state() -> dict[str, object]:
    return dict(_STATE)


def _cli() -> int:
    """`python verl_backend/fsdp2_grad_accum.py --lengths <parquet> ...` -- the gate.

    Used by scripts/30_run_sft_verl.sh before torchrun, and runnable by hand against
    any pre-tokenized parquet to answer "will this step take the unsharded-grad path".
    """
    import argparse

    ap = argparse.ArgumentParser(description="micro-batch / unsharded-grad gate")
    ap.add_argument("--lengths", help="pre-tokenized parquet with an input_ids column")
    ap.add_argument("--params-b", type=float, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--ulysses-sp", type=int, default=1)
    ap.add_argument("--max-token-len-per-gpu", type=int, required=True)
    ap.add_argument("--train-batch-size", type=int, required=True)
    ap.add_argument("--shard-degree", type=int, default=0, help="default: world size")
    ap.add_argument("--card-gib", type=float, default=79.33)
    args = ap.parse_args()

    import pandas as pd

    lengths = [int(n) for n in pd.read_parquet(args.lengths, columns=["input_ids"])
               ["input_ids"].map(len)]
    est = estimate_micro_batches(
        sample_lengths=lengths,
        train_batch_size=args.train_batch_size,
        world_size=args.world_size,
        ulysses_sp=args.ulysses_sp,
        max_token_len_per_gpu=args.max_token_len_per_gpu,
    )
    shard = args.shard_degree or args.world_size
    unsharded = unsharded_grad_gib(args.params_b)
    sharded = sharded_grad_gib(args.params_b, shard)

    print(f"[gate] dp_size={est['dp_size']} (world {args.world_size} / sp {args.ulysses_sp}), "
          f"{est['samples_per_dp']} samples per dp rank, token budget "
          f"{est['budget_per_dp_group']:,} per Ulysses group "
          f"({est['budget_per_gpu']:,}/GPU x {args.ulysses_sp})")
    print(f"[gate] micro-batches per optimizer step: {est['typical_micro_batches']} typical, "
          f"{est['worst_case_micro_batches']} worst case (same_micro_num_in_dp=True takes "
          f"the max over dp ranks)")

    if est["worst_case_micro_batches"] < 2:
        print("[gate] one micro-batch per step: verl never disables gradient sync, so the "
              "unsharded-grad path cannot be reached at all.")
        return 0

    print(f"[gate] >= 2 micro-batches, so verl WOULD call set_requires_gradient_sync(False) "
          f"and FSDP2 would retain {unsharded:.1f} GiB/GPU of unsharded fp32 gradient "
          f"on a {args.card_gib:.0f} GiB card. Raising ULYSSES_SP cannot help: it scales "
          f"the token budget and the tokens per group by the same factor.")
    if enabled():
        print(f"[gate] {ENV_FLAG}=1 (default): verl_backend/fsdp2_grad_accum.py makes every "
              f"micro-batch reduce-scatter instead, so the gradient stays sharded at "
              f"{sharded:.1f} GiB/GPU (transiently ~3x that during the reduce). Confirm "
              f"'{LOG_PREFIX}' appears in the training log.")
        if shard < 3:
            print(f"[gate] NOTE: shard degree {shard} < 3, where the reduce-scatter buffers cost "
                  f"more than the retained tensor they replace. Harmless at this scale, but do "
                  f"not read a peak-memory number from this run as the multi-node one.")
        return 0

    print(f"FATAL: {ENV_FLAG}=0 with {est['worst_case_micro_batches']} micro-batches per "
          f"step. That retains {unsharded:.1f} GiB/GPU of unsharded fp32 gradient and the "
          f"run will OOM inside loss.backward().")
    print(f"       Fix: unset {ENV_FLAG}. One extra reduce-scatter per micro-batch is the "
          f"whole cost.")
    print("       The alternative -- forcing exactly ONE micro-batch per step -- needs "
          "either")
    tokens_needed = math.ceil(sum(sorted(lengths)[-est["samples_per_dp"]:]) / args.ulysses_sp)
    largest_bsz = largest_single_micro_batch_size(
        sample_lengths=lengths, world_size=args.world_size, ulysses_sp=args.ulysses_sp,
        max_token_len_per_gpu=args.max_token_len_per_gpu,
    )
    print(f"         data.max_token_len_per_gpu >= {tokens_needed:,} (up from "
          f"{args.max_token_len_per_gpu:,}), which is {tokens_needed / args.max_token_len_per_gpu:.1f}x "
          f"the activation memory per GPU and defeats the purpose, or")
    print(f"         data.train_batch_size <= {largest_bsz} (down from "
          f"{args.train_batch_size}), which changes the effective batch size and the LR "
          f"schedule.")
    print("       Neither is robust: same_micro_num_in_dp=True means one dp rank that "
          "draws\n       long samples sets the micro-batch count for every rank, so a "
          "tight fit\n       reintroduces this OOM on some later step rather than at "
          "launch.")
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
