#!/usr/bin/env python3
"""DPO on the RST preference pairs. No sandbox, no rollouts, no reference model in memory.

    # local smoke test (small model, a handful of pairs)
    python scripts/19_train_dpo.py --pairs $BASE_FOLDER/dpo-v1 \
        --ref-logps $BASE_FOLDER/dpo-v1/ref --model-path $BASE_FOLDER/Qwen3.5-0.8B \
        --out $BASE_FOLDER/out-dpo --max-steps 4 --max-seq-len 4096

    # 4 nodes x 8 GPUs
    torchrun --nnodes 4 --nproc-per-node 8 --rdzv-backend c10d \
        --rdzv-endpoint $MASTER_ADDR:29500 scripts/19_train_dpo.py \
        --pairs $BASE_FOLDER/dpo-v1 --ref-logps $BASE_FOLDER/dpo-v1/ref \
        --model-path $BASE_FOLDER/out-hf-full --out $BASE_FOLDER/out-dpo

WHAT THIS IS AND IS NOT
    It is the container-free fallback for the RL stage: the trajectory release's own
    successes and failures, scored by the tasks' verifiers, turned into a preference
    objective. Every number it needs was paid for when the release was generated.

    It is NOT a substitute for GRPO, and the difference is not a matter of degree.
    GRPO samples from the CURRENT policy, so it can discover behaviour the data never
    contained. DPO reweights behaviour that is already in the data, and the data came
    from OTHER policies. So this can sharpen the SFT model toward the successful modes
    it already has; it cannot teach a strategy that no logged trajectory used. Say
    that in the write-up, and never report a DPO run as "RL results".

THE THREE GATES, AND WHY EACH ONE EXISTS
    1. FINGERPRINT. Reference logprobs are constants of one specific checkpoint. If
       `--model-path` is not the checkpoint `18_dpo_ref_logprobs.py` scored, then
       (pi - ref) is comparing two different models at step 0 and the implicit reward
       is garbage from the first step. Refuses to run on a mismatch.
    2. COVERAGE. Every training pair must have reference logprobs. Silently dropping
       the unscored ones would change the dataset without changing the manifest.
    3. STEP-0 CALIBRATION. At initialization the policy IS the reference, so the DPO
       loss must be exactly -log sigmoid(0) = log 2 = 0.693147. This is the single
       most valuable check in the file: it fails loudly if the mask, the tokenizer,
       the dtype, or the checkpoint changed between the reference pass and now. What
       it tolerates is float noise only (`--calibration-tol`, in nats per token);
       a real mismatch is orders of magnitude larger. Score the reference with the
       same dtype the policy's forward will use and the residual is exactly 0; a
       bf16 reference against an fp32 policy leaves ~1e-3 nats/token, and the gate
       says which of the two situations you are in rather than leaving you to guess.

MEMORY: WHY TWO BACKWARD PASSES INSTEAD OF ONE
    A textbook DPO step holds BOTH sides' autograd graphs at once. These episodes run
    to 32k tokens; with activation checkpointing one side still costs on the order of
    20 GB of saved layer inputs at 27 B scale, so both together do not fit next to
    the optimizer state.

    But the loss depends on the two logprob sums only through a scalar. With
    z = (pi_c - pi_r) - (ref_c - ref_r), the sigmoid DPO gradient is
        dL/d pi_c = -beta * sigmoid(-beta * z)      dL/d pi_r = +beta * sigmoid(-beta * z)
    so the coefficient can be computed under no_grad from two cheap forwards, and
    each side then backpropagates alone: `logp.backward(coef)`. Peak activation
    memory halves, at the cost of two extra forwards (~33% more compute per step) and
    zero approximation -- the gradient is identical. `--no-split-backward` restores
    the single-graph version for short sequences.

LENGTH BIAS
    Summed-logprob DPO prefers shorter sequences, and in this data the failures are
    the longer side (an agent that cannot solve a task keeps going until the step
    limit). `--length-normalize` divides each side's logprob sum by its supervised
    token count, which removes the length term from the objective. The pair builder's
    manifest says whether the confound is large enough to matter; whichever you pick,
    it lands in `dpo_training_summary.json` so the report can state it.

STEP SIZE: THE CLIP, NOT THE LR
    Each side's logprob is a SUM over ~3k supervised tokens, so the gradient norm is
    routinely ~1e3 against the default `--max-grad-norm 1.0`. Clipping then fires on
    every step, which means the clip -- not `--lr` -- sets how far the weights move,
    and the cosine schedule only rescales an already-normalized direction. That is a
    workable setup, but reporting `--lr 5e-7` as the step size would be wrong, so the
    summary records `clip_active_fraction` and warns when it is near 1.0.
    `--length-normalize` turns the objective per-token and drops the norms by ~1e3,
    which is the other way to make `--lr` mean what it usually means.

    Related: `--param-dtype` defaults to fp32 masters. At lr 5e-7 a bf16 master weight
    rounds most updates to zero while a few small-magnitude ones still move, so the
    loss curve moves and the model mostly does not.

OUTPUT
    <out>/dpo_training_summary.json   gates, metrics, holdout accuracy, config
    <out>/dpo_metrics.jsonl           per-step metrics
    <out>/hf/                         the trained checkpoint in HF format
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dpo_common import (  # noqa: E402
    as_f64,
    checkpoint_fingerprint,
    dpo_loss,
    load_model,
    masked_logprob_sum,
    read_ref_logps,
)

LOG2 = math.log(2.0)


# A margin this small is not a preference, it is arithmetic. At step 0 the policy IS
# the reference so the true margin is 0, and what survives is the difference between
# summing fp32 chunks in one order versus another (~1e-8 per token, larger if the
# reference used a different --logit-chunk or dtype). A trained margin is 1e-2..1, six
# orders of magnitude away, so this band cannot swallow real signal.
TIE_EPS = 1e-6


def rank_score(margin: float) -> float:
    """Ranking credit for one pair: 1 correct, 0 wrong, 0.5 for a tie.

    The tie case is not pedantry. At step 0 every margin is zero to within float
    noise; scoring those as wrong would print "accuracy 0.00 -> 0.50" and invite the
    reading that the model went from always wrong to half right, when the truth is
    that it started with no preference at all.
    """
    if margin > TIE_EPS:
        return 1.0
    return 0.5 if margin >= -TIE_EPS else 0.0


# --------------------------------------------------------------------- plumbing

def dist_info() -> tuple[int, int, int]:
    return (int(os.environ.get("RANK", 0)),
            int(os.environ.get("WORLD_SIZE", 1)),
            int(os.environ.get("LOCAL_RANK", 0)))


def log0(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def load_split(pairs: Path, name: str):
    import pandas as pd

    path = pairs / f"dpo_{name}.parquet" if pairs.is_dir() else pairs
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def shard_model(model, *, param_dtype, world_size: int):
    """FSDP2: shard every decoder layer, then the root.

    Params stay fp32 and FSDP casts to `param_dtype` for compute. That is not
    fussiness: DPO runs at lr ~5e-7, and a bf16 master weight rounds an update that
    small to exactly zero, so the run would report a falling loss while the weights
    barely moved.
    """
    import torch
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    if world_size == 1:
        return model
    policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    decoder = model.get_decoder()
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise SystemExit("cannot find decoder.layers to shard; add the right attribute here")
    for layer in layers:
        fully_shard(layer, mp_policy=policy)
    fully_shard(model, mp_policy=policy)
    return model


def save_hf(model, tokenizer_src: Path, out: Path, *, rank: int, world_size: int) -> None:
    """Write an HF-format checkpoint. Gathers the sharded state dict on rank 0."""
    import shutil

    import torch

    if world_size > 1:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        state = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    else:
        state = model.state_dict()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        state = {k: v.to(torch.bfloat16) for k, v in state.items()}
        model.save_pretrained(str(out), state_dict=state, safe_serialization=True)
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                     "special_tokens_map.json", "chat_template.jinja", "generation_config.json"):
            src = tokenizer_src / name
            if src.is_file():
                shutil.copy2(src, out / name)
    if world_size > 1:
        torch.distributed.barrier()


# ------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--ref-logps", type=Path, required=True,
                    help="dir or file written by 18_dpo_ref_logprobs.py")
    ap.add_argument("--model-path", type=Path, required=True,
                    help="policy init; MUST be the checkpoint the reference was scored on")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--variant", default="sigmoid", choices=["sigmoid", "ipo"])
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="cDPO: how often the verifier's preference is assumed wrong")
    ap.add_argument("--length-normalize", action="store_true",
                    help="divide each side's logprob sum by its supervised token count")
    ap.add_argument("--lr", type=float, default=5e-7,
                    help="DPO wants roughly an order of magnitude below the SFT lr")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--grad-accum", type=int, default=4, help="pairs per rank per step")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = one full epoch")
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="the summary reports how often this actually clipped; if that is "
                         "~100%% the schedule is not the schedule you configured")
    ap.add_argument("--param-dtype", default="fp32", choices=["fp32", "bf16"],
                    help="MASTER weight dtype. fp32 is the default for a reason: see the "
                         "note in the source. bf16 halves weight+optimizer memory and is "
                         "only safe at an lr large enough to survive bf16 rounding")
    ap.add_argument("--max-seq-len", type=int, default=32768,
                    help="pairs whose either side exceeds this are SKIPPED (and counted)")
    ap.add_argument("--logit-chunk", type=int, default=512)
    ap.add_argument("--no-split-backward", action="store_true",
                    help="hold both sides' graphs at once (short sequences only)")
    ap.add_argument("--calibration-tol", type=float, default=0.05,
                    help="max |pi - ref| in nats per token at step 0")
    ap.add_argument("--skip-calibration", action="store_true",
                    help="proceed even if step 0 does not reproduce the reference. "
                         "Only for deliberate experiments; record it in the write-up")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-pairs", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=0, help="0 = only at the end")
    ap.add_argument("--seed", type=int, default=1228)
    args = ap.parse_args()

    import torch

    rank, world_size, local_rank = dist_info()
    if world_size > 1:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank if torch.cuda.is_available() else 0)
    torch.manual_seed(args.seed)

    # ---- data + the three gates -------------------------------------------
    train = load_split(args.pairs, "train")
    if train is None:
        sys.exit(f"no dpo_train.parquet under {args.pairs}")
    holdout = load_split(args.pairs, "holdout")
    ref_table, ref_manifests = read_ref_logps(args.ref_logps)

    fingerprint = checkpoint_fingerprint(args.model_path)
    ref_fingerprints = {m.get("checkpoint_fingerprint") for m in ref_manifests}
    if ref_manifests and ref_fingerprints != {fingerprint}:
        sys.exit(
            "GATE 1 FAILED (fingerprint): the reference logprobs were computed on a "
            f"different checkpoint.\n  --model-path {args.model_path}\n    {fingerprint}\n"
            f"  reference manifests say\n    {sorted(f or 'unknown' for f in ref_fingerprints)}\n"
            "(pi - ref) would not be zero at step 0, so the implicit reward would be "
            "meaningless from the first update. Re-run 18_dpo_ref_logprobs.py against "
            "this checkpoint, or point --model-path at the one it scored."
        )

    # Length filter BEFORE the coverage gate, and with the same --max-seq-len script
    # 18 was given: scoring a 32k pair the trainer will skip is minutes of reference
    # compute at 27 B scale. The count is logged either way, so the dataset that was
    # actually trained on is never in doubt.
    too_long = train[(train.chosen_n_tokens > args.max_seq_len)
                     | (train.rejected_n_tokens > args.max_seq_len)]
    if len(too_long):
        log0(rank, f"[data] skipping {len(too_long):,} pairs longer than "
                   f"--max-seq-len {args.max_seq_len:,} (of {len(train):,})")
        train = train.drop(too_long.index)
    if train.empty:
        sys.exit("every pair exceeds --max-seq-len")

    missing = int((~train.pair_id.isin(ref_table)).sum())
    if missing:
        sys.exit(
            f"GATE 2 FAILED (coverage): {missing:,} of {len(train):,} trainable pairs have "
            f"no reference logprobs. Finish the 18_dpo_ref_logprobs.py pass (it is "
            f"resumable and shardable, and takes the same --max-seq-len) rather than "
            f"training on the subset -- a quietly reduced dataset makes every downstream "
            f"number unexplainable."
        )
    log0(rank, f"[gates] fingerprint ok; ref logprobs cover {len(train):,} trainable pairs")

    # Deterministic order, then a contiguous per-rank slice truncated so every rank
    # runs the same number of micro-steps (an uneven count deadlocks the all-reduce).
    train = train.sort_values("pair_id").reset_index(drop=True)
    per_step = world_size * args.grad_accum
    usable = (len(train) // per_step) * per_step
    if usable == 0:
        sys.exit(f"{len(train)} pairs < one step of {per_step}; lower --grad-accum")
    if usable < len(train):
        log0(rank, f"[data] dropping {len(train) - usable} pairs to align with "
                   f"{world_size} ranks x {args.grad_accum} accum")
    train = train.iloc[:usable]
    my_rows = train.iloc[rank::world_size].reset_index(drop=True)
    steps_per_epoch = len(my_rows) // args.grad_accum
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    log0(rank, f"[data] {usable:,} pairs -> {steps_per_epoch:,} steps/epoch, "
               f"training {total_steps:,} steps")

    # ---- model -------------------------------------------------------------
    # Master weights stay fp32 unless asked otherwise. DPO runs at lr ~5e-7, and bf16
    # keeps 8 mantissa bits: a weight of magnitude 0.02 has a spacing of ~7.6e-5, so an
    # AdamW step of ~lr rounds to nothing. The failure mode is worse than "no learning"
    # -- small-magnitude weights (norms, some biases) have fine enough spacing to move,
    # so the loss curve does change while most of the model is frozen, which looks like
    # a working run. Under FSDP the params are fp32 and compute is bf16 via
    # MixedPrecisionPolicy, so fp32 masters cost nothing in matmul throughput.
    runtime_warnings: list[str] = []
    load_dtype = torch.float32 if args.param_dtype == "fp32" else torch.bfloat16
    if args.param_dtype == "bf16" and args.lr < 1e-5:
        runtime_warnings.append(
            f"--param-dtype bf16 with --lr {args.lr:.1e}: an update of that size is below "
            f"the bf16 spacing of a typical weight, so most parameters cannot change at "
            f"all while a few small ones can. A falling loss here is not evidence the "
            f"model trained. Use fp32 masters, or raise the lr past ~1e-5."
        )
    model, auto_class = load_model(args.model_path, dtype=load_dtype)
    log0(rank, f"[model] {args.model_path} via {auto_class}, params {load_dtype}")
    for line in runtime_warnings:
        log0(rank, "WARNING: " + line)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if world_size > 1:
        model = shard_model(model, param_dtype=torch.bfloat16, world_size=world_size)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.0)
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.lr * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine)

    def sides(row) -> tuple[list[int], list[int], list[int], list[int]]:
        return (list(map(int, row.chosen_input_ids)), list(map(int, row.chosen_loss_mask)),
                list(map(int, row.rejected_input_ids)), list(map(int, row.rejected_loss_mask)))

    def norm(value, n: int):
        # fp64 before the division, so the policy side and the reference side (which
        # arrives from parquet as a Python float) go through identical arithmetic. In
        # fp32 the two divisions differ in the last bit, which is enough to give a
        # random sign to a margin that ought to be exactly zero.
        value = as_f64(value)
        return value / n if args.length_normalize else value

    def policy_logps(row, *, grad: bool):
        c_ids, c_mask, r_ids, r_mask = sides(row)
        c_logp, c_n = masked_logprob_sum(model, c_ids, c_mask, chunk=args.logit_chunk, grad=grad)
        r_logp, r_n = masked_logprob_sum(model, r_ids, r_mask, chunk=args.logit_chunk, grad=grad)
        return c_logp, c_n, r_logp, r_n

    def ref_of(row) -> tuple[float, float]:
        entry = ref_table[row.pair_id]
        return (norm(entry["chosen_ref_logp"], entry["chosen_ref_n"]),
                norm(entry["rejected_ref_logp"], entry["rejected_ref_n"]))

    # ---- gate 3: does the policy reproduce the reference at step 0? ---------
    calib_rows = my_rows.iloc[: min(4, len(my_rows))]
    deltas: list[float] = []
    losses0: list[float] = []
    with torch.no_grad():
        for row in calib_rows.itertuples(index=False):
            c_logp, c_n, r_logp, r_n = policy_logps(row, grad=False)
            entry = ref_table[row.pair_id]
            deltas.append(abs(float(c_logp) - entry["chosen_ref_logp"]) / max(c_n, 1))
            deltas.append(abs(float(r_logp) - entry["rejected_ref_logp"]) / max(r_n, 1))
            ref_c, ref_r = ref_of(row)
            loss, _ = dpo_loss(norm(c_logp, c_n), norm(r_logp, r_n), ref_c, ref_r,
                              beta=args.beta, variant=args.variant,
                              label_smoothing=args.label_smoothing)
            losses0.append(float(loss))
    worst = max(deltas) if deltas else 0.0
    if world_size > 1:
        buffer = torch.tensor([worst], device=device)
        torch.distributed.all_reduce(buffer, op=torch.distributed.ReduceOp.MAX)
        worst = float(buffer.item())
    mean_loss0 = sum(losses0) / max(len(losses0), 1)
    log0(rank, f"[gate 3] step-0 |pi - ref| = {worst:.4g} nats/token (tol "
               f"{args.calibration_tol}); loss = {mean_loss0:.6f} vs log 2 = {LOG2:.6f}")

    # A residual of ~1e-3 nats/token with an exactly-matching setup would be alarming;
    # with a dtype mismatch it is just arithmetic. Say which case this is, because the
    # whole value of the gate is that a reader can tell "0.002 because bf16" apart from
    # "0.002 because something is subtly wrong".
    ref_dtypes = sorted({str(m.get("dtype", "unknown")) for m in ref_manifests})
    compute_dtype = "bf16" if world_size > 1 else args.param_dtype
    dtype_match = ref_dtypes == [compute_dtype]
    if not dtype_match:
        log0(rank, f"[gate 3] note: reference scored in {'/'.join(ref_dtypes)}, policy "
                   f"forward runs in {compute_dtype}, so the residual above is a precision "
                   f"difference (expect ~1e-3 nats/token over a 248k-way softmax), not a "
                   f"mismatch. Score the reference with --dtype {compute_dtype} if you want "
                   f"step-0 loss to land on log 2 exactly.")
    elif worst == 0.0:
        log0(rank, "[gate 3] exact: same dtype, zero residual, loss is log 2 to the bit. "
                   "The policy at step 0 IS the reference.")
    if worst > args.calibration_tol:
        message = (
            f"GATE 3 FAILED (calibration): at initialization the policy differs from the "
            f"reference by {worst:.4g} nats/token, tolerance {args.calibration_tol}. The "
            f"DPO loss should be exactly log 2 = {LOG2:.6f} here and is {mean_loss0:.6f}. "
            f"Something changed between the reference pass and now -- the usual causes, in "
            f"order of likelihood: the pairs were rebuilt with a different mask or "
            f"tokenizer; --length-normalize differs from how the reference was summed (it "
            f"must not, the same flag is applied to both); a different dtype or attention "
            f"implementation. Fix it rather than raising the tolerance; --skip-calibration "
            f"exists only for deliberate experiments."
        )
        if not args.skip_calibration:
            sys.exit(message)
        log0(rank, "WARNING: " + message)

    # ---- eval --------------------------------------------------------------
    def evaluate() -> dict | None:
        """Reward accuracy on held-out pairs: how often the policy prefers the winner.

        This is the DPO analogue of a pass rate and must never be reported as one. It
        says the model assigns higher likelihood to a trajectory that succeeded than
        to one that failed on the same task -- not that the model would have solved it.
        """
        if holdout is None or holdout.empty:
            return None
        rows = holdout[holdout.pair_id.isin(ref_table)]
        rows = rows[(rows.chosen_n_tokens <= args.max_seq_len)
                    & (rows.rejected_n_tokens <= args.max_seq_len)]
        if rows.empty:
            return None
        rows = rows.sort_values("pair_id").iloc[: args.eval_pairs]
        mine = rows.iloc[rank::world_size]
        model.eval()
        correct = total = ties = 0
        margin_sum = loss_sum = 0.0
        with torch.no_grad():
            for row in mine.itertuples(index=False):
                c_logp, c_n, r_logp, r_n = policy_logps(row, grad=False)
                ref_c, ref_r = ref_of(row)
                loss, metrics = dpo_loss(norm(c_logp, c_n), norm(r_logp, r_n), ref_c, ref_r,
                                         beta=args.beta, variant=args.variant,
                                         label_smoothing=args.label_smoothing)
                score = rank_score(metrics["reward_margin"])
                correct += score
                ties += int(score == 0.5)
                margin_sum += metrics["reward_margin"]
                loss_sum += float(loss)
                total += 1
        model.train()
        stats = torch.tensor([correct, total, margin_sum, loss_sum, ties], device=device,
                             dtype=torch.float64)
        if world_size > 1:
            torch.distributed.all_reduce(stats)
        correct, total, margin_sum, loss_sum, ties = stats.tolist()
        if total == 0:
            return None
        # holdout_ties matters for reading the accuracy: 0.5 from all-ties (the policy
        # has no preference yet) and 0.5 from half-right-half-wrong (it has preferences
        # and they are coin flips) are different findings that share a number.
        return {"holdout_pairs": int(total),
                "holdout_reward_accuracy": round(correct / total, 4),
                "holdout_ties": int(ties),
                "holdout_reward_margin": round(margin_sum / total, 5),
                "holdout_loss": round(loss_sum / total, 5)}

    baseline_eval = evaluate()
    if baseline_eval:
        log0(rank, f"[eval @0] {json.dumps(baseline_eval, sort_keys=True)}")

    # ---- train -------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "dpo_metrics.jsonl"
    if rank == 0:
        metrics_path.write_text("", encoding="utf-8")
    history: list[dict] = []
    grad_norms: list[float] = []
    started = time.time()
    cursor = 0
    skipped_oom = 0

    for step in range(total_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "reward_chosen": 0.0, "reward_rejected": 0.0,
               "reward_margin": 0.0, "accuracy": 0.0, "chosen_logp_per_token": 0.0,
               "rejected_logp_per_token": 0.0}
        counted = 0

        for _ in range(args.grad_accum):
            row = my_rows.iloc[cursor % len(my_rows)]
            cursor += 1
            scale = 1.0 / args.grad_accum
            ref_c, ref_r = ref_of(row)
            try:
                if args.no_split_backward:
                    c_logp, c_n, r_logp, r_n = policy_logps(row, grad=True)
                    loss, metrics = dpo_loss(norm(c_logp, c_n), norm(r_logp, r_n),
                                             ref_c, ref_r, beta=args.beta,
                                             variant=args.variant,
                                             label_smoothing=args.label_smoothing)
                    (loss * scale).backward()
                    logp_c, logp_r = float(c_logp) / c_n, float(r_logp) / r_n
                else:
                    # Coefficient first, from two cheap no_grad forwards; then each
                    # side backpropagates alone. Same gradient, half the activations.
                    with torch.no_grad():
                        c_logp, c_n, r_logp, r_n = policy_logps(row, grad=False)
                        loss, metrics = dpo_loss(norm(c_logp, c_n), norm(r_logp, r_n),
                                                 ref_c, ref_r, beta=args.beta,
                                                 variant=args.variant,
                                                 label_smoothing=args.label_smoothing)
                        z = metrics["logits"]
                        if args.variant == "ipo":
                            base = 2.0 * (z - 1.0 / (2.0 * args.beta))
                        else:
                            sig = 1.0 / (1.0 + math.exp(min(max(args.beta * z, -60), 60)))
                            smoothing = args.label_smoothing
                            base = -args.beta * (sig * (1.0 - smoothing)
                                                 - (1.0 - sig) * smoothing)
                        logp_c, logp_r = float(c_logp) / c_n, float(r_logp) / r_n
                    c_ids, c_mask, r_ids, r_mask = sides(row)
                    for ids, mask, sign, n in ((c_ids, c_mask, +1.0, c_n),
                                               (r_ids, r_mask, -1.0, r_n)):
                        logp, _ = masked_logprob_sum(model, ids, mask,
                                                     chunk=args.logit_chunk, grad=True)
                        coefficient = base * sign * scale
                        if args.length_normalize:
                            coefficient /= n
                        logp.backward(torch.tensor(coefficient, device=logp.device,
                                                   dtype=logp.dtype))
                        del logp
            except torch.cuda.OutOfMemoryError:
                # One pair is not worth aborting a run, but a silent skip would be a
                # quietly different dataset. Counted here and reported in the summary.
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                skipped_oom += 1
                log0(rank, f"[step {step}] OOM on pair {row.pair_id[:12]} "
                           f"({row.chosen_n_tokens}/{row.rejected_n_tokens} tokens); skipped")
                continue

            agg["loss"] += float(loss)
            agg["reward_chosen"] += metrics["reward_chosen"]
            agg["reward_rejected"] += metrics["reward_rejected"]
            agg["reward_margin"] += metrics["reward_margin"]
            agg["accuracy"] += rank_score(metrics["reward_margin"])
            agg["chosen_logp_per_token"] += logp_c
            agg["rejected_logp_per_token"] += logp_r
            counted += 1

        # The pre-clip norm, recorded every step. The objective sums logprobs over
        # thousands of supervised tokens per side, so this is routinely ~1e3 against a
        # default --max-grad-norm of 1.0 -- i.e. the clip is not a safety net that fires
        # on outliers, it is the entire step-size schedule, and --lr only sets the
        # direction's length. That is a legitimate way to run, but it has to be visible:
        # the summary reports how often clipping was active so nobody reads the lr
        # schedule as the effective one.
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                         args.max_grad_norm))
        grad_norms.append(grad_norm)
        optimizer.step()

        payload = torch.tensor([counted] + [agg[k] for k in sorted(agg)], device=device,
                               dtype=torch.float64)
        if world_size > 1:
            torch.distributed.all_reduce(payload)
        values = payload.tolist()
        n = max(values[0], 1.0)
        record = {"step": step, "lr": lr_at(step), "grad_norm": round(grad_norm, 4),
                  "pairs": int(values[0]),
                  **{k: round(v / n, 5) for k, v in zip(sorted(agg), values[1:])}}
        record["elapsed_min"] = round((time.time() - started) / 60, 2)
        history.append(record)
        if rank == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"[step {step + 1}/{total_steps}] loss {record['loss']:.4f} "
                  f"margin {record['reward_margin']:+.4f} acc {record['accuracy']:.2f} "
                  f"|g| {record['grad_norm']:.3f} lr {record['lr']:.2e}", flush=True)

        if args.eval_every and (step + 1) % args.eval_every == 0:
            mid = evaluate()
            if mid:
                mid["step"] = step
                history.append(mid)
                log0(rank, f"[eval @{step + 1}] {json.dumps(mid, sort_keys=True)}")
        if args.save_every and (step + 1) % args.save_every == 0:
            save_hf(model, args.model_path, args.out / f"hf-step{step + 1}",
                    rank=rank, world_size=world_size)

    final_eval = evaluate()
    if final_eval:
        log0(rank, f"[eval final] {json.dumps(final_eval, sort_keys=True)}")
    save_hf(model, args.model_path, args.out / "hf", rank=rank, world_size=world_size)

    # ---- summary -----------------------------------------------------------
    if rank == 0:
        steps = [h for h in history if "step" in h and "loss" in h]
        first = steps[0] if steps else {}
        last = steps[-1] if steps else {}
        train_manifest = args.pairs / "manifest.json" if args.pairs.is_dir() else None
        pair_manifest = json.loads(train_manifest.read_text(encoding="utf-8")) \
            if train_manifest and train_manifest.is_file() else {}

        ordered = sorted(grad_norms)
        clipped = sum(1 for g in grad_norms if g > args.max_grad_norm)
        clip_fraction = clipped / len(grad_norms) if grad_norms else 0.0

        def quantile(values: list[float], q: float) -> float | None:
            if not values:
                return None
            return round(values[min(len(values) - 1, int(q * len(values)))], 4)

        optimization = {
            "max_grad_norm": args.max_grad_norm,
            "grad_norm_p50": quantile(ordered, 0.5),
            "grad_norm_p90": quantile(ordered, 0.9),
            "grad_norm_max": round(max(ordered), 4) if ordered else None,
            "clip_active_fraction": round(clip_fraction, 4),
            "param_dtype": args.param_dtype,
            "lr": args.lr,
        }
        if clip_fraction > 0.9 and grad_norms:
            runtime_warnings.append(
                f"gradient clipping was active on {clip_fraction:.0%} of steps (median "
                f"pre-clip norm {quantile(ordered, 0.5)} vs --max-grad-norm "
                f"{args.max_grad_norm}). The effective step size is therefore set by the "
                f"clip, not by --lr, and the cosine schedule only rescales an "
                f"already-normalized direction. This is expected when the logprob sums run "
                f"over thousands of tokens; if you want --lr to mean what it usually means, "
                f"pass --length-normalize (per-token objective, norms drop by ~1e3) or raise "
                f"--max-grad-norm above the median. Either way, report which one you used."
            )
        summary = {
            "what_this_is": "DPO on logged RST trajectories. NOT on-policy RL: it "
                            "reweights behaviour present in the data, which came from "
                            "other policies. Do not report as GRPO/RL results.",
            "model_path": str(args.model_path),
            "checkpoint_fingerprint": fingerprint,
            "config": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in sorted(vars(args).items())},
            "gates": {
                "fingerprint_match": True,
                "ref_coverage_missing": 0,
                "step0_abs_delta_nats_per_token": round(worst, 6),
                "step0_loss": round(mean_loss0, 6),
                "step0_expected_loss_log2": round(LOG2, 6),
                "calibration_passed": bool(worst <= args.calibration_tol),
                "calibration_skipped": bool(args.skip_calibration),
                "ref_dtype": ref_dtypes,
                "policy_compute_dtype": compute_dtype,
                "dtype_match": dtype_match,
            },
            "data": {
                "pairs_train_used": int(usable),
                "pairs_skipped_too_long": int(len(too_long)),
                "pairs_skipped_oom": int(skipped_oom),
                "steps": len(steps),
                "grad_accum": args.grad_accum,
                "world_size": world_size,
                "length_bias_warning_from_builder": pair_manifest.get("length_bias_warning"),
                "length_normalize": bool(args.length_normalize),
            },
            "optimization": optimization,
            "warnings": runtime_warnings,
            "metrics": {
                "first_step": first,
                "last_step": last,
                "holdout_before": baseline_eval,
                "holdout_after": final_eval,
            },
            "how_to_read_this": [
                "step0_loss must be ~0.693147 (log 2). It is the proof that the frozen "
                "reference and the policy at initialization are the same model. It lands "
                "there to the bit when gates.dtype_match is true; otherwise the residual "
                "is the bf16/fp32 difference, ~1e-3 nats/token.",
                "reward_margin rising above 0 means the policy prefers verifier-approved "
                "trajectories more than the reference did. That is the objective working, "
                "not evidence of task competence.",
                "holdout_reward_accuracy is likelihood ranking on held-out task GROUPS. "
                "It is not a pass rate and cannot be compared to terminal-bench numbers. "
                "0.5 is the no-preference value, not chance-from-nothing: an exact tie "
                "scores 0.5, which is why step 0 reads 0.5 and not 0.",
                "If the builder reported a length_bias_warning and --length-normalize was "
                "off, a rising margin may be the model learning to be brief.",
                "optimization.clip_active_fraction near 1.0 means --max-grad-norm, not "
                "--lr, set the step size. Quote the clip when describing the schedule.",
            ],
        }
        path = args.out / "dpo_training_summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\n[done] {len(steps)} steps in {(time.time() - started) / 60:.1f} min")
        print(f"       checkpoint : {args.out / 'hf'}")
        print(f"       summary    : {path}")
        print(f"       grad norm  : p50 {optimization['grad_norm_p50']} "
              f"max {optimization['grad_norm_max']}, clipped on "
              f"{clip_fraction:.0%} of steps at --max-grad-norm {args.max_grad_norm}")
        if final_eval and baseline_eval:
            print(f"       holdout reward accuracy "
                  f"{baseline_eval['holdout_reward_accuracy']:.3f} -> "
                  f"{final_eval['holdout_reward_accuracy']:.3f} "
                  f"(likelihood ranking, 0.5 = no preference, NOT a pass rate)")
        for line in runtime_warnings:
            print(f"       WARNING: {line}")

    if world_size > 1:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
