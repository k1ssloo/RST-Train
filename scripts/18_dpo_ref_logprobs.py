#!/usr/bin/env python3
"""Precompute frozen reference logprobs for the DPO pairs. One pass, then throw the
reference model away.

    # single GPU
    python scripts/18_dpo_ref_logprobs.py \
        --pairs $BASE_FOLDER/dpo-v1 --model-path $BASE_FOLDER/out-hf-full \
        --out $BASE_FOLDER/dpo-v1/ref

    # one process per GPU, no communication needed
    for i in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$i python scripts/18_dpo_ref_logprobs.py \
        --pairs $BASE_FOLDER/dpo-v1 --model-path $BASE_FOLDER/out-hf-full \
        --out $BASE_FOLDER/dpo-v1/ref --shard $i --num-shards 8 &
    done; wait

WHY PRECOMPUTE
    DPO needs log pi_ref for both sides of every pair. The textbook implementation
    keeps a second frozen copy of the model resident, which for a 27.8 B policy
    means another ~55.6 GB of bf16 weights competing with the optimizer state. But
    the reference is frozen and the data is fixed, so its logprobs are CONSTANTS.
    Computing them once removes the reference model from training entirely.

    It also makes the correctness check below possible: with the reference logprobs
    on disk, `19_train_dpo.py` can verify at step 0 that the policy reproduces them,
    which catches a mask change, a tokenizer change, or the wrong checkpoint before
    a single gradient is applied.

WHAT WOULD SILENTLY GO WRONG, AND WHAT STOPS IT
    * Wrong checkpoint. Reference logprobs are only valid for the exact weights that
      produced them, and `out-hf-full` is overwritten by every export. The manifest
      records `checkpoint_fingerprint()` (config + size + leading bytes of every
      weight shard) and the trainer refuses a mismatch.
    * Different mask. The sum is over `loss_mask == 1` only. This script asserts the
      supervised-token count it sees equals the count the builder wrote, so a
      re-tokenized or re-masked dataset cannot be silently mixed with old logprobs.
    * bf16 softmax drift. The logits are cast to fp32 before the cross-entropy. Over
      thousands of positions a bf16 logsumexp accumulates enough error to move a
      whole-sequence logprob by nats, and DPO subtracts two of these sums.
    * Nondeterminism. `--self-check` recomputes a few rows and reports the maximum
      absolute difference; anything above ~1e-3 nats means the forward pass is not
      reproducible on this hardware, and the step-0 gate downstream will be noise.

OUTPUT
    <out>/ref_logps[_shard<N>].parquet          pair_id + 4 columns
    <out>/ref_logps[_shard<N>]_manifest.json    fingerprint, dtype, self-check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dpo_common import checkpoint_fingerprint, load_model, masked_logprob_sum  # noqa: E402

DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def load_pairs(root: Path, split: str):
    """Read the pair files as one frame, tagging which split each row came from."""
    import pandas as pd

    if root.is_file():
        frame = pd.read_parquet(root)
        frame["split"] = split if split != "both" else "unknown"
        return frame
    wanted = ["dpo_train.parquet", "dpo_holdout.parquet"] if split == "both" else \
             [f"dpo_{split}.parquet"]
    frames = []
    for name in wanted:
        path = root / name
        if not path.is_file():
            if split == "both":
                continue
            sys.exit(f"missing {path}")
        part = pd.read_parquet(path)
        part["split"] = name.removeprefix("dpo_").removesuffix(".parquet")
        frames.append(part)
    if not frames:
        sys.exit(f"no dpo_*.parquet under {root}")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True,
                    help="dir written by 17_build_dpo_data.py, or a single parquet")
    ap.add_argument("--split", default="both", choices=["both", "train", "holdout"])
    ap.add_argument("--model-path", type=Path, required=True,
                    help="the REFERENCE = the checkpoint DPO will start from")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    ap.add_argument("--logit-chunk", type=int, default=512,
                    help="supervised positions scored per LM-head slice")
    ap.add_argument("--max-seq-len", type=int, default=32768,
                    help="skip pairs longer than this; pass the SAME value to "
                         "19_train_dpo.py, or its coverage gate will trip on the pairs "
                         "skipped here")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows (smoke tests)")
    ap.add_argument("--self-check", type=int, default=2,
                    help="rows to score twice as a determinism probe")
    ap.add_argument("--flush-every", type=int, default=50,
                    help="write partial results this often, so a long run is resumable")
    args = ap.parse_args()

    import pandas as pd
    import torch

    frame = load_pairs(args.pairs, args.split)
    long_rows = int(((frame.chosen_n_tokens > args.max_seq_len)
                     | (frame.rejected_n_tokens > args.max_seq_len)).sum())
    if long_rows:
        print(f"[plan] skipping {long_rows:,} pairs over --max-seq-len "
              f"{args.max_seq_len:,}; pass the same value to 19_train_dpo.py", flush=True)
        frame = frame[(frame.chosen_n_tokens <= args.max_seq_len)
                      & (frame.rejected_n_tokens <= args.max_seq_len)].reset_index(drop=True)
    if args.num_shards > 1:
        frame = frame.iloc[args.shard :: args.num_shards].reset_index(drop=True)
    if args.limit:
        frame = frame.iloc[: args.limit].reset_index(drop=True)
    if frame.empty:
        sys.exit("no rows to score")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = "" if args.num_shards == 1 else f"_shard{args.shard}"
    out_path = args.out / f"ref_logps{tag}.parquet"
    manifest_path = args.out / f"ref_logps{tag}_manifest.json"

    # Resume: a partial file is a resume point, not garbage to overwrite. A 27 B
    # reference pass over thousands of 32k-token pairs is hours of GPU time.
    done: dict[str, dict] = {}
    if out_path.is_file():
        prior = pd.read_parquet(out_path)
        done = {row.pair_id: row._asdict() for row in prior.itertuples(index=False)}
        print(f"[resume] {len(done):,} rows already scored in {out_path}", flush=True)

    todo = frame[~frame.pair_id.isin(done)].reset_index(drop=True)
    print(f"[plan] {len(todo):,} of {len(frame):,} rows to score "
          f"(shard {args.shard}/{args.num_shards})", flush=True)

    fingerprint = checkpoint_fingerprint(args.model_path)
    print(f"[model] {args.model_path} fingerprint={fingerprint[:16]}...", flush=True)

    if todo.empty:
        print("[done] nothing to do")
        return 0

    model, auto_class = load_model(args.model_path,
                                   dtype=getattr(torch, DTYPES[args.dtype]),
                                   device_map="cuda")
    model.eval()
    model.config.use_cache = False
    print(f"[model] loaded with {auto_class}", flush=True)

    def score(ids, mask, *, expect: int, side: str, pair_id: str) -> float:
        logp, n = masked_logprob_sum(model, list(int(i) for i in ids),
                                     list(int(m) for m in mask), chunk=args.logit_chunk)
        if n != int(expect):
            sys.exit(
                f"pair {pair_id} {side}: scored {n} supervised tokens but the dataset "
                f"says {int(expect)}. The mask this script used and the mask the "
                f"builder wrote disagree, so these logprobs would be for a different "
                f"objective than the one being trained. Rebuild the pairs and rerun."
            )
        return float(logp)

    rows: list[dict] = list(done.values())
    checks: list[float] = []
    started = time.time()

    def flush() -> None:
        pd.DataFrame(rows).to_parquet(out_path, index=False)

    for i, row in enumerate(todo.itertuples(index=False)):
        c_logp = score(row.chosen_input_ids, row.chosen_loss_mask,
                       expect=row.chosen_n_trained, side="chosen", pair_id=row.pair_id)
        r_logp = score(row.rejected_input_ids, row.rejected_loss_mask,
                       expect=row.rejected_n_trained, side="rejected", pair_id=row.pair_id)
        if len(checks) < args.self_check:
            again = score(row.chosen_input_ids, row.chosen_loss_mask,
                          expect=row.chosen_n_trained, side="chosen(recheck)",
                          pair_id=row.pair_id)
            checks.append(abs(again - c_logp))
        rows.append({
            "pair_id": row.pair_id,
            "split": row.split,
            "chosen_ref_logp": c_logp,
            "rejected_ref_logp": r_logp,
            "chosen_ref_n": int(row.chosen_n_trained),
            "rejected_ref_n": int(row.rejected_n_trained),
        })
        if (i + 1) % args.flush_every == 0:
            flush()
            rate = (i + 1) / max(time.time() - started, 1e-6)
            eta = (len(todo) - i - 1) / max(rate, 1e-9) / 60
            print(f"[{i + 1:,}/{len(todo):,}] {rate * 60:.1f} pairs/min "
                  f"eta {eta:.0f} min  last: chosen {c_logp / max(row.chosen_n_trained, 1):.3f} "
                  f"rejected {r_logp / max(row.rejected_n_trained, 1):.3f} nats/token",
                  flush=True)
    flush()

    # ---- plausibility, stated as numbers rather than trusted silently ---------
    scored = pd.DataFrame(rows)
    c_per_tok = (scored.chosen_ref_logp / scored.chosen_ref_n).astype(float)
    r_per_tok = (scored.rejected_ref_logp / scored.rejected_ref_n).astype(float)
    determinism = max(checks) if checks else None
    warnings: list[str] = []
    if determinism is not None and determinism > 1e-3:
        warnings.append(
            f"recomputing the same row moved its logprob by {determinism:.3g} nats. "
            f"The forward pass is not deterministic here, so the step-0 calibration "
            f"gate in 19_train_dpo.py will measure jitter as well as correctness. "
            f"Consider raising its --calibration-tol, and do not read small reward "
            f"margins as signal."
        )
    if float(c_per_tok.mean()) < -3.0:
        warnings.append(
            f"mean {float(c_per_tok.mean()):.2f} nats/token on the CHOSEN side is very "
            f"low for a model that was fine-tuned on this data's own distribution. "
            f"Check that --model-path is the SFT checkpoint and not the base model."
        )

    manifest = {
        "pairs_source": str(args.pairs),
        "split": args.split,
        "model_path": str(args.model_path),
        "checkpoint_fingerprint": fingerprint,
        "dtype": args.dtype,
        "auto_class": auto_class,
        "logit_chunk": args.logit_chunk,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "max_seq_len": args.max_seq_len,
        "pairs_skipped_too_long": long_rows,
        "rows": int(len(scored)),
        "torch_version": torch.__version__,
        "determinism_max_abs_nats": determinism,
        "logp_per_token": {
            "chosen_mean": round(float(c_per_tok.mean()), 4),
            "chosen_p05": round(float(c_per_tok.quantile(0.05)), 4),
            "rejected_mean": round(float(r_per_tok.mean()), 4),
            "rejected_p05": round(float(r_per_tok.quantile(0.05)), 4),
        },
        "warnings": warnings,
        "note": "logp is a SUM over loss_mask==1 positions, natural log, fp32 softmax. "
                "These are constants for this (checkpoint, dataset) pair only.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")

    print(f"\n[done] {len(scored):,} rows -> {out_path}")
    print(f"       chosen   {float(c_per_tok.mean()):+.4f} nats/token (mean)")
    print(f"       rejected {float(r_per_tok.mean()):+.4f} nats/token (mean)")
    if determinism is not None:
        print(f"       determinism probe: max |delta| = {determinism:.3g} nats")
    for line in warnings:
        print(f"       WARNING: {line}")
    print(f"       manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
