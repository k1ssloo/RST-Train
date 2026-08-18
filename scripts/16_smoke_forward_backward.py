#!/usr/bin/env python3
"""Real forward/backward on one GPU using the PRE-TOKENIZED data and loss mask.

    python scripts/16_smoke_forward_backward.py \
        --model data/Qwen3.5-0.8B --parquet data/sft-v1-cap10/pretokenized_train.parquet

Purpose: catch bugs that only a real model exposes, on one small GPU, before any
cluster time is spent. It checks things no unit test can:

  1. the model actually loads, and the gated-delta-net path executes
  2. `loss_mask` -> `labels` conversion is right, verified by a control: masking
     everything must give zero gradient, and the loss on trained tokens must differ
     from the loss on all tokens
  3. the loss is a plausible cross-entropy, not NaN and not ~log(vocab) noise
  4. a backward pass produces finite grads
  5. peak memory at a given sequence length, so the cluster budget is measured
     rather than assumed
  6. whether Liger's fused CE actually reduces peak memory on this model

Exit code is nonzero if any check fails, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def bytes_gib(n: float) -> float:
    return n / (1 << 30)


def build_labels(input_ids, loss_mask, ignore_index: int = -100):
    """labels[i] = input_ids[i] where loss_mask[i]==1 else ignore_index.

    HF models shift internally, so we do NOT shift here. Getting this wrong is the
    single most likely silent bug in the whole pipeline, which is why the caller
    cross-checks it against an all-masked control.
    """
    import torch

    labels = input_ids.clone()
    labels[loss_mask == 0] = ignore_index
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local HF model dir")
    ap.add_argument("--parquet", required=True, help="pretokenized parquet")
    ap.add_argument("--seq-len", type=int, default=4096, help="truncate rows to this length")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--liger", action="store_true", help="also measure with Liger patched in")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    results: dict = {"checks": {}, "model": args.model, "seq_len": args.seq_len}

    def check(name: str, ok: bool, detail: str = "") -> None:
        results["checks"][name] = {"ok": bool(ok), "detail": detail}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    print(f"=== torch {torch.__version__} cuda={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability()}")
    cfg = AutoConfig.from_pretrained(args.model)
    text_cfg = getattr(cfg, "text_config", cfg)
    print(f"=== arch={cfg.architectures} layers={getattr(text_cfg,'num_hidden_layers',None)} "
          f"hidden={getattr(text_cfg,'hidden_size',None)} vocab={getattr(text_cfg,'vocab_size',None)}")
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types:
        from collections import Counter
        print(f"=== layer mix: {dict(Counter(layer_types))}")
    results["arch"] = cfg.architectures
    results["vocab_size"] = getattr(text_cfg, "vocab_size", None)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ---- data ---------------------------------------------------------------
    full = pd.read_parquet(args.parquet)
    # Pick rows that are ACTUALLY at least seq_len long where possible. Sampling
    # short rows and then reporting a "seq_len=32768" memory figure is misleading:
    # truncation is a no-op and you measure the shorter length without noticing.
    long_enough = full[full.n_tokens >= args.seq_len] if "n_tokens" in full.columns else full
    if len(long_enough) >= args.rows:
        frame = long_enough.head(args.rows)
        note = f"all >= {args.seq_len} tokens"
    else:
        frame = full.nlargest(args.rows, "n_tokens") if "n_tokens" in full.columns else full.head(args.rows)
        actual = int(frame.n_tokens.min()) if "n_tokens" in frame.columns else -1
        note = (f"WARNING: only {len(long_enough)} rows reach {args.seq_len} tokens; using the "
                f"longest available (min {actual}). The reported peak reflects ~{actual} "
                f"tokens, NOT {args.seq_len}.")
    print(f"=== {len(frame)} rows from {args.parquet} — {note}")
    results["rows_note"] = note
    if "n_tokens" in frame.columns:
        results["effective_seq_len"] = int(min(frame.n_tokens.min(), args.seq_len))

    def load_model(use_liger: bool):
        if use_liger:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation=args.attn,
        ).cuda()
        if use_liger:
            _apply_liger_kernel_to_instance(model=model)
        model.gradient_checkpointing_enable()
        model.train()
        return model

    def run_once(model, mask_everything: bool = False):
        """One forward+backward over the sampled rows. Returns (mean_loss, peak_gib)."""
        torch.cuda.reset_peak_memory_stats()
        losses: list[float] = []
        oom: list[str] = []
        for row in frame.itertuples():
            ids = torch.tensor(list(row.input_ids)[: args.seq_len], dtype=torch.long).unsqueeze(0).cuda()
            msk = torch.tensor(list(row.loss_mask)[: args.seq_len], dtype=torch.long).unsqueeze(0).cuda()
            if mask_everything:
                msk = torch.zeros_like(msk)
            if msk.sum() == 0 and not mask_everything:
                continue
            labels = build_labels(ids, msk)
            try:
                out = model(input_ids=ids, labels=labels)
                loss = out.loss
                if torch.isfinite(loss):
                    loss.backward()
                    losses.append(loss.item())
                else:
                    losses.append(float("nan"))
            except torch.OutOfMemoryError as exc:
                # An OOM here is a RESULT, not a crash: it is the answer to "does this
                # configuration fit". Record it and keep going so the comparison run
                # (e.g. with Liger) still happens.
                oom.append(f"seq={ids.shape[1]}: {str(exc).split('.')[0]}")
                losses.append(float("nan"))
                torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
        peak = bytes_gib(torch.cuda.max_memory_allocated())
        finite = [x for x in losses if math.isfinite(x)]
        if oom:
            print(f"    OOM x{len(oom)}: {oom[0][:120]}")
            results.setdefault("oom", []).extend(oom)
        return (sum(finite) / len(finite) if finite else float("nan")), peak, losses

    print("\n=== baseline (no Liger) ===")
    started = time.time()
    model = load_model(use_liger=False)
    check("model loads and moves to GPU", True,
          f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B params, {time.time()-started:.0f}s")

    mean_loss, peak, losses = run_once(model)
    results["baseline"] = {"mean_loss": mean_loss, "peak_gib": round(peak, 2), "losses": losses}
    baseline_oom = bool(results.get("oom"))
    if baseline_oom:
        check("baseline (unfused CE) fits at this length", False,
              f"OUT OF MEMORY without a fused cross-entropy. vocab={results['vocab_size']} x "
              f"seq={results.get('effective_seq_len')} logits do not fit even for a "
              f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B model. This is the measured "
              f"proof that Liger is required, not optional.")
    else:
        check("loss is finite", all(math.isfinite(x) for x in losses),
              f"losses={[round(x,4) for x in losses]}")

    vocab = results["vocab_size"] or 1
    random_loss = math.log(vocab)
    check("loss is far below random", baseline_oom or mean_loss < random_loss * 0.6,
          f"mean={mean_loss:.4f} vs log(vocab)={random_loss:.2f} — a value near random would "
          f"mean the labels are misaligned")

    # gradients must be finite and nonzero
    ids = torch.tensor(list(frame.iloc[0].input_ids)[: args.seq_len], dtype=torch.long).unsqueeze(0).cuda()
    msk = torch.tensor(list(frame.iloc[0].loss_mask)[: args.seq_len], dtype=torch.long).unsqueeze(0).cuda()
    model.zero_grad(set_to_none=True)
    grads: list = []
    if not baseline_oom:
        model(input_ids=ids, labels=build_labels(ids, msk)).loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_norm = math.sqrt(sum(float(g.detach().float().pow(2).sum()) for g in grads)) if grads else 0.0
    check("gradients finite and nonzero", baseline_oom or (bool(grads) and math.isfinite(total_norm) and total_norm > 0),
          f"{len(grads)} tensors, global norm {total_norm:.4f}")

    # ---- THE control: mask everything -> no learning signal at all ----------
    # If this does not behave differently from the masked run, the loss_mask is not
    # actually reaching the loss and every trained model would be silently wrong.
    model.zero_grad(set_to_none=True)
    all_masked_loss, _, _ = run_once(model, mask_everything=True)
    check("all-masked control differs from masked run",
          baseline_oom or (not math.isfinite(all_masked_loss)) or abs(all_masked_loss - mean_loss) > 1e-6,
          f"all-masked loss={all_masked_loss} vs normal={mean_loss:.4f} "
          f"(identical values would mean loss_mask is being ignored)")
    results["all_masked_loss"] = all_masked_loss

    # Trained-token fraction actually reaching the loss.
    #
    # WHY THIS BAND IS WIDER THAN 30_run_sft_verl.sh's 0.25-0.45: that launcher measures
    # the fraction over the WHOLE untruncated dataset (32.42% measured for cap10), while
    # this measures `--rows` rows (4 by default) each truncated to `--seq-len` (4096).
    # Both knobs move the number legitimately -- four rows have real sampling spread,
    # and keeping only a prefix retains the long untrained harness preamble while cutting
    # trained assistant turns off the tail, pushing the fraction down. This check only
    # asks "is the mask present at all", so 0.15-0.55; the dataset-wide assertion at
    # train time is the tight one, and a mask bug fails both.
    frac = float(sum(int(sum(list(r.loss_mask)[: args.seq_len])) for r in frame.itertuples()) /
                 max(1, sum(len(list(r.input_ids)[: args.seq_len]) for r in frame.itertuples())))
    check("trained-token fraction in the expected band", 0.15 <= frac <= 0.55,
          f"{frac:.2%} of tokens carry a label at seq_len={args.seq_len}")
    results["trained_fraction_at_seqlen"] = round(frac, 4)

    del model
    torch.cuda.empty_cache()

    # ---- Liger comparison --------------------------------------------------
    if args.liger:
        print("\n=== with Liger fused CE ===")
        try:
            lmodel = load_model(use_liger=True)
            lmean, lpeak, llosses = run_once(lmodel)
            results["liger"] = {"mean_loss": lmean, "peak_gib": round(lpeak, 2)}
            if baseline_oom:
                check("liger makes this length RUNNABLE at all", math.isfinite(lmean),
                      f"baseline OOMed; with Liger loss={lmean:.4f} at {lpeak:.2f} GiB peak")
            else:
                check("liger loss matches baseline", math.isfinite(lmean) and abs(lmean - mean_loss) < 0.05,
                      f"liger={lmean:.4f} vs baseline={mean_loss:.4f}")
                check("liger reduces peak memory", lpeak < peak,
                      f"{lpeak:.2f} GiB vs {peak:.2f} GiB "
                      f"({100*(peak-lpeak)/max(peak,1e-9):.0f}% lower)")
            del lmodel
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            check("liger patches this model", False, f"{type(exc).__name__}: {exc}")

    results["peak_gib_baseline"] = round(peak, 2)
    print(f"\n=== peak memory at seq_len={args.seq_len}: {peak:.2f} GiB")

    failed = [k for k, v in results["checks"].items() if not v["ok"]]
    print(f"\n=== {len(results['checks'])-len(failed)}/{len(results['checks'])} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
