#!/usr/bin/env python3
"""Resolve a model key + detected hardware into a validated launch config.

    python scripts/model_registry.py --list
    python scripts/model_registry.py --key qwen3.5-9b --mem-class 80GB --gpus 32 --shell
    eval "$(python scripts/model_registry.py --key "$MODEL_KEY" --mem-class "$MEM_CLASS" \
             --gpus 32 --gpus-per-node 8 --max-seq-len 32768 --shell)"

The point of resolving in one place is that the invariants get checked once,
loudly, instead of being re-derived by hand in every launcher:

  * tp * pp * cp * dp == total gpus            (dp is derived, never guessed)
  * tp <= gpus_per_node                        (a tensor-parallel group must not
                                                cross a node boundary)
  * max_tokens_per_gpu * cp >= max_seq_len     (else the longest sequence in the
                                                dataset cannot be placed at all)
  * ep <= cp * dp                              (expert parallelism lives inside
                                                the non-TP/PP dimensions)

Exits non-zero with an explanation when a config is impossible, so a launcher
fails at the config step rather than 40 minutes into a run.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "configs" / "models.json"


def load() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def resolve(key: str, mem_class: str, gpus: int, gpus_per_node: int, max_seq_len: int,
            phase: str = "sft") -> dict:
    reg = load()
    models, defaults = reg["models"], reg.get("defaults", {})
    if key not in models:
        sys.exit(f"unknown model key {key!r}. Known: {', '.join(sorted(models))}")
    m = models[key]

    # RL colocates the rollout engine with the actor, so it gets its own rows when
    # present. Falling back to the SFT rows is allowed but flagged: it has not been
    # sized for a resident SGLang engine.
    rl_fallback = ""
    if phase == "rl":
        par = m.get("rl_parallelism")
        if not par:
            par = m["parallelism"]
            rl_fallback = (f"{key} has no rl_parallelism rows; falling back to the SFT rows. "
                           f"Those were NOT sized for a colocated rollout engine -- expect to "
                           f"lower max_tokens_per_gpu or raise CP.")
    else:
        par = m["parallelism"]
    if mem_class not in par:
        # 40GB-alt and friends fall back to the 40GB row
        fallback = "40GB" if mem_class.startswith("40") else "80GB"
        if fallback not in par:
            sys.exit(f"model {key} has no parallelism row for mem_class {mem_class!r}")
        print(f"# note: no '{mem_class}' row for {key}; using '{fallback}'", file=sys.stderr)
        mem_class = fallback
    p = dict(par[mem_class])

    tp, pp, cp = p["tp"], p["pp"], p["cp"]
    mtpg = p["max_tokens_per_gpu"]
    ep = p.get("ep", 1)

    # If the model is small enough to need fewer GPUs than the cluster has, we still
    # use them all via data parallelism -- dp absorbs the remainder.
    denom = tp * pp * cp
    if gpus % denom:
        sys.exit(f"{key}/{mem_class}: tp*pp*cp={denom} does not divide {gpus} GPUs. "
                 f"Adjust configs/models.json or pass a different --gpus.")
    dp = gpus // denom

    problems: list[str] = []
    if tp > gpus_per_node:
        problems.append(f"tp={tp} exceeds gpus_per_node={gpus_per_node}: a TP group would "
                        f"cross a node boundary, which is very slow (and impossible without NVLink)")
    if mtpg * cp < max_seq_len:
        problems.append(f"max_tokens_per_gpu*cp = {mtpg}*{cp} = {mtpg*cp} < max_seq_len={max_seq_len}: "
                        f"the longest sequence cannot be placed. Raise cp or max_tokens_per_gpu.")
    if m.get("moe") and ep > cp * dp:
        problems.append(f"ep={ep} > cp*dp={cp*dp}: expert parallelism must fit inside the "
                        f"non-TP/PP dimensions")
    if problems:
        sys.exit(f"invalid config for {key} @ {mem_class} on {gpus} GPUs:\n  - " + "\n  - ".join(problems))

    out = {
        "PHASE": phase,
        "MODEL_KEY": key,
        "HF_REPO": m["hf_repo"],
        "MODEL_DIR_NAME": m["hf_repo"].split("/")[-1],
        "SLIME_SPEC": m["slime_spec"],
        "LOSS_MASK_TYPE": m["loss_mask_type"],
        "PARAMS_B": m["params_b"],
        "N_LAYERS": m["n_layers"],
        "HAS_VISION": int(bool(m.get("has_vision"))),
        "HYBRID_ATTENTION": int(bool(m.get("hybrid_attention"))),
        "IS_MOE": int(bool(m.get("moe"))),
        "MEM_CLASS_USED": mem_class,
        "TP": tp, "PP": pp, "CP": cp, "DP": dp, "EP": ep,
        "MAX_TOKENS_PER_GPU": mtpg,
        "MAX_SEQ_LEN": max_seq_len,
        "TOTAL_GPUS": gpus,
        "SERVE_TP": m.get("serve_tp", tp),
        "SERVE_ENABLE_THINKING": int(bool(m.get("serve_enable_thinking"))),
        "SERVE_CONTEXT_LENGTH": defaults.get("serve_context_length", 65536),
        "GLOBAL_BATCH_SIZE": defaults.get("global_batch_size", 128),
        "NUM_EPOCH": defaults.get("num_epoch", 1),
        "LR": defaults.get("lr", "3e-6"),
        "MIN_LR": defaults.get("min_lr", "3e-7"),
        "LR_WARMUP_FRACTION": defaults.get("lr_warmup_fraction", 0.03),
        "EST_EPOCH_MINUTES": m.get("est_epoch_minutes", 0),
        "REFERENCE_CHECKPOINT": m.get("reference_checkpoint", ""),
        "DECODER_LAST_PP_LAYERS": m.get("decoder_last_pipeline_num_layers", 0),
        "MIN_GPUS": m.get("min_gpus", gpus),
        "SERVE_CHAT_TEMPLATE_REPO": m.get("serve_chat_template_repo", ""),
        "FIRST_BATCH": int(bool(m.get("first_batch"))),
        "ROLLOUT_GPUS_PER_ENGINE": p.get("rollout_gpus_per_engine", 2),
        "ROLLOUT_MAX_RESPONSE_LEN": p.get("rollout_max_response_len", 16384),
    }
    extra = list(m.get("moe_flags") or []) + list(m.get("moe_optional_flags") or [])
    out["MOE_FLAGS"] = " ".join(extra)
    unvalidated = m.get("unvalidated", "")
    if rl_fallback:
        unvalidated = (unvalidated + " " + rl_fallback).strip()
    out["_unvalidated"] = unvalidated
    out["_role"] = m.get("role", "")
    out["_paper_reference"] = m.get("paper_reference")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key")
    ap.add_argument("--mem-class", default="80GB")
    ap.add_argument("--gpus", type=int, default=32)
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--phase", default="sft", choices=["sft", "rl"])
    ap.add_argument("--shell", action="store_true", help="emit eval-able shell assignments")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        reg = load()
        print(f"{'key':20s} {'params':>8s} {'bf16':>8s} {'layers':>7s} {'moe':>4s} "
              f"{'min_gpus':>9s} {'~min/epoch':>11s}  role")
        for k, m in reg["models"].items():
            print(f"{k:20s} {m['params_b']:7.2f}B {m['bf16_gib']:6.1f}GiB {m['n_layers']:7d} "
                  f"{'yes' if m.get('moe') else 'no':>4s} {m.get('min_gpus',32):9d} "
                  f"{m.get('est_epoch_minutes',0):11d}  {m.get('role','')[:52]}")
        print("\nAll entries share one tokenizer and one training-render, so the published")
        print("cap10/cap8 datasets and --loss-mask-type qwen3_5 apply unchanged to all of them.")
        return 0

    if not args.key:
        sys.exit("--key required (or --list)")
    cfg = resolve(args.key, args.mem_class, args.gpus, args.gpus_per_node, args.max_seq_len,
                  phase=args.phase)

    if args.json:
        print(json.dumps(cfg, indent=2))
        return 0
    if args.shell:
        if cfg["_unvalidated"]:
            print(f"# WARNING [{cfg['MODEL_KEY']}]: {cfg['_unvalidated']}", file=sys.stderr)
        for k, v in cfg.items():
            if k.startswith("_"):
                continue
            print(f"export {k}={shlex.quote(str(v))}")
        return 0

    print(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, indent=2))
    if cfg["_unvalidated"]:
        print(f"\nUNVALIDATED: {cfg['_unvalidated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
