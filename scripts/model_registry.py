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

`--backend verl` RESHAPES the row first, because `configs/models.json` describes a
Megatron 3-D parallel layout and verl's FSDP engine has neither pipeline stages nor
context parallelism: PP and CP are pinned to 1 and CP is folded into
max_tokens_per_gpu, since one GPU then holds a whole sequence. Without that, the
verl launchers inherit a divisibility rule they do not obey (a 27B row of
tp4/pp2/cp2 forces the GPU count to be a multiple of 16) and a token budget that is
half the sequence length. See BACKENDS.md.

`--ulysses-sp N` then UNDOES that fold by N, because Ulysses sequence parallelism is
verl's CP: an Ulysses group of N ranks jointly holds one sequence, so the per-GPU
budget is max_seq_len/N and not max_seq_len. Leaving it at 1 while launching with
`engine.ulysses_sequence_parallel_size=N` is the silent-waste case — verl scales the
micro-batch budget by N too (`workers/engine/utils.py::prepare_micro_batches`), so
each GPU still holds a whole sequence's activations and pays N-way SP communication
for nothing. See the ULYSSES_SP block in scripts/30_run_sft_verl.sh.

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
            phase: str = "sft", backend: str = "megatron", ulysses_sp: int = 1) -> dict:
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

    # verl's FSDP engine shards PARAMETERS across the whole world. It has no pipeline
    # stages, and no context parallelism (`engine/fsdp.yaml` has no such key at all --
    # its nearest analogue is ulysses_sequence_parallel_size, which is not validated on
    # the gated-delta-net layers). So the Megatron row's pp and cp are not just unused
    # here, they are actively wrong:
    #
    #   * pp*cp inflates the divisibility rule. qwen3.5-27b/80GB is tp4/pp2/cp2, so
    #     tp*pp*cp=16 and any GPU count that is not a multiple of 16 is rejected --
    #     including the 1- and 2-GPU smoke tests, which need no such shape.
    #   * cp deflates the token budget. max_tokens_per_gpu is per-GPU-per-microbatch;
    #     with cp=2 a 32K sequence is placed as 2x16K. Under FSDP one GPU holds the
    #     whole 32K, so its budget is the row's mtpg*cp. Leaving mtpg at 16384 makes
    #     verl reject (or silently split) every sequence over 16K.
    #
    # Folding cp into mtpg keeps the "longest sequence must be placeable" assert below
    # meaningful instead of accidentally satisfied.
    # TP goes too, and for the same reason: `verl/trainer/config/engine/fsdp.yaml` has no
    # tensor_parallel_size key in verl 0.9.0, so nothing on this path splits a tensor.
    # SERVE_TP is unaffected -- serving is SGLang, not the training engine.
    row_tp = tp
    shape_note = ""
    if backend == "verl":
        if (tp, pp, cp) != (1, 1, 1):
            shape_note = (f"{key}/{mem_class} is a Megatron row (tp{tp} pp{pp} cp{cp}); verl's FSDP "
                          f"engine has no TP, PP or CP, so all three are pinned to 1, DP becomes "
                          f"the whole world, and max_tokens_per_gpu becomes {mtpg}*{cp}="
                          f"{mtpg * cp} (one GPU holds a whole sequence). Activation memory per "
                          f"GPU is therefore ~{cp}x the Megatron row's.")
            print(f"# note: {shape_note}", file=sys.stderr)
        mtpg, tp, pp, cp = mtpg * cp, 1, 1, 1

        # Ulysses SP is verl's context parallelism, so it divides the per-GPU budget the
        # same way CP does above. Doing this here rather than in the launcher is what
        # keeps the "longest sequence must be placeable" assert below honest: the thing
        # that has to hold a whole sequence is the Ulysses GROUP, not one GPU.
        #
        # Skipping it is not a small mis-sizing, it is the whole point of SP thrown away.
        # verl computes `max_token_len = max_token_len_per_gpu * sp_size`, so leaving mtpg
        # at max_seq_len means each GPU still holds max_seq_len tokens of activations at
        # SP=8 exactly as it does at SP=1 -- with 8x the pre-slice embedding tensor on the
        # VLM path (verl pads but does not slice input_ids when the config has a
        # vision_config; the slice happens after embed_tokens) and 48 GDN layers' worth of
        # cross-rank recurrent-state traffic on top.
        if ulysses_sp > 1:
            if gpus % ulysses_sp:
                sys.exit(f"--ulysses-sp {ulysses_sp} does not divide {gpus} GPUs, so "
                         f"dp_size = gpus // sp is not integral and verl cannot build the "
                         f"(dp, sp) device mesh. Pick an sp that divides the world size.")
            per_gpu = -(-mtpg // ulysses_sp)  # ceil, so per_gpu * sp >= the folded budget
            sp_note = (f"ULYSSES_SP={ulysses_sp}: an Ulysses group of {ulysses_sp} ranks jointly "
                       f"holds one sequence, so max_tokens_per_gpu becomes "
                       f"ceil({mtpg}/{ulysses_sp})={per_gpu} and the group budget stays "
                       f"{per_gpu * ulysses_sp} >= max_seq_len. Per-GPU activation memory is "
                       f"~1/{ulysses_sp} of the SP=1 figure; that reduction IS what SP buys, "
                       f"and it only happens because the budget was divided here.")
            print(f"# note: {sp_note}", file=sys.stderr)
            shape_note = f"{shape_note} {sp_note}".strip()
            mtpg = per_gpu
    elif ulysses_sp > 1:
        sys.exit(f"--ulysses-sp {ulysses_sp} is only meaningful for --backend verl; the Megatron "
                 f"path shards the sequence with cp, which this row already sets (cp={cp}).")

    # If the model is small enough to need fewer GPUs than the cluster has, we still
    # use them all via data parallelism -- dp absorbs the remainder.
    denom = tp * pp * cp
    if gpus % denom:
        sys.exit(f"{key}/{mem_class} on backend {backend}: tp*pp*cp={denom} "
                 f"(tp{tp} pp{pp} cp{cp}) does not divide {gpus} GPUs. "
                 f"Adjust configs/models.json or pass a different --gpus.")
    dp = gpus // denom

    problems: list[str] = []
    if tp > gpus_per_node:
        problems.append(f"tp={tp} exceeds gpus_per_node={gpus_per_node}: a TP group would "
                        f"cross a node boundary, which is very slow (and impossible without NVLink)")
    # Whatever shards the SEQUENCE: cp on Megatron, ulysses_sp on verl. Exactly one of the
    # two can exceed 1 (the branch above rejects the other combination), so the product is
    # the right thing to multiply the per-GPU budget by -- the unit that must hold a whole
    # sequence is the sequence-parallel group, not one rank.
    seq_shard = cp * ulysses_sp
    if mtpg * seq_shard < max_seq_len:
        knob = "cp" if backend != "verl" else "ULYSSES_SP"
        problems.append(f"max_tokens_per_gpu*{knob} = {mtpg}*{seq_shard} = {mtpg * seq_shard} < "
                        f"max_seq_len={max_seq_len}: the longest sequence cannot be placed on any "
                        f"sequence-parallel group, and bin packing cannot split one sample. Raise "
                        f"{knob} or max_tokens_per_gpu"
                        + ("" if backend != "verl" else " (verl has no cp; ULYSSES_SP is its "
                                                        "analogue, and raising max_tokens_per_gpu "
                                                        "instead costs linear activation memory)")
                        + ".")
    if m.get("moe") and ep > cp * dp:
        problems.append(f"ep={ep} > cp*dp={cp*dp}: expert parallelism must fit inside the "
                        f"non-TP/PP dimensions")
    if problems:
        sys.exit(f"invalid config for {key} @ {mem_class} on {gpus} GPUs:\n  - " + "\n  - ".join(problems))

    out = {
        "PHASE": phase,
        # Which parallelism shape the numbers below describe. Named BACKEND_SHAPE, not
        # BACKEND: 20_run_all.sh eval()s this output and already owns a BACKEND variable.
        "BACKEND_SHAPE": backend,
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
        # Echoed back, deliberately NOT named ULYSSES_SP: every launcher eval()s this
        # output, and exporting the input name would let the default (1) overwrite an
        # operator's ULYSSES_SP=8 between here and the line that reads it.
        "ULYSSES_SP_USED": ulysses_sp,
        "MAX_TOKENS_PER_GPU": mtpg,
        "MAX_SEQ_LEN": max_seq_len,
        "TOTAL_GPUS": gpus,
        "SERVE_TP": m.get("serve_tp", row_tp),
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
    # min_gpus is a memory statement, not a shape one, so it is a warning rather than a
    # rejection: CPU optimizer offload or a shorter sequence can make a smaller world
    # work. But it must be said, because the arithmetic above cannot see memory --
    # qwen3.5-27b on 8 GPUs passes every assert here and then OOMs (444.8 GB of
    # params+grads+Adam over 8 cards is 55.6 GB/GPU before a single activation).
    # Same shape of statement as tp > gpus_per_node, but a warning rather than a rejection:
    # an Ulysses group that spans nodes is slow, not impossible -- its all-to-all and (on
    # this architecture) the gated-delta-net recurrent-state exchange then cross the slowest
    # link in the cluster on every layer.
    if ulysses_sp > gpus_per_node:
        unvalidated = (unvalidated + f" ULYSSES_SP={ulysses_sp} exceeds gpus_per_node="
                                     f"{gpus_per_node}, so each Ulysses group spans nodes and "
                                     f"its per-layer all-to-all plus GDN state exchange runs "
                                     f"over the inter-node fabric. Measure tokens/s before "
                                     f"committing to it.").strip()
    min_gpus = m.get("min_gpus")
    if min_gpus and gpus < min_gpus:
        unvalidated = (unvalidated + f" {gpus} GPUs is below this model's min_gpus={min_gpus}; "
                                     f"expect OOM unless you enable optimizer offload or lower "
                                     f"max_seq_len, and say which in the report.").strip()
    out["_shape_note"] = shape_note
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
    # `slime` is spelled out because 20_run_all.sh's BACKEND is verl|slime and passes it
    # straight through; slime IS the Megatron shape.
    ap.add_argument("--backend", default="megatron", choices=["megatron", "slime", "verl"],
                    help="verl pins PP=CP=1 and folds CP into max_tokens_per_gpu")
    ap.add_argument("--ulysses-sp", type=int, default=1,
                    help="verl only: Ulysses sequence-parallel size. Divides "
                         "max_tokens_per_gpu, because an Ulysses group -- not one GPU -- is "
                         "what holds a whole sequence. Pass the SAME value the launcher "
                         "gives engine.ulysses_sequence_parallel_size")
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
    if args.ulysses_sp < 1:
        sys.exit(f"--ulysses-sp must be >= 1, got {args.ulysses_sp}")
    cfg = resolve(args.key, args.mem_class, args.gpus, args.gpus_per_node, args.max_seq_len,
                  phase=args.phase,
                  backend="megatron" if args.backend == "slime" else args.backend,
                  ulysses_sp=args.ulysses_sp)

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
