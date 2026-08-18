# Operator prompt

Copy everything inside the fence into the cluster agent's first message.
Replace the `<...>` placeholders first.

---

````text
You are operating a 4-node × 8×A100 cluster (32 GPUs) to run a supervised
fine-tune of Qwen3.5-27B on synthesized terminal-agent trajectories.

The repository at <REPO_PATH> already contains a complete, partially-verified
plan. READ THESE TWO FILES BEFORE RUNNING ANYTHING:
  - PLAN.md   — the spec: measured facts, hardware decision tables, risk register
  - README.md — status table of what is already done vs. not

Everything in PLAN.md marked "measured" was verified against the real data and
the real upstream source trees. Everything marked UNVERIFIED has not been run
yet and is your job to validate. Do not assume anything else.

## Mission

Produce a Qwen3.5-27B checkpoint fine-tuned on the RST trajectories, evaluated
on Terminal-Bench-Hard (and TB2 / LHTB if task sets are available), using
slime + Megatron for training and SGLang + Harbor + Terminus-2 for evaluation.

Success = an HF-format checkpoint that loads under SGLang, plus a scored eval
report, plus a written record of every deviation you had to make.

Reference numbers to orient by (paper Tables 3–4, pass-rate %, TB2/TB-Hard/LHTB):
  base model     41.20 / 22.67 / 18.10
  paper SFT rd.1 42.32 / 23.00 / 21.32   <- the fair comparison for a single SFT pass
  paper SFT rd.3 47.94 / 28.33 / 22.44   <- three cumulative synthesis rounds
Beating round 1 is the goal. Do not expect round 3 from one pass.

## Environment facts you must discover, not assume

GPU memory per card (80GB vs 40GB), NVLink presence, InfiniBand vs Ethernet, and
whether there is a shared filesystem are ALL UNKNOWN to whoever wrote the plan.
Do not ask a human. Run:

    bash scripts/00_preflight.sh --hostfile <HOSTFILE>

and read its output. It prints the config row to use. scripts/05_run_sft.sh
auto-detects the same things; you may override with MEM_CLASS=80GB|40GB|40GB-alt.
Log which row you selected and why.

## Hard rules — violating any of these silently corrupts the run

1. `--loss-mask-type qwen3_5`. NEVER the default `qwen`. The default mis-segments
   this chat template and trains on terminal output and the harness prompt.
2. `--qwen-gdn-backend fla`. A100 is SM80; FlashQLA requires SM90+. Do not
   install FlashQLA, and do not "fix" a GDN error by switching to flashqla.
3. No FP8 anywhere. A100 has no FP8 tensor cores. Never run
   tools/convert_hf_to_fp8.py.
4. Do not upgrade the pinned versions in scripts/01_setup_env.sh
   (torch 2.11.0+cu129, flash_attn 2.8.3, transformer_engine 2.16.1,
   flash-linear-attention 0.4.2, SGLang v0.5.15.post1, Megatron 1dcf0daf).
   That combination is version-sensitive. If a pin fails to install, report it —
   do not resolve it by bumping versions.
5. `messages[0]` has role `user`, not `system`. That is how Terminus-2 actually
   delivers the harness prompt, so it keeps training and serving identical.
   Do not "improve" this.
6. `max-tokens-per-gpu × CP ≥ max-seq-len` must hold, or the longest sequence
   cannot be placed. 05_run_sft.sh asserts it; do not weaken the assert.
7. After converting back to HF you MUST run scripts/07_restore_vision.py. A
   text-only Megatron round trip drops `model.visual.*` and `mtp.*`, and the
   checkpoint will not load as Qwen3_5ForConditionalGeneration without them.
8. Never size memory from the trajectory parquet's `input_tokens` column. It is
   a sum over turns (context is re-sent each turn), roughly 2× the true
   sequence length.

## Execute in this order. Each step is a gate.

The compute is cheap (~1.5–3 h per epoch); the integration is what fails. Steps
1–3 cost almost nothing and de-risk the expensive ones. Do not skip ahead.

STEP 1 — Preflight. `bash scripts/00_preflight.sh --hostfile <HOSTFILE>` on all
  nodes. Record: GPU mem class, compute cap, NVLink, IB, shared FS, host RAM,
  Docker, egress. GATE: host RAM must have headroom for Adam CPU offload
  (~334 GB of fp32 state, sharded across DP). If it does not, drop
  `--optimizer-cpu-offload` and raise CP, and say so.

STEP 2 — Environment. `bash scripts/01_setup_env.sh`, then `bash scripts/02_download.sh`.
  GATE: 02_download.sh sha256-verifies every dataset shard against the published
  manifest and exits non-zero on mismatch. Do not proceed past a mismatch.

STEP 3 — Checkpoint conversion. `bash scripts/04_convert_ckpt.sh to_dist`.
  This is the HIGHEST-RISK UNVERIFIED STEP: the text-only slime spec must
  tolerate the ViT (`model.visual.*`) and MTP (`mtp.*`) tensors present in the HF
  checkpoint. It is CPU/RAM-bound (~120 GB RAM, 20–40 min), so run it before
  booking all 32 GPUs. GATE: conversion completes and the output dir is ~52 GiB.
  If it fails on unexpected keys, report the exact keys and stop.

STEP 4 — Single-node smoke. `ACTOR_NUM_NODES=1` with TP4/PP2/CP1/DP1 on a
  200-example slice. GATE: loss decreases over ~20 steps and no NaN. This proves
  slime + Megatron + the GDN Triton kernels + the loss mask actually step.

STEP 5 — Context-parallel correctness check. UNVERIFIED and important: the 48
  gated-delta-net layers carry recurrent state across the sequence, so context
  parallelism is not obviously sound for them. slime ships CP4 as the default for
  this exact model, which is suggestive but not proof. Train the SAME 200-example
  slice for 20 steps at CP1 and at CP2 and compare loss curves.
  GATE: curves agree to within noise. If they diverge, CP is broken for this
  architecture — fall back to CP1, which requires MEM_CLASS=80GB with
  `--max-tokens-per-gpu 32768`. Report the comparison either way.

STEP 6 — Full run. `bash scripts/05_run_sft.sh` on 4 nodes. Defaults: the
  10,778-example set (data/sft-v1-cap10), 1 epoch, GBS 128, LR 3e-6→3e-7 cosine,
  max-seq-len 32768 → 82 optimizer steps. Watch in wandb: loss over only ~82
  points (log every step), trained tokens/step ≈ 32.6M/82 ≈ 397K, grad norm, and
  per-GPU memory. If WANDB_KEY is unset the script goes offline automatically;
  sync afterwards. On OOM use the fallback ladder in PLAN.md §1 in order
  (① halve max-tokens-per-gpu ② CP 2→4 ③ PP 2→4 ④ only as last resort reduce
  max-seq-len, since that drops the long-horizon examples the data exists to teach).

STEP 7 — Export. `04_convert_ckpt.sh to_hf <iter_dir> <out>` then
  `07_restore_vision.py`. GATE: 07_restore_vision.py refuses to write if zero
  trained tensors matched — that would mean the round trip silently failed.

STEP 8 — Evaluate. `bash scripts/06_eval.sh <out-hf-full> 4`, Docker backend.
  GATE FIRST: run the same harness against the authors' released checkpoint
  `Zhongzhi1228/Qwen3.5-27B-SFT` (`DOWNLOAD_REFERENCE=1` in 02_download.sh). If it
  does not reproduce roughly 47.9 on TB2, your harness is wrong — fix the harness
  before drawing any conclusion about your own checkpoint.

## Data

data/sft-v1-cap10/ (10,778 examples) and data/sft-v1/ (8,886, ablation) are
already built AND validated: slime's qwen3_5 mask contract passes with 0 failures
and 0 user-turn leakage, 32.6% of tokens trained. Prefer copying them over
rebuilding. If they are absent on this cluster, rebuild with
scripts/03_build_sft_data.py and then ALWAYS verify with
scripts/03b_validate_sft_data.py — it must print `contract failures : 0` and
`user-turn leakage : 0`. Treat a nonzero value as a stop condition.

## When something fails

- A GATE failure is a stop-and-report condition, not something to work around.
  Never disable a correctness check to make a step pass.
- An infrastructure failure (NCCL timeout, OOM, image pull, disk) is yours to
  retry and fix; apply the documented fallback ladder rather than inventing a new
  configuration.
- Distinguish these two in your report. Do not describe a correctness problem as
  a flaky infra problem.
- If you must deviate from PLAN.md, write the deviation, the reason, and the
  evidence into notes/DEVIATIONS.md as you go.

## Phase 2 (do NOT start until SFT is done and evaluated)

RL is specified in RL_PLAN.md with working code (rl/generate.py,
scripts/10..12). It is agentic GRPO: Harbor + Terminus-2 drive a Docker sandbox
per rollout, slime's OpenAIAdapter captures the exact sampled tokens, and reward
comes from each task's own verifier. It needs a dedicated rootless Docker daemon
and a prebuilt image pool. Two things there are load-bearing and unverified:
whether Harbor's LiteLLM client forwards the session id as an `Authorization:
Bearer` header, and whether token capture round-trips. RL_PLAN.md names the smoke
test for each. Do not begin phase 2 on your own initiative -- report SFT results
and wait.

## Report back

Append to notes/RUN_LOG.md after every step: command run, wall-clock, outcome,
and any deviation. At the end produce a summary containing:
  1. the detected hardware and the parallelism row you used;
  2. the STEP 5 CP1-vs-CP2 comparison result;
  3. final loss, step count, wall-clock, and peak per-GPU memory;
  4. eval pass rates for the reference checkpoint AND your checkpoint, side by
     side, so the harness is independently validated;
  5. every deviation from PLAN.md;
  6. anything in PLAN.md that turned out to be wrong — that matters more than a
     clean run, because the plan's author could not test these steps.

State plainly what you verified versus what you assumed. Do not report success
for a step you skipped.
````
