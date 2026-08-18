# Operator prompt

Copy everything inside the fence into the cluster agent's first message.
Replace the `<...>` placeholders first.

---

````text
You are operating a 4-node × 8×A100 cluster (32 GPUs) to fine-tune a Qwen3.5 model
on synthesized terminal-agent trajectories, then benchmark it and write a report.
Five model sizes are supported (0.8B … 35B-A3B); see "Pick a model first" below.

You are starting from an EMPTY directory. Bootstrap first:

    export BASE_FOLDER=<shared-scratch-dir>      # >= 400 GB free, ideally shared across nodes
    mkdir -p "$BASE_FOLDER" && cd "$BASE_FOLDER"
    git clone https://github.com/k1ssloo/RST-Train.git
    cd RST-Train

Then READ THESE THREE FILES BEFORE RUNNING ANYTHING:
  - README.md    — status table: what is already verified vs. never executed
  - PLAN.md      — the SFT spec: measured facts, hardware decision tables, risk register
  - RL_PLAN.md   — phase 2 only; do not act on it yet

Everything in PLAN.md marked "measured" was verified against the real data and the
real upstream source trees. Everything marked UNVERIFIED has not been run and is
your job to validate. Do not assume anything beyond that.

## Mission

An HF-format checkpoint that loads under SGLang, benchmarked on Terminal-Bench-Hard
and Terminal-Bench 2, plus `$BASE_FOLDER/REPORT.md` with your analysis. Training and
evaluation are chained automatically; you supervise and diagnose.

Reference numbers below apply to Qwen3.5-27B ONLY (paper Tables 3–4, pass rate %,
tb-hard / tb2). No other size has published numbers:
  Qwen3.5-27B base   22.67 / 41.20
  paper SFT round 1  23.00 / 42.32   <- the fair target for a single SFT pass
  paper SFT round 3  28.33 / 47.94   <- three cumulative synthesis rounds, not one
Beating round 1 is the goal. Do not expect round 3.

## Pick a model first

    python scripts/model_registry.py --list

Five models are supported; `MODEL_KEY` selects one and everything else
(parallelism, loss mask, spec file, vision handling, serving TP) follows from
configs/models.json, which VALIDATES the config and refuses impossible ones.

  qwen3.5-0.8b      0.87B   ~5 min/epoch    2+ GPUs   pipeline smoke test
  qwen3.5-4b        4.66B  ~25 min/epoch    8+ GPUs   iteration workhorse
  qwen3.5-9b        9.65B  ~50 min/epoch    8+ GPUs   primary low-cost result
  qwen3.5-27b      27.78B ~150 min/epoch   32  GPUs   the paper's model
  qwen3.5-35b-a3b  35.95B  ~40 min/epoch    8+ GPUs   MoE, ~3B active

RECOMMENDED ORDER, and the reasoning: run qwen3.5-0.8b end-to-end FIRST. It is the
same architecture as 27B, so it validates the two unverified high-risk steps
(HF<->Megatron conversion tolerating the ViT/MTP tensors, and CP correctness on the
gated-delta-net layers) at negligible cost. Then qwen3.5-27b, because it is the
only model with published numbers and therefore the only one that can validate your
eval harness. Then 4b/9b/35b-a3b for cheap iteration.

All five share one byte-identical tokenizer AND one training-time chat-template
render, so the SAME dataset and --loss-mask-type qwen3_5 apply to all of them. Do
not rebuild the data per model.

TWO MODEL-SPECIFIC THINGS THAT WILL BITE YOU:
  1. qwen3.5-0.8b must be SERVED with a thinking-on chat template. Its own template
     defaults thinking off, so its generation prompt already closes the think block
     while training targets open with "\n</think>\n\n". 20_run_all.sh fetches the
     right template automatically via the registry -- do not remove that step, and
     if you serve 0.8B by hand, pass --chat-template.
  2. qwen3.5-35b-a3b's expert-parallel rows are UNVALIDATED on A100. slime's shipped
     launcher uses TP2/EP8 on 8 GPUs; ours scale that to 32. If DeepEP misbehaves on
     SM80, drop --moe-enable-deepep and use --moe-token-dispatcher-type alltoall.
     Report what you had to change.

Only qwen3.5-27b has published reference numbers. For any other model the report
SKIPS (does not pass) the regression-vs-base and reference-reproduction checks, and
says so. Do not invent a target for 4B/9B/35B.

## The one-command path

    export MODEL_KEY=qwen3.5-0.8b            # start here, then 27b
    export MASTER_ADDR=<head-node-ip>
    export HOSTFILE="$BASE_FOLDER/hostfile"          # one node IP per line, head first
    export RST_DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock   # dedicated daemon, see below
    export WANDB_KEY=<key>                           # omit -> offline mode automatically
    bash scripts/20_run_all.sh 2>&1 | tee "$BASE_FOLDER/run_all.log"

That runs: preflight → env → download → build data → convert checkpoint → train →
export → eval (yours + the reference checkpoint) → report. Every stage writes a
marker in `$BASE_FOLDER/.stage/` and is skipped if already done, so re-running
resumes. Delete a marker to force a stage; `SKIP_STAGES="env download"` to skip by
name.

**Run stages 1–5 individually the first time** (see "Order" below). The wrapper is
for resuming and for the long unattended stretch, not for hiding the risky steps.

## Getting the data: download, don't rebuild

The training data is already built and validated. Prefer downloading it:

    hf download NiuNiu0110/RST-SFT-Qwen3.5-27B --repo-type dataset \
      --local-dir "$BASE_FOLDER/sft-hf"
    mkdir -p "$BASE_FOLDER/sft-v1-cap10"
    cp "$BASE_FOLDER/sft-hf/data/cap10/train.parquet"   "$BASE_FOLDER/sft-v1-cap10/rst_sft_train.parquet"
    cp "$BASE_FOLDER/sft-hf/data/cap10/holdout.parquet" "$BASE_FOLDER/sft-v1-cap10/rst_sft_holdout.parquet"
    cp "$BASE_FOLDER/sft-hf/manifest_cap10.json"        "$BASE_FOLDER/sft-v1-cap10/manifest.json"

`cap10` = 10,778 examples (the paper's exact count); `cap8` = 8,886 (ablation).
Then `SKIP_STAGES="data"` so the wrapper does not rebuild it.

If you do rebuild (`scripts/03_build_sft_data.py`), you MUST then run
`scripts/03b_validate_sft_data.py`. It must print `contract failures : 0` and
`user-turn leakage : 0`. Anything else is a stop condition — the training target
would be wrong and no amount of training fixes that.

## Hard rules — violating any of these silently corrupts the run

1. `--loss-mask-type qwen3_5`. NEVER the default `qwen`. The default mis-segments
   this chat template and trains on terminal output and the harness prompt.
2. `--qwen-gdn-backend fla`. A100 is SM80; FlashQLA requires SM90+. Do not install
   FlashQLA, and do not "fix" a GDN error by switching to flashqla.
3. No FP8 anywhere. A100 has no FP8 tensor cores. Never run convert_hf_to_fp8.py.
4. Do not upgrade the pinned versions in scripts/01_setup_env.sh (torch
   2.11.0+cu129, flash_attn 2.8.3, transformer_engine 2.16.1,
   flash-linear-attention 0.4.2, SGLang v0.5.15.post1, Megatron 1dcf0daf). That
   combination is version-sensitive. If a pin fails to install, report it — do not
   resolve it by bumping versions.
5. `messages[0]` has role `user`, not `system`. That is how Terminus-2 delivers the
   harness prompt, so it keeps training and serving identical.
6. `max-tokens-per-gpu × CP ≥ max-seq-len` must hold or the longest sequence cannot
   be placed. scripts/model_registry.py asserts this (plus tp·pp·cp·dp == gpus and
   tp within one node) and exits non-zero. Do not weaken those asserts; fix
   configs/models.json instead, and say what you changed.
7. After converting back to HF you MUST run scripts/07_restore_vision.py for any
   model with a vision tower (all five here do). A text-only Megatron round trip
   drops `model.visual.*` and `mtp.*`, and the checkpoint will not then load as a
   ConditionalGeneration model. 20_run_all.sh does this conditionally on the
   registry's has_vision.
8. Never size memory from the trajectory parquet's `input_tokens` column — it is a
   sum over turns, roughly 2× the true sequence length.
9. Evaluation needs a DEDICATED (ideally rootless) Docker daemon. Benchmark tasks
   build untrusted third-party Dockerfiles; do not build them on the host's default
   daemon.

## Environment facts you must discover, not assume

GPU memory per card (80GB vs 40GB), NVLink, InfiniBand vs Ethernet, and whether
there is a shared filesystem are ALL UNKNOWN to whoever wrote the plan. Do not ask
a human. Run:

    bash scripts/00_preflight.sh --hostfile "$HOSTFILE"

It prints the parallelism row to use. 05_run_sft.sh auto-detects the same things;
override with MEM_CLASS=80GB|40GB|40GB-alt. Log which row you used and why.

## Order. Each step is a gate.

Compute is cheap (~1.5–3 h per epoch); integration is what fails. Steps 1–3 cost
almost nothing and de-risk the expensive ones. Do not skip ahead.

STEP 1 — Preflight. GATE: host RAM must have headroom for Adam CPU offload (~334 GB
  fp32 state, sharded across DP). If not, drop `--optimizer-cpu-offload`, raise CP,
  and say so.

STEP 2 — Environment + download (`01_setup_env.sh`, `02_download.sh`). GATE:
  02_download.sh sha256-verifies every dataset shard against the published manifest
  and exits non-zero on mismatch. Do not proceed past a mismatch.

STEP 3 — Checkpoint conversion (`04_convert_ckpt.sh to_dist`). HIGHEST-RISK
  UNVERIFIED STEP: the text-only slime spec must tolerate the ViT
  (`model.visual.*`) and MTP (`mtp.*`) tensors in the HF checkpoint. CPU/RAM-bound
  (~120 GB RAM, 20–40 min), so run it BEFORE booking all 32 GPUs. GATE: completes,
  output ~52 GiB. If it fails on unexpected keys, report the exact keys and stop.

STEP 4 — Single-node smoke. `ACTOR_NUM_NODES=1`, TP4/PP2/CP1/DP1, ~200 examples.
  GATE: loss decreases over ~20 steps, no NaN. Proves slime + Megatron + the GDN
  Triton kernels + the loss mask actually step.

STEP 5 — Context-parallel correctness. UNVERIFIED and important: the 48
  gated-delta-net layers carry recurrent state across the sequence, so CP is not
  obviously sound for them. slime ships CP4 as the default for this exact model,
  which is suggestive, not proof. Train the SAME slice for 20 steps at CP1 and CP2
  and compare loss curves. GATE: curves agree within noise. If they diverge, CP is
  broken here — fall back to CP1, which needs MEM_CLASS=80GB with
  `--max-tokens-per-gpu 32768`. Report the comparison either way.

STEP 6 — Full run + automatic eval + report: `bash scripts/20_run_all.sh`.
  Defaults: cap10 (10,778 ex), 1 epoch, GBS 128, LR 3e-6→3e-7 cosine, seq 32768 →
  ~82 optimizer steps. Watch in wandb: loss over ~82 points, trained tokens/step
  ≈ 32.6M/82 ≈ 397K, grad norm, per-GPU memory. On OOM apply the fallback ladder in
  PLAN.md §1 IN ORDER (① halve max-tokens-per-gpu ② CP 2→4 ③ PP 2→4 ④ only as a
  last resort reduce max-seq-len, which drops the long-horizon examples the data
  exists to teach).

## Evaluation (automatic, but understand it)

`scripts/06_eval.py` serves the checkpoint under SGLang and drives Harbor with
Terminus-2 on Docker, 3 independent runs, reporting mean ± std.

- **tb-hard** 100 tasks, verifiers ship → scored
- **tb2** 89 tasks, verifiers ship → scored
- **lhtb** 46 tasks, verifiers are WITHHELD upstream (0/46 ship `tests/`) →
  reported as `unscorable`. **Do not produce an LHTB number.** If you find a way to
  score it, say exactly how; otherwise leaving it blank is the correct answer.

Infrastructure failures are excluded from the pass-rate denominator and reported as
their own rate. A Docker build failure is not a wrong answer. Never "fix" a low
score by counting infra failures as model failures, or vice versa.

**The reference eval is what makes your number interpretable.** 20_run_all.sh also
scores the authors' released `Zhongzhi1228/Qwen3.5-27B-SFT` through the SAME
harness. If it does not land near 28.33 / 47.94, THE HARNESS IS WRONG, not your
training — fix the harness before interpreting anything about your own checkpoint.
Do not skip this to save time.

## The report — your main deliverable

`scripts/14_make_report.py` runs automatically and writes `$BASE_FOLDER/REPORT.md`:
a results table against the paper's numbers, a findings table from mechanical
checks (loss mask, GDN backend, parallelism arithmetic, sequence placement, data
provenance, NaN/grad-norm/step-count, infra-failure rate, regression vs base,
reference-harness validity), and an empty **Analysis** section.

**You must fill in the Analysis section.** The tool reports what it can check; it
does not diagnose. For every 🔴 FAIL and 🟡 WARN:

- Name the evidence — log line, file path, command output. If you are guessing,
  write "hypothesis:" and say what would confirm it. Never present speculation as
  fact.
- Say explicitly whether you are claiming a **correctness** problem or an
  **infrastructure** problem. Do not describe a correctness bug as flakiness.
- If a check is a false positive, say why its threshold is wrong in this case
  rather than silently ignoring it.
- State plainly whether the headline numbers are trustworthy.

Then re-run the generator so the report on disk includes your analysis, and report
the final verdict. The generator exits 2 when any FAIL finding exists — a non-zero
exit with a written explanation is a perfectly good outcome; a clean exit you
achieved by disabling a check is not.

## When something fails

- A GATE failure is stop-and-report. Never disable a correctness check to make a
  step pass.
- An infrastructure failure (NCCL timeout, OOM, image pull, disk) is yours to retry
  and fix; apply the documented fallback ladder rather than inventing a new config.
- Distinguish the two in everything you write.
- Log deviations to `notes/DEVIATIONS.md` as you go, and per-step progress to
  `notes/RUN_LOG.md`.

## Phase 2 (do NOT start on your own initiative)

RL is specified in RL_PLAN.md with working code (rl/generate.py, scripts/10–12):
agentic GRPO where Harbor + Terminus-2 drive a Docker sandbox per rollout, slime's
OpenAIAdapter captures exact sampled tokens, and reward comes from each task's own
verifier. Two load-bearing things there are unverified: whether Harbor's LiteLLM
client forwards the session id as an `Authorization: Bearer` header, and whether
token capture round-trips. RL_PLAN.md names the smoke test for each. Report SFT
results and wait for instruction.

## Report back

Final summary must contain:
  1. detected hardware and the parallelism row used;
  2. the STEP 5 CP1-vs-CP2 comparison result;
  3. final loss, step count, wall-clock, peak per-GPU memory;
  4. eval table for BOTH your checkpoint and the reference, side by side;
  5. every deviation from PLAN.md;
  6. anything in PLAN.md that turned out to be WRONG — that matters more than a
     clean run, because the plan's author could not test these steps.

State plainly what you verified versus what you assumed. Never report success for a
step you skipped.
````
