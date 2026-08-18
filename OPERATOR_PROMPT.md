# Kickoff prompt for the cluster-side LLM

Copy everything below the line into the operator LLM's first message.

---

You are operating a 4-node × 8×A100 cluster (32 GPUs). Deliverable, for
**qwen3.5-27b and then qwen3.5-9b**: an SFT checkpoint, then a **DPO** checkpoint on
top of it, each measured against the same base model, plus a report containing your own
analysis of anything abnormal. Agentic GRPO is a separate later phase — do not start it
in this pass.

You start from an empty directory.

```bash
export BASE_FOLDER=<shared-scratch>          # >= 400 GB, visible from every node
mkdir -p "$BASE_FOLDER" && cd "$BASE_FOLDER"
git clone https://github.com/k1ssloo/RST-Train.git && cd RST-Train
```

Read before running anything: **README.md** (status table — verified vs never executed),
**BACKENDS.md** §Decision (verl+FSDP is the primary backend, *not* slime), **PLAN.md**
(the SFT spec; this is the contract), **DPO_PLAN.md** (the DPO stage). `RL_PLAN.md` only
if you later get a sandbox.

Anything those files call "measured" was verified against the real data. Anything marked
UNVERIFIED has never been executed — validating it is your job. Assume nothing beyond
that, and never report success for a step you skipped.

## Run it

```bash
bash scripts/00_preflight.sh                      # prints the parallelism row to use
export MODEL_KEY=qwen3.5-27b
export MASTER_ADDR=<head-ip> HOSTFILE=$BASE_FOLDER/hostfile   # one node IP per line, head first
export WANDB_KEY=<key>                            # omit -> offline mode
bash scripts/20_run_all.sh 2>&1 | tee $BASE_FOLDER/run_all.log
```

Chain: preflight → env → download → data → train → export → eval → report → **DPO** →
offline eval of the DPO checkpoint. Every stage writes `$BASE_FOLDER/.stage/<name>.done`
and is skipped next time, so after fixing anything just re-run the same command. Delete
a marker to force one stage; `SKIP_STAGES="env download"` skips by name.

**Do a throwaway `MODEL_KEY=qwen3.5-0.8b` run first** (~5 min/epoch, 2 GPUs). Same
architecture as 27B, so it clears the integration risks in minutes instead of after
booking 32 GPUs for a day. It is a smoke test — never report its numbers as a result.

Then 27B, then:

```bash
MODEL_KEY=qwen3.5-9b bash scripts/20_run_all.sh
```

**The gate between models.** `20_run_all.sh` writes `verdict.json` with `in_range`,
which means the report produced **zero FAIL findings**. WARNs do not block — a WARN is a
caveat for a human to weigh, a FAIL means a number is wrong.

- **In range** → continue on your own initiative, including the next model. Do not ask.
- **Out of range** → stop that model, write the analysis, wait for a human. Do not start
  another long run on top of an unexplained failure, and do not "fix" it by loosening a
  check.

DPO refuses to start if the SFT report has FAILs, deliberately: it uses that checkpoint
as both the policy *and* the frozen reference, so an untrustworthy checkpoint makes every
implicit reward meaningless — and unlike a bad benchmark number, that failure is
invisible in the loss curve.

## Data: download it, do not rebuild it

Both datasets are published, public (**no HF token needed**) and already validated. The
tokenizer is byte-identical across all five model sizes, so one copy serves every model —
**do not rebuild per model.** The base weights do need a token if your HF account has not
accepted the model licence; `02_download.sh` fails with a 401/403 there, not a hash error.

```bash
hf download NiuNiu0110/RST-SFT-Qwen3.5-27B --repo-type dataset --local-dir $BASE_FOLDER/sft-hf
mkdir -p $BASE_FOLDER/sft-v1-cap10
cp $BASE_FOLDER/sft-hf/data/cap10/train.parquet   $BASE_FOLDER/sft-v1-cap10/rst_sft_train.parquet
cp $BASE_FOLDER/sft-hf/data/cap10/holdout.parquet $BASE_FOLDER/sft-v1-cap10/rst_sft_holdout.parquet
cp $BASE_FOLDER/sft-hf/manifest_cap10.json        $BASE_FOLDER/sft-v1-cap10/manifest.json
```

Then pass `SKIP_STAGES="data"`. `cap10` = 10,778 examples (the paper's exact count);
`cap8` = 8,886 (ablation); `cap10_pretokenized` = `input_ids` + `loss_mask`, which is
what the verl path consumes.

**The DPO pairs need no action from you.** `33_run_dpo.sh` fetches
[`NiuNiu0110/RST-DPO-Qwen3.5-27B`](https://huggingface.co/datasets/NiuNiu0110/RST-DPO-Qwen3.5-27B)
— 2,673 pairs, 48 MB — into `$BASE_FOLDER/dpo-v2/` on its own, so this stage works even
on a pod that never downloaded the 23 GB trajectory release. Three sources are tried in
cost order: local `$BASE_FOLDER/dpo-v2/` → the published set → rebuild from
`rst-trajectories` (~25 min CPU, same 2,673 pairs, seed 1228). Knobs:
`DPO_PAIRS_DIR`, `DPO_PAIRS_REPO`, `DPO_FETCH_HF=0` to force the local rebuild.

If you do rebuild the SFT data, `scripts/03b_validate_sft_data.py` **must** print
`contract failures : 0` and `user-turn leakage : 0`. Anything else is a stop condition —
the training target would be wrong, and no amount of training fixes that.

## You are authorized to edit any file here

This repo was written without access to your cluster and several steps have never
executed. Fix launcher bugs, wrong paths, NCCL/Ray/filesystem problems, OOM configs,
renamed upstream flags, unbuildable pins, anything in `configs/models.json` that is
wrong for your hardware. Do not sit blocked on plumbing.

Rules: fix the cause rather than the symptom; change one variable at a time; prefer a
flag over a version bump over a code edit; record every change in `notes/DEVIATIONS.md`
with the system fact that forced it ("driver 535 → cu121 wheel, cu128 failed at cuda
init"); commit locally, do not push. **If a fix changes what the numbers mean** — shorter
sequences, fewer eval runs, dropped tasks, sdpa instead of FA2 — say so in the report. A
silent scope reduction reads as a clean result and is the worst possible outcome.

### Never change these to make a run start

Each one silently invalidates the result, so a run that "succeeds" after touching one is
worse than no run at all. If you believe one is genuinely wrong, say so and wait.

- `--loss-mask-type qwen3_5` (the default `qwen` mis-segments this template and trains
  on terminal output), or recomputing the mask with another implementation
- the data gate in `20_run_all.sh` (contract failures / user-turn leakage == 0)
- `verl`'s `ignore_input_ids_mismatch: True` — it silences the check, not the bug
- `model.use_liger=True`: with a 248,320-row vocab the logits alone are 16–33 GB at 32K.
  This is what makes the run fit, not an optimization
- the registry asserts (`max_tokens_per_gpu × CP ≥ max_seq_len`, `tp·pp·cp·dp == gpus`)
- infrastructure-vs-model-failure separation in `06_eval.py` / `rl/generate.py`
- the reference-checkpoint eval, when the model has one
- the DPO tolerances in `19_train_dpo.py` (see the three gates below)

Hard floors, non-negotiable: `transformers >= 5.15.0` (older versions do not know
`qwen3_5` at all); no FP8 anywhere (A100 has no FP8 tensor cores); `--qwen-gdn-backend
fla`, never FlashQLA (SM90+ only).

## When something fails

A GATE failure is stop-and-report. An infrastructure failure is yours to retry and fix.
Say which one you are claiming — never describe a correctness bug as flakiness.

| symptom | what it actually is | do this |
|---|---|---|
| Megatron / TransformerEngine / apex will not build | A100 needs a cuDNN swap a shared cluster will not allow | already handled: `BACKEND=verl` is the default. Do not fight it |
| OOM during SFT | expected at 32K | ladder **in order**: ① halve `max-tokens-per-gpu` ② CP 2→4 ③ PP 2→4 ④ only last, reduce `max-seq-len` — that drops the long-horizon examples the data exists to teach |
| loss sits near `log(vocab)` = 12.42 and flat | the model is predicting noise — usually the loss mask never reached the loss | stop. `16_smoke_forward_backward.py` checks exactly this on one GPU and prints the comparison; then re-run `03b_validate_sft_data.py` |
| `MODEL_DIR_NAME: unbound variable` from the registry | the registry rejected your GPU count for its TP/PP/CP plan | fine for DPO (FSDP2 only, no TP/PP/CP) — `33_run_dpo.sh` already falls through. For SFT, fix `configs/models.json` |
| 0.8B generates nothing sane at eval | its own template defaults thinking **off** while training targets open with `\n</think>\n\n` | the launcher fetches the thinking-on template via the registry — do not remove that step; if you serve by hand, pass `--chat-template` |
| `02_download.sh` sha256 mismatch | a corrupted or changed shard | stop. Do not proceed past it |
| tokenizer checksum mismatch | drift between models | stop. Do not "fix" it by re-tokenizing |
| no container runtime for eval | expected; **SFT and DPO need none** | `bash scripts/00b_setup_sandbox.sh --diagnose` names the one case you are in. See below |
| every `mount` returns **EACCES** under AppArmor `docker-default (enforce)` | an LSM denial; nothing inside the pod fixes it (profiles survive `unshare`/`execve`) | ask ops for exactly **one** flag: `--security-opt apparmor=unconfined` (k8s ≥1.30: `securityContext.appArmorProfile.type: Unconfined`). Do **not** ask for `podman`+`uidmap` (already installed), `SYS_ADMIN`, `/dev/fuse`, or `--privileged` |
| podman fails on 31 % of task images ("Unknown instruction: IF") | podman < 4.4 cannot do Dockerfile heredocs; Ubuntu 22.04 ships 3.4.4 | install the static rootless build (32 MB, no root): `curl -sSL -o /tmp/podman.tgz https://github.com/mgoltzsche/podman-static/releases/latest/download/podman-linux-amd64.tar.gz && mkdir -p $HOME/.local && tar xzf /tmp/podman.tgz -C $HOME/.local --strip-components=1 && export PATH=$HOME/.local/bin:$PATH` |
| podman fails on 16 % of images with a `SHELL` error | podman's default OCI format ignores `SHELL`, then errors | `11_prebuild_images.py` already passes `--format docker`; add it if you build by hand |
| DPO **gate 1** (fingerprint) fails | `POLICY` is not the checkpoint the reference logprobs were scored on | delete `$BASE_FOLDER/dpo-v2/ref` and re-run, or point `POLICY` back |
| DPO **gate 2** (coverage) fails | the reference pass did not finish | just re-run `33_run_dpo.sh`; stage 2 is resumable and keeps scored rows. The per-shard cause is in `$BASE_FOLDER/logs/dpo_ref_shard*.log` |
| DPO **gate 3** (calibration) fails | something changed between the reference pass and training: mask, tokenizer, dtype, or `--logit-chunk` | fix the cause. **Never raise the tolerance to get past it** — that gate is the only proof the frozen reference and the initial policy are the same model |
| DPO OOMs on 8 GPUs at 27B | fp32 masters + Adam is 444.8 GB sharded; 8 is tight, 16 is comfortable | go to 2 nodes — `DPO_NNODES=2` for `20_run_all.sh`, or `NNODES=2` if you call `33_run_dpo.sh` directly — plus `MASTER_ADDR` and one launch per node. Or raise `DPO_GRAD_ACCUM` |
| DPO `holdout_reward_accuracy` = 0.5 | at step 0 this is *correct* — an exact tie scores 0.5, and `holdout_ties` tells you which kind of 0.5 it is | not a failure. Read `holdout_ties`: nonzero = ties, zero = genuinely no preference |
| DPO `clip_active_fraction` = 1.0 | `--max-grad-norm`, not `--lr`, is setting your real step size | expected with summed logprobs (\|g\| ≈ 1e3). `--length-normalize` is on by default and drops it to ~1. Quote whichever one actually set the step size |

### If eval has nowhere to run containers

SFT and DPO are unaffected — neither needs a container. Two unblocks that need no local
privilege, in order of preference: a managed backend
(`export RST_HARBOR_ENV=daytona|e2b|modal` + its API key, then `source
scripts/00b_setup_sandbox.sh` — the task Dockerfile builds provider-side; `BACKENDS.md`
prices them at roughly $1.4–2.0 per GRPO step), or the one-flag ops ask above.

If neither lands, run the container-free eval — it is a weaker signal, but "unmeasured"
is not an acceptable deliverable:

```bash
python scripts/06b_eval_offline.py \
  --model-path $BASE_FOLDER/out-hf-full --base-model $BASE_FOLDER/$MODEL_DIR_NAME \
  --holdout $BASE_FOLDER/sft-v1-cap10/rst_sft_holdout.parquet \
  --tokenizer $BASE_FOLDER/$MODEL_DIR_NAME --out $BASE_FOLDER/eval/offline
```

The report still records a **FAIL** on benchmark coverage. That is deliberate: it must
never let "we could not measure it" read as "it looks good".

## DPO: what it is, and the two ways to misreport it

It trains on 2,673 logged pairs — two runs on the same task, one that the task's own
verifier scored reward 1 and one it scored 0. It needs no container, no network and no
privilege, which is why it is the default continuation. Artifacts:

```
$BASE_FOLDER/out-dpo/dpo_training_summary.json     read this one, and quote from it
$BASE_FOLDER/eval/dpo-offline/offline_results.json
$BASE_FOLDER/out-dpo/hf                            the checkpoint
```

In the summary, `gates.step0_loss` must be **0.693147** (= log 2, exact when
`gates.dtype_match` is true): with policy == reference the loss has no choice, so any
other value means the two were not the same model. `warnings[]` is empty on a clean run;
if it is not, it belongs in the report.

Two things you must not write:

1. **DPO is off-policy.** It reweights behaviour already present in other policies'
   trajectories, so it cannot discover a strategy none of them used. It is not "our RL
   result".
2. `holdout_reward_accuracy` is **likelihood ranking** — how often the model prefers the
   run that passed over the run that failed. **0.5 means no preference**, not a 50 % pass
   rate. Never put it in the same table as terminal-bench numbers.

A DPO checkpoint with no agentic eval is an untested checkpoint, and the report has to
say so.

Agentic GRPO is not gone, just not this pass: it is `RUN_RL=1` on the same launcher, it
needs a sandbox per rollout, and `RL_PLAN.md` lists its gates. Finish SFT + DPO on both
models first.

## Benchmarks

`06_eval.py` serves the checkpoint under SGLang and drives Harbor + Terminus-2, 3
independent runs, mean ± std. **tb-hard** (100 tasks) and **tb2** (89) are scored;
**lhtb** (46) has its verifiers withheld upstream, so it is reported `unscorable` — do
not produce an LHTB number. Infrastructure failures are excluded from the pass-rate
denominator and reported separately; a Docker build failure is not a wrong answer.

Reference numbers exist for **Qwen3.5-27B only** (pass rate %, tb-hard / tb2):

```
base                 22.67 / 41.20
paper SFT round 1    23.00 / 42.32     <- the fair target for one SFT pass
paper SFT round 3    28.33 / 47.94     <- three cumulative synthesis rounds
```

Beat round 1; do not expect round 3. For any other size the report *skips* the
regression and reference checks rather than passing them — do not invent a target for
4B/9B/35B.

`20_run_all.sh` also scores the authors' released `Zhongzhi1228/Qwen3.5-27B-SFT` through
the same harness. **If that does not land near 28.33 / 47.94, the harness is wrong, not
your training** — fix the harness before interpreting anything about your own
checkpoint. Do not skip it to save time.

## The report — your main deliverable

`14_make_report.py` runs automatically and writes `$BASE_FOLDER/REPORT.md`: results
against the paper's numbers, a findings table from mechanical checks, and an empty
**Analysis** section. It exits 2 when any FAIL exists — a non-zero exit with a written
explanation is a good outcome; a clean exit you got by disabling a check is not.

**Fill in the Analysis section yourself.** For every 🔴 FAIL and 🟡 WARN: name the
evidence (log line, file, command output); if you are guessing, write "hypothesis:" and
say what would confirm it; state whether you are claiming a correctness or an
infrastructure problem; and say plainly whether the headline numbers are trustworthy.

Keep your analysis in `notes/ANALYSIS.md` and append it to `REPORT.md` **last** — the
generator overwrites `REPORT.md` on every run, so anything written only there is lost
the next time a stage re-runs.

Final summary must contain:

1. detected hardware and the parallelism row you used;
2. SFT: final loss, step count, wall-clock, peak per-GPU memory;
3. the eval table for your checkpoint **and** the reference, side by side;
4. DPO: `gates.step0_loss`, whether `--length-normalize` was on, `holdout_reward_accuracy`
   with `holdout_ties`, and the explicit statement that it is off-policy and agentically
   unevaluated;
5. `in_range` per model, and for anything out of range, your analysis and what you need
   from a human;
6. every deviation and code edit, with reasons;
7. anything in `PLAN.md` / `DPO_PLAN.md` that turned out to be **wrong** — that matters
   more than a clean run, because the author could not test these steps.
