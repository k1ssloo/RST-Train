# DPO on logged RST trajectories — the container-free fallback for RL

Companion to `RL_PLAN.md`. Read that first: this exists because the GRPO path has a
single hard prerequisite — somewhere to run task containers — and on the target pod
that prerequisite is currently unmet (`AppArmor docker-default` denies `mount(2)`;
`RL_PLAN.md` §1). SFT is unaffected. RL is blocked outright until either ops grants
`--security-opt apparmor=unconfined` or an off-machine backend is wired
(`BACKENDS.md`).

This path needs **no container, no network, no privilege**. It is not a substitute
for GRPO and the code says so in its own output.

## What it is, stated so the report cannot overclaim

The trajectory release already contains successes *and* failures on the same tasks,
scored by those tasks' own verifiers. That is a preference dataset, paid for once
when the release was generated, replayable forever.

* **DPO reweights behaviour already in the data.** The data came from *other*
  policies (`Qwen3.5-27B`, `Qwen3.6-27B-base`, `gpt-oss-120b`,
  `qwen35-27b-iter0000161-hf`), so it is off-policy. It can sharpen successful modes
  the model already has. It **cannot discover a strategy no logged trajectory used** —
  that is exactly what on-policy sampling buys, and it is what is being given up.
* **`holdout_reward_accuracy` is likelihood ranking, not a pass rate.** It measures
  how often the model assigns higher likelihood to a verifier-approved trajectory
  than to a failing one on the same task. `0.5` is *no preference*. It cannot be
  compared with terminal-bench numbers.
* **A DPO checkpoint with no agentic eval is an untested checkpoint.** DPO removes
  the container requirement from *training* only. Say that in the report.

## Status

| piece | state |
|---|---|
| Pair builder (`scripts/17_build_dpo_data.py`) | ✅ **run**: 231,092 clean trajectories → 2,673 pairs, 25 min CPU |
| Shared logprob core (`scripts/dpo_common.py`) | ✅ **run**: bit-exact determinism, fp64 accumulation |
| Reference pass (`scripts/18_dpo_ref_logprobs.py`) | ✅ **run** on 0.8B/H100: 30–67 pairs/min, max \|Δ\| = 0 nats on recompute |
| Trainer (`scripts/19_train_dpo.py`) | ✅ **run** single-GPU: step-0 loss = 0.693147 = log 2 exactly; split-backward verified identical to single-graph (\|g\| 1427.2365 both) |
| Launcher (`scripts/33_run_dpo.sh`) | ✅ **run** end to end on the 0.8B smoke set |
| 27B multi-node FSDP2 | ⚠️ **never executed** — no cluster. The FSDP2 sharding and `get_model_state_dict` gather are written, not run |

Everything above marked ✅ was measured on one H100 with `Qwen3.5-0.8B`. The
memory arithmetic for 27B is arithmetic, not measurement.

## Run it

```bash
export BASE_FOLDER=/shared/rst MODEL_KEY=qwen3.5-27b
bash scripts/33_run_dpo.sh                      # 8 GPUs by default (nvidia-smi -L)
```

Three stages, each skipped if its output already exists, so a re-run resumes:

| stage | script | cost | output |
|---|---|---|---|
| 1 | `17_build_dpo_data.py` | ~25 min CPU (tar I/O bound), cached | `$BASE_FOLDER/dpo-v1/dpo_{train,holdout}.parquet` |
| 2 | `18_dpo_ref_logprobs.py` | 2 forwards/pair, sharded 1 process/GPU, resumable | `$BASE_FOLDER/dpo-v1/ref/ref_logps*.parquet` |
| 3 | `19_train_dpo.py` | FSDP2, one side's activations at a time | `$BASE_FOLDER/out-dpo/{hf,dpo_training_summary.json}` |

Knobs that matter (all env vars on `33_run_dpo.sh`):

```bash
PER_SIDE=14           # candidates per (group, model) per outcome. THE yield lever; see below
DPO_BETA=0.1          # KL strength. 0.1 is the usual starting point
DPO_LR=5e-7           # ~an order of magnitude below the SFT lr
DPO_LENGTH_NORM=1     # 1 = per-token objective (default, see "step size" below)
PARAM_DTYPE=fp32      # master weights. Do not set bf16 at this lr; see below
DPO_LOGIT_CHUNK=512   # positions per LM-head slice. Same value reaches stages 2 and 3
POLICY=$BASE_FOLDER/out-hf-full   # the SFT checkpoint; also the frozen reference
```

## The three gates, and what each one caught

`19_train_dpo.py` refuses to train rather than produce an unexplainable number.

1. **Fingerprint.** Reference logprobs are constants of one exact checkpoint, and
   `out-hf-full` is overwritten by every export, so a path string is not identity.
   `checkpoint_fingerprint()` hashes `config.json` plus the size and leading 1 MiB of
   every weight shard. A mismatch means `(π − ref)` compares two different models at
   step 0 and the implicit reward is garbage from the first update.
2. **Coverage.** Every trainable pair must have reference logprobs. Silently
   dropping the unscored ones would change the dataset without changing the manifest.
   The length filter runs *before* this gate and takes the same `--max-seq-len` as
   stage 2, so the two stages skip the same rows.
3. **Step-0 calibration.** At initialization the policy *is* the reference, so the
   loss must be exactly `−log σ(0) = log 2 = 0.693147`. **Measured: 0.693147, with
   `|π − ref| = 0` nats/token.** This is the most valuable check in the file: it
   fails loudly if the mask, tokenizer, dtype or checkpoint changed between the
   reference pass and now.

   It is exact only when stages 2 and 3 agree on dtype *and* chunk size. A bf16
   reference against an fp32 policy leaves ~1e-3 nats/token; a different
   `--logit-chunk` regroups the sum and leaves ~1e-8. Both are benign, and the gate
   prints which situation you are in instead of leaving a residual to guess about.

## Four numerical decisions that are load-bearing

**fp32 softmax, always.** Vocab is 248,320. A bf16 `logsumexp` over that, accumulated
across thousands of positions, drifts by nats over a whole sequence — and DPO
*subtracts two such sums*.

**fp64 accumulation, and fp64 before every comparison against the reference.** The
per-chunk cross-entropy is fp32, but the running total reaches ~1e3 nats where an
fp32 ulp is 6e-5, and `logits = (π_c − π_r) − (ref_c − ref_r)` is the difference of
two quantities that are *equal* at step 0. That is textbook catastrophic cancellation.
Before this was fixed the step-0 margins came out at ~1e-8 with random signs, so a
pair that should have been an exact tie was scored as a preference and holdout
accuracy printed 0.375 instead of 0.5.

**fp32 master weights (`PARAM_DTYPE=fp32`).** At lr 5e-7 a bf16 master weight rounds
the update to zero: bf16 keeps 8 mantissa bits, so a weight of magnitude 0.02 has a
spacing of ~7.6e-5. The failure is worse than "no learning" — small-magnitude weights
(norms, some biases) have fine enough spacing to move, so **the loss curve changes
while most of the model is frozen**, which looks like a working run. Under FSDP2 the
params are fp32 and compute is bf16 via `MixedPrecisionPolicy`, so this costs nothing
in matmul throughput. `19` warns if you override it below lr 1e-5.

**Chunked LM head + activation checkpointing.** One 32k row of fp32 logits at vocab
248,320 is 29.85 GiB, so the head is applied in `--logit-chunk` slices. With
gradients on, each slice is wrapped in `torch.utils.checkpoint` — *without* that,
holding every chunk's logits for backward costs exactly as much as never chunking,
which is the trap the chunking exists to avoid.

## Memory: two backward passes instead of one

A textbook DPO step holds both sides' autograd graphs at once. These episodes run to
32k tokens, so at 27B that does not fit next to the optimizer state. But the loss
depends on the two logprob sums only through a scalar:

```
z = (π_c − π_r) − (ref_c − ref_r)
dL/dπ_c = −β·σ(−βz)        dL/dπ_r = +β·σ(−βz)
```

So the coefficient is computed under `no_grad` from two cheap forwards, and each side
then backpropagates alone via `logp.backward(coef)`. Peak activation memory halves
for ~33% more compute and **zero approximation** — verified: split-backward and
`--no-split-backward` produce `|g| = 1427.2365` on the same data and weights.

## Step size: the clip, not the lr

Each side's logprob is a **sum** over ~3k supervised tokens, so the gradient norm runs
~1e3 against the default `--max-grad-norm 1.0`. Clipping then fires on *every* step,
which means the clip — not `--lr` — sets how far the weights move, and the cosine
schedule only rescales an already-normalized direction. Measured: `|g|` p50 = 1427,
`clip_active_fraction` = 1.0.

`--length-normalize` (on by default in the launcher) makes the objective per-token and
drops the norm to ~1.0, at which point `--lr` means what it usually means. Measured:
`|g|` p50 = 1.05, `clip_active_fraction` = 0.5. It also removes the length term the
summed objective would otherwise reward.

Either choice is defensible. Reporting `--lr 5e-7` as the step size while the clip is
active on 100% of steps is not, which is why `optimization.clip_active_fraction` is in
the summary and warns at >0.9.

## Yield: why 2,673 pairs out of 231,092 trajectories

The funnel, all measured:

```
231,092 clean trajectories (61,575 success / 169,517 failure)
  2,246 task groups have any clean data
  1,290 groups contain BOTH a success and a failure      <- the usable pool
 43,814 candidates selected at --per-side 14
 40,422 reconstructed (3,392 dropped: unparseable or missing keys)
  5,759 distinct CANONICAL prompts
 └ 3,884 of them (67%) hold only ONE outcome and pair nothing
  2,820 candidate pairs
  2,673 after the length filter        -> 2,448 train / 225 holdout
```

The ceiling is structural, not a bug. Selection is per `(group, model)` because that
is all the trajectory metadata knows; **pairing is per task variant, which is only
knowable after reconstruction**. So most sampled buckets turn out to be one-sided.
`--per-side` is the only real lever, and it costs reconstruction I/O linearly:

| `--per-side` | candidates/side | pairs | note |
|---|---|---|---|
| 2 | ~1,400 | ~2 | effectively nothing |
| 5 | 9,286 | 1,330 | |
| 14 | 21,907 | **2,673** | launcher default |

`--pairs-per-prompt` barely helps (2 → 8 moved 1,330 → 1,533): the binding constraint
is one-sided buckets, not pairs per bucket.

## The pairing trap, and why the fix is narrow

DPO compares `log π(y_w|x)` against `log π(y_l|x)` for the **same** `x`. The
trajectory metadata has no `task_id` — only `task_group_id`, which spans several task
*variants with different instructions*. Pairing on `task_group_id` would silently
compare answers to two different questions.

So pairs are formed on the actual first user message. Hashing it **verbatim pairs
nothing**: measured 2,064 trajectories → 2,064 distinct prompts → **0 pairs**, because
the prompt ends with a live terminal screen carrying the container's per-run UUID
hostname. The hash is therefore taken over a canonicalized copy that masks UUIDs and
docker's 12-hex hostnames — which collapses 2,064 → 360 prompts and yields 241 pairs
on that shard.

Two guards keep that from over-merging:

* Every group that still splits after canonicalization splits on **genuinely rewritten
  instructions** — checked by hand; real variants, which must stay unpaired.
* The residual prompt divergence is measured at the token level, not hidden:
  `prompt_divergence_tokens` median 40, max 56 — a handful of hostname tokens in
  *masked context on both sides, never a target*. `--max-prompt-divergence` drops
  anything worse.

**Canonicalization is used for grouping only. The trained text is always the verbatim
bytes the model saw.**

## Length bias: measured, not assumed

Summed-logprob DPO prefers shorter sequences, and the folk expectation here is that
failures are the longer side (an agent that cannot solve a task keeps going until the
step limit). On the built pairs that is **only weakly true**: 46.09% have the longer
side rejected, median rejected/chosen supervised-token ratio 0.9758. So
`length_bias_warning: null`. The mechanism is still real, which is why
`--length-normalize` exists and why the flag lands in the summary.

## Reading `dpo_training_summary.json`

| field | what it means |
|---|---|
| `what_this_is` | the off-policy caveat, in the artifact so it cannot be dropped |
| `gates.step0_loss` | must be ~0.693147. Exact when `gates.dtype_match` |
| `gates.dtype_match` | false ⇒ the step-0 residual is bf16-vs-fp32, ~1e-3 nats/token |
| `metrics.holdout_reward_accuracy` | likelihood ranking. 0.5 = no preference |
| `metrics.holdout_*.holdout_ties` | 0.5 from all-ties (no preference) and 0.5 from half-right (preferences that are coin flips) are different findings sharing a number |
| `optimization.clip_active_fraction` | ~1.0 ⇒ the clip set the step size, not `--lr` |
| `data.pairs_skipped_oom` | a skipped pair is a quietly different dataset; counted, never silent |
| `warnings` | empty on a clean run. If not, it belongs in the report |

## When it fails

| symptom | cause | fix |
|---|---|---|
| `GATE 1 FAILED (fingerprint)` | `POLICY` is not the checkpoint the reference was scored on (usually `out-hf-full` was re-exported) | `rm -rf $BASE_FOLDER/dpo-v1/ref` and re-run, or point `POLICY` back |
| `GATE 2 FAILED (coverage)` | the reference pass did not finish | re-run `33_run_dpo.sh`; stage 2 resumes from the partial parquet |
| `GATE 3 FAILED (calibration)` | mask / tokenizer / dtype changed between stages | fix the cause. Do **not** raise `--calibration-tol` to get past it |
| `--holdout-groups N consumed every group` | too few groups paired | already clamped to `--max-holdout-fraction` (0.15) with a loud message |
| `REFUSING TO TRAIN: ...` | the pairs parquet itself is malformed (mask misaligned, empty mask, no shared prefix, duplicate `pair_id`, token-identical sides) | rebuild stage 1; the gate reads the tensors, not the manifest |
| OOM on one pair | a 32k×2 pair at 27B | counted in `data.pairs_skipped_oom` and reported; lower `DPO_LOGIT_CHUNK` or `--max-seq-len` |

## Order of operations

1. `bash scripts/33_run_dpo.sh` with `--max-steps 2` on a small `PER_SIDE` build.
   **GATE:** step-0 loss = 0.693147 and `gates.dtype_match` true.
2. Full run, 1 epoch. **GATE:** `holdout_reward_accuracy` rises above its step-0
   value *and* `holdout_ties` at step 0 equals `holdout_pairs` (proof the baseline was
   "no preference", not "coin flips").
3. Offline eval of the DPO checkpoint against the SFT one — no container needed:
   ```bash
   python scripts/06b_eval_offline.py --model-path $BASE_FOLDER/out-dpo/hf \
     --base-model $BASE_FOLDER/out-hf-full \
     --holdout $BASE_FOLDER/sft-v1-cap10/pretokenized_holdout.parquet \
     --out $BASE_FOLDER/eval/dpo-offline
   ```
4. Agentic eval **if and when** a sandbox exists. Until then the report says the
   checkpoint is untested on tasks, because it is.

## DPO risk register

| risk | severity | mitigation / test |
|---|---|---|
| Reported as RL / GRPO results | **high** | `what_this_is` is the first field of the summary and `33_run_dpo.sh` prints the caveat on exit. This is a writing risk, not a code risk |
| `holdout_reward_accuracy` read as a pass rate | **high** | named in the field's own docstring, the summary's `how_to_read_this`, and the launcher's exit text |
| 2,673 pairs is small for 27B | med | 1 epoch, lr 5e-7, β 0.1 — a deliberately small nudge. Raise `PER_SIDE` before raising the lr |
| Off-policy data + a strong SFT model ⇒ DPO pushes toward *worse* trajectories the SFT model already beats | med | the holdout margin is the detector; a falling holdout accuracy with a rising train margin is the signature. Stop and report it |
| ⚠️ 27B FSDP2 path never executed | **high** | step 1 above on the real cluster before committing GPU days |
| bf16 masters silently freeze most weights | med | `PARAM_DTYPE=fp32` default + a warning below lr 1e-5 |
| A rising margin is just "learn to be brief" | med | `length_bias_warning` from the builder + `--length-normalize` recorded in the summary |
