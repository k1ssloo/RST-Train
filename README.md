# RST-Train

SFT (then RL) of **Qwen3.5-27B** on the *Recursive Synthesis for Long-Horizon
Terminal Tasks* release, targeting **4 nodes × 8 A100 = 32 GPUs**.

**Read [`PLAN.md`](PLAN.md) first** — it is the executable spec, written for the
operator LLM on the cluster. [`OPERATOR_PROMPT.md`](OPERATOR_PROMPT.md) is the
copy-paste kickoff message for that LLM. This file is just the map.

## Status

| stage | state |
|---|---|
| Dataset audit (327 K trajectories) | ✅ verified end-to-end on the full release |
| SFT data pipeline | ✅ **built and validated locally** on the full 23 GB release |
| slime loss-mask contract (`qwen3_5`) | ✅ verified: 0 failures, 0 leakage, 32.6 % trained |
| Conda env recipe | ✅ written, A100-adapted — ⏳ not yet executed |
| HF → Megatron conversion | ⏳ **highest-risk unverified step**, run it on the H100 first |
| 32-GPU SFT launch | ⏳ written, needs cluster |
| Eval harness (`06_eval.py`, 3 runs, mean±std, infra-separated) | ⏳ written, needs cluster |
| Report generator + anomaly checks | ✅ **tested** on synthetic healthy/faulty runs |
| Multi-model registry (5 models) | ✅ **tested**: all rows resolve, 4 negative tests reject |
| Pre-tokenized export (backend-agnostic) | ✅ **run**: 10,578 rows, 0 drops, 32.42 % trained |
| verl+FSDP SFT path (**primary**) | ⚠️ dataset core unit-tested; launcher not executed |
| Rootless-podman sandbox (no Docker needed) | ✅ **verified**: build 26.6 s, run/exec/tmux/no-net all OK |
| Real fwd/bwd on Qwen3.5-0.8B + measured memory | ✅ **run**: 8/8 checks; unfused CE OOMs at 32 K, Liger 13.1 GiB |
| verl Harbor AgentLoop | ⚠️ assembly logic tested; never run against verl |
| RL task pool + leak guard | ✅ **run locally**: 5,140 tasks / 999 groups materialized, 0 verifier leaks |
| RL rollout code (`rl/generate.py`) | ⚠️ written against real slime APIs, **never executed** |
| RL image prebuild / launcher | ⚠️ written, needs a rootless Docker daemon + cluster |
| FSDP2 unsharded-gradient-accumulation fix (`verl_backend/fsdp2_grad_accum.py`) | ✅ **measured** on one H100: verl's path retains fp32 unsharded gradients covering every parameter, the patched path retains none, gradients differ by 0.000e+00. Projects to 93.8 GiB/GPU freed at shard 32 — ⏳ the 4×8 launch it unblocks has not rerun yet |
| Container-free offline eval (`06b_eval_offline.py`) | ⚠️ **crashed on the cluster** — `device_map="auto"` + a tied head put logits and targets on different cards, so the 4B run that finished all 82 steps has *no* eval at all (`BUG.md` BUG-11). Fixed, and the chunked scoring arithmetic is now checked against a full-logits computation (`tests/test_offline_eval_scoring.py`); ⏳ not yet rerun on the checkpoint |
| Self-improvement loop: rollouts -> SFT data (`03h`, `21_rsi_round.sh`) | ⚠️ **the loop was broken and is now closed in code**: nothing turned a verifier-passed rollout of our own checkpoint back into training data, and a naive attempt would have produced wrong data — Terminus-2 stores its own `Analysis:/Plan:` rendering, not the model's completion, unless asked for `raw_content` (`BUG.md` BUG-19, measured: 65 of 70 agent steps rendered on the one local rollout). Fixed via `--export-trajectories`; the refusal path, the salvage path and the ATIF reconstruction are tested, but **no rollout has yet been run with the flag** |
| LLM supervision on the curation band (`rst_common/judge.py`, `03g_curate_sft.py`) | ✅ **run end to end** against a local sglang server on real Nemotron rows: heuristics band 30,536 corpus rows into 81.0 % clear_keep / **18.0 % borderline** / 1.1 % clear_drop, and only the borderline reaches the judge. Cache verified (rerun: 31 hits, 9 calls). ⚠️ judged by Qwen3.5-0.8B — this proves the plumbing, not the verdict quality |
| DPO on logged trajectories — **the default post-SFT stage** (`DPO_PLAN.md`) | ✅ **run end to end SINGLE-GPU** on 0.8B/H100: 2,673 pairs, step-0 loss = log 2 exactly; off-policy, so not an RL result. ⚠️ the multi-rank path was **never executed** and was broken until `BUG.md` BUG-2 — `shard_model` sharded the FSDP2 root while the DPO forward calls `decoder(...)`/`lm_head(...)` directly, so every `torchrun` run died on `aten.embedding` with mixed Tensor/DTensor. Fixed and layout-tested (`tests/test_dpo_sharding.py`); still not run on >1 rank |

## Supported models

`MODEL_KEY` selects one; parallelism, loss mask, spec file, vision handling and
serving TP all follow from `configs/models.json`.

```bash
python scripts/model_registry.py --list
MODEL_KEY=qwen3.5-9b bash scripts/20_run_all.sh            # SFT -> eval -> report -> DPO
MODEL_KEY=qwen3.5-9b RUN_RL=1 bash scripts/20_run_all.sh   # ... plus agentic GRPO, if a sandbox exists
```

DPO runs by default because it needs no container; agentic GRPO is opt-in (`RUN_RL=1`)
because every rollout needs one. `RUN_DPO=0` turns DPO off.

`20_run_all.sh` evaluates the **base model on the same harness** as well as the
fine-tuned one, so "did this help?" is answerable even for sizes the paper never
published. It writes `verdict.json` with two flags: `in_range` (zero FAIL findings of any
kind) gates GRPO and the next model, and `checkpoint_trustworthy` (zero FAILs *about the
checkpoint*) gates DPO — DPO uses that checkpoint as its own frozen reference, so an
untrustworthy one makes every implicit reward meaningless.

The two differ in exactly one case, deliberately: "the benchmarks never ran" — no container
runtime, or no sglang to serve with — is a FAIL about the *measurement*, not the weights,
and DPO needs neither a container nor a server. So on a pod that cannot run containers, DPO
still runs, prints the FAILs it is carrying forward, and both checkpoints must be reported
as *not agentically evaluated*. That exemption is one entry in `POSTTRAIN_EXEMPT_FAILS` in
`scripts/14_make_report.py`; nothing else is exempt.

The launchers enter the conda env themselves (`scripts/lib_env.sh`) and verify it by
locating `torch`/`transformers`/`pandas`/`pyarrow`, because the `micromamba activate` inside
`01b_setup_env_verl.sh` only affects that script's own process. To do it by hand:
`source $BASE_FOLDER/env-rstverl.sh` in a script, `micromamba activate rstverl` in a shell.

| key | params | ~min/epoch | min GPUs | note |
|---|---|---|---|---|
| `qwen3.5-0.8b` | 0.87 B | ~5 | 2 | smoke test; needs a thinking-on serving template |
| `qwen3.5-4b` | 4.66 B | ~25 | 8 | four runs fit on four nodes |
| `qwen3.5-9b` | 9.65 B | ~50 | 8 | primary low-cost result |
| `qwen3.5-27b` | 27.78 B | ~150 | 32 | the paper's model (only one with published numbers) |
| `qwen3.5-35b-a3b` | 35.95 B / ~3 B active | ~40 | 8 | MoE; EP rows unvalidated on A100 |

All five share one byte-identical tokenizer **and** one training-time chat-template
render, so the published datasets and `--loss-mask-type qwen3_5` apply unchanged.

## Layout

```
BUG.md                         defects found reviewing the 4x8 OOM: cause, evidence, fix,
                               what is still open, and what was checked and IS correct
PLAN.md                        SFT plan; hardware decision tables; risk register
BACKENDS.md                    slime+Megatron vs verl+FSDP; why others were rejected
RL_PLAN.md                     agentic GRPO: architecture, prerequisites, gates
DPO_PLAN.md                    DPO on logged trajectories: the container-free default stage
OPERATOR_PROMPT.md             copy-paste kickoff message for the cluster LLM
scripts/
  00_preflight.sh              detect GPU mem / NVLink / IB / shared FS / RAM → config row
  lib_env.sh                   sourced by every launcher: enter the conda env, then prove it
  siblings.py                  ONE loader for the digit-prefixed scripts (they cannot be imported)
  sft_common.py                ONE dedup / template gate / group split / token-stats tail
  taskpool_common.py           ONE tier table, FROM-line reader and verifier-leak rule
  hf_publish.py                ONE card check + upload path behind every 13* script
  01_setup_env.sh              slime/Megatron env (secondary path)
  01b_setup_env_verl.sh        PRIMARY env: verl+FSDP, driver-adaptive torch build
  16_smoke_forward_backward.py real fwd/bwd on 1 GPU; measures peak memory
  02_download.sh               model + datasets, sha256-verified against the manifests
  03_build_sft_data.py         327,189 trajectories → slime `messages` parquet
  03b_validate_sft_data.py     ports slime's qwen3_5 mask; asserts the training target
  03d_build_openthoughts_sft.py  OpenThoughts-Agent-v1 → this format, via the same normalizer
  03e_build_tmax_sft.py        AI2 TMax trajectories → this format (native tool-calling source)
  03f_build_nemotron_sft.py    NVIDIA Nemotron-Terminal-Corpus → this format, 366k rows sharded
  03g_curate_sft.py            reward 1 is not "worth imitating": heuristics band every row,
                               an LLM judge sees the borderline 18% only, decisions archived
  03h_build_rollout_sft.py     OUR OWN rollouts (Harbor job dirs / TerminalEvo golden episodes)
                               → the same messages parquet. The arrow that closes the loop
  04_convert_ckpt.sh           HF ↔ Megatron torch_dist
  05_run_sft.sh                32-GPU SFT; auto-picks the 80GB/40GB parallelism row
  06_eval.py                   SGLang + Harbor/Terminus-2 on Docker; 3 runs, mean±std
  07_restore_vision.py         splice trained text weights back into the ViT/MTP checkpoint
  08_prepare_eval_ckpt.sh      verl FSDP shards (local or on the Hub) -> a checkpoint eval
                               can serve: shard-completeness gate, merge, sidecars, vision
                               restore, base-diff, load+generate smoke test
  10_build_rl_taskset.py       difficulty-tiered GRPO task pool + verifier-leak guard
  10b_build_termigen_taskset.py  AI2 open-instruct-termigen → a task pool (zero assistant turns)
  10c_build_swegym_taskset.py  SWE-Gym → a tiered pool, tiered from its own rollouts
  11_prebuild_images.py        prebuild/cache task Docker images (refuses default daemon)
  12_run_grpo.sh               32-GPU agentic GRPO (Harbor/Terminus-2 rollout)
  13_upload_hf.py              publish the derived datasets (sanitizes local paths)
  13c_upload_openthoughts_hf.py  publish the OpenThoughts build; gates the card on manifests
  14_make_report.py            markdown report + mechanical anomaly checks
  20_run_all.sh                one command: preflight -> ... -> train -> eval -> report
  15_export_pretokenized.py    bake the verified mask into input_ids+loss_mask
  00b_setup_sandbox.sh         find/start a container runtime (rootless podman)
  30_run_sft_verl.sh           PRIMARY backend: verl + FSDP (no Megatron)
  17_build_dpo_data.py         logged successes/failures on one task → preference pairs
  18_dpo_ref_logprobs.py       frozen reference logprobs, once, sharded across GPUs
  19_train_dpo.py              DPO with a step-0 = log 2 calibration gate (FSDP2)
  dpo_common.py                the one logprob implementation both 18 and 19 use
  21_rsi_round.sh              ONE self-improvement round: roll out -> harvest -> curate ->
                               pretokenize. Rejection sampling, not RL; mixes the seed corpus
                               back in so the policy cannot narrow onto its own output
  33_run_dpo.sh                the three DPO stages, resumable, container-free
  resume_guard.py              refuses a resume whose lr schedule changed under it
  34_diagnose_oom.py           why an OOM does not move when you cut the token budget
  35_probe_fsdp2_grad_accum.py measures the unsharded-gradient claim on one GPU
verl_backend/                  verl dataset + Harbor AgentLoop bridge
  model_registry.py            resolve+validate a model's launch config
  fsdp2_grad_accum.py          stops FSDP2 retaining a full unsharded fp32 gradient
rst_common/                    definitions that must be identical in eval and RL
  harbor.py                    the infra-vs-budget split, plus the ONE `harbor run` command line
  judge.py                     LLM supervision: cached, budgeted, off the critical path,
                               and a NullJudge when no endpoint is configured
tests/                         no-GPU, no-cluster unit tests (see "Tests" below)
configs/models.json            the model registry
rl/generate.py                 slime --custom-generate-function-path implementation
data/
  rst-trajectories/            23 GB source release (66 tars, all verified)
  Qwen3.5-27B-tokenizer/       tokenizer only (for local data work)
  sft-v1-cap10/                ★ primary: 10,778 examples = the paper's exact count
  sft-v1/                      ablation: cap 8, 8,886 examples
  rl-sweet/                    5,140 materialized RL tasks (pass-rate 10-90% band)
  dpo-v2/                      ★ adopted DPO pairs: 2,673 (--per-side 14)
  dpo-v1/                      first DPO build, --per-side 5 → 1,330; kept for the yield table
  openthoughts-agent-v1/       second SFT source, same format: 14,312 examples
  rst-tasks/                   3.7 GB task release (8 tars)
probe/                         paper.pdf + the upstream sources I read
.venv/                         local CPU-only env for the data pipeline
```

## Getting the data

The derived datasets are published:

| dataset | contents |
|---|---|
| [`NiuNiu0110/RST-SFT-Qwen3.5-27B`](https://huggingface.co/datasets/NiuNiu0110/RST-SFT-Qwen3.5-27B) | configs `cap10` (10,778 ex), `cap8` (8,886, ablation), `cap10_pretokenized` (`input_ids`+`loss_mask`) |
| [`NiuNiu0110/RST-DPO-Qwen3.5-27B`](https://huggingface.co/datasets/NiuNiu0110/RST-DPO-Qwen3.5-27B) | config `v2`: 2,673 pre-tokenized preference pairs (2,448 train / 225 holdout), 48 MB |
| `NiuNiu0110/RST-RL-Taskset` (private) | GRPO task selection metadata, 5,140 `sweet`-tier tasks |
| [`NiuNiu0110/OpenThoughts-Agent-v1-SFT-terminus`](https://huggingface.co/datasets/NiuNiu0110/OpenThoughts-Agent-v1-SFT-terminus) | configs `default` (14,312 ex) and `pretokenized`; a second source in **this same format**, mixable row-for-row with `cap10` |

```python
from datasets import load_dataset
ds = load_dataset("NiuNiu0110/RST-SFT-Qwen3.5-27B", "cap10", split="train")
# or, with the verified loss mask already applied (what the verl path consumes):
ds = load_dataset("NiuNiu0110/RST-SFT-Qwen3.5-27B", "cap10_pretokenized", split="train")
```

`33_run_dpo.sh` fetches the DPO pairs itself when `$BASE_FOLDER/dpo-v2/` is empty, so
that stage needs neither the 23 GB trajectory release nor a local rebuild
(`DPO_FETCH_HF=0` forces the rebuild instead).

The RL taskset is metadata only — task *bodies* are rebuilt from upstream with
`scripts/10_build_rl_taskset.py --materialize` (~15 s), which also rewrites
`metadata.task_dir` to local absolute paths.

`data/` is **not in this repo** (27 GB). To rebuild everything from the public release — the
whole pipeline is deterministic, and `manifest.json` records every count so you can
check you got the same thing:

```bash
export BASE_FOLDER=/path/to/scratch
bash   scripts/02_download.sh                      # sha256-verified against the release manifests
python scripts/03_build_sft_data.py --traj-root $BASE_FOLDER/rst-trajectories \
       --tokenizer $BASE_FOLDER/Qwen3.5-27B --out-dir $BASE_FOLDER/sft-v1-cap10 --per-group 10
python scripts/10_build_rl_taskset.py --tasks-root $BASE_FOLDER/rst-tasks \
       --traj-root $BASE_FOLDER/rst-trajectories --out $BASE_FOLDER/rl-sweet --tier sweet --materialize
```

## The two datasets

| | `sft-v1-cap10` ★ | `sft-v1` |
|---|---|---|
| per-group cap | 10 | 8 |
| final examples | **10,778** (= paper) | 8,886 |
| train / holdout | 10,578 / 200 | 8,686 / 200 |
| total tokens | 99.9 M | 82.4 M |
| trained tokens | 32.6 M (32.6 %) | 27.3 M (33.0 %) |
| task groups | 1,329 | 1,327 |
| steps/epoch @ GBS 128 | 82 | 67 |

Both **published** holdouts — and the 200-example counts above — come from a uniform row
shuffle, so up to `--per-group − 1` siblings of each held-out trajectory sit in train:
that loss is closer to a memorization check than to a transfer measurement. A local
rebuild now defaults to `--holdout-mode group`, which puts no `task_group_id` in both
splits. `manifest.json` records `holdout_mode` either way; quote it next to any holdout
number.

## A second SFT source in the same format

`open-thoughts/OpenThoughts-Agent-v1-SFT` is already the same agent contract —
`terminus-2`, the same `{analysis, plan, commands}` assistant JSON, the same
`New Terminal Output:` observations. So `03d_build_openthoughts_sft.py` is a
*converter*, not another builder: it puts those turns through the **same**
`normalize_assistant` (loaded by path out of `03_build_sft_data.py`, never
reimplemented) and the same chat-template contract gate, so one canonical form and
one mask cover both datasets and they concatenate safely.

```bash
curl -L -o data/openthoughts-agent-v1/source-train.parquet \
  https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT/resolve/main/data/train-00000-of-00001.parquet
python scripts/03d_build_openthoughts_sft.py \
  --source data/openthoughts-agent-v1/source-train.parquet \
  --tokenizer data/Qwen3.5-27B-tokenizer --out-dir data/openthoughts-agent-v1
```

15,209 rows → 14,372 reconstructed (837 dropped: 645 unparseable, 192 keyless) →
14,312 after the 32,768-token gate → 14,112 train / 200 holdout, 100.2 M tokens,
31.16 % trained. Validated: **0 contract failures, 0 user-turn leakage.**

| | this | `sft-v1-cap10` |
|---|---|---|
| examples | 14,312 | 10,778 |
| assistant turns / row | 7.46 mean | 12.0 mean |
| tasks | 14,312, **one trajectory each** | 1,329 groups, ~10 each |
| source models | 1 (`GLM-4.6-AWQ`) | 4 |
| steps/epoch @ GBS 128 | 110 | 82 |

Complementary rather than redundant: breadth of task here, depth of horizon there.
Holdout is group-disjoint for free — upstream `task` is unique per row, asserted at
build time rather than assumed.

Three things this converter does that a naive reformat would not:

1. **A literal newline inside a JSON string is repaired, not dropped.** Agents emit
   `"analysis": "step one⏎step two"`. Invalid JSON by spec, unambiguous in meaning,
   so it is re-dumped escaped — every assistant turn in the output parses under
   *strict* `json.loads` (verified: 106,828/106,828).
2. **A stale warning preamble is stripped.** When a turn is renormalized, the next
   observation opens `Previous response had warnings: - Extra text detected before
   JSON object` — a complaint about an error no longer in the data. 572 repaired.
   Kept whenever the previous turn was already clean.
3. **A failing turn drops the whole trajectory.** Truncating at the last good turn
   was measured and rejected: median salvageable fraction 0.15, and 323 of the 837
   fail on the *first* assistant turn.

Both fixes in (1) live in the shared normalizer, so they apply to the RST builder
too. `data/sft-v1-cap10/` predates them — all 126,630 of its assistant turns are
byte-identical under the current code, so it is not wrong, but a rebuild would
recover ~492 turns it dropped.

## Training on our own rollouts

Every dataset above came from somebody else's policy. `scripts/21_rsi_round.sh` is the
arrow back: serve a checkpoint, roll it out on tasks whose verifier can score it, keep
what passed, curate it, and hand back a parquet the next SFT reads.

```bash
export BASE_FOLDER=/shared/rst MODEL_KEY=qwen3.5-9b
export RST_DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock   # or RST_HARBOR_ENV=daytona
bash scripts/21_rsi_round.sh --ckpt $BASE_FOLDER/out-hf-full \
     --tasks $BASE_FOLDER/rl-sweet/tasks --runs 4 --round 1
# -> $BASE_FOLDER/rsi/round1/pretokenized_train.parquet  + rsi_round.json
DATA_DIR=$BASE_FOLDER/rsi/round1 RUN_NAME=qwen3.5-9b-rsi-r1 bash scripts/30_run_sft_verl.sh
```

This is rejection sampling (expert iteration), **not** RL: no advantage, no importance
ratio, no gradient from a failure. `12_run_grpo.sh` is that, and it is strictly more
informative per rollout. This path exists because it needs no trainer-side rollout
plumbing — one sglang server, one Harbor loop, one parquet — so on a pod that can run
containers but cannot stand up a colocated actor, it is the only loop that closes.

**The flag that makes it possible at all.** Terminus-2 writes its own `Analysis:/Plan:`
rendering into the trajectory and drops the model's completion unless asked for
`raw_content` (`BUG.md` BUG-19; measured 65 of 70 agent steps on the one local rollout).
`06_eval.py --export-trajectories` sends that kwarg and keeps the job dirs;
`RST_EXPORT_TRAJECTORIES=1` does the same for both RL paths. `03h_build_rollout_sft.py`
**refuses** a rendered trajectory rather than inventing text from it.

Measured on real (salvaged) rollout data, stages 2-4 chained: 24 rows survive curation and
pretokenize to a **32.98 % trained-token fraction** — inside the 0.25-0.45 band
`30_run_sft_verl.sh` gates on, and next to the release corpus's 32.42 %. Self-produced data
passes the launcher's own check without widening any threshold.

**Two defences against self-training narrowing the policy**, both on by default:
`--mix-ratio` concatenates the seed corpus back in (1:1 by default) so the next SFT still
sees the distribution the first one was trained on, and `rsi_round.json` records the pass
rate per round — a round whose pass rate did not move, or whose kept-row count collapsed,
taught nothing.

### Reward 1 is not the same as "worth imitating"

`scripts/03g_curate_sft.py` is the second gate, and it runs on any messages parquet, not
just rollouts. Deterministic heuristics — repeated commands, error-laden observations,
harness complaints, completion claims the agent then retracted, turn count against the
task's own siblings — band every row; an LLM judge sees only the ambiguous band, plus a
small random audit of the clear-keep band for calibration. Measured over the 30,536 rows
of `cap10` + OpenThoughts + TMax + Nemotron-holdout:

| band | rows | share |
|---|---|---|
| clear_keep | 24,722 | 81.0 % |
| **borderline** (the only band the judge sees) | **5,493** | **18.0 %** |
| clear_drop | 321 | 1.1 % |

That 18 % is the whole speed argument: judge cost tracks the borderline, not the corpus.
Verdicts are cached by content hash, so re-curating with different thresholds is free.

```bash
export RST_JUDGE_BASE_URL=http://127.0.0.1:30000/v1 RST_JUDGE_MODEL=my-judge
export RST_JUDGE_API_KEY=... RST_JUDGE_MAX_CALLS=2000        # see rst_common/judge.py
python scripts/03g_curate_sft.py --parquet data/sft-v1-cap10/rst_sft_train.parquet \
       --out-dir data/curated-v1
```

With no endpoint configured it runs to completion on heuristics alone and writes
`judge.enabled: false` into its manifest — it never pretends the borderline was reviewed.
**The judge is never consulted for a reward, a loss mask or an eval score**: the task's
verifier is the reward, and an LLM opinion in any of those makes the numbers
incomparable with the paper and with each other. See the header of `rst_common/judge.py`
for where the line is drawn and why.

## Local quick start (no cluster)

```bash
python scripts/03b_validate_sft_data.py \
  --parquet data/sft-v1-cap10/rst_sft_train.parquet \
  --tokenizer data/Qwen3.5-27B-tokenizer --sample 300 --show 1
```

The data pipeline and the checkpoint conversion are CPU/RAM-bound and can be
validated on a single machine before booking the cluster — see `PLAN.md` §4.

## Tests

```bash
python -m pytest tests/ -q     # 333 pass / 18 skip, ~2 s (those needing torch/numpy say SKIP)
python tests/run_tests.py      # same tests, for an env without pytest
```

They need no GPU, no cluster, no container runtime and no dataset. What they cover:

| file | what is pinned |
|---|---|
| `tests/test_loss_mask.py` | the two ports of slime's `qwen3_5` mask (producer in `15_export_pretokenized.py`, independent auditor in `03b_validate_sft_data.py`) stay behaviourally identical, on a synthetic tokenizer and on a per-character one; plus the semantics — no prompt/header/user token is ever a target, the `<think>\n` opener is prompt, `step_loss_mask=0` turns are excluded |
| `tests/test_harbor_outcomes.py` | the `HARNESS_INFRA` vs `AGENT_BUDGET` split in `rst_common/harbor.py`: marker precedence, all three `result.json` layouts, "unmeasured ≠ reward 0", escalate-only stdout refinement, the proxy policy |
| `tests/test_verl_dataset.py` | `build_row` padding never becomes a training target, and an oversized row is an error rather than a silent truncation; plus `RSTPretokenizedSFTDataset.__init__` raising on a misaligned mask, an oversized table or an empty one *before* the first forward pass (those need torch) |
| `tests/test_launcher_memory_flags.py` | that `30_run_sft_verl.sh` still passes verl's own fused-CE switch (`model.use_fused_kernels` + `impl_backend=torch`) *inside* the torchrun invocation, keeps `use_liger` for swiglu/rms_norm without relying on it for the cross-entropy, gates on the config keys existing before launching 32 processes, names the log line that proves the kernel was used, and — the OOM that no data change fixes — computes the 16 B/param static footprint and refuses a launch whose rendezvous would silently make each node its own `world_size=8` job |
| `tests/test_launcher_correctness_gates.py` | the gates that protect *correctness*, not memory: FLA's real `chunk_gated_delta_rule` (the PyPI wheel ships no `fla/ops`, and the pure-torch fallback silently drops `cu_seqlens`, so packed documents would share one recurrent state), the `transformers>=5.11,<5.15` window checked by looking for the attribute verl calls rather than by version string, `flash_attn` being mandatory whenever `use_remove_padding=True`, and the SP note saying what SP *cannot* fix |
| `tests/test_model_registry.py` | `--backend verl` reshapes the Megatron row before validating it — TP/PP/CP pinned to 1 and CP folded into `max_tokens_per_gpu`, announced not silent, `slime` still shaped like Megatron, `tp*pp*cp*dp == world`, an undividable GPU count rejected, and the 27B-on-8-GPUs case warned about (it passes every shape assert and then OOMs) |
| `tests/test_assistant_normalization.py` | `normalize_assistant` — the single canonical form every dataset here shares. Two cases are regressions from real data: a **literal newline inside a JSON string** is repaired (invalid by spec, unambiguous in meaning) and `find . -exec ls {} \;` in a prose preamble no longer **balances as `{}` and masks the real response two lines down**. The boundary is also pinned: an unescaped inner quote stays a drop, because guessing where the string ends invents training content |
| `tests/test_openthoughts_convert.py` | the upstream→ours mapping in `03d_build_openthoughts_sft.py`, where the shapes are so close that the dangerous failures are quiet: a `system` turn shifting the rendered prefix (refused, not folded in), a trailing user turn with nothing to predict, one bad turn dropping the whole trajectory rather than being skipped, and the stale `Previous response had warnings:` preamble being stripped **only** when the turn it complains about was actually renormalized |
| `tests/test_fsdp2_grad_accum.py` | the second, distinct 27B OOM: verl skipping FSDP gradient sync on non-final micro-batches makes FSDP2 retain a full **unsharded** fp32 gradient (`params × 4 B` = 103.5 GiB/GPU at 27B) until the final backward. Pins the micro-batch estimator that decides whether a config reaches that path at all — including that raising `ULYSSES_SP` **cannot** change the count, because `prepare_micro_batches` scales the token budget by `sp_size` too — that the worst case is taken over dp ranks (`same_micro_num_in_dp=True`), that the retained tensor does not shrink with the shard degree, and that the patch is still applied from the module verl loads in every rank rather than from a launcher a cluster-side wrapper can bypass. The allocation and numerics claims are *measured* instead, by `scripts/35_probe_fsdp2_grad_accum.py` |
| `tests/test_dpo_sharding.py` | `BUG.md` BUG-2: the multi-rank DPO path sharded the FSDP2 *root* while the forward calls `decoder(...)`/`lm_head(...)` directly, so every rank died on `aten.embedding` with mixed Tensor/DTensor before its first gradient. Asserts the sharding *layout* (which submodules are wrapped, what stays unsharded and frozen), so it runs without CUDA |
| `tests/test_offline_eval_scoring.py` | the container-free fallback in `06b_eval_offline.py` — the eval that has to work when the sandbox is blocked, and which crashed on the finished 4B run: `device_map="auto"` plus Qwen3.5's **tied** output head puts `head(hidden)` and `targets` on different cards, so `cross_entropy` raised and the one checkpoint that trained to completion was recorded as unmeasured. Pins the `.to(logits.device)` fix at the source level (a single-GPU box cannot reproduce a sharded load) and checks the chunked scoring arithmetic on CPU against an independent full-logits computation: position i is predicted from hidden i−1, position 0 is never a target however the mask is set, and `--chunk` moves memory without moving the number |
| `tests/test_dpo_shard_failure_report.py` | `BUG.md` BUG-13: the DPO launcher used to reduce sixteen background reference shards to one bit (`wait "$pid" \|\| rc=1`), so the 27B failure named no shard and no exit status while all sixteen logs it pointed at ended in `[done]`. Runs the real `rst_explain_shard_failures` from `scripts/lib_env.sh` against hand-written logs and pins the three cases apart: a `[done]` log with a non-zero status (the work is on disk, re-running is nearly free), an empty or absent log (python never printed), and a log that stops short (show the tail). Plus a source assertion that the collapsing wait loop stays gone |
| `tests/test_dpo_noise_floor.py` | `BUG.md` BUG-15: both finished DPO runs reported `holdout_reward_accuracy 0.5 -> 0.5938` (4B) and `0.5 -> 0.5391` (9B) with `warnings: []`, off final margins of 6e-05 and 1e-05 nats — an accuracy that is a sign test on a margin below the bf16 noise of the reference logprobs it is differenced against. Pins `dpo_common.noise_floor_warning` to those two summaries: it fires on both, scales with beta because the reward does, and stays silent for a genuinely trained margin, a real negative margin, and a run with no holdout eval. Plus a source assertion that the warning is appended to `runtime_warnings` before the summary is built, so it is archived and not just printed |
| `tests/test_nccl_timeout_report.py` | `BUG.md` BUG-14: when the 4B DPO trainer hung ten minutes in an NCCL watchdog timeout, the launcher answered with GATE 1 / GATE 2 / GATE 3 — three hypotheses that are all checked *before* the first collective — and the trainer's output was never on disk to read. Runs the real `rst_explain_nccl_timeout` against the observed log text and pins the two readings apart: per-rank `NumelIn` that differs (the ranks are running different collectives, a code divergence) versus identical everywhere (one rank never arrived — host OOM killer first). Also that a `[rank*]: *Error` raised before the timeout is surfaced as the first failure, that a log with no timeout stays silent, and that the launcher tees and classifies |
| `tests/test_report_loss_curve.py` | `BUG.md` BUG-17: every verl report so far WARNed "no loss values scraped from logs" while 110 usable steps sat in `$BASE_FOLDER/logs/run.log` — `--run-dir` pointed at the trainer's *output* directory, which under verl/FSDP holds only `global_step_*/`. Scrapes the real verl line format (`step:41 - train/loss:… - train/lr:…`, progress-bar prefix and all, including the short `lr` spelling the long pattern never matched) and pins the stage slice: one appended `run.log` holds SFT, GRPO and DPO, and DPO's loss is log 2 by construction, so scraping the file whole turns two healthy runs into a curve that ends at 0.693. Plus source assertions that the launcher passes its own log and names the stage for each report |
| `tests/test_resume_schedule_gate.py` | `BUG.md` BUG-18: the 4B tmax model came out of two launches of one `RUN_NAME` — the first ended at step 42 with `train/lr` 3.0e-07, its `min_lr_ratio` floor; the second, relaunched with `total_epochs=3`, was resumed by `trainer.resume_mode=auto` and started step 43 at 2.35e-06, because `total_training_steps` is re-derived every launch and the cosine rebuilt over the new total. Both exited 0, nothing warned, and the 42 extra steps moved the loss 0.1954 → 0.1965. Pins `scripts/resume_guard.py` against that exact pair: the epoch change on a resume is exit 2 naming the knob and the measured lr jump, an honest same-schedule resume and a pre-gate resume are allowed, the escape hatch *records* the new schedule instead of ignoring it, every curve-shaping knob (including one that disappears) is compared while `save_freq` and the wandb project are not, and the step comes from `global_step_*/` rather than the stale `latest_checkpointed_iteration.txt`. Plus the report-side `find_lr_restart`, which catches the same thing from the log alone without flagging warmup |
| `tests/test_sft_common.py` | the shared builder tail against the inline code it replaced -- the test re-implements the old group split and asserts the shared one reproduces it exactly, and checks `quantile` pointwise against `numpy.quantile`. Plus that no builder grew its own dedup or percentile back |
| `tests/test_taskpool_common.py` | that the three RL pools share one TIERS *object* (asserted with `is`: a copied tuple can drift), and the verifier-leak decision -- byte-identical beats name-only whatever the file order, and a pool's own verifier names are honoured (termigen grades with `test_outputs.py`, not `test_state.py`) |
| `tests/test_hf_publish.py` | that a missing input stops a publish BEFORE any repo is created, and that none of the six `13*` scripts kept a private `check_card` or a direct `HfApi` |
| `tests/test_judge.py` | the LLM client's contract on a fake transport: cache before network, budget exhaustion degrading to counted refusals rather than an exception, one JSON-only retry billed as one budget unit, bearer vs Azure `api-key`, failures never cached, 429 retried and 401 not |
| `tests/test_curate_sft.py` | the curation bands, on both dialects. Two are regressions from the first real run: a `<tool_call>` turn is parseable (the curator had grown a JSON-only parser and called every TMax row unparseable), and **two consecutive completion claims are a normal ending** -- Terminus-2 asks the agent to repeat the claim -- so only a claim followed by more work is a retraction. Before that fix, `clear_keep` was 156 rows of 30,536; after, 24,722 |
| `tests/test_rollout_sft.py` | `BUG.md` BUG-19: a rendered trajectory is refused by default and counted, the salvage path reconstructs the action JSON from the rendering plus tool calls and marks every turn it invented, compaction boundaries split into the linear segments Harbor would have written, and an infra failure never becomes a reward-0 training row |
| `tests/test_harbor_invocation.py` | that all three rollout launchers build their `harbor run` through one `run_argv`, and that asking for exportable trajectories on a harbor without `--agent-kwarg` refuses the run instead of producing a jobs tree that fails hours later in `03h` |
| `tests/test_siblings.py` | that the test helper and the scripts get the same module object, and that no script kept its own `spec_from_file_location` |
| `tests/test_restore_vision.py` | `07_restore_vision.py` end to end on synthetic 2-shard checkpoints: vision/MTP preserved, dtype cast back, a missing text tensor refused with nothing written, `--allow-original-fallback` recorded, shape and naming mismatches refused. Skips without torch |

What they do **not** cover, and no test in this repo does: anything that needs the
real tokenizer (`03b_validate_sft_data.py --sample 300` is that check, run by hand),
a GPU (`16_smoke_forward_backward.py`), a container runtime (`00b_setup_sandbox.sh`,
`06_eval.py`), or a multi-node launch. The `⏳`/`⚠️` rows in the status table above
are exactly those; a green test run says nothing about them.
