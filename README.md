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
| DPO on logged trajectories — **the default post-SFT stage** (`DPO_PLAN.md`) | ✅ **run end to end** on 0.8B/H100: 2,673 pairs, step-0 loss = log 2 exactly; off-policy, so not an RL result |

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
PLAN.md                        SFT plan; hardware decision tables; risk register
BACKENDS.md                    slime+Megatron vs verl+FSDP; why others were rejected
RL_PLAN.md                     agentic GRPO: architecture, prerequisites, gates
DPO_PLAN.md                    DPO on logged trajectories: the container-free default stage
OPERATOR_PROMPT.md             copy-paste kickoff message for the cluster LLM
scripts/
  00_preflight.sh              detect GPU mem / NVLink / IB / shared FS / RAM → config row
  lib_env.sh                   sourced by every launcher: enter the conda env, then prove it
  01_setup_env.sh              slime/Megatron env (secondary path)
  01b_setup_env_verl.sh        PRIMARY env: verl+FSDP, driver-adaptive torch build
  16_smoke_forward_backward.py real fwd/bwd on 1 GPU; measures peak memory
  02_download.sh               model + datasets, sha256-verified against the manifests
  03_build_sft_data.py         327,189 trajectories → slime `messages` parquet
  03b_validate_sft_data.py     ports slime's qwen3_5 mask; asserts the training target
  03d_build_openthoughts_sft.py  OpenThoughts-Agent-v1 → this format, via the same normalizer
  04_convert_ckpt.sh           HF ↔ Megatron torch_dist
  05_run_sft.sh                32-GPU SFT; auto-picks the 80GB/40GB parallelism row
  06_eval.py                   SGLang + Harbor/Terminus-2 on Docker; 3 runs, mean±std
  07_restore_vision.py         splice trained text weights back into the ViT/MTP checkpoint
  10_build_rl_taskset.py       difficulty-tiered GRPO task pool + verifier-leak guard
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
  33_run_dpo.sh                the three DPO stages, resumable, container-free
  34_diagnose_oom.py           why an OOM does not move when you cut the token budget
verl_backend/                  verl dataset + Harbor AgentLoop bridge
  model_registry.py            resolve+validate a model's launch config
rst_common/                    definitions that must be identical in eval and RL
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
python -m pytest tests/ -q     # 94 tests, ~1 s (10 of them skip without torch)
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
| `tests/test_restore_vision.py` | `07_restore_vision.py` end to end on synthetic 2-shard checkpoints: vision/MTP preserved, dtype cast back, a missing text tensor refused with nothing written, `--allow-original-fallback` recorded, shape and naming mismatches refused. Skips without torch |

What they do **not** cover, and no test in this repo does: anything that needs the
real tokenizer (`03b_validate_sft_data.py --sample 300` is that check, run by hand),
a GPU (`16_smoke_forward_backward.py`), a container runtime (`00b_setup_sandbox.sh`,
`06_eval.py`), or a multi-node launch. The `⏳`/`⚠️` rows in the status table above
are exactly those; a green test run says nothing about them.
