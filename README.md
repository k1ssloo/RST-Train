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
| verl Harbor AgentLoop | ⚠️ assembly logic tested; never run against verl |
| RL task pool + leak guard | ✅ **run locally**: 5,140 tasks / 999 groups materialized, 0 verifier leaks |
| RL rollout code (`rl/generate.py`) | ⚠️ written against real slime APIs, **never executed** |
| RL image prebuild / launcher | ⚠️ written, needs a rootless Docker daemon + cluster |

## Supported models

`MODEL_KEY` selects one; parallelism, loss mask, spec file, vision handling and
serving TP all follow from `configs/models.json`.

```bash
python scripts/model_registry.py --list
MODEL_KEY=qwen3.5-9b RUN_RL=1 bash scripts/20_run_all.sh   # SFT -> eval -> report -> RL
```

`20_run_all.sh` evaluates the **base model on the same harness** as well as the
fine-tuned one, so "did this help?" is answerable even for sizes the paper never
published. It writes `verdict.json` with `in_range`, and the RL stage refuses to
start unless the SFT report has zero FAIL findings.

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
OPERATOR_PROMPT.md             copy-paste kickoff message for the cluster LLM
scripts/
  00_preflight.sh              detect GPU mem / NVLink / IB / shared FS / RAM → config row
  01_setup_env.sh              micromamba `slime` env, A100 deltas (no FlashQLA, no FP8)
  02_download.sh               model + datasets, sha256-verified against the manifests
  03_build_sft_data.py         327,189 trajectories → slime `messages` parquet
  03b_validate_sft_data.py     ports slime's qwen3_5 mask; asserts the training target
  04_convert_ckpt.sh           HF ↔ Megatron torch_dist
  05_run_sft.sh                32-GPU SFT; auto-picks the 80GB/40GB parallelism row
  06_eval.py                   SGLang + Harbor/Terminus-2 on Docker; 3 runs, mean±std
  07_restore_vision.py         splice trained text weights back into the ViT/MTP checkpoint
  10_build_rl_taskset.py       difficulty-tiered GRPO task pool + verifier-leak guard
  11_prebuild_images.py        prebuild/cache task Docker images (refuses default daemon)
  12_run_grpo.sh               32-GPU agentic GRPO (Harbor/Terminus-2 rollout)
  13_upload_hf.py              publish the derived datasets (sanitizes local paths)
  14_make_report.py            markdown report + mechanical anomaly checks
  20_run_all.sh                one command: preflight -> ... -> train -> eval -> report
  15_export_pretokenized.py    bake the verified mask into input_ids+loss_mask
  00b_setup_sandbox.sh         find/start a container runtime (rootless podman)
  30_run_sft_verl.sh           PRIMARY backend: verl + FSDP (no Megatron)
verl_backend/                  verl dataset + Harbor AgentLoop bridge
  model_registry.py            resolve+validate a model's launch config
configs/models.json            the model registry
rl/generate.py                 slime --custom-generate-function-path implementation
data/
  rst-trajectories/            23 GB source release (66 tars, all verified)
  Qwen3.5-27B-tokenizer/       tokenizer only (for local data work)
  sft-v1-cap10/                ★ primary: 10,778 examples = the paper's exact count
  sft-v1/                      ablation: cap 8, 8,886 examples
  rl-sweet/                    5,140 materialized RL tasks (pass-rate 10-90% band)
  rst-tasks/                   3.7 GB task release (8 tars)
probe/                         paper.pdf + the upstream sources I read
.venv/                         local CPU-only env for the data pipeline
```

## Getting the data

The derived datasets are published:

| dataset | contents |
|---|---|
| [`NiuNiu0110/RST-SFT-Qwen3.5-27B`](https://huggingface.co/datasets/NiuNiu0110/RST-SFT-Qwen3.5-27B) | SFT conversations, configs `cap10` (10,778 ex) and `cap8` (8,886 ex) |
| `NiuNiu0110/RST-RL-Taskset` (private) | GRPO task selection metadata, 5,140 `sweet`-tier tasks |

```python
from datasets import load_dataset
ds = load_dataset("NiuNiu0110/RST-SFT-Qwen3.5-27B", "cap10", split="train")
```

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

## Local quick start (no cluster)

```bash
python scripts/03b_validate_sft_data.py \
  --parquet data/sft-v1-cap10/rst_sft_train.parquet \
  --tokenizer data/Qwen3.5-27B-tokenizer --sample 300 --show 1
```

The data pipeline and the checkpoint conversion are CPU/RAM-bound and can be
validated on a single machine before booking the cluster — see `PLAN.md` §4.
