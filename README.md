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
| Eval harness (SGLang + Harbor + Docker) | ⏳ written, needs cluster |
| RL task pool + leak guard | ✅ **run locally**: 5,140 tasks / 999 groups materialized, 0 verifier leaks |
| RL rollout code (`rl/generate.py`) | ⚠️ written against real slime APIs, **never executed** |
| RL image prebuild / launcher | ⚠️ written, needs a rootless Docker daemon + cluster |

## Layout

```
PLAN.md                        SFT plan; hardware decision tables; risk register
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
  06_eval.sh                   SGLang server + Harbor/Terminus-2 on Docker + scoring
  07_restore_vision.py         splice trained text weights back into the ViT/MTP checkpoint
  10_build_rl_taskset.py       difficulty-tiered GRPO task pool + verifier-leak guard
  11_prebuild_images.py        prebuild/cache task Docker images (refuses default daemon)
  12_run_grpo.sh               32-GPU agentic GRPO (Harbor/Terminus-2 rollout)
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

`data/` is **not in this repo** (27 GB). Rebuild it from the public release — the
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
