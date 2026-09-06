# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SFT → post-training (DPO by default, agentic GRPO opt-in) of Qwen3.5 models on the
*Recursive Synthesis for Long-Horizon Terminal Tasks* release, targeting 4 nodes × 8 A100.
It is an **operations repo, not a library**: numbered shell launchers plus python scripts,
no package, no `pyproject.toml`, no install step. The deliverable is a reproducible cluster
run and a report that can be argued with.

`PLAN.md` is the executable spec. `README.md` is the map and carries the **status table** —
it distinguishes what was *measured* from what was *written but never executed*. Keep that
distinction alive: many things here are `⏳`/`⚠️`, and a green test run says nothing about them.

## Commands

```bash
# Tests — no GPU, no cluster, no container, no dataset. This is the gate for local work.
.venv/bin/python -m pytest tests/ -q          # ~2 s
python tests/run_tests.py                     # same tests, for an env without pytest
python -m pytest tests/test_loss_mask.py::test_two_ports_agree_on_every_conversation -q   # one test
python tests/run_tests.py test_loss_mask      # one module, pytest-free runner

# Curate a corpus (heuristics alone without RST_JUDGE_BASE_URL; see rst_common/judge.py)
python scripts/03g_curate_sft.py --parquet data/sft-v1-cap10/rst_sft_train.parquet \
  --out-dir /tmp/curated

# Model registry — resolve + validate a launch config before booking GPUs
python scripts/model_registry.py --list
python scripts/model_registry.py --key qwen3.5-27b --mem-class 80GB --gpus 32
python scripts/model_registry.py --key qwen3.5-9b --backend verl --gpus 8

# Local data-pipeline check (CPU, needs the real tokenizer — not covered by tests/)
python scripts/03b_validate_sft_data.py \
  --parquet data/sft-v1-cap10/rst_sft_train.parquet \
  --tokenizer data/Qwen3.5-27B-tokenizer --sample 300 --show 1

# Cluster: everything, resumable
export BASE_FOLDER=/shared/rst MASTER_ADDR=<head-ip> HOSTFILE=$BASE_FOLDER/hostfile
MODEL_KEY=qwen3.5-9b bash scripts/20_run_all.sh              # SFT -> eval -> report -> DPO
MODEL_KEY=qwen3.5-9b RUN_RL=1 bash scripts/20_run_all.sh     # ... plus agentic GRPO
```

`.venv/` is the local CPU-only env for data work; `.venv-gpu/` and `.venv-sglang/` are local
GPU scratch envs. None of them is the cluster env — see "Entering the env" below.

Stage markers live in `$BASE_FOLDER/.stage/`. Delete one to force a re-run, or use
`SKIP_STAGES="env download"`. `20_run_all.sh` writes a report even when a stage fails,
because a report explaining a failure beats no report; the exit code still reflects it.

## Architecture

### `MODEL_KEY` drives everything
`configs/models.json` + `scripts/model_registry.py` are the single source of truth for
parallelism (TP/PP/CP/DP), `max_tokens_per_gpu`, loss-mask type, vision handling, serving TP
and the thinking-template quirk. The registry *validates* (`tp·pp·cp·dp == gpus`,
`tp ≤ gpus_per_node`, `max_tokens_per_gpu·cp ≥ max_seq_len`) and exits non-zero with an
explanation rather than letting an impossible config reach the trainer. `--backend verl`
reshapes a Megatron row (TP/PP/CP → 1, CP folded into the token budget) and says so.

All five models share one byte-identical tokenizer **and** one training-time chat-template
render, so one dataset and `--loss-mask-type qwen3_5` serve every entry. Do not re-tokenize
per model.

### Two backends
verl + FSDP is **primary** (`01b_setup_env_verl.sh`, `30_run_sft_verl.sh`); slime + Megatron
is secondary (`01_setup_env.sh`, `05_run_sft.sh`, `12_run_grpo.sh`). The reason is the
cluster, not preference: Megatron on A100 needs a cuDNN swap a shared cluster will not
permit. `BACKENDS.md` has the full trade, including what leaving slime costs (its
`OpenAIAdapter`, which is why `verl_backend/harbor_agent_loop.py` exists).

### The pre-tokenized data contract
verl's built-in `MultiTurnSFTDataset` tokenizes turns separately and concatenates; for
Qwen3.5 that produces one `<think>` block *per assistant turn* instead of one per
conversation (measured: 200/200 rows mismatch). So the verified mask is baked into the data
once by `scripts/15_export_pretokenized.py` (`input_ids` + `loss_mask`), and verl consumes it
through `data.custom_cls` → `verl_backend/rst_sft_dataset.py`. That module applies
`fsdp2_grad_accum.apply()` **as an import side effect on purpose** — verl loads it in every
rank, so no launcher can bypass the patch. `[rst-fsdp2]` in the log is the proof it ran.

### One definition per measured quantity
This is the organising rule of the repo, and it has a test-enforced boundary. `rst_common/`
holds anything whose duplication could make two numbers non-comparable: `harbor.py` (the
`HARNESS_INFRA` vs `AGENT_BUDGET` split — infra failure is *excluded from the denominator*,
budget failure is *reward 0 inside it* — plus the one `harbor run` command line every
rollout launcher uses), `paper.py` (the published reference numbers), `judge.py` (LLM
supervision). Four more live in `scripts/` because the scripts are their only consumers:

| module | the one thing it owns |
|---|---|
| `siblings.py` | loading a digit-prefixed script by path. Five scripts had their own copy |
| `sft_common.py` | dedup, the chat-template gate, the group-disjoint split, `token_stats` |
| `taskpool_common.py` | the `TIERS` table, the `FROM`-line reader, the verifier-leak rule |
| `hf_publish.py` | `check_card` plus the create-repo/upload sequence for all six `13*` |

`normalize_assistant` stays in `03_build_sft_data.py` and every converter (`03d`/`03e`/`03f`/
`03h`) loads it through `siblings.load_script` rather than reimplementing it, so one
canonical assistant form covers every dataset. `reconstruct_trajectory` in that same file
turns one ATIF record into `messages` and is shared by the release tars and our own Harbor
rollouts. Tests assert the sharing (`test_taskpool_common.py` checks `is`, not `==`, because
a copied tuple can drift), and assert that no script grew a private copy back.

Not everything that looks duplicated is: `command_signature` exists three times on purpose
(RST reads the JSON `commands` key, TMax reads native `tool_calls` arguments, Nemotron must
strip a `<think>` prefix first) because they act on different shapes at different pipeline
stages. The comments say so; don't "fix" it.

### `scripts/` numbering is pipeline order, and the modules are not importable
`00*` preflight/sandbox · `01*` env · `02` download · `03*` SFT data (build, converters,
`03g` curation, `03h` our own rollouts) · `04` ckpt convert · `05`/`30` SFT · `06`/`06b` eval ·
`07`/`08` checkpoint prep · `10*` RL task pools · `11`/`12` GRPO · `13*` HF upload ·
`14` report · `15` pretokenize · `16`–`19` smoke/DPO · `20` run-all · `21` one RSI round ·
`33`–`35` DPO launcher + diagnostics. Unnumbered files (`siblings.py`, `sft_common.py`,
`taskpool_common.py`, `hf_publish.py`, `dpo_common.py`, `model_registry.py`, `resume_guard.py`,
`lib_env.sh`) are libraries, not stages.
Filenames start with digits, so `import scripts.15_export_pretokenized` is a syntax error.
Tests load them with `tests/_util.py::load_script("15_export_pretokenized")`. Scripts that
need `rst_common` insert the repo root into `sys.path` themselves.

### Entering the env
`micromamba activate` inside `01b_setup_env_verl.sh` only affects that script's process, so
setup writes `$BASE_FOLDER/env-<name>.sh` and every launcher `source`s `scripts/lib_env.sh`
and calls `rst_enter_env`, which enters **and then proves it** by locating
`torch`/`transformers`/`pandas`/`pyarrow` (+`verl` on the verl path — a stock ML image
satisfies the first four by accident). `lib_env.sh` is sourced, never executed; it also
holds the failure explainers (`rst_explain_shard_failures`, `rst_explain_nccl_timeout`) that
tests assert against.

### The data loop, and where an LLM may speak
`21_rsi_round.sh` is one self-improvement round: `06_eval.py --export-trajectories` rolls
the checkpoint out and **keeps** the job dirs, `03h_build_rollout_sft.py` turns the
verifier-passed ones into the same messages parquet every other builder writes,
`03g_curate_sft.py` filters them, `15_export_pretokenized.py` bakes the mask. It is
rejection sampling, not RL — `12_run_grpo.sh` is the RL path.

The export flag is load-bearing: Terminus-2 stores its own `Analysis:/Plan:` rendering and
drops the model's completion unless asked for `raw_content` (BUG-19), so `03h` **refuses**
a rendered trajectory rather than inventing text from it.

`rst_common/judge.py` is the only place an LLM opinion enters the pipeline, and its header
states the boundary: never a reward, a loss mask or an eval score — the task's verifier is
the reward, and an LLM in any of those makes the numbers incomparable. It is consulted only
on `03g`'s borderline band (measured: 18% of 30,536 corpus rows), cached by content hash,
budgeted, and degrades to a `NullJudge` that says `judge.enabled: false` in the manifest
rather than pretending the band was reviewed.

### Gates, not hope
Launchers refuse rather than proceed: `30_run_sft_verl.sh` checks the fused-CE switch, the
`transformers` window, `flash_attn`, FLA's real kernel, the rendezvous world size and the
static footprint before starting 32 processes; `08_prepare_eval_ckpt.sh` requires every FSDP
shard and diffs the merged text stack against the base; `resume_guard.py` refuses a resume
whose lr schedule changed under it; `19_train_dpo.py` gates on step-0 loss = log 2.
`14_make_report.py` emits FAIL/WARN/OK findings and `verdict.json` with two flags:
`in_range` (zero FAILs of any kind) gates GRPO, `checkpoint_trustworthy` (zero FAILs *about
the checkpoint*) gates DPO. The one deliberate exemption — "the benchmarks never ran", a
FAIL about the measurement rather than the weights — is `POSTTRAIN_EXEMPT_FAILS`; nothing
else is exempt.

## Conventions that matter here

- **`BUG.md` is the regression index.** Numbered entries (`BUG-1`…`BUG-18`) each state how
  the defect was established — "read the source" and "measured it" are different claims and
  only one survives a version bump — plus what is still open (`OPEN-1`…) and what was
  *checked and found correct, do not re-litigate*. When you fix a real defect, add the entry
  and a test that pins it, and name the BUG in the test's docstring; that is the existing
  pattern for most of `tests/`.
- **Docstrings carry the why and the number.** Every non-trivial module opens with what it
  does, what was measured, and which alternative was rejected and on what evidence. Match
  that density; a bare "converts X to Y" header is out of place here.
- **Tests must run anywhere.** No fixtures (so `run_tests.py` stays a few lines and cannot
  drift from pytest), stdlib-only helpers, and `_util.need("torch")` to *skip loudly* rather
  than fail. A run that skipped everything must not look like a run that verified everything.
- **Never soften an unverified claim.** `UNVERIFIED` / `⏳` / `⚠️` markers in the docs are
  load-bearing; promote a row only when it has actually been executed, and say what executed it.
- **Commit messages are prose imperative sentences** describing intent, not conventional-commit
  prefixes — e.g. "Refuse a resume that silently restarts the lr schedule".
- `data/` (27 GB) and `probe/` are gitignored and rebuilt from the public release; derived
  datasets are published to the Hub (see README). `manifest.json` records every count, so a
  rebuild is checkable rather than trusted. **Before changing a builder, rebuild and diff**:
  the RST/OpenThoughts/TMax parquets are reproducible byte-for-byte, so a refactor that
  moves a published dataset is caught in minutes rather than at the next upload.
- Local envs: `.venv` is the CPU data pipeline (pandas + transformers, no torch),
  `.venv-gpu` and `.venv-sglang` are GPU scratch, and `~/.venvs/AgentGen` carries `harbor`
  and `ruff` but *not* pandas. Pick per task; none of them is the cluster env.
