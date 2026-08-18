#!/usr/bin/env python3
"""Publish the derived datasets to the Hugging Face Hub.

    export HF_TOKEN=hf_...
    python scripts/13_upload_hf.py --owner <hf-user> --data-root data [--dry-run]

Three repos, deliberately different visibility:

  <owner>/RST-SFT-Qwen3.5-27B   PUBLIC   two configs (cap10 default, cap8 ablation)
  <owner>/RST-DPO-Qwen3.5-27B   PUBLIC   preference pairs, pre-tokenized
  <owner>/RST-RL-Taskset        PRIVATE  selection metadata only, no task bodies

`--only sft|dpo|rl` uploads one of them; the default touches all three.

Why the RL repo is metadata-only: the materialized task dirs contain each task's
`solution/solve.sh` and `tests/` (reference solution + verifier). Upstream already
publishes those under CC-BY-4.0, so re-hosting them leaks nothing new — but a
ready-made public copy makes "did the model see the answer?" harder to argue for
anyone evaluating on these tasks. `scripts/10_build_rl_taskset.py` rebuilds the
directories from upstream in ~15s, so the copy buys nothing and costs clarity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SFT_REPO = "RST-SFT-Qwen3.5-27B"
DPO_REPO = "RST-DPO-Qwen3.5-27B"
RL_REPO = "RST-RL-Taskset"

SFT_CARD = """---
license: cc-by-4.0
task_categories:
- text-generation
language:
- en
tags:
- terminal-agent
- agentic
- sft
- qwen3.5
- recursive-task-synthesis
pretty_name: RST SFT trajectories for Qwen3.5-27B
size_categories:
- 10K<n<100K
configs:
- config_name: cap10
  data_files:
  - split: train
    path: data/cap10/train.parquet
  - split: holdout
    path: data/cap10/holdout.parquet
- config_name: cap8
  data_files:
  - split: train
    path: data/cap8/train.parquet
  - split: holdout
    path: data/cap8/holdout.parquet
- config_name: cap10_pretokenized
  data_files:
  - split: train
    path: data/cap10_pretokenized/train.parquet
  - split: holdout
    path: data/cap10_pretokenized/holdout.parquet
---

# RST SFT trajectories for Qwen3.5-27B

Multi-turn terminal-agent conversations distilled from
[Zhongzhi1228/Recursive-Task-Synthesis-Trajectories](https://huggingface.co/datasets/Zhongzhi1228/Recursive-Task-Synthesis-Trajectories),
ready for supervised fine-tuning of `Qwen/Qwen3.5-27B`.

Pipeline, launchers, and the full plan: **https://github.com/k1ssloo/RST-Train**

## `cap10` reproduces the paper's SFT example count exactly

The source release has 327,189 trajectories. `cap10` ends at **10,778 examples** —
the count [arXiv:2608.05466v3](https://arxiv.org/abs/2608.05466v3) states it trained
on. That was not tuned toward; it fell out of the filter chain below. Read it as
strong corroboration that the paper used a per-group cap of ~10 with essentially
this filtering, not as proof.

```
327,189 trajectories
  ├─ gate: status=completed ∧ has_trajectory ∧ ¬has_exception
  │        ∧ reward=1.0 ∧ task_present_in_task_dataset
  │   → 60,932 clean successes over 1,338 task groups
  ├─ per-group cap (round-robin across the 4 generating models) → 11,582
  ├─ reconstruct ATIF-v1.7 → messages, normalize assistant JSON → 11,090
  ├─ dedup (exact + per-group command signature)                → 11,010
  └─ drop > 32,768 tokens                                       → 10,778
```

| config | examples | train / holdout | tokens | groups | steps/epoch @ GBS 128 |
|---|---|---|---|---|---|
| `cap10` (default) | **10,778** | 10,578 / 200 | 99.9 M | 1,329 | 82 |
| `cap8` (ablation) | 8,886 | 8,686 / 200 | 82.4 M | 1,327 | 67 |
| `cap10_pretokenized` | 10,778 | 10,578 / 200 | 99.9 M | 1,329 | 82 |

Group-capping is the point: successes per group are median 28, max 284, so
uncapped training would be dominated by a handful of lineages.

## Schema

| field | type | notes |
|---|---|---|
| `messages` | `list<{role, content}>` | the conversation; see the shape below |
| `trajectory_id` | string | upstream id, for provenance |
| `task_group_id` | string | upstream group; use it for grouped splits |
| `model_name` | string | which model generated the trajectory |
| `n_tokens` | int | full sequence length under the Qwen3.5 chat template |
| `n_assistant_turns` | int | mean 12.0, max 60 |
| `n_rewritten_turns` | int | assistant turns whose JSON was renormalized |

```
messages[0]  role=user       full Terminus-2 harness prompt + task + initial screen
messages[1]  role=assistant  canonical JSON {analysis, plan, commands[, task_complete]}
messages[2]  role=user       terminal observation
...
```

`messages[0]` is **`user`, not `system`**, because that is how Terminus-2 delivers
the harness prompt (`steps[0].source == "user"` upstream). Keeping it as `user`
makes training and serving identical — changing it introduces a train/serve skew.

## Two processing details that matter

**1. Assistant JSON was renormalized.** 62.6 % of upstream assistant turns are
wrapped in ```` ```json ```` fences or carry extra prose; 0.1 % are unparseable
(dropped). Turns are re-serialized to canonical `indent=2` JSON preserving key
order.

**2. The warning preamble was repaired.** When an upstream turn was fenced, the
*following* observation begins `Previous response had warnings: - Extra text
detected before JSON object`. Normalizing the assistant turn without stripping
that preamble trains the model to accept "you had warnings" feedback for clean
output. **50,169 observations needed this repair** in `cap10`.

## `cap10_pretokenized`: the same data with the mask already applied

| field | type | meaning |
|---|---|---|
| `input_ids` | `list[int]` | the exact tokens of the whole-conversation render |
| `loss_mask` | `list[int]` | 1 = train on this token, 0 = context only. Aligned 1:1 with `input_ids`. |

Same 10,778 examples, 99,939,485 tokens, **32,402,050 trained tokens (32.42 %)**.
It is *smaller* than the `messages` version (75 MB vs 87 MB) because token ids
compress better than JSON text.

To get `labels`, set `labels[i] = input_ids[i]` where `loss_mask[i] == 1` else
`-100`. Do **not** shift — HuggingFace models shift internally.

### Why you may want this instead of `messages`

Building the mask yourself is the easiest place in this pipeline to be silently
wrong: a bad mask still trains, the loss still falls, and the model just comes out
worse. Two concrete traps:

**1. Do not let a trainer re-tokenize turn-by-turn.** verl's `MultiTurnSFTDataset`
templates each message separately and concatenates. Measured on 200 rows of this
dataset, **200/200 disagree** with the whole-conversation render, because the Qwen3.5
template injects an empty `<think>\n\n</think>\n\n` before the **last** assistant
turn — so turn-by-turn building makes every turn "last" and a 21-turn conversation
ends up with 21 think blocks instead of 1. verl's `ignore_input_ids_mismatch: True`
silences the assertion, not the bug. Using `cap10_pretokenized` avoids this entirely.

**2. Budget for the logits, not the model.** This tokenizer's vocab is 248,320, and
the loss upcasts logits to fp32. Measured on one H100-80GB with **Qwen3.5-0.8B**
(0.75 B params!), a real forward/backward over these rows:

| sequence length | peak, unfused CE | peak, fused CE (Liger) |
|---|---|---|
| 4,096 | 14.98 GiB | 5.52 GiB |
| 8,192 | 28.43 GiB | 6.75 GiB |
| ~16,000 | 48.34 GiB | 8.57 GiB |
| 32,329 | **out of memory** | 13.14 GiB |

At 32,329 tokens the unfused cross-entropy asks for a single **29.85 GiB** tensor
(`seq × 248,320 × 4 bytes`) and dies. That term is independent of model size, so a
fused/chunked cross-entropy is effectively required at long sequence length
regardless of which model you train.

## Loss masking

Built for [slime](https://github.com/THUDM/slime)'s `--loss-mask-type qwen3_5`,
which trains only assistant content. Verified before release: **0 chat-template
contract failures, 0 user-turn leakage, 32.6 % of tokens trained**. The default
`--loss-mask-type qwen` mis-segments this template and would train on terminal
output — do not use it.

Each message may also carry `step_loss_mask: 0` to exclude a single assistant turn
from the loss while keeping it as context. Unused here; available as a lever.

## Limitations

- **Reward-verified, not exact-environment-replay-verified.** In a 500-sample
  check only 46 instructions mapped exactly to a public task. These are
  verifier-passing trajectories; they are not a claim that each was replayed in a
  byte-identical environment.
- **Source mix is skewed** toward one iterated model (`qwen35-27b-iter0000161-hf`,
  ~63 %). Rebalance via `--models` in the builder if that matters to you.
- 232 trajectories were dropped for exceeding 32,768 tokens, which biases mildly
  against the longest-horizon episodes.
- Success-only. The 166,660 clean *failures* are not here; 1,279 groups have both
  successes and failures and are a ready-made offline preference set.

## Attribution

Derived from `Zhongzhi1228/Recursive-Task-Synthesis-Trajectories`
(CC-BY-4.0) by Zhongzhi1228 et al., *Recursive Synthesis for Long-Horizon Terminal
Tasks* (arXiv:2608.05466). Released under the same license. Trajectories were
filtered, reconstructed, normalized, deduplicated, and re-serialized; no new
rollouts were generated.
"""

DPO_CARD = """---
license: cc-by-4.0
task_categories:
- text-generation
language:
- en
tags:
- terminal-agent
- agentic
- dpo
- preference
- qwen3.5
- recursive-task-synthesis
pretty_name: RST DPO preference pairs for Qwen3.5
size_categories:
- 1K<n<10K
configs:
- config_name: v2
  data_files:
  - split: train
    path: data/v2/train.parquet
  - split: holdout
    path: data/v2/holdout.parquet
---

# RST DPO preference pairs for Qwen3.5

**2,673 preference pairs** (2,448 train / 225 holdout) built from
[Zhongzhi1228/Recursive-Task-Synthesis-Trajectories](https://huggingface.co/datasets/Zhongzhi1228/Recursive-Task-Synthesis-Trajectories).
Each pair is two agent runs **on the same task**: one whose trajectory the task's own
verifier scored reward 1, one it scored 0.

Builder, trainer, and the numerical gates: **https://github.com/k1ssloo/RST-Train**
(`scripts/17_build_dpo_data.py`, `scripts/19_train_dpo.py`, `DPO_PLAN.md`).

## What the preference actually encodes

**Outcome, not model identity.** Cross-model pairs are excluded
(`cross_model_allowed: false`), so both sides of every pair come from the *same*
generating model — 2,673 / 2,673 are same-model. Without that rule a "preference"
could be learned as "prefer whatever the stronger model wrote", which is a different
and much less useful signal than "prefer the run that solved the task".

Source models, identical on both sides by construction: `qwen35-27b-iter0000161-hf`
1,419 / `Qwen3.5-27B` 591 / `Qwen3.6-27B-base` 412 / `gpt-oss-120b` 251.

## Pre-tokenized, with the verified loss mask

| column | meaning |
|---|---|
| `chosen_input_ids` / `rejected_input_ids` | `list[int]`, **whole-conversation** render |
| `chosen_loss_mask` / `rejected_loss_mask` | aligned 1:1 with `input_ids`, no offset; `1` = a token the policy produced |
| `pair_id`, `prompt_sha256`, `task_group_id` | content-addressed ids |
| `chosen_trajectory_id` / `rejected_trajectory_id` | upstream trajectory ids |
| `chosen_model` / `rejected_model`, `same_model` | provenance |
| `chosen_n_trained` / `rejected_n_trained` | supervised token counts |
| `common_prefix_tokens`, `prompt_tokens`, `prompt_divergence_tokens` | prompt-agreement evidence |

Tokenizer: `Qwen/Qwen3.5-27B` — **byte-identical across the five Qwen3.5 sizes**
(0.8B / 4B / 9B / 27B / 35B-A3B), so these ids are valid for any of them. The mask is
slime's `gen_multi_turn_loss_mask_qwen3_5`, the same implementation the SFT export
verified, so a consumer never recomputes it and never disagrees with it.

## Yield, and why it is not larger

231,092 clean trajectories → 1,290 task groups with **both** outcomes → 5,759
canonical prompts (3,884 one-sided, i.e. 67 % have only wins or only losses) → 2,820
candidate pairs → **2,673** after dropping 147 over-length sides at 32,768 tokens.
Pairing must be per *task variant*, and a variant is only known after reconstruction;
that is the bottleneck, not the filter thresholds. A `--per-side 5` build yields 1,330
pairs, `--per-side 14` (this set) yields 2,673.

**The prompt trap, if you rebuild:** hashing prompts verbatim yields ~0 pairs, because
each run's prompt ends with a per-container UUID hostname. UUIDs and 12-hex docker
hostnames are masked **for grouping only** — the trained text is verbatim, always.
Median residual divergence inside a pair's prompt is 40 tokens (max 57), which is
exactly that hostname.

## Length bias is measured, not assumed

DPO's objective rewards length unless you check: 47.18 % of pairs have the longer side
as `rejected` (a coin flip is the healthy value), median rejected/chosen token ratio
0.9815. `length_bias_warning: null`. Train with per-token normalization
(`--length-normalize`) unless you have a reason not to, and say which you used.

## Read the two caveats before quoting a number

- **This is off-policy.** It reweights behaviour already present in other policies'
  logged trajectories, so it can sharpen modes a model already has and **cannot**
  discover a strategy no logged trajectory used. It is not an RL result.
- `holdout_reward_accuracy` from the trainer is **likelihood ranking** on held-out
  task groups: how often the model assigns higher likelihood to the run that passed
  than to the run that failed. **0.5 means no preference**, not 50 % of tasks solved.
  It is not a pass rate and must not be compared with terminal-bench numbers.

The 225 holdout pairs come from **80 task groups disjoint from train**, so a holdout
number is not measuring memorized prompts.

## Attribution

Derived from `Zhongzhi1228/Recursive-Task-Synthesis-Trajectories` (CC-BY-4.0) by
Zhongzhi1228 et al., *Recursive Synthesis for Long-Horizon Terminal Tasks*
(arXiv:2608.05466). Same license. Trajectories were filtered, reconstructed, paired
and tokenized; no new rollouts were generated and no verifier was re-run.
"""

RL_CARD = """---
license: cc-by-4.0
task_categories:
- reinforcement-learning
language:
- en
tags:
- terminal-agent
- agentic-rl
- grpo
- recursive-task-synthesis
pretty_name: RST GRPO task pool (difficulty-tiered)
configs:
- config_name: default
  data_files:
  - split: train
    path: rl_tasks.jsonl
---

# RST GRPO task pool (difficulty-tiered)

Task **selection metadata** for agentic GRPO on
[Zhongzhi1228/Recursive-Task-Synthesis](https://huggingface.co/datasets/Zhongzhi1228/Recursive-Task-Synthesis).
Code: **https://github.com/k1ssloo/RST-Train** (`scripts/10_build_rl_taskset.py`,
`rl/generate.py`).

**Metadata only — no task bodies.** Rebuild the task directories from upstream in
~15 s with `scripts/10_build_rl_taskset.py --materialize`. They are omitted on
purpose: each contains `solution/solve.sh` and `tests/`, and a ready-made public
copy would make evaluation-contamination arguments harder for everyone, without
adding anything upstream doesn't already provide.

## Why tiering by difficulty is the point

GRPO computes advantage *within* a group of rollouts on the same prompt. If all
rollouts score the same, the advantage is identically zero — the group costs a full
set of sandboxes and contributes nothing to the gradient. The paper's reward curve
sat at 0.11 → 0.14, i.e. most of its groups were all-fail.

The trajectory release lets you avoid that. Empirical pass rate per group, measured
over 231,092 clean trajectories (only 2,246 of 12,010 groups have any data):

| pass rate | groups | tier |
|---|---|---|
| 0 % (all fail) | 897 | `hard` — zero advantage, pure cost |
| 0–10 % | 252 | `hard` |
| 10–35 % | 469 | **`sweet`** |
| 35–65 % | 394 | **`sweet`** |
| 65–90 % | 144 | **`sweet`** |
| > 90 % | 90 | `easy` — little left to learn |
| no data | 9,764 | `unknown` |

Task-level tiers over the 37,484 validated tasks: **sweet 5,140 / hard 4,089 /
easy 196 / unknown 28,059**. This file contains the **5,140 `sweet` tasks** across
999 groups, mean pass rate 0.393.

## Fields

One JSON object per line, in slime's prompt-data shape
(`--input-key prompt --label-key label --metadata-key metadata`):

| field | notes |
|---|---|
| `prompt` | the task instruction (the real prompt is built by Terminus-2 at rollout time) |
| `label` | `task_id` |
| `metadata.task_id` / `task_group_id` | upstream ids |
| `metadata.task_dir` | local path, **rewrite after materializing** |
| `metadata.task_content_sha256` | upstream content hash |
| `metadata.base_image` | Docker `FROM` (68 distinct across this pool) |
| `metadata.tier` | `sweet` here |
| `metadata.empirical_pass_rate` / `n_reference_trials` | the difficulty signal |

## Rollout prerequisites (measured)

- **99 % of the 37,484 task Dockerfiles install packages at build time.** Prebuild
  and cache; lazy building turns every rollout into a network-bound build.
- **710 of these 5,140 tasks (13.8 %) are docker-compose multi-service**; they need
  `docker compose` and more RAM per rollout.
- 68 distinct base images (`ubuntu:22.04` 2,677 / `ubuntu:20.04` 356 /
  `python:3.11-slim` 320 / `centos:7` 200 / …).
- Untrusted third-party build scripts — build on a dedicated/rootless daemon.
- **Verifier-leak check: 0 leaks across all 5,140 build contexts.** (46 tasks do
  contain `environment/tests/`, but that is the *project's own* fixtures — a PHPUnit
  suite, Ansible playbooks, JSON files — not the RST verifier, which lives at
  task-root `tests/`. The guard checks verifier filenames and byte-identical
  content, not directory names.)

## Attribution

Derived from `Zhongzhi1228/Recursive-Task-Synthesis` and
`…-Trajectories` (CC-BY-4.0), Zhongzhi1228 et al., arXiv:2608.05466. Same license.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True, help="HF user or org")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", choices=("all", "sft", "dpo", "rl"), default="all",
                        help="upload one repo instead of all three")
    parser.add_argument("--dpo-dir", default="dpo-v2",
                        help="which local DPO build to publish (default: the adopted one)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    root = args.data_root

    sft_repo = f"{args.owner}/{SFT_REPO}"
    dpo_repo = f"{args.owner}/{DPO_REPO}"
    rl_repo = f"{args.owner}/{RL_REPO}"

    uploads: list[tuple[Path, str]] = [
        (root / "sft-v1-cap10/rst_sft_train.parquet", "data/cap10/train.parquet"),
        (root / "sft-v1-cap10/rst_sft_holdout.parquet", "data/cap10/holdout.parquet"),
        (root / "sft-v1-cap10/manifest.json", "manifest_cap10.json"),
        (root / "sft-v1/rst_sft_train.parquet", "data/cap8/train.parquet"),
        (root / "sft-v1/rst_sft_holdout.parquet", "data/cap8/holdout.parquet"),
        (root / "sft-v1/manifest.json", "manifest_cap8.json"),
        # Pre-tokenized variant of cap10: input_ids + loss_mask with the verified
        # qwen3_5 mask already applied. Smaller than the messages version (token ids
        # compress better than JSON) and it removes any chance of a consumer
        # recomputing the mask differently.
        (root / "sft-v1-cap10/pretokenized_train.parquet", "data/cap10_pretokenized/train.parquet"),
        (root / "sft-v1-cap10/pretokenized_holdout.parquet", "data/cap10_pretokenized/holdout.parquet"),
        (root / "sft-v1-cap10/pretokenized_train_manifest.json", "manifest_cap10_pretokenized.json"),
    ]
    # The local jsonl carries absolute `task_dir` paths from whatever machine built
    # it. Publish a portable copy instead: `tasks/<task_id>`, to be resolved against
    # the root that `10_build_rl_taskset.py --materialize` writes.
    rl_src = root / "rl-sweet/rl_tasks.jsonl"
    rl_portable = root / "rl-sweet/rl_tasks.portable.jsonl"
    if rl_src.is_file() and args.only in ("all", "rl"):
        rewritten = 0
        with rl_src.open(encoding="utf-8") as fin, rl_portable.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                md = row.get("metadata") or {}
                if md.get("task_dir"):
                    md["task_dir"] = f"tasks/{md['task_id']}"
                    rewritten += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[sanitize] rewrote {rewritten} absolute task_dir paths -> tasks/<task_id>")

    rl_uploads: list[tuple[Path, str]] = [
        (rl_portable, "rl_tasks.jsonl"),
        (root / "rl-sweet/manifest.json", "manifest.json"),
    ]

    # The pairs are pre-tokenized ids plus content-addressed provenance -- no text, no
    # absolute paths (checked: the manifest's paths are repo-relative), and nothing from
    # any task's solution/ or tests/. Same source and license as the SFT repo, so this
    # is public. The reconstructed.jsonl.gz cache is deliberately NOT published: it is
    # 226 MB of intermediate state that 17_build_dpo_data.py regenerates.
    dpo_root = root / args.dpo_dir
    dpo_uploads: list[tuple[Path, str]] = [
        (dpo_root / "dpo_train.parquet", "data/v2/train.parquet"),
        (dpo_root / "dpo_holdout.parquet", "data/v2/holdout.parquet"),
        (dpo_root / "manifest.json", "manifest.json"),
    ]

    jobs: list[tuple[str, str, bool, str, list[tuple[Path, str]]]] = [
        ("SFT", sft_repo, False, SFT_CARD, uploads),
        ("DPO", dpo_repo, False, DPO_CARD, dpo_uploads),
        ("RL ", rl_repo, True, RL_CARD, rl_uploads),
    ]
    if args.only != "all":
        jobs = [j for j in jobs if j[0].strip().lower() == args.only]

    for _, _, _, _, files in jobs:
        for src, _dst in files:
            if not src.is_file():
                sys.exit(f"missing input: {src}")

    for name, repo, private, _card, files in jobs:
        vis = "PRIVATE" if private else "public"
        print(f"{name} -> {repo}  ({vis})")
        for src, dst in files:
            print(f"   {src}  ->  {dst}  ({src.stat().st_size/2**20:.1f} MB)")
    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return 0

    for _name, repo, private, card, files in jobs:
        api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
        print(f"\n[{repo}] created/exists (private={private})")
        for src, dst in files:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                            repo_id=repo, repo_type="dataset")
            print(f"  uploaded {dst}")
        api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                        repo_id=repo, repo_type="dataset")
        print("  uploaded README.md")

    for _name, repo, private, _card, _files in jobs:
        print(f"\nhttps://huggingface.co/datasets/{repo}" + ("  (private)" if private else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
