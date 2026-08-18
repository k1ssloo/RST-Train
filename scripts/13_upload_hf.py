#!/usr/bin/env python3
"""Publish the derived datasets to the Hugging Face Hub.

    export HF_TOKEN=hf_...
    python scripts/13_upload_hf.py --owner <hf-user> --data-root data [--dry-run]

Two repos, deliberately different visibility:

  <owner>/RST-SFT-Qwen3.5-27B   PUBLIC   two configs (cap10 default, cap8 ablation)
  <owner>/RST-RL-Taskset        PRIVATE  selection metadata only, no task bodies

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
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    root = args.data_root

    sft_repo = f"{args.owner}/{SFT_REPO}"
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
    if rl_src.is_file():
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

    for src, _ in uploads + rl_uploads:
        if not src.is_file():
            sys.exit(f"missing input: {src}")

    print(f"SFT  -> {sft_repo}  (public)")
    for src, dst in uploads:
        print(f"   {src}  ->  {dst}  ({src.stat().st_size/2**20:.1f} MB)")
    print(f"RL   -> {rl_repo}  (PRIVATE, metadata only)")
    for src, dst in rl_uploads:
        print(f"   {src}  ->  {dst}  ({src.stat().st_size/2**20:.1f} MB)")
    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return 0

    for repo, private, card, files in (
        (sft_repo, False, SFT_CARD, uploads),
        (rl_repo, True, RL_CARD, rl_uploads),
    ):
        api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
        print(f"\n[{repo}] created/exists (private={private})")
        for src, dst in files:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                            repo_id=repo, repo_type="dataset")
            print(f"  uploaded {dst}")
        api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                        repo_id=repo, repo_type="dataset")
        print("  uploaded README.md")

    print(f"\nhttps://huggingface.co/datasets/{sft_repo}")
    print(f"https://huggingface.co/datasets/{rl_repo}  (private)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
