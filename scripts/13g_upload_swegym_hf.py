#!/usr/bin/env python3
"""Publish the tiered SWE-Gym RL task pool to the Hugging Face Hub.

    export HF_TOKEN=hf_...
    python scripts/13g_upload_swegym_hf.py --owner <hf-user> [--dry-run]

  <owner>/SWE-Gym-RL-Taskset   PRIVATE   prompts + verifier spec + measured tiers

Private by default, and for a sharper reason than `Termigen-RL-Taskset`: this pool
carries `test_patch`, `FAIL_TO_PASS` and `PASS_TO_PASS` for every instance, which is
the complete verifier. It has to -- without them nothing can score a rollout. The gold
`patch` is excluded outright, and `hints_text` never appears as text.

Upstream (`SWE-Gym/SWE-Gym`, MIT) is public, so nothing here is a new disclosure. What
`--public` would add is a second, tier-annotated copy of 2,438 SWE-bench verifiers
sitting next to the prompts, which makes "was this contaminated?" harder to answer for
everyone who later evaluates on SWE-Gym. Pass it only deliberately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hf_publish import check_card, publish  # noqa: E402

REPO = "SWE-Gym-RL-Taskset"

CARD = """---
license: mit
language:
- en
tags:
- swe-bench
- software-engineering
- agentic
- reinforcement-learning
- rlvr
- grpo
pretty_name: SWE-Gym GRPO pool, tiered by 6,055 measured rollouts
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: rl_tasks.jsonl
---

# SWE-Gym GRPO pool, tiered by 6,055 measured rollouts

**2,438 SWE-bench task instances** from
[`SWE-Gym/SWE-Gym`](https://huggingface.co/datasets/SWE-Gym/SWE-Gym), in the
prompt-data shape this repo's GRPO stage consumes, with a **measured pass rate on every
single one**.

Builder and tests: **https://github.com/k1ssloo/RST-Train**
(`scripts/10c_build_swegym_taskset.py`, `tests/test_swegym_taskset.py`).

## This is not SFT data

`SWE-Gym/SWE-Gym` is SWE-bench format — `instance_id`, `problem_statement`, `patch`,
`test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `repo`, `base_commit` — with **no
`messages` or `conversations` column at all**. There is nothing to supervise on. The
builder asserts the absence rather than assuming it, so if upstream ever adds
trajectories the run fails loudly instead of silently discarding them.

Its trajectories do exist, in
[`SWE-Gym/OpenHands-SFT-Trajectories`](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories)
— which is exactly the 491 `resolved` rollouts out of
`OpenHands-Sampled-Trajectories`' 6,055. They are **not** converted into this repo's
SFT format, deliberately: they are in the OpenHands/CodeAct action space,

```
<function=str_replace_editor>
<parameter=command>view</parameter>
<parameter=path>/workspace/...</parameter>
</function>
```

which has no terminus-2 equivalent. Re-expressing it as
`{"analysis", "plan", "commands": [{"keystrokes", "duration"}]}` would mean inventing
the analysis and plan text and turning structured edits into shell heredocs — writing a
guess into supervision. Converted faithfully it would instead be a **491-row** corpus
(4 optimizer steps per epoch at global batch 128) in an action space this repo's
Harbor/Terminus-2 eval cannot drive, distilled from `gpt-4o-2024-08-06`.

## The number to plan around: 88 % of this pool cannot produce a gradient

GRPO's advantage is computed *within* a group of rollouts on one prompt. If every
rollout in a group scores the same, the advantage is identically zero and the group
contributes **nothing** to the gradient while paying the full sandbox cost.

`OpenHands-Sampled-Trajectories` carries a `resolved` boolean per rollout and covers
**all 2,438** instances, so unlike most task pools this one is fully screened:

| tier | pass rate | instances | share | |
|---|---|---|---|---|
| `hard` | 0 – 10 % | **2,144** | 88.0 % | all-fail — zero gradient, pure cost |
| `sweet` | 10 – 90 % | **187** | 7.7 % | the only band with reliable within-group variance |
| `easy` | ≥ 90 % | 107 | 4.4 % | near-saturated, little left to learn |

Mean pass rate **7.4 %**, median **0.0 %**. A naive GRPO run over `default` spends
roughly nine tenths of its sandbox budget on groups that cannot teach it anything.
Build the pool with `--tier sweet` unless you have a reason not to.

**These rates are `gpt-4o-2024-08-06`'s.** A pass rate is not a property of a task
alone. They transfer to a stronger policy as an *ordering*, not as absolutes — a better
model shifts everything up and moves instances out of `hard` into `sweet`. Do not quote
them as intrinsic difficulty. The `run_id`s they came from are in `manifest.json`.

## Two things left out of the output

**The gold patch.** All 2,438 instances ship `patch`, the reference solution. Nothing
needs it at rollout time, and a ready-made copy next to the prompts makes "did the model
see the answer?" impossible to argue. Excluded entirely — it is one `hf download` from
upstream for anyone who legitimately needs it.

**`hints_text`.** Non-empty on **1,528 of 2,438** instances (63 %), and it is maintainer
discussion that frequently names the fix. Every rollout that produced the pass rates
above came from a `no-hint` run, so putting hints in the prompt would also make the
measured tiers describe a different task than the one being served. Recorded as the
boolean `hints_text_available_upstream`, never as text.

`test_patch`, `FAIL_TO_PASS` and `PASS_TO_PASS` **are** included — they are what scores
a rollout, and the pool is not runnable without them. That is why this repo is private.

They live in **`verifier_spec.parquet`** (2,438 rows, keyed by `instance_id`), not inline
in the jsonl. `PASS_TO_PASS` averages 751 test names and peaks at 29,737, so inlining it
made a 191 MiB jsonl of which 173 MiB was that one field; columnar + compressed it is
34 MiB, and `rl_tasks.jsonl` stays 7.7 MiB and greppable. It also restores this repo's
convention that the prompt file carries prompt plus metadata while the verifier sits
beside it.

```python
import pandas as pd
spec = pd.read_parquet("verifier_spec.parquet").set_index("instance_id")
row  = spec.loc["getmoto__moto-7365"]          # test_patch, FAIL_TO_PASS, PASS_TO_PASS
```

## Schema

`rl_tasks.jsonl`, one JSON object per line:

| field | meaning |
|---|---|
| `prompt` | the `problem_statement`, alone |
| `label` | `instance_id`, passed as `--label-key label` |
| `metadata.task_group_id` | the instance — each SWE-bench instance is its own prompt. Not `repo`: 2,438 instances share only 11 repos, and grouping by repo would make a group-disjoint split discard ~9 % of the pool per held-out repo |
| `metadata.repo` / `base_commit` / `version` | what the swebench harness needs to build the environment |
| `metadata.environment_setup` | a reminder that **this pool ships no Dockerfile** |
| `metadata.n_fail_to_pass` / `n_pass_to_pass` / `verifier_sha256` | verifier size and identity; the spec itself is in `verifier_spec.parquet` |
| `metadata.tier` / `empirical_pass_rate` / `n_reference_trials` | measured, from the rollouts above |
| `metadata.hints_text_available_upstream` | boolean |
| `metadata.gold_patch_excluded` | always `true` |
| `metadata.problem_statement_sha256` | so a pool can be pinned |

**No Dockerfile, unlike the termigen pool.** SWE-bench environments are built by the
`swebench` harness from `repo` + `base_commit` + `version`, so this is not drop-in for
this repo's Harbor/Terminus-2 rollout path — it needs a SWE-bench adapter. That work
does not exist here yet; the pool is published so it is ready when it does.

Repo coverage is concentrated: `pandas-dev/pandas` 737, `Project-MONAI/MONAI` 374,
`getmoto/moto` 343, `python/mypy` 257, `iterative/dvc` 225, then a tail — 11 repos
total. Worth knowing before reading a per-repo result as a general one.

## Reproducing

```bash
hf download SWE-Gym/SWE-Gym --repo-type dataset --local-dir swegym-pool
hf download SWE-Gym/OpenHands-Sampled-Trajectories --repo-type dataset --local-dir swegym-rollouts
python scripts/10c_build_swegym_taskset.py \\
    --pool swegym-pool/data/train-00000-of-00001.parquet \\
    --rollouts 'swegym-rollouts/data/*.parquet' \\
    --out data/swegym --tier sweet
```

## Attribution

Derived from [`SWE-Gym/SWE-Gym`](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) and
`SWE-Gym/OpenHands-Sampled-Trajectories` (MIT), and released under the same licence.
Instances were gated, tiered from measured rollout outcomes, stripped of the gold patch
and hint text, and rewritten into this repo's GRPO prompt-data shape. No rollouts were
generated and no verifier was executed.
"""

CLAIMS: list[tuple[str, str, tuple[str, ...], object]] = [
    ("pool instances", "manifest.json", ("pool_instances",), 2438),
    ("tasks selected", "manifest.json", ("tasks_selected",), 2438),
    ("hard tier", "manifest.json", ("tier_counts_all_instances", "hard"), 2144),
    ("sweet tier", "manifest.json", ("tier_counts_all_instances", "sweet"), 187),
    ("easy tier", "manifest.json", ("tier_counts_all_instances", "easy"), 107),
    ("zero-gradient fraction", "manifest.json", ("zero_gradient_fraction",), 0.8794),
    ("rollouts used", "manifest.json", ("rollouts_used",), 6055),
    ("distinct repos", "manifest.json", ("distinct_repos",), 11),
    ("hints upstream", "manifest.json", ("hints_text_available_upstream_count",), 1528),
    ("gold patch excluded", "manifest.json", ("gold_patch_excluded",), True),
    ("pairable instances", "manifest.json", ("dpo_verdict", "pairable_instances"), 188),
    ("dpo pairs", "manifest.json", ("dpo_verdict", "pairs_at_min_per_instance"), 291),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", required=True, help="HF user or org")
    parser.add_argument("--src", type=Path, default=Path("data/swegym"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--public", action="store_true",
                        help="publish publicly. Read the docstring first: the default is "
                             "private because this pool's verifiers are the reward signal "
                             "for anything evaluated on it.")
    args = parser.parse_args()

    files: list[tuple[Path, str]] = [
        (args.src / "rl_tasks.jsonl", "rl_tasks.jsonl"),
        (args.src / "verifier_spec.parquet", "verifier_spec.parquet"),
        (args.src / "manifest.json", "manifest.json"),
    ]
    if check_card(args.src, CLAIMS):
        sys.exit("the card disagrees with the manifests; fix one of them before publishing")
    return publish(repo=f"{args.owner}/{REPO}", files=files, card=CARD,
                   private=not args.public, dry_run=args.dry_run, kind="taskset",
                   notes=("gold patches excluded; hints_text never included as text",),
                   missing_hint="Run scripts/10c_build_swegym_taskset.py first")


if __name__ == "__main__":
    raise SystemExit(main())
