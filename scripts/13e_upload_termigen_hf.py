#!/usr/bin/env python3
"""Publish the termigen GRPO task pool to the Hugging Face Hub.

    export HF_TOKEN=hf_...
    python scripts/13e_upload_termigen_hf.py --owner <hf-user> [--dry-run]

  <owner>/Termigen-RL-Taskset   PRIVATE   selection metadata only, no task bodies

WHY PRIVATE, AND WHY METADATA ONLY
----------------------------------
Same reasoning `13_upload_hf.py` applies to `RST-RL-Taskset`, and it applies harder
here. The materialized task dirs contain each task's `tests/test.sh` and
`tests/test_outputs.py` -- the verifier. Upstream already publishes all of it inside
`task-data.tar.gz`, so re-hosting leaks nothing genuinely new, but a ready-made
public copy of 3,541 graders makes "did the model see the answer?" harder to argue
for anyone evaluating on these tasks. `10b_build_termigen_taskset.py` rebuilds the
directories from upstream in about a minute, so the copy buys nothing and costs
clarity.

What goes up is `rl_tasks.jsonl` -- instruction text plus selection metadata, which
is what makes this pool reproducible -- and the leak report, which is the part
someone else would otherwise have to rediscover.

THIS IS NOT SFT DATA
--------------------
Stated in the card because the name of the upstream dataset does not say so and two
sibling datasets have already caused this confusion. All 3,556 upstream rows are
`(system, user)` with zero assistant turns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = "Termigen-RL-Taskset"

CARD = """---
license: odc-by
language:
- en
tags:
- terminal-agent
- agentic
- reinforcement-learning
- rlvr
- grpo
pretty_name: Termigen GRPO task pool, screened for verifier leaks
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: rl_tasks.jsonl
---

# Termigen GRPO task pool, screened for verifier leaks

**3,541 terminal-agent RL tasks** derived from
[`allenai/open-instruct-termigen`](https://huggingface.co/datasets/allenai/open-instruct-termigen),
in the prompt-data shape this repo's GRPO stage consumes.

Builder and tests: **https://github.com/k1ssloo/RST-Train**
(`scripts/10b_build_termigen_taskset.py`, `tests/test_termigen_taskset.py`).

## This is not SFT data

Upstream has **zero assistant turns** — all 3,556 rows are exactly
`(system, user)` plus a `ground_truth` task id and an `env_config`. It is the
open-instruct RLVR format: a task pool with verifiers. There are no responses to
supervise on and no pairs to build a preference set from. The builder *asserts* the
absence rather than assuming it, so if upstream ever adds responses the run fails
loudly instead of silently throwing them away.

(The same is true of `allenai/TMax-15K` and `allenai/tmax-15k-open-instruct`. The
TMax *trajectories*, which are real SFT data, are at
[`NiuNiu0110/TMax-Agent-SFT-terminus`](https://huggingface.co/datasets/NiuNiu0110/TMax-Agent-SFT-terminus).)

## 15 tasks were excluded for exposing their own verifier

`environment/` is the Docker build context, so anything in it is readable by the
agent. A verifier that lands there makes the task's reward hackable — and that
failure shows up as a suspiciously good RL curve and nothing else.

| | |
|---|---|
| tasks upstream | 3,556 |
| **byte-identical verifier copy in the build context** | **10** |
| name-only match (a project file sharing the verifier's name) | 5 |
| tasks kept | **3,541** |

Both kinds are excluded. For an RL pool a false exclusion costs one task; a false
inclusion costs the meaning of every reward that task produces. The 10 byte-identical
cases are unambiguous — the agent can read its own grader. The excluded ids are in
`verifier_leaks.json`.

The verifier is identified **by name** (`tests/test.sh`, `tests/test_outputs.py` —
both present in all 3,556 tasks) rather than by globbing `tests/`. Globbing was tried
and is wrong: it excluded 11 sound tasks, because tasks *about* testing (pytest
fixtures, k6 load tests, robotframework, ctest) legitimately ship
`environment/tests/conftest.py`, `environment/tests/*.test.js` and the like as the
very thing the agent is meant to fix.

## Read this before spending a sandbox budget on it

**These tasks have no measured pass rates, and that is the main thing separating this
pool from a usable one.** GRPO's advantage is computed within a group of rollouts on
the same prompt, so a group whose rollouts all score the same contributes *exactly
zero* gradient while paying the full sandbox cost. `10_build_rl_taskset.py` exists
mostly to sort the RST pool into tiers by empirical pass rate for that reason.

Upstream ships no trial data, so every row here is:

```json
"tier": "unknown", "empirical_pass_rate": null, "n_reference_trials": 0
```

left explicitly null rather than defaulted, so nothing here reads as measured. Run a
cheap sampling pass first and tier it yourself; do not treat this as equivalent to
the RST `sweet` tier.

**One Docker image per task.** 3,541 distinct `hamishi740/termigen:*` tags, so a
pre-build/pre-pull pass is a 3,541-image job, not a handful. Budget for it before
scheduling rollouts.

## Schema

`rl_tasks.jsonl`, one JSON object per line:

| field | meaning |
|---|---|
| `prompt` | the task's `instruction.md`. slime's loader requires it; the *real* prompt is built by Terminus-2 inside Harbor from the same file |
| `label` | task id, passed as `--label-key label` |
| `metadata.task_id` / `task_group_id` | upstream ships one instance per task, so these are equal |
| `metadata.task_dir` | where `--materialize` puts the task; **does not exist until you run the builder** |
| `metadata.task_content_sha256` | over every file in the task dir, so a pool can be pinned |
| `metadata.base_image` | from `image.txt`, falling back to the Dockerfile's `FROM` |
| `metadata.env_name` | upstream `env_config.env_name` |
| `metadata.tier` / `empirical_pass_rate` / `n_reference_trials` | `"unknown"` / `null` / `0` — see above |

## Reproducing the task directories

Task bodies are deliberately **not** re-hosted here (see the script's docstring).
Rebuild them from upstream:

```bash
hf download allenai/open-instruct-termigen --repo-type dataset --local-dir termigen-src
python scripts/10b_build_termigen_taskset.py \\
    --parquet   termigen-src/data/train-00000-of-00001.parquet \\
    --task-data termigen-src/task-data.tar.gz \\
    --out data/termigen --materialize
```

That regenerates `rl_tasks.jsonl` byte-for-byte and materializes 3,541 task dirs.

## Attribution

Derived from [`allenai/open-instruct-termigen`](https://huggingface.co/datasets/allenai/open-instruct-termigen)
by Ivison et al., released under the same licence. Tasks were screened for verifier
leaks, given content hashes, and rewritten into this repo's GRPO prompt-data shape.
No rollouts were generated and no verifier was executed.
"""

CLAIMS: list[tuple[str, tuple[str, ...], object]] = [
    ("source rows", ("source_rows",), 3556),
    ("assistant turns", ("source_assistant_turns",), 0),
    ("tasks selected", ("tasks_selected",), 3541),
    ("task dirs in tarball", ("task_dirs_in_tarball",), 3556),
    ("leaks excluded", ("verifier_leaks_excluded",), 15),
    ("byte-identical leaks", ("verifier_leaks_byte_identical",), 10),
    ("name-only leaks", ("verifier_leaks_name_only",), 5),
    ("distinct images", ("distinct_base_images",), 3541),
]


def check_card(src_dir: Path) -> int:
    manifest = json.loads((src_dir / "manifest.json").read_text(encoding="utf-8"))
    bad = 0
    for label, keys, expected in CLAIMS:
        node: object = manifest
        for key in keys:
            assert isinstance(node, dict)
            node = node[key]
        if node != expected:
            print(f"  MISMATCH {label}: card says {expected}, manifest says {node}")
            bad += 1
    print(f"[check] {len(CLAIMS) - bad}/{len(CLAIMS)} card claims match the manifest")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", required=True, help="HF user or org")
    parser.add_argument("--src", type=Path, default=Path("data/termigen"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--public", action="store_true",
                        help="publish publicly. Read the docstring first: the default is "
                             "private because this pool's verifiers are the reward signal "
                             "for anything evaluated on it.")
    args = parser.parse_args()

    repo = f"{args.owner}/{REPO}"
    files: list[tuple[Path, str]] = [
        (args.src / "rl_tasks.jsonl", "rl_tasks.jsonl"),
        (args.src / "manifest.json", "manifest.json"),
        (args.src / "verifier_leaks.json", "verifier_leaks.json"),
    ]
    for src, _dst in files:
        if not src.is_file():
            sys.exit(f"missing input: {src}. Run scripts/10b_build_termigen_taskset.py first.")
    if check_card(args.src):
        sys.exit("the card disagrees with the manifest; fix one of them before publishing")

    print(f"taskset -> {repo}  ({'PUBLIC' if args.public else 'private'})")
    for src, dst in files:
        print(f"   {src}  ->  {dst}  ({src.stat().st_size / 2**20:.1f} MB)")
    print("   (task bodies deliberately not uploaded)")
    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=not args.public, exist_ok=True)
    for src, dst in files:
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                        repo_id=repo, repo_type="dataset")
        print(f"  uploaded {dst}")
    api.upload_file(path_or_fileobj=CARD.encode("utf-8"), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    print("  uploaded README.md")
    print(f"\nhttps://huggingface.co/datasets/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
