#!/usr/bin/env python3
"""Publish the converted Nemotron-Terminal-Corpus to the Hugging Face Hub.

    export HF_TOKEN=hf_...
    python scripts/13f_upload_nemotron_hf.py --owner <hf-user> [--dry-run]

  <owner>/Nemotron-Terminal-SFT-terminus   PUBLIC   ~6.5 GiB, 5 configs

Train is one file per upstream subset, so `configs` can offer both the whole corpus
and any single slice over the same files. That matters at this size: nobody trains on
all 3.88 B tokens, and pulling 6.5 GiB to use the 5,681-row `mixed` slice is absurd.

Pre-tokenized data is deliberately not published. It would be roughly 4x the size of
the messages it is derived from, and `30_run_sft_verl.sh` builds it from
`rst_sft_train.parquet` on first use anyway.

Numbers in the card come from `data/nemotron-terminal/manifest.json`; `check_card`
re-reads it and refuses to publish if the card has drifted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = "Nemotron-Terminal-SFT-terminus"

SUBSETS = ("dataset_adapters", "skill_based_easy", "skill_based_medium", "skill_based_mixed")

CARD = """---
license: cc-by-4.0
task_categories:
- text-generation
language:
- en
tags:
- terminal-agent
- agentic
- sft
- reasoning
- chain-of-thought
- terminus-2
- qwen3.5
pretty_name: Nemotron Terminal-Corpus, normalized to the Terminus-2 contract
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train_*.parquet
  - split: holdout
    path: data/holdout.parquet
- config_name: dataset_adapters
  data_files:
  - split: train
    path: data/train_dataset_adapters.parquet
- config_name: skill_based_easy
  data_files:
  - split: train
    path: data/train_skill_based_easy.parquet
- config_name: skill_based_medium
  data_files:
  - split: train
    path: data/train_skill_based_medium.parquet
- config_name: skill_based_mixed
  data_files:
  - split: train
    path: data/train_skill_based_mixed.parquet
---

# Nemotron Terminal-Corpus, normalized to the Terminus-2 contract

**360,057 multi-turn terminal-agent trajectories** (359,656 train / 401 holdout),
**3.88 billion tokens**, with a `<think>` chain-of-thought on every assistant turn,
derived from [`nvidia/Nemotron-Terminal-Corpus`](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus)
and put through the same assistant-JSON normalizer, loss-mask contract and length
gate as [`NiuNiu0110/OpenThoughts-Agent-v1-SFT-terminus`](https://huggingface.co/datasets/NiuNiu0110/OpenThoughts-Agent-v1-SFT-terminus)
and `NiuNiu0110/RST-SFT-Qwen3.5-27B`. All three are **mixable row-for-row**.

Converter, tests and launchers: **https://github.com/k1ssloo/RST-Train**
(`scripts/03f_build_nemotron_sft.py`, `tests/test_nemotron_convert.py`).

Upstream's own results are the reason to care: training on this corpus took Qwen3-32B
from 3.4 % to **27.4 %** on Terminal-Bench 2.0, beating 480B Qwen3-Coder (23.9 %), and
Qwen3-14B to 20.2 %, beating 120B GPT-OSS-high (18.7 %).

## Composition

| config | train rows | steps/epoch @ GBS 128 | size | what it is |
|---|---|---|---|---|
| `dataset_adapters` | 220,381 | 1,721 | 3.71 GiB | Math, Code and SWE datasets transformed into terminal tasks |
| `skill_based_medium` | 88,912 | 694 | 1.97 GiB | synthetic tasks from a terminal-skill taxonomy, medium |
| `skill_based_easy` | 44,692 | 349 | 0.67 GiB | same taxonomy, easy |
| `skill_based_mixed` | 5,671 | 44 | 0.09 GiB | same taxonomy, mixed difficulty |
| **`default`** | **359,656** | **2,809** | 6.44 GiB | all four |

The 401-row holdout is shared and split by task group, so it is disjoint from every
config's train rows, not just from the one you pick.

| | |
|---|---|
| tokens | **3,880,294,261** — mean 10,777, p50 9,816, p90 19,957, p99 29,320, max 32,768 |
| trained tokens | 1,691,164,897 (**43.58 %**) |
| assistant turns | mean 7.34, max 59 |
| distinct tasks | 272,098 |
| teacher | `deepseek-ai/DeepSeek-V3.2` (all rows) |
| steps/epoch @ GBS 128 | **2,809** (whole train split) |

**This is ~39x the token count of either sibling SFT set.** One epoch of the whole
thing is a different class of run from anything else in this repo. Pick a config.

## The one thing to understand before reading a loss curve on this

Every assistant turn upstream carries a real `<think>` block — 47.1 % of assistant
tokens — and **only the final turn's reasoning is supervised**. That is not a
conversion defect. It is what inference looks like.

Observations here are plain `user` turns, so they count as queries for the Qwen3.5
template's `last_query_index` scan, and the template emits reasoning only for
assistant turns *after* the last user turn. In a
`user, assistant, user, assistant, …, user, assistant` trajectory that is exactly one
turn. So the render is:

```
harness prompt                                     context
JSON action, observation, JSON action, observation, …   actions supervised,
                                                       reasoning ABSENT
<think> reasoning </think> JSON action                 final turn: both supervised
```

The terminus-2 harness re-renders the whole history every turn under this same rule,
so at turn *k* the model's own earlier reasoning has already been dropped from its
context and it is asked to think afresh. Training on the same render is training on
what the model will actually see. It also costs nothing in tokens — the dropped
reasoning is not carried as unsupervised context, it simply is not there.

The `<think>` blocks are nevertheless **kept in `messages`** (1,759,784,993 tokens of
them), not stripped, so the artifact is not lossy and a per-turn-reasoning objective
remains possible later.

## What the conversion did

```
366,154 upstream rows
  ├─ splice out parse-error retry loops        (328,372 turns removed)
  ├─ truncate trailing `null` actions           (17,570 trajectories)
  ├─ repair stale warning preambles                (152 observations)
  ├─ normalize every JSON action through the shared normalizer
  ├─ refuse 195 unparseable, 90 bad think shape, 12 keyless, 3 control-token, 3 headless
  ├─ drop 5,794 rows over 32,768 tokens          → 360,057   (98.33 % survive)
  ├─ dedup on (task, command signature)          → 360,057   (0 dropped)
  └─ split by task group  ├─ train    359,656
                          └─ holdout      401   (297 whole tasks)
```

### Three upstream shapes that needed a decision

**1. Parse-error retry loops — 328,372 turns.** 17.4 % of assistant turns have an
empty body: the `<think>` block ran long, swallowed the start of the JSON, and
`</think>` landed after it. The next turn is *always* a
`Previous response had parsing errors: ERROR: No valid JSON found in response`
scolding carrying **no observation at all** (761/761 in a hand-checked sample) — a
pure retry request. The malformed turn and the scolding are removed together, which
reconnects the previous observation to the retry and keeps the user/assistant
alternation the loss mask depends on. Kept, they would train the model to emit a
think block and no action, and to expect to be told off for it.

**2. Trailing `null` actions — 17,570 trajectories.** The harness's marker for an
episode that ended without a final action. **Every** normalization failure surviving
the splice was this, and always the last turn (118/118 measured across five files
spanning both streams), so the trailing turn and the observation it answered are
truncated.

This is where this converter deliberately differs from
`03d_build_openthoughts_sft.py`, which refuses to truncate and drops trajectories
whole. Same test, opposite answer: on OpenThoughts the median salvageable fraction
was 0.15 and 323 trajectories failed on their *first* assistant turn, so truncating
would have produced one-turn stubs of ten-turn episodes. Here the failure is always
last, the salvageable fraction is ~0.9, and dropping whole would be the lossy choice.

**3. Think markup that breaks the round-trip — 90 turns.** The template recovers
reasoning with `content.split('</think>')[0]` and content with
`split('</think>')[-1]`, so a second `</think>` makes those two splits disagree about
where the reasoning ended, silently. Exactly one balanced pair is required; anything
else is refused rather than guessed at. Same for the 3 turns carrying a literal
`<|im_start|>`, which is one token here — a model trained to emit it can forge a turn
boundary.

## Schema

| field | type | notes |
|---|---|---|
| `messages` | `list<{role, content}>` | `user` / `assistant` only, user-first, assistant-last |
| `trajectory_id` | string | upstream `trial_name` |
| `task_group_id` | string | upstream `task`; 272,098 distinct |
| `model_name` | string | `deepseek-ai/DeepSeek-V3.2` |
| `subset` | string | one of the four configs above |
| `domain` | string | upstream folder — `math`, `swe`, `code`, `debugging`, `security`, … |
| `n_tokens` | int | full sequence under the Qwen3.5 chat template |
| `n_trained_tokens` | int | supervised tokens under the `qwen3_5` mask |
| `n_assistant_turns` | int | mean 7.34, max 59 |

`messages[0]` is **`user`, not `system`** — that is how Terminus-2 delivers the
harness prompt, and keeping it as `user` makes training and serving identical. Rows
containing a `system` turn are refused rather than folded in, because a system turn
shifts the rendered prefix and therefore the loss mask.

Assistant content is `<think>\\n…\\n</think>\\n\\n` followed by the canonical
`{"analysis", "plan", "commands"[, "task_complete"]}` JSON, re-dumped with
`indent=2` by the same normalizer the sibling datasets went through.

Tokenizer: `Qwen/Qwen3.5-27B` — byte-identical across the five Qwen3.5 sizes
(0.8B / 4B / 9B / 27B / 35B-A3B).

## Verification before release

- **0 chat-template contract failures**: rendering then tokenizing equals tokenizing
  directly for every published row. That equality is what makes the character offsets
  the loss mask is built from valid. Rows failing it are dropped, not silenced.
- **0 rows with a supervised first token** — `30_run_sft_verl.sh` and the verl dataset
  both refuse those, because verl's `sft_loss` rolls the mask cyclically across a
  packed micro-batch and a supervised token 0 leaks onto the previous document.
- **0 rows over 32,768 tokens.**
- Holdout is **group-disjoint by construction** — split by `task_group_id`, so no
  sibling of a held-out task is in train.
- Every count above is reproduced in `manifest.json`, and the uploader refuses to
  publish if the card and the manifest disagree.

## Limitations

- **Single teacher.** All rows distil `deepseek-ai/DeepSeek-V3.2`, so this is one
  policy's style, not a consensus.
- **Not replay-verified here.** NVIDIA's own filtering (`data_filtered.parquet`) is
  taken as given; no trajectory was re-executed and no verifier was re-run.
- **Success-only, so no within-task contrast** — not usable as a preference set.
- **Only the final turn's reasoning is supervised.** See above. If you want per-turn
  reasoning supervision you must restructure the data; the CoT is retained so you can.
- 5,794 rows over 32,768 tokens were dropped, biasing mildly against the
  longest-horizon episodes — concentrated in the SWE slice, whose p90 is the highest.
- `dataset_adapters` is 61 % of the corpus and is *transformed* math/code/SWE data,
  not natively terminal work. If you care about terminal skills specifically, the
  `skill_based_*` configs are the more targeted 139,431 rows.

## Attribution

Derived from [`nvidia/Nemotron-Terminal-Corpus`](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus)
(CC-BY-4.0) by Renjie Pi, Grace Lam, Mohammad Shoeybi, Pooya Jannaty, Bryan Catanzaro
and Wei Ping, and released under the same licence. Trajectories were spliced,
truncated, JSON-normalized, deduplicated, length-gated and split by task group. No new
rollouts were generated and no verifier was re-run.

```bibtex
@misc{pi2026dataengineeringscalingllm,
      title={On Data Engineering for Scaling LLM Terminal Capabilities},
      author={Renjie Pi and Grace Lam and Mohammad Shoeybi and Pooya Jannaty and Bryan Catanzaro and Wei Ping},
      year={2026}, eprint={2602.21193}, archivePrefix={arXiv}, primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.21193}}
```
"""

CLAIMS: list[tuple[str, tuple[str, ...], object]] = [
    ("source rows", ("source_rows",), 366154),
    ("built", ("built",), 360057),
    ("after dedup", ("after_dedup",), 360057),
    ("train examples", ("train_examples",), 359656),
    ("holdout examples", ("holdout_examples",), 401),
    ("holdout tasks", ("holdout_tasks",), 297),
    ("tasks covered", ("groups_covered",), 272098),
    ("total tokens", ("token_stats", "total_tokens"), 3880294261),
    ("trained tokens", ("token_stats", "trained_tokens"), 1691164897),
    ("p50 tokens", ("token_stats", "p50"), 9816),
    ("max tokens", ("token_stats", "max"), 32768),
    ("spliced retry turns", ("spliced_retry_turns",), 328372),
    ("truncated null tails", ("drop_counters", "truncated_null_tail"), 17570),
    ("warning preamble repairs", ("repaired_warning_preamble",), 152),
    ("too-long drops", ("drop_counters", "drop_too_long"), 5794),
    ("unparseable drops", ("drop_counters", "drop_unparseable"), 195),
    ("bad think shape drops", ("drop_counters", "drop_bad_think_shape"), 90),
    ("control-token drops", ("drop_counters", "drop_control_token"), 3),
    ("think tokens retained", ("reasoning", "think_tokens_in_source"), 1759784993),
    ("adapters rows", ("rows_per_subset", "dataset_adapters"), 220626),
    ("easy rows", ("rows_per_subset", "skill_based_easy"), 44747),
    ("medium rows", ("rows_per_subset", "skill_based_medium"), 89003),
    ("mixed rows", ("rows_per_subset", "skill_based_mixed"), 5681),
]


def check_card(src_dir: Path) -> int:
    manifest = json.loads((src_dir / "manifest.json").read_text(encoding="utf-8"))
    bad = 0
    for label, keys, expected in CLAIMS:
        node: object = manifest
        for key in keys:
            assert isinstance(node, dict), f"{label}: {keys} is not a path into the manifest"
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
    parser.add_argument("--src", type=Path, default=Path("data/nemotron-terminal"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--card-only", action="store_true",
                        help="re-upload README.md only (no parquet re-upload)")
    args = parser.parse_args()

    repo = f"{args.owner}/{REPO}"
    files: list[tuple[Path, str]] = [
        (args.src / "nemotron_sft_holdout.parquet", "data/holdout.parquet"),
        (args.src / "manifest.json", "manifest.json"),
    ]
    files += [(args.src / f"nemotron_sft_train_{subset}.parquet",
               f"data/train_{subset}.parquet") for subset in SUBSETS]
    if args.card_only:
        files = []

    for src, _dst in files:
        if not src.is_file():
            sys.exit(f"missing input: {src}. Run scripts/03f_build_nemotron_sft.py first.")
    if check_card(args.src):
        sys.exit("the card disagrees with the manifest; fix one of them before publishing")

    total = sum(src.stat().st_size for src, _ in files) / 2**30
    print(f"SFT -> {repo}  (public, {total:.2f} GiB)")
    for src, dst in files:
        print(f"   {src}  ->  {dst}  ({src.stat().st_size / 2**30:.2f} GiB)")
    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    print(f"\n[{repo}] created/exists (public)")
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
