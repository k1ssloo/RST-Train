#!/usr/bin/env python3
"""Publish the converted OpenThoughts-Agent trajectories to the Hugging Face Hub.

    export HF_TOKEN=hf_...
    python scripts/13c_upload_openthoughts_hf.py --owner <hf-user> [--dry-run]

One public repo, two configs:

  <owner>/OpenThoughts-Agent-v1-SFT-terminus   PUBLIC
      default        messages      14,112 train / 200 holdout
      pretokenized   input_ids + loss_mask, same rows

Separate from `RST-SFT-Qwen3.5-27B` on purpose: same format contract, different
source dataset and different license (apache-2.0 here, cc-by-4.0 there). Folding
it in as another config would make provenance a footnote instead of the repo name.

Numbers in the card come from `data/openthoughts-agent-v1/manifest.json` and the
two pretokenized manifests; `--check` re-reads them and fails if the card has
drifted from what is on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = "OpenThoughts-Agent-v1-SFT-terminus"

CARD = """---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- terminal-agent
- agentic
- sft
- terminus-2
- qwen3.5
pretty_name: OpenThoughts-Agent v1 SFT, normalized to the Terminus-2 contract
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/messages/train.parquet
  - split: holdout
    path: data/messages/holdout.parquet
- config_name: pretokenized
  data_files:
  - split: train
    path: data/pretokenized/train.parquet
  - split: holdout
    path: data/pretokenized/holdout.parquet
---

# OpenThoughts-Agent v1 SFT, normalized to the Terminus-2 contract

**14,312 multi-turn terminal-agent trajectories** (14,112 train / 200 holdout),
derived from [`open-thoughts/OpenThoughts-Agent-v1-SFT`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT)
and put through the same assistant-JSON normalizer, loss-mask contract and
length gate as [`NiuNiu0110/RST-SFT-Qwen3.5-27B`](https://huggingface.co/datasets/NiuNiu0110/RST-SFT-Qwen3.5-27B).
The two are therefore **mixable row-for-row** in one SFT run.

Converter, tests, and launchers: **https://github.com/k1ssloo/RST-Train**
(`scripts/03d_build_openthoughts_sft.py`, `tests/test_openthoughts_convert.py`).

## What was actually done

Upstream is already the right contract — `agent=terminus-2`, assistant turns are
`{"analysis", "plan", "commands"[, "task_complete"]}` JSON, observations are
`New Terminal Output:` blocks. So this is not a reformat; it is a normalization
pass plus a set of gates:

```
15,209 upstream rows
  ├─ reconstruct + normalize every assistant turn   → 14,372   (837 dropped)
  ├─ dedup (exact content + per-task command signature) → 14,372  (0 dropped)
  └─ drop > 32,768 tokens under the Qwen3.5 template    → 14,312  (60 dropped)
       ├─ train    14,112
       └─ holdout     200
```

| | |
|---|---|
| examples | **14,312** (14,112 / 200) |
| tokens | 100,185,231 — mean 7,000, p50 6,078, p90 11,923, p99 22,530, max 32,739 |
| assistant turns | mean 7.46, max 32 |
| distinct tasks | 14,312 — **one trajectory per task** |
| source model | `QuantTrio/GLM-4.6-AWQ` (all rows) |
| steps/epoch @ GBS 128 | 110 |

### The 837 dropped rows, and why they are dropped whole

645 had an assistant turn that will not parse as JSON (overwhelmingly an
unescaped inner quote: `"made directories "1" and "2" here"`), 192 had one
missing a required key. A trajectory is dropped **entirely** when any of its
turns fails.

Truncating at the last good turn was measured and rejected: across those 837 the
median salvageable fraction is **0.15**, and **323 fail on the very first
assistant turn**. Truncating would mostly contribute one-turn stubs of ten-turn
episodes and skew the length distribution toward short, easy prefixes.

Unescaped inner quotes are **not** repaired. Where a broken string ends is a
guess, and a guess here writes invented content into supervision.

### Two normalizations that are easy to get wrong

**1. Literal newlines inside JSON strings are repaired, not dropped.** Agents
emit `"analysis": "step one⏎step two"` with a real newline. That is invalid JSON
by spec and `json.loads` rejects it as an "Invalid control character", but the
meaning is unambiguous, so it is re-dumped with the newline properly escaped.
Every assistant turn in this dataset parses under **strict** `json.loads`.

**2. The stale warning preamble is stripped.** When an assistant turn had to be
renormalized, the *next* observation begins `Previous response had warnings: -
Extra text detected before JSON object` — a complaint about a formatting error
that no longer exists in the data. Left in, it trains the model to expect a
scolding for output it was just shown as correct. **572 observations** needed
this repair. The preamble is kept whenever the preceding turn was already clean,
because then the complaint is about something real.

5.82 % of assistant turns were rewritten in some way (`n_rewritten_turns`).

## Schema

`default` config:

| field | type | notes |
|---|---|---|
| `messages` | `list<{role, content}>` | `user` / `assistant` only, user-first, assistant-last |
| `trajectory_id` | string | upstream `trial_name` |
| `task_group_id` | string | upstream `task`; unique per row here |
| `model_name` | string | `QuantTrio/GLM-4.6-AWQ` |
| `n_tokens` | int | full sequence under the Qwen3.5 chat template |
| `n_assistant_turns` | int | mean 7.46, max 32 |
| `n_rewritten_turns` | int | assistant turns whose JSON was renormalized |

`messages[0]` is **`user`, not `system`** — that is how Terminus-2 delivers the
harness prompt, and keeping it as `user` makes training and serving identical. A
`system` turn would shift the rendered prefix and therefore the loss mask; rows
containing one are refused rather than folded in (upstream has none).

`pretokenized` config:

| field | type | meaning |
|---|---|---|
| `input_ids` | `list[int]` | tokens of the whole-conversation render |
| `loss_mask` | `list[int]` | 1 = train on this token, 0 = context only; aligned 1:1, no offset |

Same 14,112 / 200 rows, **98,567,847 tokens, 30,711,017 trained (31.16 %)**.

To get `labels`, set `labels[i] = input_ids[i]` where `loss_mask[i] == 1` else
`-100`, and do **not** shift — HuggingFace CausalLMs and Liger's fused CE shift
internally.

## Verification before release

- **0 chat-template contract failures.** `apply_chat_template(tokenize=False)`
  then tokenizing equals `apply_chat_template(tokenize=True)` for every row —
  that equality is what makes the character offsets the mask is built from valid.
  Rows failing it are dropped, not silenced.
- **0 user-turn leakage**: no terminal observation token is ever trained on.
- 31.05 % of tokens trained (300-row independent audit), 31.16 % over the full
  pretokenized export.
- Holdout is **group-disjoint** by construction: upstream `task` is unique per
  row (15,209 distinct over 15,209 rows), so no sibling of a held-out task is in
  train. Asserted at build time rather than assumed.

Tokenizer: `Qwen/Qwen3.5-27B` — byte-identical across the five Qwen3.5 sizes
(0.8B / 4B / 9B / 27B / 35B-A3B), so `pretokenized` is valid for any of them.

## Mixing with RST-SFT

Identical canonical assistant form and identical mask semantics, so
concatenation is safe. What differs, and matters for interpreting a mixed run:

| | this dataset | `RST-SFT-Qwen3.5-27B` (`cap10`) |
|---|---|---|
| examples | 14,312 | 10,778 |
| tokens | 100.2 M | 99.9 M |
| assistant turns / row | 7.46 mean | 12.0 mean |
| tasks | 14,312, one trajectory each | 1,329 groups, capped at ~10 each |
| source models | 1 | 4 |
| license | apache-2.0 | cc-by-4.0 |

The task/trajectory ratio is the real difference: RST is many trajectories over
few task lineages, this is one trajectory over many distinct tasks. Mixed, they
are complementary — breadth of task from here, depth of horizon from there.

> **Note for anyone diffing against `RST-SFT-Qwen3.5-27B`:** that dataset was
> built before the two normalizations above. Every row it contains is
> byte-identical under the current normalizer (verified over all 126,630 of its
> assistant turns), so it is not wrong — it simply predates the recovery of ~492
> previously-dropped turns and would gain rows if rebuilt.

## Limitations

- **Single source model.** All rows come from `QuantTrio/GLM-4.6-AWQ`, so this
  distills one policy's style, not a consensus.
- **Not replay-verified here.** Upstream's own filtering is taken as given; no
  trajectory was re-executed and no verifier was re-run in producing this copy.
- **Success-only, and one sample per task** — there is no within-task contrast,
  so this is not usable as a preference set.
- 60 rows over 32,768 tokens were dropped, biasing mildly against the
  longest-horizon episodes.

## Attribution

Derived from [`open-thoughts/OpenThoughts-Agent-v1-SFT`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT)
(Apache-2.0) by the OpenThoughts team, and released under the same license.
Trajectories were reconstructed, assistant JSON normalized, stale warning
preambles repaired, deduplicated, length-gated and pre-tokenized. No new
rollouts were generated and no verifier was re-run.
"""

# (card claim, manifest path, key path) — checked against disk before upload so the
# card cannot quietly drift from the data it describes.
CLAIMS: list[tuple[str, str, tuple[str, ...], object]] = [
    ("source rows", "manifest.json", ("source_rows",), 15209),
    ("reconstructed", "manifest.json", ("reconstructed",), 14372),
    ("final examples", "manifest.json", ("final_examples",), 14312),
    ("train examples", "manifest.json", ("train_examples",), 14112),
    ("holdout examples", "manifest.json", ("holdout_examples",), 200),
    ("total tokens", "manifest.json", ("token_stats", "total_tokens"), 100185231),
    ("max tokens", "manifest.json", ("token_stats", "max"), 32739),
    ("unparseable drops", "manifest.json", ("drop_counters", "drop_unparseable"), 645),
    ("keyless drops", "manifest.json", ("drop_counters", "drop_missing_required_keys"), 192),
    ("too-long drops", "manifest.json", ("drop_counters", "drop_too_long"), 60),
    ("warning repairs", "manifest.json", ("drop_counters", "repaired_warning_preamble"), 572),
    ("pretok rows", "pretokenized_train_manifest.json", ("rows_out",), 14112),
    ("pretok tokens", "pretokenized_train_manifest.json", ("total_tokens",), 98567847),
    ("pretok trained", "pretokenized_train_manifest.json", ("trained_tokens",), 30711017),
    ("pretok fraction", "pretokenized_train_manifest.json", ("trained_fraction",), 0.3116),
]


def check_card(src_dir: Path) -> int:
    """Fail loudly if any number in the card disagrees with the manifests."""
    cache: dict[str, dict] = {}
    bad = 0
    for label, filename, keys, expected in CLAIMS:
        if filename not in cache:
            cache[filename] = json.loads((src_dir / filename).read_text(encoding="utf-8"))
        node: object = cache[filename]
        for key in keys:
            assert isinstance(node, dict)
            node = node[key]
        if node != expected:
            print(f"  MISMATCH {label}: card says {expected}, {filename} says {node}")
            bad += 1
    print(f"[check] {len(CLAIMS) - bad}/{len(CLAIMS)} card claims match the manifests")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", required=True, help="HF user or org")
    parser.add_argument("--src", type=Path, default=Path("data/openthoughts-agent-v1"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--card-only", action="store_true",
                        help="re-upload README.md only (no parquet re-upload)")
    args = parser.parse_args()

    repo = f"{args.owner}/{REPO}"
    files: list[tuple[Path, str]] = [
        (args.src / "ota_sft_train.parquet", "data/messages/train.parquet"),
        (args.src / "ota_sft_holdout.parquet", "data/messages/holdout.parquet"),
        (args.src / "manifest.json", "manifest.json"),
        (args.src / "pretokenized_train.parquet", "data/pretokenized/train.parquet"),
        (args.src / "pretokenized_holdout.parquet", "data/pretokenized/holdout.parquet"),
        (args.src / "pretokenized_train_manifest.json", "manifest_pretokenized_train.json"),
        (args.src / "pretokenized_holdout_manifest.json", "manifest_pretokenized_holdout.json"),
    ]
    if args.card_only:
        files = []

    for src, _dst in files:
        if not src.is_file():
            sys.exit(f"missing input: {src}")
    if check_card(args.src):
        sys.exit("the card disagrees with the manifests; fix one of them before publishing")

    print(f"SFT -> {repo}  (public)")
    for src, dst in files:
        print(f"   {src}  ->  {dst}  ({src.stat().st_size / 2**20:.1f} MB)")
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
