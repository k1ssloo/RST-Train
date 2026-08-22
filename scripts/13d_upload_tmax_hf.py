#!/usr/bin/env python3
"""Publish the converted TMax terminal-agent trajectories to the Hugging Face Hub.

    export HF_TOKEN=hf_...
    python scripts/13d_upload_tmax_hf.py --owner <hf-user> [--dry-run]

One public repo, two configs:

  <owner>/TMax-Agent-SFT-terminus   PUBLIC
      default        messages      5,445 train / 200 holdout
      pretokenized   input_ids + loss_mask, same rows

Separate from `OpenThoughts-Agent-v1-SFT-terminus` because the source, the licence
(ODC-BY here, Apache-2.0 there) and the shape of the supervision all differ: this
one trains a `<think>` block per turn, which the other has none of.

Numbers in the card come from `data/tmax-sft/manifest.json` and the two
pretokenized manifests; `--check` re-reads them and fails if the card has drifted
from what is on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = "TMax-Agent-SFT-terminus"

CARD = """---
license: odc-by
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
- qwen3.5
pretty_name: TMax agent trajectories, rendered for Qwen3.5 thinking SFT
size_categories:
- 1K<n<10K
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

# TMax agent trajectories, rendered for Qwen3.5 thinking SFT

**5,645 multi-turn terminal-agent trajectories** (5,445 train / 200 holdout) with
**a reasoning block on every assistant turn**, derived from
[`allenai/tmax-sft`](https://huggingface.co/datasets/allenai/tmax-sft) and put
through the same loss-mask contract and length gate as
[`OpenThoughts-Agent-v1-SFT-terminus`](https://huggingface.co/datasets/NiuNiu0110/OpenThoughts-Agent-v1-SFT-terminus).

Converter, tests and launchers: **https://github.com/k1ssloo/RST-Train**
(`scripts/03e_build_tmax_sft.py`, `tests/test_tmax_convert.py`).

## Read this first if you came here from TMax-15K

[`allenai/TMax-15K`](https://huggingface.co/datasets/allenai/TMax-15K) contains
**zero assistant turns**, and so do `allenai/tmax-15k-open-instruct` and — despite
the name — `allenai/TMax-SFT-16.5K`. They are RL *environment instances*: task
text plus graders, which is what the [TMax paper](https://arxiv.org/abs/2606.23321)
says they are. There is nothing in them to do SFT on and nothing to build
preference pairs from. The trajectories are in `allenai/tmax-sft`, config
`skill_tax_20260505_2.2k_combined_balanced_thinking_only_success` — the
verifier-labelled subset — and that is the source of this dataset.

Note that `has_task_complete` does **not** identify the failures: among the 4,931
trajectories present in the `_thinking_all` config but absent from
`_thinking_only_success`, it is 2,506 `True` / 2,425 `False`. It records that the
agent *claimed* completion, not that it succeeded.

## Why there is no preference split here

A preference pair needs two responses to the **same** prompt that differ in
quality. Keying the success and failure configs on `task`:

| | |
|---|---|
| tasks | 2,020 |
| tasks with ≥ 1 success | 1,136 |
| tasks with ≥ 1 failure | 1,175 |
| tasks with **both** — i.e. pairable | **291 (14.4 %)** |

`1,136 + 1,175 − 291 = 2,020` exactly, and that identity is the whole story: a
TMax task is almost always *either* always solved *or* never solved, so the
success/failure contrast mostly measures task difficulty rather than response
quality. Pairing gives 444 pairs at `min(successes, failures)` per task, or 1,746
as a full cross-product over just 291 distinct prompts. So this is an SFT dataset
and nothing else.

## What the conversion did

```
5,795 upstream rows (only_success config)
  ├─ splice out "Format error:" turns + the scoldings   (333 turns removed)
  ├─ repair 468 stray </think>, 34 mis-split tool scaffoldings
  ├─ refuse 45 ambiguous multi-THOUGHT turns, 4 control-token CoTs
  ├─ drop 80 rows over 32,768 tokens                    → 5,666
  ├─ dedup on (task, command signature)                 → 5,645  (21 dropped)
  └─ split by task group  ├─ train    5,445
                          └─ holdout    200  (41 whole tasks)
```

| | |
|---|---|
| examples | **5,645** (5,445 / 200) |
| tokens | 42,850,932 — mean 7,591, p50 5,957, p90 13,919, p99 26,666, max 32,755 |
| trained tokens | 19,432,729 (**45.35 %**) |
| assistant turns | mean 13.60, max 64 |
| distinct tasks | 1,115 — ~5.1 trajectories per task |
| source model | `Qwen/Qwen3.6-27B` (all rows) |
| steps/epoch @ GBS 128 | 42 |

### The four upstream defects, and which ones are repairable

Upstream is already this repo's agent contract — one persistent `bash` tool, one
command per turn, a THOUGHT before the action. What needed work is that the
generating model's own markup leaked into the wrong fields. Measured over all
82,203 assistant turns:

**1. "Format error" retry loops (409 turns).** Every mid-conversation `user` turn
in the dataset — all 409, no exceptions — is the harness scolding
`Format error: Your last response did not include a bash tool call.`, and 407 of
them directly follow an assistant turn with no `tool_calls`. The malformed turn
and the scolding are both removed, splicing the observation before it onto the
retry.

This is not cosmetic. The Qwen3.5 template emits reasoning **only for assistant
turns after the last `user` turn**, and 71.3 % of assistant tokens here are
reasoning. A surviving mid-conversation user turn silently deletes the CoT of
every turn before it. After the splice all 5,795 trajectories have exactly one
user turn, so every assistant turn keeps its reasoning.

**2. Stray `</think>` in content (528 turns).** These turns carry a `</think>`
inside `content` while *also* carrying `reasoning_content`, so the tag is surplus —
rendered as-is, the turn closes a think block the template already closed, which
trains two `</think>` per turn. 483 are one THOUGHT paragraph then the tag with
nothing but whitespace after it, and are repaired. **45 are two to six THOUGHT
paragraphs each with their own tag** — an upstream sampling loop that emitted the
thought several times — and are refused, because which paragraph was meant is
unknowable and picking one writes a guess into supervision.

**3. Mis-split tool scaffolding in `reasoning_content` (269 turns).** The CoT ends
in `</parameter></function></tool_call>` while the call itself parsed correctly
into `tool_calls`, so those closing tags are a duplicate. Stripping the trailing
run recovers real prose on 34 turns; where the reasoning *is* the scaffolding, the
row is refused. Most of the 269 never reach this gate — they sit in turns the
format-error splice already removed, which is coherent, since a leaked tool call
is exactly what triggers the scolding.

**4. Control tokens in `reasoning_content` (4 turns).** A CoT that is the bare
literal `<|im_start|>` — one token under this tokenizer. A model trained to emit it
can forge a turn boundary mid-answer and inject its own system or tool turn, which
ends a rollout in a way no reward signal attributes to the right cause. Refused,
and gated for on every field of every row.

## Schema

`default` config:

| field | type | notes |
|---|---|---|
| `messages` | `list<{role, content}>` | `system` / `user` / `assistant` |
| `trajectory_id` | string | upstream `trial_name` |
| `task_group_id` | string | upstream `task`; ~5.1 rows share one |
| `model_name` | string | `Qwen/Qwen3.6-27B` |
| `n_tokens` | int | full sequence under the Qwen3.5 chat template |
| `n_trained_tokens` | int | supervised tokens under the `qwen3_5` mask |
| `n_assistant_turns` | int | mean 13.60, max 64 |
| `n_spliced_turns` | int | malformed turns removed from this trajectory |

**`messages` is flattened to `{role, content}` on purpose, and it is byte-exact.**
The `bash` schema is baked into the `system` turn, the tool-call XML and the
`<think>` block are inlined into assistant content, and each `role="tool"`
observation is a `user` turn carrying `<tool_response>...</tool_response>`. So
`apply_chat_template` needs no `tools` argument and every existing consumer reads
this data unchanged. For **every** row the converter asserts

```
apply_chat_template(flattened) == apply_chat_template(native, tools=tools)
```

5,795 / 5,795 pass, so the model sees exactly the tokens the native tool-calling
shape would have produced. Carrying observations as `user` turns costs nothing:
the template's own scan for the last real query *skips* a user turn that starts
with `<tool_response>` and ends with `</tool_response>`, so `last_query_index`
still lands on the task description.

`pretokenized` config:

| field | type | meaning |
|---|---|---|
| `input_ids` | `list[int]` | tokens of the whole-conversation render |
| `loss_mask` | `list[int]` | 1 = train on this token, 0 = context only; aligned 1:1, no offset |

Same 5,445 / 200 rows, **41,432,572 tokens, 18,781,144 trained (45.33 %)** on
train; 1,418,360 / 651,585 (45.94 %) on holdout.

To get `labels`, set `labels[i] = input_ids[i]` where `loss_mask[i] == 1` else
`-100`, and do **not** shift — HuggingFace CausalLMs and Liger's fused CE shift
internally.

## Verification before release

- **0 chat-template contract failures** over all rows: rendering then tokenizing
  equals tokenizing directly, which is what makes the character offsets the mask
  is built from valid.
- **0 observation leakage.** Measured across all 5,445 train rows: of 22,305,351
  tokens inside non-assistant turns, **none** is supervised.
- **0 forgeable control tokens in supervision** — no `<|im_start|>` or
  `<|endoftext|>` at any position where `loss_mask == 1`.
- The `<think>\\n` opener is **not** trained on; everything after it, including
  `</think>` and the tool call, is. That matches serving: the template's
  `add_generation_prompt` emits `<|im_start|>assistant\\n<think>\\n`, so the opener
  is given to the model at every step of a rollout.
- Holdout is **group-disjoint by construction** — split by `task_group_id`, which
  matters here because ~5.1 trajectories share a task, so a row-wise split would
  put siblings of a held-out task in train.

Tokenizer: `Qwen/Qwen3.5-27B` — byte-identical across the five Qwen3.5 sizes
(0.8B / 4B / 9B / 27B / 35B-A3B), so `pretokenized` is valid for any of them.

## Mixing with the other two SFT sets

Identical mask semantics, so concatenation is safe. What differs, and matters:

| | this dataset | `OpenThoughts-Agent-v1-SFT-terminus` | `RST-SFT-Qwen3.5-27B` |
|---|---|---|---|
| examples | 5,645 | 14,312 | 10,778 |
| tokens | 42.9 M | 100.2 M | 99.9 M |
| assistant turns / row | 13.60 | 7.46 | 12.0 |
| tasks | 1,115, ~5.1 each | 14,312, one each | 1,329 groups, ~10 each |
| assistant format | `<think>` CoT + `bash` tool call | `{analysis, plan, commands}` JSON | same JSON |
| trained fraction | 45.35 % | 31.16 % | ~32.6 % |
| licence | ODC-BY | Apache-2.0 | CC-BY-4.0 |

**The assistant format is genuinely different, and that is the thing to decide
about before mixing.** The other two train a JSON action object with the reasoning
in an `analysis` *field*; this one trains a real `<think>` block followed by a
native tool call. Mixed in one run, the model learns both surface forms and has to
infer which is wanted from the system prompt — workable, since the prompts differ,
but it is a genuine multi-format objective and not a free lunch. Trained alone,
this set is small: 42 steps per epoch at global batch 128.

## Limitations

- **Single teacher.** All rows come from `Qwen/Qwen3.6-27B`, so this distils one
  policy's style, not a consensus.
- **Small.** 5,645 rows / 42.9 M tokens — roughly 43 % of either sibling set by
  token count, and enough for only ~42 optimizer steps per epoch at GBS 128.
- **Not replay-verified here.** Upstream's verifier labels are taken as given; no
  trajectory was re-executed.
- **Success-only, so no within-task contrast** is usable as a preference signal —
  see the table above for why the pairs that do exist were not worth building.
- 80 rows over 32,768 tokens were dropped, biasing mildly against the
  longest-horizon episodes; 45 more were refused for ambiguous THOUGHT duplication.

## Attribution

Derived from [`allenai/tmax-sft`](https://huggingface.co/datasets/allenai/tmax-sft)
(ODC-BY) by Ivison, Yin, Shao, Xiao, Lambert and Hajishirzi, and released under the
same licence. Trajectories were spliced, markup-repaired, flattened to a
byte-equivalent `{role, content}` render, deduplicated, length-gated and
pre-tokenized. No new rollouts were generated and no verifier was re-run.

```bibtex
@misc{ivison2026tmaxsimplerecipeterminal,
      title={Tmax: A simple recipe for terminal agents},
      author={Hamish Ivison and Junjie Oscar Yin and Rulin Shao and Teng Xiao and Nathan Lambert and Hannaneh Hajishirzi},
      year={2026}, eprint={2606.23321}, archivePrefix={arXiv}, primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.23321}}
```
"""

# (card claim, manifest filename, key path, value asserted in the card)
CLAIMS: list[tuple[str, str, tuple[str, ...], object]] = [
    ("source rows", "manifest.json", ("source_rows",), 5795),
    ("built", "manifest.json", ("built",), 5666),
    ("after dedup", "manifest.json", ("after_dedup",), 5645),
    ("train examples", "manifest.json", ("train_examples",), 5445),
    ("holdout examples", "manifest.json", ("holdout_examples",), 200),
    ("holdout tasks", "manifest.json", ("holdout_tasks",), 41),
    ("tasks covered", "manifest.json", ("groups_covered",), 1115),
    ("total tokens", "manifest.json", ("token_stats", "total_tokens"), 42850932),
    ("trained tokens", "manifest.json", ("token_stats", "trained_tokens"), 19432729),
    ("max tokens", "manifest.json", ("token_stats", "max"), 32755),
    ("p50 tokens", "manifest.json", ("token_stats", "p50"), 5957),
    ("spliced turns", "manifest.json", ("spliced_malformed_turns",), 333),
    ("stray-think repairs", "manifest.json", ("repaired_stray_think",), 468),
    ("scaffolding repairs", "manifest.json", ("repaired_tool_scaffolding",), 34),
    ("ambiguous refusals", "manifest.json", ("drop_counters", "drop_ambiguous_stray_think"), 45),
    ("control-token refusals", "manifest.json", ("drop_counters", "drop_markup_in_reasoning"), 4),
    ("too-long drops", "manifest.json", ("drop_counters", "drop_too_long"), 80),
    ("dedup drops", "manifest.json", ("drop_counters", "dedup_command_signature"), 21),
    ("pretok train rows", "pretokenized_train_manifest.json", ("rows_out",), 5445),
    ("pretok train tokens", "pretokenized_train_manifest.json", ("total_tokens",), 41432572),
    ("pretok train trained", "pretokenized_train_manifest.json", ("trained_tokens",), 18781144),
    ("pretok holdout rows", "pretokenized_holdout_manifest.json", ("rows_out",), 200),
    ("pretok holdout tokens", "pretokenized_holdout_manifest.json", ("total_tokens",), 1418360),
    ("pretok holdout trained", "pretokenized_holdout_manifest.json", ("trained_tokens",), 651585),
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
    parser.add_argument("--src", type=Path, default=Path("data/tmax-sft"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--card-only", action="store_true",
                        help="re-upload README.md only (no parquet re-upload)")
    args = parser.parse_args()

    repo = f"{args.owner}/{REPO}"
    files: list[tuple[Path, str]] = [
        (args.src / "tmax_sft_train.parquet", "data/messages/train.parquet"),
        (args.src / "tmax_sft_holdout.parquet", "data/messages/holdout.parquet"),
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
