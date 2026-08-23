#!/usr/bin/env python3
"""Convert NVIDIA's Nemotron-Terminal-Corpus into this repo's SFT format.

    python scripts/03f_build_nemotron_sft.py \
        --source data/nemotron-terminal/source \
        --tokenizer data/Qwen3.5-27B-tokenizer \
        --out-dir data/nemotron-terminal --workers 20

WHY THIS IS A CONVERTER AND NOT A BUILDER
-----------------------------------------
Upstream is already this repo's contract, more exactly than any other source it has
ingested: `agent=terminus-2`, the first turn is the harness prompt as a `user` turn,
observations are `New Terminal Output:` blocks, and assistant turns are
`{"analysis", "plan", "commands"[, "task_complete"]}` JSON. The column names are the
same ones `03d_build_openthoughts_sft.py` reads. So the JSON goes through the *same*
`normalize_assistant` the RST and OpenThoughts data went through, and one canonical
form plus one loss-mask contract covers all four datasets.

One thing is genuinely new, and it is the whole reason this file is not just `03d`
with a different path: **every assistant turn is prefixed with a real `<think>`
block** (`enable_thinking=True` upstream, teacher `deepseek-ai/DeepSeek-V3.2`).
Measured: 47.1 % of assistant tokens are reasoning.

WHAT HAPPENS TO THAT REASONING, EXACTLY
---------------------------------------
This is the part to read before interpreting a loss curve trained on this data.

Observations here are plain `user` turns, so they count as queries for the Qwen3.5
template's `last_query_index` scan (unlike TMax, whose `tool`-role observations the
template skips). The template emits reasoning only for assistant turns *after* the
last user turn. In a `user, assistant, user, assistant, ..., user, assistant`
trajectory that is exactly one turn: the final one.

So the render is:

    harness prompt                          (context)
    JSON action, observation, JSON action, observation, ...   (actions supervised,
                                                              reasoning ABSENT)
    <think> reasoning </think> JSON action  (final turn: both supervised)

That is **not a defect and not a loss** — it is what inference looks like. The
terminus-2 harness re-renders the whole history every turn, so at turn k the model's
own earlier reasoning has already been dropped from its context by this same
template rule, and it is asked to think afresh. Training on the same render is
training on what the model will actually see. It also costs no tokens: the dropped
reasoning is not carried as unsupervised context, it is simply not there.

The `<think>` blocks are nevertheless **kept in `messages`**, not stripped, because
stripping would destroy them in the published artifact for no gain — the render is
identical either way for every turn but the last, and keeping them leaves the door
open to a per-turn-reasoning objective later.

FOUR UPSTREAM SHAPES THAT NEEDED A DECISION
-------------------------------------------
Every count below was measured over a multi-file sample, not assumed.

  1. **Parse-error retry loops.** 17.4 % of assistant turns have an empty body: the
     `<think>` block ran long, swallowed the start of the JSON, and `</think>` landed
     after it, leaving nothing parseable. The next turn is *always* a
     `Previous response had parsing errors:\nERROR: No valid JSON found in response`
     scolding carrying **no observation at all** (761/761 in the sample) -- a pure
     retry request. `splice_retries` drops the malformed turn and the scolding
     together, which reconnects the previous observation to the retry and keeps the
     user/assistant alternation intact. Kept, they would train the model to emit a
     think block and no action, and to expect to be told off for it.

  2. **Stale warning preambles.** `Previous response had warnings: ...` followed by a
     real observation. Handled exactly as `03d` handles it -- the repo's own
     `_WARNING_PREAMBLE` is reused, and it is stripped only when the preceding turn
     was renormalized, because only then is the complaint about a formatting error
     that no longer exists in the data.

  3. **`Current terminal state:` observations.** The terminus-2 "are you sure you
     want to mark the task complete?" prompt. A legitimate observation, left alone;
     the repo's preamble regex already anticipates it.

  4. **Think markup that breaks the round-trip.** The template recovers reasoning
     from content with `content.split('</think>')[-1]`, so a body containing a second
     `</think>` would silently take the wrong half. Refused rather than guessed at,
     along with any assistant turn carrying a literal control token.

Upstream is CC-BY-4.0. Trajectories are NOT re-executed and no verifier is re-run;
NVIDIA's own filtering (`data_filtered.parquet`) is taken as given.

Pre-tokenized data is deliberately NOT emitted here. At this scale it is ~4x the
size of the messages it is derived from, and `30_run_sft_verl.sh` builds it from
`rst_sft_train.parquet` on first use anyway.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

SOURCE_DATASET = "nvidia/Nemotron-Terminal-Corpus"
SOURCE_LICENSE = "cc-by-4.0"

PARSE_ERROR_PREFIX = "Previous response had parsing errors:"
OBSERVATION_PREFIXES = ("New Terminal Output:", "Current terminal state:")

# Each of these is one token under this tokenizer, so a literal in a source field
# does not train markup -- it trains the control token itself. `<|im_start|>` is the
# one that costs something: a model that can emit it can forge a turn boundary.
FORBIDDEN_IN_SOURCE = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")

# Set once per worker process. A tokenizer costs ~1 s to build and cannot be pickled
# cheaply, so it is created on first use in each worker rather than per file.
_TOKENIZER: Any = None
_BUILDER: Any = None
_EXPORTER: Any = None


def load_script(stem: str) -> Any:
    """Load `scripts/<stem>.py` by path.

    By path because `scripts/` is not a package and the filenames start with a digit.
    Shared rather than reimplemented: `normalize_assistant` is the definition of this
    repo's canonical assistant form, and a second copy of it would mean two canonical
    forms in one training mixture.
    """
    path = Path(__file__).resolve().parent / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_nemo_{stem}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        sys.exit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_think(content: str) -> tuple[str | None, str, str]:
    """Split an assistant turn into `(think, json_body, reason)`.

    `think` is None when the turn must be refused. The shape gate is strict on
    purpose: the chat template recovers reasoning with
    `content.split('</think>')[0]` and content with `content.split('</think>')[-1]`,
    so anything other than exactly one balanced pair makes those two splits disagree
    about where the reasoning ended, and the disagreement is silent.
    """
    opens, closes = content.count("<think>"), content.count("</think>")
    if (opens, closes) == (0, 0):
        return "", content, "ok"
    if (opens, closes) != (1, 1) or content.index("<think>") > content.index("</think>"):
        return None, "", "bad_think_shape"
    think = content.split("<think>", 1)[1].split("</think>", 1)[0]
    body = content.split("</think>", 1)[1]
    return think, body, "ok"


def command_signature(messages: list[dict[str, str]]) -> str:
    """Digest of the executed keystroke sequence, ignoring the `<think>` prefix.

    The repo's own `command_signature` does `json.loads(message["content"])`, which
    cannot work here because content opens with a think block. Same idea, same
    purpose (near-duplicate detection within a task), applied after the split.
    """
    keystrokes: list[str] = []
    for message in messages:
        if message["role"] != "assistant":
            continue
        _, body, reason = split_think(message["content"])
        if reason != "ok":
            continue
        try:
            obj = json.loads(body.strip() or "{}")
        except (json.JSONDecodeError, ValueError):
            continue
        for command in obj.get("commands") or []:
            if isinstance(command, dict):
                keystrokes.append(str(command.get("keystrokes", "")))
    return hashlib.sha256("\x00".join(keystrokes).encode("utf-8")).hexdigest()


def splice_retries(conversation: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Drop each parse-error scolding together with the turn that caused it.

    The scolding carries no observation, so removing both messages reconnects the
    previous observation directly to the retry and leaves the user/assistant
    alternation the rest of this file (and the loss mask) depends on.
    """
    out: list[dict[str, str]] = []
    removed = 0
    for message in conversation:
        if (message["role"] == "user" and out
                and message["content"].startswith(PARSE_ERROR_PREFIX)):
            if out and out[-1]["role"] == "assistant":
                out.pop()
                removed += 1
            continue
        out.append(message)
    return out, removed


def truncate_dead_tail(conversation: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Drop a trailing `null` action, and any observation left dangling after it.

    Measured over 1,600 trajectories spanning both upstream streams: **every** single
    normalization failure that survives `splice_retries` is the LAST assistant turn,
    and its body is the literal string `null` (118 of 118) -- the harness's marker for
    an episode that ended without a final action. Everything before it is valid
    supervision.

    So this truncates where `03d_build_openthoughts_sft.py` deliberately refuses to.
    That is not an inconsistency, it is the same test reaching the opposite answer:
    `03d` rejected truncation because on its data the median salvageable fraction was
    0.15 and 323 trajectories failed on their *first* assistant turn, so truncating
    would have contributed one-turn stubs of ten-turn episodes. Here the failure is
    always the last turn, so the salvageable fraction is (n-1)/n -- about 0.9 -- and
    dropping the whole trajectory would be the lossy choice.

    The second pop also cleans up after `splice_retries`: when a scolding was the last
    message, removing it and its malformed turn leaves an observation with no answer,
    which is context carrying no target.
    """
    out = list(conversation)
    removed = 0
    if out and out[-1]["role"] == "assistant":
        _, body, reason = split_think(out[-1]["content"])
        if reason == "ok" and body.strip() in ("", "null"):
            out.pop()
            removed = 1
    if out and out[-1]["role"] == "user":
        out.pop()
    return out, removed


def reconstruct(conversation: list[dict[str, str]], builder: Any,
                stats: Counter) -> list[dict[str, str]] | None:
    """Turn one upstream `conversations` list into our `messages` list."""
    if not conversation:
        stats["drop_empty_conversation"] += 1
        return None
    if any(turn.get("role") not in ("user", "assistant") for turn in conversation):
        stats["drop_unexpected_role"] += 1
        return None
    if any(not isinstance(turn.get("content"), str) for turn in conversation):
        stats["drop_nonstring_message"] += 1
        return None

    spliced, removed = splice_retries(conversation)
    stats["spliced_retry_turns"] += removed
    spliced, truncated = truncate_dead_tail(spliced)
    stats["truncated_null_tail"] += truncated
    if not spliced or spliced[0]["role"] != "user":
        stats["drop_no_user_prompt"] += 1
        return None

    messages: list[dict[str, str]] = []
    previous_rewritten = False
    for turn in spliced:
        content = turn["content"]
        if turn["role"] == "user":
            if previous_rewritten:
                repaired = builder._WARNING_PREAMBLE.sub("", content)
                if repaired != content:
                    stats["repaired_warning_preamble"] += 1
                content = repaired
            if not content.strip():
                stats["drop_empty_user_turn"] += 1
                return None
            messages.append({"role": "user", "content": content})
            previous_rewritten = False
            continue

        if any(literal in content for literal in FORBIDDEN_IN_SOURCE):
            stats["drop_control_token"] += 1
            return None
        think, body, reason = split_think(content)
        if think is None:
            stats[f"drop_{reason}"] += 1
            return None
        if "<think>" in body or "</think>" in body:
            stats["drop_think_markup_in_body"] += 1
            return None
        canonical, rewritten, reason = builder.normalize_assistant(body)
        if canonical is None:
            stats[f"drop_{reason}"] += 1
            return None
        think = think.strip()
        messages.append({
            "role": "assistant",
            "content": f"<think>\n{think}\n</think>\n\n{canonical}" if think else canonical,
        })
        previous_rewritten = rewritten

    if messages[-1]["role"] != "assistant":
        stats["drop_not_assistant_terminated"] += 1
        return None
    if len(messages) < 2:
        stats["drop_too_short"] += 1
        return None
    return messages


def process_file(job: dict[str, Any]) -> dict[str, Any]:
    """Convert one source parquet. Runs in a worker process.

    Writes surviving rows to its own shard and returns only lightweight per-row
    metadata, so the parent can do a global dedup and a group-disjoint split without
    ever holding 366k trajectories in memory at once.
    """
    global _TOKENIZER, _BUILDER, _EXPORTER
    import pyarrow.parquet as pq
    import pandas as pd

    if _BUILDER is None:
        _BUILDER = load_script("03_build_sft_data")
        _EXPORTER = load_script("15_export_pretokenized")
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(job["tokenizer"])

    stats: Counter = Counter()
    rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    max_seq_len = job["max_seq_len"]

    reader = pq.ParquetFile(job["path"])
    parts, part = job["parts"], job["part"]
    for ordinal, batch in enumerate(reader.iter_batches(batch_size=64)):
        # Round-robin batches across the parts of one file. Every part decodes the
        # whole file and tokenizes a `1/parts` slice of it, which is the right trade:
        # `math.parquet` alone is 162,692 of the corpus's 366,154 rows in a single row
        # group, so without this one worker does 44 % of the job while 19 idle, and
        # parquet decode is an order of magnitude cheaper than tokenization.
        if parts > 1 and ordinal % parts != part:
            continue
        for record in batch.to_pylist():
            messages = reconstruct(record["conversations"], _BUILDER, stats)
            if messages is None:
                continue
            try:
                ids, mask = _EXPORTER.qwen3_5_mask(_TOKENIZER, messages)
            except ValueError as exc:
                stats["drop_contract" if "contract" in str(exc) else "drop_mask_error"] += 1
                continue
            if len(ids) > max_seq_len:
                stats["drop_too_long"] += 1
                continue
            if not sum(mask):
                stats["drop_no_trained_tokens"] += 1
                continue
            if mask[0] != 0:
                stats["drop_supervised_first_token"] += 1
                continue

            think_tokens = sum(
                len(_TOKENIZER(m["content"].split("</think>")[0], add_special_tokens=False)
                    ["input_ids"])
                for m in messages
                if m["role"] == "assistant" and "</think>" in m["content"]
            )
            rows.append({
                "messages": messages,
                "trajectory_id": str(record["trial_name"]),
                "task_group_id": str(record["task"]),
                "model_name": str(record["model"]),
                "subset": job["subset"],
                "domain": job["domain"],
                "n_tokens": len(ids),
                "n_trained_tokens": sum(mask),
                "n_assistant_turns": sum(1 for m in messages if m["role"] == "assistant"),
            })
            meta.append({
                "task_group_id": str(record["task"]),
                "trajectory_id": str(record["trial_name"]),
                "content_hash": hashlib.sha256(
                    json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
                "command_signature": command_signature(messages),
                "n_tokens": len(ids),
                "n_trained_tokens": sum(mask),
                "n_assistant_turns": rows[-1]["n_assistant_turns"],
                "think_tokens": think_tokens,
                "subset": job["subset"],
                "shard": job["shard"],
                "index": len(rows) - 1,
            })
            stats["built"] += 1

    shard_path = Path(job["shard_dir"]) / f"{job['shard']}.parquet"
    if rows:
        pd.DataFrame(rows).to_parquet(shard_path, index=False)
    # `source_rows` is attributed to part 0 only, or a split file would count its rows
    # once per part and the manifest would claim more input than upstream has.
    return {"shard": job["shard"], "subset": job["subset"], "domain": job["domain"],
            "source_rows": reader.metadata.num_rows if part == 0 else 0,
            "stats": dict(stats), "meta": meta, "shard_path": str(shard_path)}


def discover(root: Path, rows_per_part: int = 12000) -> list[dict[str, Any]]:
    """Map every source parquet to its (subset, domain), splitting the large ones."""
    import pyarrow.parquet as pq

    jobs: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.parquet")):
        parts = path.relative_to(root).parts
        if parts[0] == "dataset_adapters":
            subset, domain = "dataset_adapters", path.stem
        elif parts[0] == "synthetic_tasks" and len(parts) >= 4:
            subset, domain = f"skill_based_{parts[2]}", parts[3]
        else:
            continue
        num_rows = pq.ParquetFile(path).metadata.num_rows  # footer only, no decode
        splits = max(1, -(-num_rows // rows_per_part))
        for part in range(splits):
            jobs.append({"path": str(path), "subset": subset, "domain": domain,
                         "part": part, "parts": splits,
                         "shard": f"{subset}__{domain}__{part:02d}"})
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help=f"local clone of {SOURCE_DATASET}")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--holdout", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1228)
    ap.add_argument("--workers", type=int, default=min(20, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    jobs = discover(args.source)
    if not jobs:
        sys.exit(f"no source parquet under {args.source}")
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        job.update(tokenizer=str(args.tokenizer), out_dir=str(args.out_dir),
                   shard_dir=str(shard_dir), max_seq_len=args.max_seq_len)
    print(f"[source] {len(jobs)} files, {args.workers} workers", flush=True)

    stats: Counter = Counter()
    meta: list[dict[str, Any]] = []
    per_subset: dict[str, Counter] = defaultdict(Counter)
    source_rows = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(process_file, jobs):
            stats.update(result["stats"])
            per_subset[result["subset"]].update(result["stats"])
            source_rows += result["source_rows"]
            meta.extend(result["meta"])
            print(f"  [{result['shard']}] source={result['source_rows']:,} "
                  f"built={result['stats'].get('built', 0):,}", flush=True)
    print(f"[build] source={source_rows:,} built={stats['built']:,} "
          f"spliced_retry_turns={stats['spliced_retry_turns']:,}", flush=True)
    if not meta:
        sys.exit("nothing survived the gates")

    # ---- global dedup --------------------------------------------------------
    # Keyed WITH the task, as everywhere else in this repo: two different tasks that
    # need the same commands are two instruction->action mappings, not a duplicate.
    # It bites here because upstream runs several episodes per task.
    seen_content: set[str] = set()
    seen_command: set[tuple[str, str]] = set()
    owners: dict[str, set[str]] = defaultdict(set)
    kept: list[dict[str, Any]] = []
    for row in sorted(meta, key=lambda r: (r["subset"], r["trajectory_id"])):
        owners[row["command_signature"]].add(row["task_group_id"])
        if row["content_hash"] in seen_content:
            stats["dedup_exact"] += 1
            continue
        key = (row["task_group_id"], row["command_signature"])
        if key in seen_command:
            stats["dedup_command_signature"] += 1
            continue
        seen_content.add(row["content_hash"])
        seen_command.add(key)
        kept.append(row)
    cross_task = sum(1 for tasks in owners.values() if len(tasks) > 1)
    print(f"[dedup] kept={len(kept):,} exact={stats['dedup_exact']:,} "
          f"cmd={stats['dedup_command_signature']:,} "
          f"(signatures shared across tasks, not dropped: {cross_task:,})", flush=True)

    # ---- split, disjoint by task --------------------------------------------
    # Upstream runs several episodes per task, so a row-wise split would put siblings
    # of a held-out task in train and the holdout loss would be reading a task the
    # model had already been shown.
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        by_task[row["task_group_id"]].append(row)
    task_ids = sorted(by_task)
    rng = random.Random(args.seed)
    rng.shuffle(task_ids)
    holdout_tasks: set[str] = set()
    holdout_rows = 0
    for task_id in task_ids:
        if holdout_rows >= args.holdout:
            break
        holdout_tasks.add(task_id)
        holdout_rows += len(by_task[task_id])
    want: dict[tuple[str, int], str] = {
        (row["shard"], row["index"]): ("holdout" if row["task_group_id"] in holdout_tasks
                                      else "train")
        for row in kept
    }
    print(f"[split] train={len(kept) - holdout_rows:,} holdout={holdout_rows:,} "
          f"(holdout covers {len(holdout_tasks):,} whole tasks)", flush=True)

    # ---- second pass: stream the kept rows into the two outputs --------------
    # Train is written one file PER SUBSET, holdout as a single file. 3.88 B tokens in
    # one 6.5 GiB parquet is not a usable artifact: nobody trains on all of it, and
    # pulling the whole thing to use the 5,681-row `mixed` slice is absurd. Hugging
    # Face `configs` can then present both views -- whole corpus and one subset --
    # over the same files, with nothing duplicated on disk.
    columns = ["messages", "trajectory_id", "task_group_id", "model_name", "subset",
               "domain", "n_tokens", "n_trained_tokens", "n_assistant_turns"]
    writers: dict[str, Any] = {}
    counts: Counter = Counter()
    paths: dict[str, Path] = {
        "holdout": args.out_dir / "nemotron_sft_holdout.parquet",
        **{f"train_{subset}": args.out_dir / f"nemotron_sft_train_{subset}.parquet"
           for subset in sorted({job["subset"] for job in jobs})},
    }
    for job in jobs:
        shard_path = Path(job["shard_dir"]) / f"{job['shard']}.parquet"
        if not shard_path.is_file():
            continue
        frame = pd.read_parquet(shard_path)
        for key, split in (("holdout", "holdout"), (f"train_{job['subset']}", "train")):
            take = [i for i in range(len(frame)) if want.get((job["shard"], i)) == split]
            if not take:
                continue
            part = frame.iloc[take][columns]
            table = pa.Table.from_pandas(part, preserve_index=False)
            if key not in writers:
                writers[key] = pq.ParquetWriter(paths[key], table.schema)
            writers[key].write_table(table)
            counts[key] += len(part)
    for writer in writers.values():
        writer.close()
    for key in sorted(paths):
        if counts[key]:
            print(f"[write] {paths[key]} rows={counts[key]:,}", flush=True)

    lengths = [r["n_tokens"] for r in kept]
    trained = [r["n_trained_tokens"] for r in kept]
    turns = [r["n_assistant_turns"] for r in kept]
    think = [r["think_tokens"] for r in kept]
    subset_rows = Counter(r["subset"] for r in kept)
    manifest = {
        "source_dataset": SOURCE_DATASET,
        "source_license": SOURCE_LICENSE,
        "source_rows": source_rows,
        "source_models": ["deepseek-ai/DeepSeek-V3.2"],
        "built": stats["built"],
        "after_dedup": len(kept),
        "train_examples": sum(v for k, v in counts.items() if k.startswith("train_")),
        "holdout_examples": counts["holdout"],
        "holdout_tasks": len(holdout_tasks),
        "holdout_mode": "group (task_group_id); upstream runs several episodes per task",
        "groups_covered": len(by_task),
        "rows_per_subset": dict(sorted(subset_rows.items())),
        "command_signatures_shared_across_tasks": cross_task,
        "spliced_retry_turns": stats["spliced_retry_turns"],
        "repaired_warning_preamble": stats["repaired_warning_preamble"],
        "max_seq_len": args.max_seq_len,
        "tokenizer": str(args.tokenizer),
        "reasoning": {
            "kept_in_messages": True,
            "supervised_turns": "the final assistant turn only",
            "why": "observations are plain `user` turns, so the Qwen3.5 template's "
                   "last_query_index lands on the last observation and reasoning is "
                   "emitted for the final assistant turn alone. That matches "
                   "inference: the terminus-2 harness re-renders history every turn "
                   "under the same rule, so the model never sees its own earlier "
                   "reasoning either.",
            "think_tokens_in_source": sum(think),
        },
        "token_stats": {
            "mean": statistics.mean(lengths),
            "p50": statistics.median(lengths),
            "p90": statistics.quantiles(lengths, n=10)[8] if len(lengths) > 10 else max(lengths),
            "p99": statistics.quantiles(lengths, n=100)[98] if len(lengths) > 100 else max(lengths),
            "max": max(lengths),
            "total_tokens": sum(lengths),
            "trained_tokens": sum(trained),
            "trained_fraction": round(sum(trained) / max(1, sum(lengths)), 4),
        },
        "turns": {"mean": statistics.mean(turns), "max": max(turns)},
        "drop_counters": dict(sorted(stats.items())),
        "per_subset_counters": {k: dict(sorted(v.items())) for k, v in sorted(per_subset.items())},
        "seed": args.seed,
        "train_parquet_per_subset": {k.removeprefix("train_"): str(v)
                                     for k, v in sorted(paths.items())
                                     if k.startswith("train_") and counts[k]},
        "train_rows_per_subset_after_split": {k.removeprefix("train_"): counts[k]
                                              for k in sorted(paths) if counts[k]
                                              and k.startswith("train_")},
        "holdout_parquet": str(paths["holdout"]),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"[manifest] {manifest_path}", flush=True)
    print(json.dumps(manifest["token_stats"], indent=2, sort_keys=True))
    print(json.dumps(manifest["rows_per_subset"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
