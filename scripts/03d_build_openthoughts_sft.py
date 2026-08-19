#!/usr/bin/env python3
"""Convert `open-thoughts/OpenThoughts-Agent-v1-SFT` into this repo's SFT format.

    python scripts/03d_build_openthoughts_sft.py \
        --source data/openthoughts-agent-v1/source-train.parquet \
        --tokenizer data/Qwen3.5-27B-tokenizer \
        --out-dir data/openthoughts-agent-v1 --max-seq-len 32768

Why this is a converter and not another builder: the upstream dataset is already
the same agent contract this repo trains on. It is `terminus-2` output, the
assistant turns are the same ``{"analysis", "plan", "commands", "task_complete"}``
JSON, and the observations are the same ``New Terminal Output:`` blocks. What
differs is packaging:

    upstream                          this repo
    --------                          ---------
    conversations [{role, content}]   messages [{role, content}]     (identical shape)
    task            (unique per row)  task_group_id
    trial_name      (unique per row)  trajectory_id
    model                             model_name
    agent, model_provider, date,      dropped (single-valued or provenance-only,
    episode, run_id                   kept in the manifest instead)

So the real work is not the renaming. It is putting the upstream assistant turns
through the *same* normalizer the RST pipeline uses, so that one canonical JSON
form and one loss-mask contract cover both datasets:

  * `normalize_assistant` is loaded by path out of `03_build_sft_data.py` rather
    than reimplemented, for the same reason `06b_eval_offline.py` does it — two
    copies of a format contract are two contracts.
  * the warning-feedback repair applies here too: when a turn is renormalized,
    the "Previous response had warnings:" preamble in the *next* observation is
    complaining about a formatting error that no longer exists in the data, so it
    is stripped. Left in, it trains the model to expect a scolding for output it
    was just shown as correct.
  * the same `apply_chat_template` render-vs-tokenize contract gate runs, because
    that is what makes `--loss-mask-type qwen3_5` and
    `15_export_pretokenized.py` applicable to the result.

A trajectory is dropped whole when any assistant turn fails to normalize. The
alternative — truncating at the last good turn — was measured on this dataset and
rejected: of the failing trajectories the median salvageable fraction is 0.15 and
323 of them fail on the first assistant turn, so it would mostly add one-turn
stubs of ten-turn episodes and skew the length distribution.

Upstream is Apache-2.0. Every count this prints goes into `manifest.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SOURCE_DATASET = "open-thoughts/OpenThoughts-Agent-v1-SFT"
SOURCE_FILE = "data/train-00000-of-00001.parquet"


def load_builder() -> Any:
    """Load `03_build_sft_data.py` by path and return it as a module.

    By path because `scripts/` is not a package and the filename starts with a
    digit. The point is to share `normalize_assistant`, `command_signature` and
    `_WARNING_PREAMBLE` with the RST pipeline, not to copy them: a divergence
    between the two would silently mean two different canonical forms in one
    training mixture.
    """
    path = Path(__file__).resolve().parent / "03_build_sft_data.py"
    spec = importlib.util.spec_from_file_location("_rst_sft_builder", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        sys.exit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct(conversation: list[dict[str, str]], builder: Any,
                stats: Counter) -> dict[str, Any] | None:
    """Turn one upstream `conversations` list into our `messages` list.

    Returns None and counts a reason when the trajectory cannot be represented.
    """
    if not conversation:
        stats["drop_empty_conversation"] += 1
        return None
    roles = [turn.get("role") for turn in conversation]
    # Unexpected roles are checked FIRST so the counter names the actual novelty. A
    # `system` turn arriving upstream would otherwise be filed as "does not start
    # with a user turn", which points at the wrong thing: the problem is that a
    # system turn changes the rendered prefix, and therefore the loss mask.
    if any(role not in ("user", "assistant") for role in roles):
        stats["drop_unexpected_role"] += 1
        return None
    if roles[0] != "user":
        stats["drop_no_user_prompt"] += 1
        return None

    messages: list[dict[str, str]] = []
    rewritten_turns = 0
    previous_rewritten = False

    for turn in conversation:
        content = turn.get("content")
        if not isinstance(content, str):
            stats["drop_nonstring_message"] += 1
            return None

        if turn["role"] == "user":
            text = content
            if previous_rewritten:
                repaired = builder._WARNING_PREAMBLE.sub("", text)
                if repaired != text:
                    stats["repaired_warning_preamble"] += 1
                text = repaired
            if not text.strip():
                stats["drop_empty_user_turn"] += 1
                return None
            messages.append({"role": "user", "content": text})
            continue

        canonical, rewritten, reason = builder.normalize_assistant(content)
        if canonical is None:
            stats[f"drop_{reason}"] += 1
            return None
        messages.append({"role": "assistant", "content": canonical})
        rewritten_turns += int(rewritten)
        previous_rewritten = rewritten

    if messages[-1]["role"] != "assistant":
        stats["drop_not_assistant_terminated"] += 1
        return None
    if len(messages) < 2:
        stats["drop_too_short"] += 1
        return None

    return {
        "messages": messages,
        "n_assistant_turns": sum(m["role"] == "assistant" for m in messages),
        "n_rewritten_turns": rewritten_turns,
        "command_signature": builder.command_signature(messages),
        "content_hash": hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help=f"the upstream parquet ({SOURCE_DATASET}, {SOURCE_FILE})")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1228)
    args = ap.parse_args()

    import pandas as pd

    builder = load_builder()
    if not args.source.exists():
        sys.exit(f"{args.source} not found. Download it with:\n"
                 f"  curl -L -o {args.source} "
                 f"https://huggingface.co/datasets/{SOURCE_DATASET}/resolve/main/{SOURCE_FILE}")

    frame = pd.read_parquet(args.source)
    required = {"conversations", "task", "trial_name", "model"}
    missing = required - set(frame.columns)
    if missing:
        sys.exit(f"{args.source} is missing {sorted(missing)}; upstream schema changed")
    print(f"[source] {len(frame)} rows from {SOURCE_DATASET}", flush=True)

    stats: Counter = Counter()
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        record = reconstruct(list(row.conversations), builder, stats)
        if record is None:
            continue
        record["trajectory_id"] = str(row.trial_name)
        record["task_group_id"] = str(row.task)
        record["model_name"] = str(row.model)
        records.append(record)
        stats["built"] += 1
    print(f"[reconstruct] built={stats['built']} "
          f"dropped={len(frame) - stats['built']} "
          f"repaired_warning_preamble={stats['repaired_warning_preamble']}", flush=True)

    # ---- dedup ---------------------------------------------------------------
    # Exact content is always safe to drop. The command signature is keyed WITH the
    # task, exactly as in the RST pipeline: two different tasks that happen to need
    # the same commands are two different instruction->action mappings, not a
    # duplicate. Upstream tasks are unique, so that key drops nothing here -- the
    # cross-task collision count is reported instead of acted on, because it is a
    # property of the task pool worth knowing and not a defect.
    seen_content: set[str] = set()
    seen_cmd: set[tuple[str, str]] = set()
    signature_owners: dict[str, set[str]] = defaultdict(set)
    kept: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: r["trajectory_id"]):
        signature_owners[record["command_signature"]].add(record["task_group_id"])
        if record["content_hash"] in seen_content:
            stats["dedup_exact"] += 1
            continue
        key = (record["task_group_id"], record["command_signature"])
        if key in seen_cmd:
            stats["dedup_command_signature"] += 1
            continue
        seen_content.add(record["content_hash"])
        seen_cmd.add(key)
        kept.append(record)
    cross_task = sum(1 for owners in signature_owners.values() if len(owners) > 1)
    print(f"[dedup] kept={len(kept)} exact_dropped={stats['dedup_exact']} "
          f"cmd_dropped={stats['dedup_command_signature']} "
          f"(command signatures shared across tasks, not dropped: {cross_task})", flush=True)

    # ---- tokenize + verify the chat-template contract ------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    final: list[dict[str, Any]] = []
    lengths: list[int] = []
    for record in kept:
        messages = record["messages"]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, return_dict=False)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        expected = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
        if ids != expected:
            # render-then-tokenize must equal tokenize-directly, or the character
            # offsets 15_export_pretokenized.py builds the mask from do not apply.
            stats["drop_template_contract_mismatch"] += 1
            continue
        if len(ids) > args.max_seq_len:
            stats["drop_too_long"] += 1
            continue
        record["n_tokens"] = len(ids)
        lengths.append(len(ids))
        final.append(record)
    print(f"[tokenize] kept={len(final)} "
          f"contract_mismatch={stats['drop_template_contract_mismatch']} "
          f"too_long={stats['drop_too_long']}", flush=True)
    if not final:
        sys.exit("nothing survived the gates")

    # ---- split ---------------------------------------------------------------
    # Upstream `task` is unique per row, so a row-wise split is already
    # group-disjoint -- there are no siblings of a held-out task in train. That is
    # the property the RST pipeline has to work for with --holdout-mode group; it
    # is free here, and the manifest records it so no one has to re-derive it.
    final.sort(key=lambda r: r["trajectory_id"])
    rng = random.Random(args.seed)
    shuffled = list(final)
    rng.shuffle(shuffled)
    holdout = shuffled[: args.holdout]
    train = shuffled[args.holdout :]
    groups_train = {r["task_group_id"] for r in train}
    groups_holdout = {r["task_group_id"] for r in holdout}
    overlap = groups_train & groups_holdout
    if overlap:
        sys.exit(f"{len(overlap)} task_group_id(s) are in both splits; upstream `task` "
                 f"is no longer unique per row and the split needs the group logic")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["messages", "trajectory_id", "task_group_id", "model_name",
               "n_tokens", "n_assistant_turns", "n_rewritten_turns"]
    paths = {}
    for name, rows in (("train", train), ("holdout", holdout)):
        frame_out = pd.DataFrame([{k: r[k] for k in columns} for r in rows])
        path = args.out_dir / f"ota_sft_{name}.parquet"
        frame_out.to_parquet(path, index=False)
        paths[name] = str(path)
        print(f"[write] {path} rows={len(frame_out)}", flush=True)

    turns = [r["n_assistant_turns"] for r in final]
    manifest = {
        "source_dataset": SOURCE_DATASET,
        "source_file": SOURCE_FILE,
        "source_license": "apache-2.0",
        "source_rows": len(frame),
        "source_agent": sorted({str(v) for v in frame["agent"]}),
        "source_models": sorted({str(v) for v in frame["model"]}),
        "reconstructed": stats["built"],
        "after_dedup": len(kept),
        "final_examples": len(final),
        "train_examples": len(train),
        "holdout_examples": len(holdout),
        "holdout_mode": "row (upstream task is unique per row, so this is group-disjoint)",
        "groups_covered": len({r["task_group_id"] for r in final}),
        "command_signatures_shared_across_tasks": cross_task,
        "max_seq_len": args.max_seq_len,
        "tokenizer": str(args.tokenizer),
        "token_stats": {
            "mean": statistics.mean(lengths),
            "p50": statistics.median(lengths),
            "p90": statistics.quantiles(lengths, n=10)[8] if len(lengths) > 10 else max(lengths),
            "p99": statistics.quantiles(lengths, n=100)[98] if len(lengths) > 100 else max(lengths),
            "max": max(lengths),
            "total_tokens": sum(lengths),
        },
        "turns": {"mean": statistics.mean(turns), "max": max(turns)},
        "rewritten_turn_fraction": (
            sum(r["n_rewritten_turns"] for r in final) / max(1, sum(turns))
        ),
        "drop_counters": dict(sorted(stats.items())),
        "seed": args.seed,
        "train_parquet": paths["train"],
        "holdout_parquet": paths["holdout"],
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"[manifest] {manifest_path}", flush=True)
    print(json.dumps(manifest["token_stats"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
