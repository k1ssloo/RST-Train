#!/usr/bin/env python3
"""The tail every SFT builder shares: dedup, the template gate, the split, the stats.

Four converters feed one training mixture -- the RST release (`03_build_sft_data.py`),
OpenThoughts (`03d`), TMax (`03e`), Nemotron (`03f`) -- and now the repo's own
rollouts (`03h`). What makes them concatenable is that every one applies the SAME
gates to the SAME canonical form. Until now each carried its own copy of those gates,
and the copies had already started to disagree in the one place nobody looks: the
manifests' `p90`/`p99` came from `numpy.quantile` in one script and from
`statistics.quantiles(method="exclusive")` in the other three, so the same lengths
reported different percentiles depending on which builder wrote them.

Everything here is behaviour-preserving for the parquet outputs of the existing
builders (checked by rebuilding and diffing -- see WORKLOG.md). Only the descriptive
percentiles in three manifests move, to the numpy-equivalent definition the RST
manifests and PLAN.md's quoted p50/p90/p99 already use.

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sft_common import dedup_records, contract_length_gate, group_disjoint_split, token_stats
"""

from __future__ import annotations

import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from typing import Any

Record = dict[str, Any]


# ------------------------------------------------------------------------ dedup

def dedup_records(
    records: Iterable[Record],
    stats: Counter,
    *,
    key: Callable[[Record], Any] = lambda r: r["trajectory_id"],
    group: Callable[[Record], str] = lambda r: r["task_group_id"],
) -> tuple[list[Record], int]:
    """Drop exact duplicates, then near-duplicates within one task.

    Two keys, in this order:
      * `content_hash`       -- byte-identical canonical conversation, always dropped;
      * (task, command_signature) -- the same command sequence on the same task is
        the same demonstration. Keyed WITH the task on purpose: two different tasks
        that happen to need the same commands are two instruction->action mappings,
        not a duplicate.

    Records are visited in `key` order so the survivor of a duplicate set is
    deterministic. Returns `(kept, cross_task)`, where `cross_task` counts command
    signatures that occur under more than one task -- reported, never acted on,
    because it is a property of the task pool and not a defect. Increments
    `stats["dedup_exact"]` and `stats["dedup_command_signature"]`.
    """
    seen_content: set[str] = set()
    seen_command: set[tuple[str, str]] = set()
    owners: dict[str, set[str]] = defaultdict(set)
    kept: list[Record] = []
    for record in sorted(records, key=key):
        task = group(record)
        owners[record["command_signature"]].add(task)
        if record["content_hash"] in seen_content:
            stats["dedup_exact"] += 1
            continue
        command_key = (task, record["command_signature"])
        if command_key in seen_command:
            stats["dedup_command_signature"] += 1
            continue
        seen_content.add(record["content_hash"])
        seen_command.add(command_key)
        kept.append(record)
    cross_task = sum(1 for tasks in owners.values() if len(tasks) > 1)
    return kept, cross_task


# ------------------------------------------------------------- template gate

def contract_length_gate(tokenizer, messages: list[dict], max_seq_len: int
                         ) -> tuple[int | None, str | None]:
    """slime's chat-template contract plus the length cap, as one decision.

    Returns `(n_tokens, None)` when the row may ship, else `(None, reason)` with
    `reason` in {"contract_mismatch", "too_long"}.

    The contract: render-then-tokenize must equal tokenize-directly. If it does not,
    the character offsets `15_export_pretokenized.py` builds the loss mask from do
    not apply to the ids the model will actually see, and the mask would be silently
    misaligned. No offset mapping is taken here on purpose -- this stage needs the
    LENGTH and the contract, nothing else; the mask is built once, later.
    """
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, return_dict=False)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    expected = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    if ids != expected:
        return None, "contract_mismatch"
    if len(ids) > max_seq_len:
        return None, "too_long"
    return len(ids), None


# ------------------------------------------------------------------- split

def group_disjoint_split(
    records: list[Record],
    *,
    holdout: int,
    seed: int,
    group: str = "task_group_id",
    max_fraction: float | None = None,
) -> tuple[list[Record], list[Record], set[str], int]:
    """Hold out whole groups, in a seed-determined order, until the target is met.

    No group appears in both splits, so the holdout loss measures transfer to an
    unseen task rather than memorization of a task whose siblings are in train.
    `max_fraction` caps the holdout as a fraction of the pool so a coarse group
    structure (few groups, many rows each) cannot swallow the training set; the
    effective target is returned so the caller can say when it was reduced.

    Order is preserved: `train` keeps the input order, `held` is in the order groups
    were taken. Callers that want sorted output sort it themselves -- the existing
    builders differ on that, and this function must reproduce each of them exactly.
    """
    by_group: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_group[record[group]].append(record)
    order = sorted(by_group)
    random.Random(seed).shuffle(order)
    target = holdout
    if max_fraction is not None:
        target = min(holdout, int(len(records) * max_fraction))
    held: list[Record] = []
    groups: set[str] = set()
    for group_id in order:
        if len(held) >= target:
            break
        held.extend(by_group[group_id])
        groups.add(group_id)
    train = [record for record in records if record[group] not in groups]
    return train, held, groups, target


# ------------------------------------------------------------------- stats

def quantile(values: list[float], q: float) -> float:
    """numpy.quantile(values, q) with the default linear interpolation, stdlib only."""
    if not values:
        raise ValueError("quantile of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * fraction)


def token_stats(lengths: list[int], trained: list[int] | None = None) -> dict[str, Any]:
    """The manifest's length block, computed one way for every builder.

    `p50/p90/p99` are numpy-style linear-interpolation quantiles -- the definition
    behind every number PLAN.md quotes -- rather than `statistics.quantiles`'s
    default exclusive method, which three converters used to disagree with.
    """
    if not lengths:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0, "total_tokens": 0}
    out: dict[str, Any] = {
        "mean": float(statistics.fmean(lengths)),
        "p50": quantile(lengths, 0.50),
        "p90": quantile(lengths, 0.90),
        "p99": quantile(lengths, 0.99),
        "max": int(max(lengths)),
        "total_tokens": int(sum(lengths)),
    }
    if trained is not None:
        out["trained_tokens"] = int(sum(trained))
        out["trained_fraction"] = round(sum(trained) / max(1, sum(lengths)), 4)
    return out
