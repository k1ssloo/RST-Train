"""`scripts/sft_common.py` -- the tail every SFT builder shares, pinned to what the
builders did inline before it existed.

The refactor's promise is behaviour-preserving parquet output for the RST,
OpenThoughts, TMax and Nemotron builders. The oracle for each function here is the
inline code it replaced, re-implemented in the test, so a change to the shared copy
that would move a published dataset is caught without the datasets.
"""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, need  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))

import sft_common as sc  # noqa: E402


def rec(tid, group, sig, content, **extra):
    return {"trajectory_id": tid, "task_group_id": group, "command_signature": sig,
            "content_hash": content, **extra}


# ------------------------------------------------------------------ dedup


def test_exact_duplicates_drop_and_the_first_in_key_order_survives():
    stats: Counter = Counter()
    kept, cross = sc.dedup_records(
        [rec("b", "g1", "s1", "h1"), rec("a", "g1", "s2", "h1")], stats)
    assert [r["trajectory_id"] for r in kept] == ["a"], "sorted by trajectory_id, not input order"
    assert stats["dedup_exact"] == 1 and stats["dedup_command_signature"] == 0
    assert cross == 0


def test_the_command_signature_key_is_scoped_to_the_task():
    stats: Counter = Counter()
    kept, cross = sc.dedup_records(
        [rec("a", "g1", "same", "h1"), rec("b", "g1", "same", "h2"), rec("c", "g2", "same", "h3")],
        stats)
    assert [r["trajectory_id"] for r in kept] == ["a", "c"], (
        "same commands on the same task is a duplicate; on another task it is not")
    assert stats["dedup_command_signature"] == 1
    assert cross == 1, "one signature is shared across two tasks -- reported, not acted on"


def test_the_nemotron_key_orders_by_subset_first():
    stats: Counter = Counter()
    kept, _ = sc.dedup_records(
        [rec("a", "g", "s", "h", subset="z"), rec("b", "g", "s", "h", subset="a")],
        stats, key=lambda r: (r["subset"], r["trajectory_id"]))
    assert [r["trajectory_id"] for r in kept] == ["b"]


# ------------------------------------------------------------------ split


def _inline_group_split(records, holdout, seed, max_fraction=None):
    """The code 03/03e/03f carried before the shared function, verbatim in spirit."""
    by_group = defaultdict(list)
    for r in records:
        by_group[r["task_group_id"]].append(r)
    order = sorted(by_group)
    rng = random.Random(seed)
    rng.shuffle(order)
    target = holdout if max_fraction is None else min(holdout, int(len(records) * max_fraction))
    held, groups = [], set()
    for g in order:
        if len(held) >= target:
            break
        held.extend(by_group[g])
        groups.add(g)
    train = [r for r in records if r["task_group_id"] not in groups]
    return train, held, groups


def _pool(n_groups=37, per_group=(1, 2, 3, 5), seed=3):
    rng = random.Random(seed)
    out = []
    for g in range(n_groups):
        for k in range(rng.choice(per_group)):
            out.append(rec(f"t{g:03d}_{k}", f"g{g:03d}", f"s{g}{k}", f"h{g}{k}"))
    rng.shuffle(out)
    return out


def test_group_split_reproduces_the_inline_algorithm_exactly():
    pool = _pool()
    for seed in (1228, 7, 99):
        for holdout in (0, 5, 20, 200):
            train, held, groups, target = sc.group_disjoint_split(pool, holdout=holdout, seed=seed)
            t2, h2, g2 = _inline_group_split(pool, holdout, seed)
            assert (train, held, groups) == (t2, h2, g2), (seed, holdout)
            assert target == holdout


def test_group_split_with_the_fraction_cap_matches_the_rst_builder():
    pool = _pool()
    train, held, groups, target = sc.group_disjoint_split(
        pool, holdout=200, seed=1228, max_fraction=0.15)
    t2, h2, g2 = _inline_group_split(pool, 200, 1228, max_fraction=0.15)
    assert (train, held, groups) == (t2, h2, g2)
    assert target == int(len(pool) * 0.15) < 200, "the cap bound, and the caller is told"


def test_no_group_is_on_both_sides_and_nothing_is_lost():
    pool = _pool()
    train, held, groups, _ = sc.group_disjoint_split(pool, holdout=30, seed=5)
    assert {r["task_group_id"] for r in train}.isdisjoint(groups)
    assert {r["task_group_id"] for r in held} == groups
    assert len(train) + len(held) == len(pool)
    assert train == [r for r in pool if r["task_group_id"] not in groups], "input order kept"


# ------------------------------------------------------------------ stats


def test_quantiles_are_numpy_linear_interpolation():
    np = need("numpy")
    for values in ([3], [1, 2], [5, 1, 9, 3, 3, 7], list(range(1, 101)), [2.5, 1.5, 100.0]):
        for q in (0.5, 0.9, 0.99):
            assert abs(sc.quantile(values, q) - float(np.quantile(values, q))) < 1e-9, (values, q)


def test_token_stats_has_the_manifest_shape_and_the_trained_block_when_asked():
    out = sc.token_stats([10, 20, 30, 40])
    assert set(out) == {"mean", "p50", "p90", "p99", "max", "total_tokens"}
    assert out["mean"] == 25.0 and out["p50"] == 25.0 and out["max"] == 40
    assert out["total_tokens"] == 100
    with_trained = sc.token_stats([10, 20, 30, 40], [1, 2, 3, 4])
    assert with_trained["trained_tokens"] == 10 and with_trained["trained_fraction"] == 0.1


def test_token_stats_of_nothing_is_zeros_not_an_exception():
    assert sc.token_stats([])["total_tokens"] == 0


# ------------------------------------------------------------------ gate


class _FakeTokenizer:
    """render == tokenize-directly by construction; one token per character."""

    def __init__(self, contract_ok=True):
        self.contract_ok = contract_ok

    def apply_chat_template(self, messages, tokenize, return_dict):
        text = "".join(m["content"] for m in messages)
        if not tokenize:
            return text
        ids = [ord(c) for c in text]
        return ids if self.contract_ok else ids + [0]

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) for c in text]}


def test_the_gate_returns_the_length_when_both_checks_pass():
    n, reason = sc.contract_length_gate(_FakeTokenizer(), [{"content": "abcd"}], 10)
    assert (n, reason) == (4, None)


def test_the_gate_names_the_contract_before_the_length():
    n, reason = sc.contract_length_gate(_FakeTokenizer(contract_ok=False), [{"content": "a" * 50}], 10)
    assert (n, reason) == (None, "contract_mismatch")
    n, reason = sc.contract_length_gate(_FakeTokenizer(), [{"content": "a" * 50}], 10)
    assert (n, reason) == (None, "too_long")


# ------------------------------------------------------------------ wiring


def test_every_sft_builder_imports_the_shared_tail_instead_of_inlining_it():
    for stem in ("03_build_sft_data", "03d_build_openthoughts_sft",
                 "03e_build_tmax_sft", "03f_build_nemotron_sft"):
        source = (ROOT / "scripts" / f"{stem}.py").read_text(encoding="utf-8")
        assert "from sft_common import" in source, stem
        assert "statistics.quantiles(" not in source, f"{stem} grew its own percentile again"
        assert "seen_content" not in source, f"{stem} grew its own dedup again"


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
