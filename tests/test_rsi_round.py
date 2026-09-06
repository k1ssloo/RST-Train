"""`scripts/21_rsi_round.sh` -- one self-improvement round, checked at the source level.

Nothing here runs the script: it needs a GPU, a sandbox and a checkpoint. What it pins
are the four claims that make the round's data trustworthy, each of which is a quiet
failure if it regresses -- the round would still finish and still write a parquet.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT  # noqa: E402

SOURCE = (ROOT / "scripts" / "21_rsi_round.sh").read_text(encoding="utf-8")


def test_the_rollout_asks_for_exportable_trajectories():
    # Without it Harbor stores its own rendering of each turn and 03h refuses the lot
    # (BUG-19). The round would burn a full rollout and harvest nothing.
    assert "--export-trajectories" in SOURCE
    assert "06_eval.py" in SOURCE


def test_sampling_is_not_greedy():
    # Rejection sampling needs independent samples: at temperature 0 every one of
    # --runs samples is the same trajectory, and the round harvests one row per task
    # however many times it rolls out.
    assert re.search(r'TEMPERATURE="\$\{TEMPERATURE:-([\d.]+)\}"', SOURCE), SOURCE
    default = float(re.search(r'TEMPERATURE="\$\{TEMPERATURE:-([\d.]+)\}"', SOURCE).group(1))
    assert default > 0, "a greedy default makes --runs > 1 meaningless"
    assert "--temperature" in SOURCE, "the value must reach harbor, not just the shell"


def test_the_serving_template_override_is_honoured():
    # Qwen3.5-0.8B defaults thinking OFF, so serving it with its own template puts the
    # generation prompt one block ahead of what training produced. 20_run_all.sh handles
    # this for eval; a rollout that skipped it would harvest data from a mis-served
    # policy, and nothing downstream could tell.
    assert "SERVE_CHAT_TEMPLATE_REPO" in SOURCE
    assert "--chat-template" in SOURCE
    assert "SERVE_TEMPLATE_ARG" in SOURCE


def test_the_seed_corpus_is_mixed_back_in_by_default():
    # Training round N+1 on round N's successes alone narrows the policy onto its own
    # output. The default must not be "new data only".
    match = re.search(r'MIX_RATIO="\$\{MIX_RATIO:-([\d.]+)\}"', SOURCE)
    assert match, SOURCE
    assert float(match.group(1)) > 0, "the default must keep the seed distribution in view"
    assert "SEED_DATA" in SOURCE and "mix.json" in SOURCE


def test_an_empty_round_stops_instead_of_training_on_nothing():
    assert "exit 3" in SOURCE, "a round that harvested no rows needs its own exit status"
    assert "do NOT train on an empty round" in SOURCE


def test_every_stage_is_a_script_in_this_repo_and_they_are_wired_in_order():
    order = ["06_eval.py", "03h_build_rollout_sft.py", "03g_curate_sft.py",
             "15_export_pretokenized.py"]
    positions = [SOURCE.index(name) for name in order]
    assert positions == sorted(positions), f"stages out of order: {order}"
    for name in order:
        assert (ROOT / "scripts" / name).is_file(), name


def test_the_round_writes_a_summary_naming_the_pass_rate():
    assert "rsi_round.json" in SOURCE
    assert "pass_rate_mean" in SOURCE, (
        "the one number that says whether the round taught anything must be recorded")


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
