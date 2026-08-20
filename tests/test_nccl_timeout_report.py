"""What `33_run_dpo.sh` says when the trainer dies in an NCCL watchdog timeout.

The 4B DPO attempt on 2026-08-20T00:11 ended with six ranks stuck on `SeqNum=1,
OpType=BROADCAST` with a DIFFERENT `NumelIn` each (9687 / 14042 / 3328 / 5678 / 8365 /
6697) while ranks 0 and 2 were already at `SeqNum=2, OpType=ALLREDUCE` — then SIGABRT
everywhere, `ChildFailedError`, and a launcher message offering GATE 1 / GATE 2 / GATE 3
as explanations. None of the three was the cause, and the detail that IS the cause was
buried under ~500 lines of C++ watchdog frames.

The sizes are the whole diagnosis, and they read two ways:

  * sizes (or sequence numbers) differ per rank -> the ranks are not running the same
    sequence of collectives, because the size depends on each rank's own data. A code
    divergence; re-running reproduces it.
  * sizes identical -> the collective was well formed and one rank never arrived. Host
    OOM killer first (every rank builds the model on CPU before FSDP shards it), then a
    rank that raised on its own, then a short rendezvous.

`rst_explain_nccl_timeout` is run here as real bash against real log text, so the two
readings are pinned apart and a log with no timeout in it stays silent.
"""

from __future__ import annotations

import subprocess
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "scripts" / "lib_env.sh"
LAUNCHER = ROOT / "scripts" / "33_run_dpo.sh"


def _watchdog(rank: int, seq: int, op: str, numel: int) -> str:
    return (f"[rank{rank}]:[E820 00:23:06.4558022{rank:02d} ProcessGroupNCCL.cpp:757] "
            f"[Rank {rank}] Watchdog caught collective operation timeout: "
            f"WorkNCCL(SeqNum={seq}, OpType={op}, NumelIn={numel}, NumelOut={numel}, "
            f"Timeout(ms)=600000) ran for 600027 milliseconds before timing out.\n")


def _explain(log_text: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "dpo_train.log"
        log.write_text(log_text, encoding="utf-8")
        script = textwrap.dedent(
            f"""
            set -uo pipefail
            source {LIB}
            rst_explain_nccl_timeout {log}
            """
        )
        done = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=120
        )
    assert done.stdout == "", done.stdout
    assert done.returncode == 0, done.stderr
    return done.stderr


# The real 4B failure: one collective size per rank.
DIVERGED = "".join(
    [_watchdog(1, 1, "BROADCAST", 9687), _watchdog(7, 1, "BROADCAST", 6697),
     _watchdog(6, 1, "BROADCAST", 5678), _watchdog(3, 1, "BROADCAST", 14042),
     _watchdog(4, 1, "BROADCAST", 3328), _watchdog(5, 1, "BROADCAST", 8365),
     _watchdog(0, 2, "ALLREDUCE", 15047680), _watchdog(2, 2, "ALLREDUCE", 7767040)]
)

# The other reading: same collective everywhere, one rank simply absent.
ONE_MISSING = "".join(_watchdog(r, 4, "ALLREDUCE", 15047680) for r in (0, 1, 2, 3, 4, 5, 6))


def test_a_log_without_a_timeout_says_nothing_at_all():
    quiet = "[step 12/76] loss 0.6931 margin +0.0000\nDone.\n"
    assert _explain(quiet) == ""


def test_per_rank_sizes_are_read_as_a_code_divergence_not_a_memory_problem():
    out = _explain(DIVERGED)
    assert "NCCL WATCHDOG TIMEOUT" in out, out
    assert "none of the three gates above is the cause" in out, out
    # Every rank and its own size, so the reader can see the pattern rather than be told.
    assert "rank 3 waiting on BROADCAST (SeqNum=1) of 14042 elements" in out, out
    assert "rank 0 waiting on ALLREDUCE (SeqNum=2) of 15047680 elements" in out, out
    assert "DIFFER between ranks" in out, out
    assert "only a code change fixes it" in out, out
    # The opposite diagnosis must not also be offered: two hypotheses is none.
    assert "OOM killer" not in out, out


def test_identical_sizes_are_read_as_a_rank_that_never_arrived():
    out = _explain(ONE_MISSING)
    assert "SAME collective" in out, out
    assert "OOM killer" in out, out
    assert "params x bytes x local_ranks" in out, out
    assert "DIFFER between ranks" not in out, out


def test_a_rank_that_raised_first_is_surfaced_above_the_watchdog_noise():
    first = ("[rank5]: RuntimeError: aten.embedding.default got mixed torch.Tensor and "
             "DTensor, need to convert all torch.Tensor to DTensor before calling "
             "distributed operators!\n")
    out = _explain(first + ONE_MISSING)
    assert "raised before the timeout, which is the first failure" in out, out
    assert "aten.embedding.default got mixed" in out, out


def test_the_re_run_cost_is_stated_because_19_is_not_resumable():
    out = _explain(ONE_MISSING)
    assert "NOT resumable" in out, out
    assert "reference parquets from 18 are kept" in out, out


def test_the_launcher_captures_the_trainer_output_and_classifies_a_failure():
    src = LAUNCHER.read_text(encoding="utf-8")
    # Without the capture there is nothing to classify: the watchdog lines only ever
    # existed on the console.
    assert 'TRAIN_LOG="$BASE_FOLDER/logs/dpo_train.log"' in src, src
    assert 'tee "$TRAIN_LOG"' in src, src
    assert 'rst_explain_nccl_timeout "$TRAIN_LOG"' in src, src
    # `RC=$?` has to read the pipeline, and pipefail is what makes it torchrun's status
    # rather than tee's.
    assert "set -uo pipefail" in src, src
    train = src.index("torchrun \\")
    assert src.index('rst_explain_nccl_timeout "$TRAIN_LOG"') > train, (
        "the classifier must run after the trainer, on what the trainer wrote"
    )
