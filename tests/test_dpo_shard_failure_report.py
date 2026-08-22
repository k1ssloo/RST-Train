"""The message `33_run_dpo.sh` prints when a reference shard exits non-zero.

The 27B run failed at the DPO stage with "a reference shard failed; see
$BASE_FOLDER/logs/dpo_ref_shard*.log" -- and all sixteen of those logs ended in a
complete `[done] N rows -> ref_logps_shard<n>.parquet` with a determinism probe of
zero. The old wait loop, `for pid in "${pids[@]}"; do wait "$pid" || rc=1; done`,
discarded both which shard failed and what status it returned, so the only evidence
left pointed at files that all look successful. There was nowhere to go from there.

These tests run the real bash helper (`rst_explain_shard_failures` in
`scripts/lib_env.sh`) against hand-written logs, and pin the three cases an operator
has to be able to tell apart:

  * a log that ENDS in [done] with a non-zero status -> the work is on disk and the
    failure came after the last flush; re-running is nearly free
  * an EMPTY log -> python never printed, so it died before it started
  * anything else -> show the tail, which is the only place the cause can be

No GPU, no python interpreter and no shard process is involved: the helper reads
logs and prints prose, and that prose is the whole deliverable.
"""

from __future__ import annotations

import subprocess
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "scripts" / "lib_env.sh"
LAUNCHER = ROOT / "scripts" / "33_run_dpo.sh"


def _explain(logs: dict[str, str], failed: list[str]) -> str:
    """Write `logs` into a temp dir, run the helper on `failed`, return its stderr."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in logs.items():
            (Path(tmp) / name).write_text(text, encoding="utf-8")
        script = textwrap.dedent(
            f"""
            set -uo pipefail
            source {LIB}
            rst_explain_shard_failures {tmp}/dpo_ref_shard {" ".join(failed)}
            """
        )
        done = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=120
        )
    # Everything here is diagnostics, so it belongs on stderr and nowhere else.
    assert done.stdout == "", done.stdout
    return done.stderr


def test_a_done_log_with_a_nonzero_status_says_the_parquet_is_already_on_disk():
    out = _explain(
        {"dpo_ref_shard7.log": "[plan] 153 rows to score\n"
                               "[done] 153 rows -> /r/ref/ref_logps_shard7.parquet\n"},
        ["7=1"],
    )
    assert "shard 7 exited 1" in out, out
    assert "ENDS IN [done]" in out, out
    assert "do not rebuild the reference pass" in out, out


def test_an_empty_log_is_reported_as_python_never_having_printed():
    out = _explain({"dpo_ref_shard3.log": ""}, ["3=127"])
    assert "shard 3 exited 127" in out, out
    assert "EMPTY" in out, out
    # The [done] wording is the opposite diagnosis and must not appear here.
    assert "ENDS IN [done]" not in out, out


def test_an_unfinished_log_shows_its_tail_because_that_is_where_the_cause_is():
    out = _explain(
        {"dpo_ref_shard0.log": "[plan] 153 rows to score\n"
                               "torch.OutOfMemoryError: CUDA out of memory\n"},
        ["0=1"],
    )
    assert "does not reach [done]" in out, out
    assert "CUDA out of memory" in out, out


def test_every_failed_shard_is_named_not_just_the_first():
    out = _explain(
        {"dpo_ref_shard1.log": "[done] 10 rows -> a.parquet\n",
         "dpo_ref_shard9.log": ""},
        ["1=1", "9=137"],
    )
    assert "shard 1 exited 1" in out, out
    assert "shard 9 exited 137" in out, out


def test_a_missing_log_file_is_treated_as_the_empty_case_and_does_not_crash():
    out = _explain({}, ["5=1"])
    assert "shard 5 exited 1" in out, out
    assert "EMPTY" in out, out


def test_the_launcher_no_longer_collapses_sixteen_shards_into_one_bit():
    src = LAUNCHER.read_text(encoding="utf-8")
    assert 'for pid in "${pids[@]}"; do wait "$pid" || rc=1; done' not in src, (
        "the wait loop is back to discarding which shard failed and with what status"
    )
    # The shard id has to be recorded next to the pid, or the status cannot be
    # attributed to a shard no matter how it is captured.
    assert "shard_ids=()" in src, src
    assert 'shard_ids+=("$shard")' in src, src
    assert 'rst_explain_shard_failures "$BASE_FOLDER/logs/dpo_ref_shard"' in src, src
    # `wait || code=$?` is the only way to keep the status; `|| code=1` would lose
    # the difference between 137 (signalled) and 1 (python raised).
    assert 'wait "${pids[$i]}" || code=$?' in src, src
