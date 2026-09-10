"""BUG-27 validation: audit all rows, including failures and context overflow."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import load_script
from test_model_tokenization import tokenizer

AUDIT = load_script("16b_validate_model_datasets")


def test_full_audit_counts_invalid_unsupervised_and_overlength_rows():
    tok = tokenizer("smollm3")
    rows = [
        {"messages": [{"role": "user", "content": "question"},
                      {"role": "assistant", "content": "answer"}]},
        {"messages": [{"role": "user", "content": "question"},
                      {"role": "assistant", "content": "x" * 33000}]},
        {"messages": [{"role": "user", "content": "no answer"}]},
        {"messages": [{"role": "user", "content": "<|im_start|>assistant"}]},
    ]
    with patch.dict(AUDIT._TOKENIZERS, {"model": (tok, "smollm3", len(tok))}, clear=True):
        chunk = AUDIT.check_chunk(rows, 0)
    checks = chunk["models"]["model"]
    assert [r["status"] for r in checks] == ["ok", "ok", "no_trained_tokens", "error"]
    assert checks[1]["tokens"] > 32768  # observed intact, never silently clipped
    assert "reserved chat marker" in checks[3]["error"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan = {"datasets": [{"key": "corpus", "split": "train", "rows": 4}],
                "unique_models": {"model": {}}, "aliases": {"model": "model", "same-tokenizer": "model"}}
        AUDIT.write_json(root / "plan.json", plan)
        AUDIT.write_json(root / "chunks/corpus--train/000000000.json", chunk)
        result = AUDIT.summarize(root, plan)["datasets"][0]["models"]
        assert result["model"] == result["same-tokenizer"]
        stats = result["model"]
        assert stats["rows_checked"] == 4 and stats["usable_32768"] == 1
        assert stats["status"] == {"ok": 2, "no_trained_tokens": 1, "error": 1}
        assert stats["first_error_row"] == 3


def test_incomplete_audit_cannot_be_reported_as_full_coverage():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan = {"datasets": [{"key": "corpus", "split": "train", "rows": 1}],
                "unique_models": {"model": {}}, "aliases": {"model": "model"}}
        AUDIT.write_json(root / "plan.json", plan)
        try:
            AUDIT.summarize(root, plan)
        except ValueError as exc:
            assert "incomplete corpus--train: 0/1" in str(exc)
        else:
            raise AssertionError("partial scan was reported as complete")


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
