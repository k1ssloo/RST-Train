"""`verl_backend.rst_sft_dataset.build_row` is where a good mask can still be ruined.

It pads and truncates. Both operations can quietly destroy the training target:
padding that is not excluded from the loss teaches the model to emit pad tokens,
and truncation that keeps the prompt end but drops the answer produces rows with
zero trained tokens that still contribute a (meaningless) loss. The default is
therefore `truncation="error"` -- a row that does not fit is a data-pipeline bug to
be counted in the manifest, not something to silently shorten here.

Torch-free: `build_row` is deliberately kept out of the torch-dependent half of
that module so it can be tested anywhere.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import load_repo_module, need  # noqa: E402

MODULE = load_repo_module("verl_backend.rst_sft_dataset")
build_row = MODULE.build_row

IDS = [10, 11, 12, 13, 14, 15]
MASK = [0, 0, 0, 1, 1, 1]


def test_no_padding_mode_returns_the_row_untouched():
    row = build_row(IDS, MASK, max_length=32, pad_mode="no_padding", pad_token_id=0)
    assert row["input_ids"] == IDS
    assert row["loss_mask"] == MASK
    assert row["position_ids"] == list(range(len(IDS)))
    assert "attention_mask" not in row, "no_padding rows must not carry an attention mask"


def test_padding_keeps_pad_tokens_out_of_the_loss():
    row = build_row(IDS, MASK, max_length=9, pad_mode="max_length", pad_token_id=7)
    assert row["input_ids"] == IDS + [7, 7, 7]
    assert row["attention_mask"] == [1] * 6 + [0] * 3
    assert row["loss_mask"] == MASK + [0, 0, 0], "padding became a training target"
    assert len(row["position_ids"]) == 9
    assert all(len(value) == 9 for value in row.values()), "fields have unequal length"


def test_an_oversized_row_is_an_error_by_default():
    try:
        build_row(IDS, MASK, max_length=4, pad_mode="no_padding", pad_token_id=0)
    except ValueError as exc:
        assert "max_length" in str(exc)
        return
    raise AssertionError("a row longer than max_length was silently truncated")


def test_explicit_truncation_sides_do_what_they_say():
    left = build_row(IDS, MASK, max_length=4, pad_mode="no_padding", pad_token_id=0,
                     truncation="left")
    assert left["input_ids"] == IDS[-4:] and left["loss_mask"] == MASK[-4:]
    assert sum(left["loss_mask"]) == 3, "left truncation should keep the answer"

    right = build_row(IDS, MASK, max_length=4, pad_mode="no_padding", pad_token_id=0,
                      truncation="right")
    assert right["input_ids"] == IDS[:4] and right["loss_mask"] == MASK[:4]


def test_mismatched_lengths_and_empty_rows_are_rejected():
    for ids, mask, why in (
        ([1, 2, 3], [1, 1], "length mismatch"),
        ([], [], "empty row"),
    ):
        try:
            build_row(ids, mask, max_length=8, pad_mode="no_padding", pad_token_id=0)
        except ValueError:
            continue
        raise AssertionError(f"{why} was accepted")


def test_inputs_are_copied_not_aliased():
    ids, mask = list(IDS), list(MASK)
    row = build_row(ids, mask, max_length=6, pad_mode="no_padding", pad_token_id=0)
    row["input_ids"].append(999)
    row["loss_mask"].append(1)
    assert ids == IDS and mask == MASK, "build_row mutated its caller's lists"


# --------------------------------------------------------------------------
# The same two errors, raised at CONSTRUCTION time instead of inside a DataLoader
# worker mid-epoch. These need torch (the Dataset half of the module refuses to
# construct without it) and pandas, so they skip on a torch-free environment.
# --------------------------------------------------------------------------


class _Tokenizer:
    pad_token_id = 7
    eos_token_id = 7


def _parquet(directory: str, rows: list[tuple[list[int], list[int]]]) -> str:
    pandas = need("pandas")
    frame = pandas.DataFrame(
        {"input_ids": [ids for ids, _ in rows], "loss_mask": [mask for _, mask in rows]}
    )
    path = Path(directory) / "pretokenized.parquet"
    frame.to_parquet(path)
    return str(path)


def _build(rows, **config):
    need("torch")
    with tempfile.TemporaryDirectory() as directory:
        return MODULE.RSTPretokenizedSFTDataset(
            _parquet(directory, rows), _Tokenizer(), config or None
        )


def test_construction_rejects_a_mask_that_does_not_line_up():
    try:
        _build([(IDS, MASK), ([1, 2, 3], [1, 1])], max_length=32)
    except ValueError as exc:
        assert "different lengths" in str(exc)
        return
    raise AssertionError("a misaligned mask survived construction and would train wrong")


def test_construction_refuses_oversized_rows_before_any_gpu_time():
    try:
        _build([(IDS, MASK)], max_length=4)
    except ValueError as exc:
        message = str(exc)
        assert "max_length=4" in message and "6 tokens" in message, message
        return
    raise AssertionError("an oversized row was only going to be discovered mid-epoch")


def test_deliberate_truncation_constructs_and_truncates():
    dataset = _build([(IDS, MASK)], max_length=4, truncation="left")
    assert len(dataset) == 1
    row = dataset[0]
    assert row["input_ids"].tolist() == IDS[-4:]
    assert row["loss_mask"].tolist() == MASK[-4:]


def test_train_max_samples_is_applied_before_validation():
    # Otherwise a small debug run (`train_max_samples=1`) would be blocked by a bad row
    # it is never going to load.
    dataset = _build([(IDS, MASK), (IDS * 3, MASK * 3)], max_length=8, train_max_samples=1)
    assert len(dataset) == 1


def test_an_empty_table_is_refused():
    try:
        _build([], max_length=8)
    except ValueError as exc:
        assert "empty" in str(exc)
        return
    raise AssertionError("an empty dataset was accepted; the run would train on nothing")


def test_a_supervised_first_token_is_refused_because_packing_rolls_cyclically():
    # verl's sft_loss aligns the mask to the log-probs with
    # `torch.roll(loss_mask_flatten, shifts=-1, dims=0)` over the WHOLE packed
    # micro-batch, not per sample. So mask[0]==1 on any row means the preceding
    # document's last hidden state is trained to predict this document's first token,
    # and the batch's final position is trained on the batch's first. Nothing in the
    # loss curve shows it, which is why it has to be refused rather than warned about.
    try:
        _build([(IDS, [1, 0, 0, 1, 1, 1])], max_length=32)
    except ValueError as exc:
        message = str(exc)
        assert "loss_mask[0]" in message, message
        assert "cyclically" in message or "rolls" in message, message
        return
    raise AssertionError("a supervised first token was accepted; under pad_mode=no_padding "
                         "that leaks supervision across a packed document boundary")


def test_the_ordinary_mask_from_the_exporter_still_constructs():
    # Guard against the check above being written so tightly that real data trips it:
    # every row 15_export_pretokenized.py emits opens on <|im_start|> with mask 0.
    dataset = _build([(IDS, MASK), (IDS, [0, 0, 1, 1, 1, 0])], max_length=32)
    assert len(dataset) == 2


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
