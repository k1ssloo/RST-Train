"""verl SFT dataset that consumes PRE-TOKENIZED RST data.

Wire in through verl's custom dataset hook:

    data.custom_cls.path=verl_backend/rst_sft_dataset.py
    data.custom_cls.name=RSTPretokenizedSFTDataset
    data.train_files=$BASE_FOLDER/sft-v1-cap10/pretokenized_train.parquet

WHY NOT verl's BUILT-IN MultiTurnSFTDataset
-------------------------------------------
It tokenizes each message separately and concatenates. For Qwen3.5 that does not
reproduce the whole-conversation render, and the divergence is not cosmetic.
Measured on 200 real rows from our training set:

    identical  : 0
    MISMATCHED : 200  (100%)

Mechanism: the Qwen3.5 chat template injects an empty `<think>\\n\\n</think>\\n\\n`
block before the LAST assistant turn. Building the sequence turn-by-turn makes
every assistant turn "last" at some point, so a 21-turn conversation ends up with
**21 think blocks instead of 1** (+22 tokens in the example we checked). verl's own
config warns about this and offers `ignore_input_ids_mismatch: True`, which
silences the assertion rather than fixing the sequence -- you would then train on
a token sequence that serving never produces.

So we pre-tokenize once with the mask we verified token-by-token against
`slime/utils/mask_utils.py` (see scripts/15_export_pretokenized.py) and let verl
consume the result. `input_ids` and `loss_mask` are already correct and aligned;
this class only batches them.

Expected columns: `input_ids` (list[int]), `loss_mask` (list[int]).
Returned dict matches verl's MultiTurnSFTDataset:
  pad_mode="no_padding"  -> {input_ids, position_ids, loss_mask}
  otherwise              -> {input_ids, attention_mask, position_ids, loss_mask}
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Pure-python core, kept free of torch so it can be unit-tested anywhere.
# --------------------------------------------------------------------------


def build_row(
    input_ids: list[int],
    loss_mask: list[int],
    *,
    max_length: int,
    pad_mode: str,
    pad_token_id: int,
    truncation: str = "error",
) -> dict[str, list[int]]:
    """Pad/truncate one pre-tokenized row into verl's expected field set."""
    if len(input_ids) != len(loss_mask):
        raise ValueError(f"input_ids ({len(input_ids)}) and loss_mask ({len(loss_mask)}) differ")
    if not input_ids:
        raise ValueError("empty input_ids")

    if len(input_ids) > max_length:
        if truncation == "error":
            raise ValueError(
                f"sequence of {len(input_ids)} tokens exceeds max_length={max_length}. "
                f"Raise data.max_length, or re-export with a smaller --max-seq-len so the "
                f"drop is counted in the manifest instead of happening silently here."
            )
        if truncation == "left":
            input_ids, loss_mask = input_ids[-max_length:], loss_mask[-max_length:]
        else:  # "right"
            input_ids, loss_mask = input_ids[:max_length], loss_mask[:max_length]

    length = len(input_ids)
    if pad_mode == "no_padding":
        return {
            "input_ids": list(input_ids),
            "position_ids": list(range(length)),
            "loss_mask": list(loss_mask),
        }

    pad = max_length - length
    return {
        "input_ids": list(input_ids) + [pad_token_id] * pad,
        "attention_mask": [1] * length + [0] * pad,
        # Padded positions get 0; attention_mask/loss_mask keep them out of the loss.
        "position_ids": list(range(length)) + [0] * pad,
        "loss_mask": list(loss_mask) + [0] * pad,
    }


# --------------------------------------------------------------------------
# verl-facing Dataset
# --------------------------------------------------------------------------

try:  # torch is present in any real verl environment; absent in doc/test contexts
    import torch
    from torch.utils.data import Dataset

    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    _HAVE_TORCH = False

    class Dataset:  # type: ignore[no-redef]
        pass


class RSTPretokenizedSFTDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config=None, processor=None, **kwargs: Any):
        if not _HAVE_TORCH:  # pragma: no cover
            raise RuntimeError("torch is required to use RSTPretokenizedSFTDataset")
        import pandas as pd

        cfg = config or {}
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        frames = [pd.read_parquet(p) for p in parquet_files]
        self.frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        for column in ("input_ids", "loss_mask"):
            if column not in self.frame.columns:
                raise ValueError(
                    f"column {column!r} missing from {parquet_files}. This dataset expects "
                    f"PRE-TOKENIZED data from scripts/15_export_pretokenized.py, not a "
                    f"`messages` parquet."
                )

        self.max_length = int(get("max_length", 32768) or 32768)
        self.pad_mode = str(get("pad_mode", "no_padding") or "no_padding")
        self.truncation = str(get("truncation", "error") or "error")
        self.pad_token_id = (
            tokenizer.pad_token_id
            if getattr(tokenizer, "pad_token_id", None) is not None
            else getattr(tokenizer, "eos_token_id", 0)
        )

        limit = int(get("train_max_samples", -1) or -1)
        if limit and limit > 0:
            self.frame = self.frame.head(limit)

        # Validate the whole table HERE, at configuration time. build_row() raises the
        # same two errors, but it runs inside a DataLoader worker partway through an
        # epoch: the job dies after minutes or hours of GPU time, with a stack trace
        # from a subprocess, and (for `truncation="error"`) for a reason that was
        # already knowable before the first forward pass. 30_run_sft_verl.sh checks
        # lengths before launching; this makes the dataset itself do it too, so a
        # hand-run launch or a different launcher cannot skip the check.
        id_lengths = self.frame["input_ids"].map(len)
        mask_lengths = self.frame["loss_mask"].map(len)
        mismatched = int((id_lengths != mask_lengths).sum())
        if mismatched:
            raise ValueError(
                f"{mismatched} of {len(self.frame)} rows have input_ids and loss_mask of "
                f"different lengths. The mask does not describe this sequence; re-export "
                f"with scripts/15_export_pretokenized.py."
            )
        if len(self.frame) == 0:
            raise ValueError("the dataset is empty after filtering; nothing to train on")
        over = int((id_lengths > self.max_length).sum())
        if over and self.truncation == "error":
            raise ValueError(
                f"{over} of {len(self.frame)} rows are longer than max_length="
                f"{self.max_length} (longest {int(id_lengths.max()):,} tokens), and "
                f"truncation='error'. Raise data.max_length, re-export with a smaller "
                f"--max-seq-len so the drop is counted in the manifest, or set "
                f"data.truncation=left|right deliberately -- truncating a trajectory "
                f"changes what is being trained on."
            )
        if over:
            print(f"[RSTPretokenizedSFTDataset] WARNING: {over} row(s) exceed "
                  f"max_length={self.max_length} and will be truncated "
                  f"({self.truncation}); trained-token counts below are pre-truncation.")

        trained = int(self.frame["loss_mask"].map(sum).sum())
        total = int(self.frame["input_ids"].map(len).sum())
        print(
            f"[RSTPretokenizedSFTDataset] rows={len(self.frame)} tokens={total:,} "
            f"trained={trained:,} ({trained / max(1, total):.2%}) "
            f"pad_mode={self.pad_mode} max_length={self.max_length}"
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        built = build_row(
            list(row["input_ids"]),
            list(row["loss_mask"]),
            max_length=self.max_length,
            pad_mode=self.pad_mode,
            pad_token_id=self.pad_token_id,
            truncation=self.truncation,
        )
        return {key: torch.tensor(value, dtype=torch.long) for key, value in built.items()}
