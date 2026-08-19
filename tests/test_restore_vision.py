"""`scripts/07_restore_vision.py` on tiny synthetic checkpoints.

The script splices a trained text stack into the original multimodal checkpoint. Its
dangerous failure mode is silence: if the trained export is missing text tensors, the
base model's values get shipped in their place and the result loads, runs and
evaluates as if it were trained. So the checks here are mostly about *refusal*, and
about the streaming path (one output shard resident at a time) actually working --
`--shard-size-gb` is set absurdly small so the tiny fixture still splits.

Needs torch + safetensors; skips cleanly without them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, need  # noqa: E402

torch = need("torch")
need("safetensors")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

SCRIPT = ROOT / "scripts" / "07_restore_vision.py"
# small enough that ~64-byte tensors land in different shards
TINY_SHARD_GB = "0.00000006"


def build(root: Path, tensors: dict, shards: int = 2, config: dict | None = None) -> None:
    """Write a minimal HF-style safetensors checkpoint (index + config + tokenizer)."""
    root.mkdir(parents=True, exist_ok=True)
    keys = sorted(tensors)
    groups = [g for g in ([keys[i::shards] for i in range(shards)] if shards > 1 else [keys]) if g]
    weight_map = {}
    for position, group in enumerate(groups, 1):
        name = f"model-{position:05d}-of-{len(groups):05d}.safetensors"
        save_file({k: tensors[k] for k in group}, str(root / name), metadata={"format": "pt"})
        for key in group:
            weight_map[key] = name
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
    (root / "config.json").write_text(json.dumps({"model_type": "fake", **(config or {})}))
    (root / "tokenizer_config.json").write_text("{}")


def read_all(root: Path) -> dict:
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(root / shard, framework="pt") as handle:
            for key in handle.keys():
                out[key] = handle.get_tensor(key)
    return out


def original_tensors() -> dict:
    return {
        "model.language_model.a": torch.zeros(4, 8, dtype=torch.bfloat16),
        "model.language_model.b": torch.zeros(4, 8, dtype=torch.bfloat16),
        "model.language_model.c": torch.zeros(2, 2, dtype=torch.float32),  # dtype-cast path
        "model.visual.v": torch.full((3, 3), 7.0, dtype=torch.bfloat16),
        "mtp.h": torch.full((2, 2), 9.0, dtype=torch.bfloat16),
    }


def trained_tensors() -> dict:
    return {
        "model.language_model.a": torch.ones(4, 8, dtype=torch.bfloat16),
        "model.language_model.b": torch.ones(4, 8, dtype=torch.bfloat16) * 2,
        "model.language_model.c": torch.ones(2, 2, dtype=torch.bfloat16) * 3,  # bf16 vs fp32
        "some.optimizer.leftover": torch.ones(2, dtype=torch.float32),
    }


class Case:
    """One temp directory holding `original/`, a trained checkpoint and outputs."""

    def __init__(self, original_config: dict | None = None) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        build(self.root / "original", original_tensors(), shards=2, config=original_config)

    def run(self, *extra: str, trained: str = "trained", out: str = "out"):
        return subprocess.run(
            [sys.executable, str(SCRIPT),
             "--trained", str(self.root / trained), "--original", str(self.root / "original"),
             "--out", str(self.root / out), "--shard-size-gb", TINY_SHARD_GB, *extra],
            capture_output=True, text=True, check=False)

    def close(self) -> None:
        self._tmp.cleanup()


def test_happy_path_takes_trained_text_and_original_vision():
    case = Case()
    try:
        build(case.root / "trained", trained_tensors(), shards=2)
        done = case.run()
        assert done.returncode == 0, done.stdout + done.stderr
        merged = read_all(case.root / "out")
        assert set(merged) == set(original_tensors()), "the key set changed"
        assert merged["model.language_model.a"].eq(1).all()
        assert merged["model.language_model.b"].eq(2).all()
        assert merged["model.language_model.c"].eq(3).all()
        assert merged["model.language_model.c"].dtype is torch.float32, \
            "trained tensor was not cast back to the original dtype"
        assert merged["model.visual.v"].eq(7).all(), "vision weights were not preserved"
        assert merged["mtp.h"].eq(9).all(), "mtp head was not preserved"

        manifest = json.loads((case.root / "out/restore_vision_manifest.json").read_text())
        assert manifest["tensors_from_trained"] == 3
        assert manifest["tensors_copy_always"] == 2
        assert manifest["tensors_unexpected_fallback"] == 0
        assert manifest["trained_only_keys_dropped"] == ["some.optimizer.leftover"]
        assert manifest["shards"] > 1, "shard-size-gb did not split; streaming path untested"
        assert (case.root / "out/config.json").is_file()
        assert (case.root / "out/tokenizer_config.json").is_file()
    finally:
        case.close()


def test_a_missing_text_tensor_is_refused_and_nothing_is_written():
    case = Case()
    try:
        partial = {k: v for k, v in trained_tensors().items() if k != "model.language_model.b"}
        build(case.root / "trained_missing", partial, shards=1)
        done = case.run(trained="trained_missing", out="out_missing")
        assert done.returncode != 0, "a partly-untrained checkpoint was accepted"
        blob = done.stdout + done.stderr
        assert "REFUSING TO WRITE" in blob and "model.language_model.b" in blob
        assert not (case.root / "out_missing/model.safetensors.index.json").exists(), \
            "a refused run still left a loadable checkpoint behind"
    finally:
        case.close()


def test_the_escape_hatch_writes_but_records_every_affected_key():
    case = Case()
    try:
        partial = {k: v for k, v in trained_tensors().items() if k != "model.language_model.b"}
        build(case.root / "trained_missing", partial, shards=1)
        done = case.run("--allow-original-fallback", trained="trained_missing", out="out_forced")
        assert done.returncode == 0, done.stdout + done.stderr
        manifest = json.loads((case.root / "out_forced/restore_vision_manifest.json").read_text())
        assert manifest["unexpected_fallback_keys"] == ["model.language_model.b"]
        assert manifest["allow_original_fallback"] is True
        assert read_all(case.root / "out_forced")["model.language_model.b"].eq(0).all(), \
            "the fallback did not actually take the original value"
        assert "partly untrained" in done.stdout, "the warning has to be printed, not just filed"
    finally:
        case.close()


def test_a_shape_mismatch_is_refused():
    case = Case()
    try:
        bad = trained_tensors()
        bad["model.language_model.b"] = torch.ones(4, 4, dtype=torch.bfloat16)
        build(case.root / "trained_shape", bad, shards=1)
        done = case.run(trained="trained_shape", out="out_shape")
        assert done.returncode != 0
        assert "shape" in (done.stdout + done.stderr).lower()
    finally:
        case.close()


# --------------------------------------------------------------------------
# WHY THE mtp.* COPY IS UNCONDITIONAL, AND MUST STAY THAT WAY.
#
# Reviewing the 4B SFT checkpoint it looks wrong. Measured:
#
#   Qwen/Qwen3.5-4B (base)            mtp_num_hidden_layers: 1  + 15 mtp.* tensors
#   its verl SFT ckpt huggingface/    mtp_num_hidden_layers: 0  +  0 mtp.* tensors
#     (the training log's "Loading weights: .../723" is 738 - 15: verl never read them)
#
# So it reads as "the script copies a head the trained config says does not exist". It is
# not, because the non-weight files -- config.json included -- are copied from
# `--original`, NOT from `--trained`. The shipped config therefore declares 1 MTP layer
# and 15 mtp.* tensors are written to match it. Consistent.
#
# Making the copy conditional on the TRAINED config, which is the obvious "fix", drops
# those 15 tensors while still shipping the base config that asks for them -- and
# transformers then silently RANDOM-INITIALIZES the MTP head. These tests pin both halves
# of the invariant so that edit fails here instead of at serve time.
# --------------------------------------------------------------------------


def test_the_shipped_config_comes_from_the_original_not_the_trained_checkpoint():
    case = Case(original_config={"mtp_num_hidden_layers": 1, "dtype": "bfloat16"})
    try:
        build(case.root / "trained", trained_tensors(), shards=2,
              config={"mtp_num_hidden_layers": 0, "dtype": "float32"})
        done = case.run()
        assert done.returncode == 0, done.stdout + done.stderr
        config = json.loads((case.root / "out/config.json").read_text())
        assert config["mtp_num_hidden_layers"] == 1, (
            "config.json was taken from --trained. verl's exported config drops the MTP "
            "head and says fp32; the base's is the one the restored weight set matches."
        )
        assert config["dtype"] == "bfloat16", (
            "the shipped config says fp32 while the weights were cast to the original's "
            "bf16, so every loader would upcast the model for nothing"
        )
    finally:
        case.close()


def test_the_mtp_head_is_written_even_when_the_trained_export_dropped_it():
    """The 4B case. mtp.* is absent from --trained and present in the shipped config, so
    it must come from --original -- otherwise the head is randomly initialized."""
    case = Case(original_config={"mtp_num_hidden_layers": 1})
    try:
        # trained_tensors() has no mtp.* at all, exactly like the merged verl checkpoint
        build(case.root / "trained", trained_tensors(), shards=2,
              config={"mtp_num_hidden_layers": 0})
        done = case.run()
        assert done.returncode == 0, done.stdout + done.stderr
        merged = read_all(case.root / "out")
        assert "mtp.h" in merged, (
            "the mtp head was dropped while config.json (from --original) still declares "
            "mtp_num_hidden_layers=1; transformers would random-initialize it and no eval "
            "number would look wrong"
        )
        assert merged["mtp.h"].eq(9).all(), "the mtp head is not the original's values"
        assert merged["model.visual.v"].eq(7).all(), "the ViT must be restored too"
        assert merged["model.language_model.a"].eq(1).all(), "text weights must be trained"
        manifest = json.loads((case.root / "out/restore_vision_manifest.json").read_text())
        assert manifest["tensors_copy_always"] == 2, "ViT + mtp head"
    finally:
        case.close()


def test_a_total_naming_mismatch_is_named_as_such():
    """Zero overlap is a naming/conversion bug, not a training failure."""
    case = Case()
    try:
        build(case.root / "trained_alien", {"totally.different": torch.ones(2)}, shards=1)
        done = case.run(trained="trained_alien", out="out_alien")
        assert done.returncode != 0
        assert "not one tensor name" in (done.stdout + done.stderr)
    finally:
        case.close()


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
