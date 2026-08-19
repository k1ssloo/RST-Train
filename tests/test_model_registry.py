"""`scripts/model_registry.py` — the one place a launch shape is decided.

`configs/models.json` describes a Megatron 3-D layout. verl's FSDP engine has no
pipeline stages and no context parallelism (`verl/trainer/config/engine/fsdp.yaml`
in 0.9.0 has neither key, nor a tensor_parallel_size), so passing a Megatron row
through unchanged produces two wrong answers at once:

  * `tp*pp*cp` becomes a divisibility rule the verl path does not obey — the 27B
    80GB row is tp4/pp2/cp2, so any GPU count that is not a multiple of 16 is
    rejected for a reason that does not exist on this backend;
  * `max_tokens_per_gpu` is per-GPU-per-microbatch, so a cp=2 row budgets half a
    sequence. Under FSDP one GPU holds the whole sequence, and verl then rejects
    (`rearrange_micro_batches` asserts max_token_len >= max_seq_len) or silently
    splits every sequence over the budget.

These tests pin the reshaping and the failure messages, because both are consumed
by a shell `eval` on a cluster where a wrong number costs a 40-minute run.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, load_script  # noqa: E402

reg = load_script("model_registry")


def resolve(**kw):
    args = dict(key="qwen3.5-27b", mem_class="80GB", gpus=32, gpus_per_node=8,
                max_seq_len=32768, phase="sft", backend="megatron")
    args.update(kw)
    err = io.StringIO()
    with redirect_stderr(err):
        out = reg.resolve(**args)
    out["_stderr"] = err.getvalue()
    return out


def test_the_megatron_row_for_the_27b_is_still_three_dimensional():
    # If this ever becomes tp1/pp1/cp1 the reshaping below is a no-op and the tests
    # that follow would pass vacuously.
    m = reg.load()["models"]["qwen3.5-27b"]["parallelism"]["80GB"]
    assert (m["tp"], m["pp"], m["cp"]) != (1, 1, 1), (
        "the 27B 80GB row no longer has a Megatron shape, so these tests no longer "
        "prove that --backend verl reshapes anything"
    )


def test_verl_pins_the_megatron_dimensions_to_one():
    out = resolve(backend="verl")
    assert (out["TP"], out["PP"], out["CP"]) == (1, 1, 1), (
        f"verl got TP={out['TP']} PP={out['PP']} CP={out['CP']}; none of the three "
        f"exists on the FSDP path, and a non-1 value is a divisibility rule the "
        f"launcher cannot satisfy"
    )


def test_verl_folds_the_rows_cp_into_the_token_budget():
    row = reg.load()["models"]["qwen3.5-27b"]["parallelism"]["80GB"]
    out = resolve(backend="verl")
    assert out["MAX_TOKENS_PER_GPU"] == row["max_tokens_per_gpu"] * row["cp"], (
        f"MAX_TOKENS_PER_GPU={out['MAX_TOKENS_PER_GPU']} but the row is "
        f"{row['max_tokens_per_gpu']}*cp{row['cp']}. Under FSDP one GPU holds the "
        f"whole sequence, so the row's per-CP-slice budget is not the per-GPU budget"
    )
    assert out["MAX_TOKENS_PER_GPU"] >= out["MAX_SEQ_LEN"], (
        "the folded budget is still below max_seq_len, so verl's "
        "rearrange_micro_batches assert will fire on the first long sample"
    )


def test_the_reshaping_is_announced_not_silent():
    out = resolve(backend="verl")
    note = out["_shape_note"]
    assert note, "the reshaping happens with no note; the operator cannot tell the " \
                 "printed numbers differ from configs/models.json"
    assert "FSDP" in note and "max_tokens_per_gpu" in note


def test_megatron_keeps_its_own_shape():
    out = resolve(backend="megatron")
    row = reg.load()["models"]["qwen3.5-27b"]["parallelism"]["80GB"]
    assert (out["TP"], out["PP"], out["CP"]) == (row["tp"], row["pp"], row["cp"])
    assert out["MAX_TOKENS_PER_GPU"] == row["max_tokens_per_gpu"]
    assert not out["_shape_note"], "megatron should not be reshaped at all"


def test_slime_is_shaped_like_megatron_not_like_verl():
    # 20_run_all.sh passes BACKEND=verl|slime straight through; slime *is* Megatron.
    a = reg.resolve(key="qwen3.5-27b", mem_class="80GB", gpus=32, gpus_per_node=8,
                    max_seq_len=32768, backend="megatron")
    b = resolve(backend="megatron")
    assert (a["TP"], a["PP"], a["CP"]) == (b["TP"], b["PP"], b["CP"])
    src = (ROOT / "scripts" / "model_registry.py").read_text(encoding="utf-8")
    assert 'backend="megatron" if args.backend == "slime" else args.backend' in src, (
        "--backend slime no longer maps to the megatron shape; it would then fall "
        "through to the verl reshaping and pin TP=1 for a Megatron job"
    )


def test_dp_is_derived_and_the_product_is_the_world():
    for backend in ("megatron", "verl"):
        out = resolve(backend=backend)
        product = out["TP"] * out["PP"] * out["CP"] * out["DP"]
        assert product == out["TOTAL_GPUS"], (
            f"{backend}: tp*pp*cp*dp={product} != {out['TOTAL_GPUS']} GPUs"
        )


def test_a_world_too_small_for_the_static_footprint_is_warned_about():
    """8 GPUs passes every shape assert and then OOMs.

    27.78B params x 16 B/param (fp32 master + fp32 grad + Adam m + v) / 8 ranks is
    ~52 GiB per card before a single activation. The arithmetic in resolve() cannot
    see memory, so min_gpus is the only thing that can say so.
    """
    out = resolve(backend="verl", gpus=8, max_seq_len=8192)
    assert "min_gpus" in out["_unvalidated"], (
        "resolving the 27B onto 8 GPUs produced no warning; that is the exact "
        "configuration that OOMs in backward with a peak that ignores the token budget"
    )


def test_an_impossible_shape_exits_rather_than_returning_a_number():
    # tp*pp*cp=16 for the 27B Megatron row, so 24 GPUs cannot be divided.
    try:
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            reg.resolve(key="qwen3.5-27b", mem_class="80GB", gpus=24, gpus_per_node=8,
                        max_seq_len=32768, backend="megatron")
    except SystemExit as exc:
        assert "does not divide" in str(exc)
    else:
        raise AssertionError("an undividable GPU count was accepted")


def test_the_same_count_is_fine_once_verl_has_reshaped_it():
    out = resolve(backend="verl", gpus=24)
    assert out["DP"] == 24, "FSDP has no divisibility constraint left to violate"


def test_the_shell_output_is_evalable_and_quotes_its_values():
    out = resolve(backend="verl")
    buf = io.StringIO()
    with redirect_stdout(buf):
        reg.emit_shell(out) if hasattr(reg, "emit_shell") else None
    text = buf.getvalue()
    if not text:  # emitted inline in main(); assert on the mechanism instead
        src = (ROOT / "scripts" / "model_registry.py").read_text(encoding="utf-8")
        assert "shlex.quote" in src, (
            "the --shell output is built without shlex.quote; a value containing a "
            "space would be eval'd as two words by every launcher"
        )
        return
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        assert "=" in line


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
