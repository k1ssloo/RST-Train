"""The second 27B OOM: an UNSHARDED fp32 gradient retained across micro-batches.

The cluster's first OOM was the static footprint (four nodes that never rendezvoused,
so the shard degree was 8). That was fixed, provably -- `NumelIn=11982408,
NumelOut=383437056` in the NCCL log is exactly 32.0x -- and the run OOMed again anyway,
this time inside `loss.backward()` holding 75.36 GiB of a 79.33 GiB card while asking
for 552 MiB.

That second failure is `verl_backend/fsdp2_grad_accum.py`'s subject: verl calls
`set_requires_gradient_sync(False)` on every non-final micro-batch, FSDP2 answers by
upcasting each parameter's gradient to the fp32 reduce dtype and keeping it UNSHARDED
until the final backward, and 27.78e9 x 4 B = 103.5 GiB does not fit next to anything.

Two properties are worth pinning in a test that needs no GPU:

  * the micro-batch estimator, because the whole failure is gated on ">= 2 micro-batches
    per step" and the arithmetic has a trap in it -- `prepare_micro_batches` scales the
    token budget by `sp_size`, so ULYSSES_SP moves the numerator and the denominator
    together and CANNOT change the count. Someone will eventually try to fix this OOM by
    raising it.
  * that the launcher still runs the gate before torchrun, and that the patch is still
    applied from the module verl loads in every rank rather than from a launcher that a
    cluster-side wrapper can bypass -- which is exactly what happened: the failing run
    used its own launcher, not scripts/30_run_sft_verl.sh.

The allocation and numerics claims are measured instead, by
`scripts/35_probe_fsdp2_grad_accum.py`, which needs a real GPU and real FSDP2.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT  # noqa: E402

BACKEND = ROOT / "verl_backend"
sys.path.insert(0, str(BACKEND))

_spec = importlib.util.spec_from_file_location(
    "rst_fsdp2_grad_accum", BACKEND / "fsdp2_grad_accum.py")
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
sys.modules["rst_fsdp2_grad_accum"] = mod
_spec.loader.exec_module(mod)

LAUNCHER = (ROOT / "scripts" / "30_run_sft_verl.sh").read_text(encoding="utf-8")
DATASET = (BACKEND / "rst_sft_dataset.py").read_text(encoding="utf-8")

# The cluster's real configuration, from the launch line in logs/train_rank1.log.
CLUSTER = {
    "train_batch_size": 128,
    "world_size": 32,
    "ulysses_sp": 8,
    "max_token_len_per_gpu": 32768,
}
# Measured over data/sft-v1-cap10/pretokenized_train.parquet: 10,578 rows, mean 9,448.
CAP10_MEAN = 9448


def lengths(n: int = 1000, mean: int = CAP10_MEAN) -> list[int]:
    return [mean] * n


def test_the_cluster_configuration_really_does_split_into_two_micro_batches():
    est = mod.estimate_micro_batches(sample_lengths=lengths(), **CLUSTER)
    assert est["dp_size"] == 4, "world 32 / sp 8; if this changes the whole estimate does"
    assert est["samples_per_dp"] == 32
    assert est["budget_per_dp_group"] == 32768 * 8
    assert est["typical_micro_batches"] == 2, (
        "the failing run had exactly 2 -- the minimum that reaches the unsharded-grad "
        "path at all, which is why a smaller run of the same config looks fine"
    )


def test_raising_ulysses_sp_cannot_change_the_micro_batch_count():
    """The trap. `prepare_micro_batches` does `max_token_len = max_token_len_per_gpu *
    sp_size`, so tokens-per-group and budget-per-group scale together. Sequence
    parallelism is the fix for activations, and it is not a fix for this."""
    counts = {
        sp: mod.estimate_micro_batches(
            sample_lengths=lengths(),
            train_batch_size=CLUSTER["train_batch_size"],
            world_size=CLUSTER["world_size"],
            ulysses_sp=sp,
            max_token_len_per_gpu=CLUSTER["max_token_len_per_gpu"],
        )["typical_micro_batches"]
        for sp in (1, 2, 4, 8, 16, 32)
    }
    assert len(set(counts.values())) == 1, (
        f"the estimator thinks ULYSSES_SP changes the micro-batch count: {counts}. "
        f"It does not, and believing it does sends the next person tuning a knob that "
        f"cannot help."
    )


def test_a_small_enough_global_batch_reaches_one_micro_batch():
    # The alternative fix, so it has to be reachable and reported honestly.
    small = {**CLUSTER, "train_batch_size": 16}
    est = mod.estimate_micro_batches(sample_lengths=lengths(), **small)
    assert est["typical_micro_batches"] == 1
    assert est["worst_case_micro_batches"] == 1


def test_the_worst_case_is_not_below_the_typical_case():
    # same_micro_num_in_dp=True pads every dp rank to the largest count any rank needs,
    # so the gate has to reason about the unluckiest rank, not the average one.
    mixed = [2000] * 900 + [32000] * 100
    est = mod.estimate_micro_batches(sample_lengths=mixed, **CLUSTER)
    assert est["worst_case_micro_batches"] >= est["typical_micro_batches"]
    assert est["worst_case_micro_batches"] > 1, (
        "a length tail that long must show up as a worst case above one, or a run that "
        "passes the gate will OOM on the first unlucky step instead of at launch"
    )


def test_the_retained_gradient_does_not_shrink_with_more_gpus():
    """The whole reason this is a separate failure mode from the static footprint.

    `unsharded_accumulated_grad` is a whole-parameter tensor, so the only lever is not
    allocating it.
    """
    assert round(mod.unsharded_grad_gib(27.78), 1) == 103.5, (
        "the number quoted in the diagnosis, the launcher gate and OPERATOR_PROMPT.md"
    )
    # It takes no shard degree at all, by construction -- so nothing about the cluster
    # topology can reduce it. Contrast the sharded gradient the patch keeps instead.
    assert round(mod.sharded_grad_gib(27.78, 32), 1) == 3.2
    assert mod.sharded_grad_gib(27.78, 32) < mod.sharded_grad_gib(27.78, 8)
    assert mod.sharded_grad_gib(27.78, 1) == mod.unsharded_grad_gib(27.78), (
        "at shard degree 1 the two must coincide; if they do not, one of the two "
        "formulas has picked up a factor and the gate's projections are wrong"
    )
    # bf16 reduce halves it and is still fatal on an 80 GiB card -- the reason the patch
    # is structural rather than a dtype change.
    assert 50 < mod.unsharded_grad_gib(27.78, reduce_bytes=2) < 55


def test_the_env_switch_defaults_to_on_and_accepts_the_usual_falsehoods():
    import os

    saved = os.environ.pop(mod.ENV_FLAG, None)
    try:
        assert mod.enabled(), "the patch must be ON by default; the failure it prevents is fatal"
        for off in ("0", "false", "False", "no", "OFF", " 0 "):
            os.environ[mod.ENV_FLAG] = off
            assert not mod.enabled(), f"{off!r} should disable the patch"
        for on in ("1", "true", "yes"):
            os.environ[mod.ENV_FLAG] = on
            assert mod.enabled(), f"{on!r} should enable the patch"
    finally:
        os.environ.pop(mod.ENV_FLAG, None)
        if saved is not None:
            os.environ[mod.ENV_FLAG] = saved


def test_apply_is_a_no_op_without_verl_rather_than_a_crash():
    # The estimator and the gate are used on machines with no verl at all (this test run
    # is one), and importing the dataset module must not become a hard dependency.
    status = mod.apply(verbose=False)
    assert isinstance(status, str) and status
    assert mod.state()["applied"] is True


def test_the_patch_is_applied_from_the_module_verl_loads_in_every_rank():
    """Not from a launcher. The run that failed used a cluster-side wrapper script, so a
    fix that lives in scripts/30_run_sft_verl.sh would have been bypassed."""
    assert "fsdp2_grad_accum" in DATASET, (
        "verl_backend/rst_sft_dataset.py no longer applies the patch, so any launcher "
        "that does not know about it silently gets the 103.5 GiB path back"
    )
    assert "fsdp2_grad_accum.apply()" in DATASET
    assert "data.custom_cls.path" in LAUNCHER


def test_the_launcher_gates_this_before_torchrun():
    gate_at = LAUNCHER.index("unsharded-gradient-accumulation gate")
    torchrun_at = LAUNCHER.index("\ntorchrun")
    assert gate_at < torchrun_at, "the gate runs after the launch, which is 32 wasted starts"
    window = LAUNCHER[gate_at:torchrun_at]
    assert "verl_backend/fsdp2_grad_accum.py" in window
    for flag in ("--ulysses-sp", "--train-batch-size", "--max-token-len-per-gpu", "--params-b"):
        assert flag in window, f"the gate is not given {flag}, so its estimate is not this run's"


def test_the_launcher_names_the_log_line_that_proves_the_patch_ran():
    # "it started" is not evidence. One printed line is, and it is the only thing that
    # distinguishes a patched run from one about to OOM 90% into a backward pass.
    assert mod.LOG_PREFIX in LAUNCHER, (
        f"the launcher's report notes do not tell the operator to grep for "
        f"{mod.LOG_PREFIX}"
    )


def test_the_diagnosis_script_knows_this_failure_mode():
    text = (ROOT / "scripts" / "34_diagnose_oom.py").read_text(encoding="utf-8")
    assert "grad_accum_verdict" in text
    assert mod.LOG_PREFIX in text, (
        "34_diagnose_oom.py cannot tell a patched run's OOM from an unpatched one's "
        "without looking for that line"
    )


def test_a_sequence_longer_than_the_group_budget_is_refused_not_counted():
    """`ceil(total_tokens / budget)` happily returns a number for a config where one
    SAMPLE does not fit, because bin packing cannot split a sample. verl discovers that
    inside `rearrange_micro_batches`, mid-step, after 32 ranks have read the weights. The
    estimator is used as a launcher gate, so it has to be the thing that says no."""
    # 4096/GPU at sp=1 is the mistake this catches: it looks like a memory-saving budget
    # and is 8x too small for the longest cap10 row. The fix is sp=8, which is exactly
    # what SP is for -- the same 4096/GPU then adds up to a placeable 32768 per group.
    too_small = {**CLUSTER, "ulysses_sp": 1, "max_token_len_per_gpu": 4096}
    try:
        mod.estimate_micro_batches(sample_lengths=lengths() + [32329], **too_small)
    except ValueError as exc:
        assert "longest sample" in str(exc) and "32,329" in str(exc), str(exc)
    else:
        raise AssertionError("a budget below the longest sequence was certified as placeable")

    ok = {**CLUSTER, "ulysses_sp": 8, "max_token_len_per_gpu": 4096}
    est = mod.estimate_micro_batches(sample_lengths=lengths() + [32329], **ok)
    assert est["budget_per_dp_group"] == 32768
    assert est["longest_sample"] == 32329


def test_the_gate_reports_an_unplaceable_config_instead_of_a_traceback():
    # It runs as `python verl_backend/fsdp2_grad_accum.py ... || exit 2` in the launcher,
    # where a traceback reads as "the gate is broken", not "the config is".
    src = (BACKEND / "fsdp2_grad_accum.py").read_text(encoding="utf-8")
    assert "except ValueError as exc:" in src and "cannot be placed at all" in src, (
        "_cli lets estimate_micro_batches' ValueError escape as a traceback"
    )


def test_an_empty_or_inconsistent_configuration_is_refused_not_guessed():
    for kwargs in (
        {"sample_lengths": [], **CLUSTER},
        {"sample_lengths": lengths(), **{**CLUSTER, "ulysses_sp": 0}},
        {"sample_lengths": lengths(), **{**CLUSTER, "ulysses_sp": 5}},  # 32 % 5 != 0
        {"sample_lengths": lengths(), **{**CLUSTER, "max_token_len_per_gpu": 0}},
    ):
        try:
            mod.estimate_micro_batches(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"estimate_micro_batches accepted {kwargs.keys()} silently")


def test_the_single_micro_batch_batch_size_is_computed_from_the_length_tail():
    # Reported by the gate as the cost of the alternative fix, so it must be packed from
    # the tail, not from the mean: a batch size that fits the average and overflows on the
    # long rows is exactly the "OOMs on some later step" failure this file is about.
    budget = 32768 * 8
    uniform = mod.largest_single_micro_batch_size(
        sample_lengths=[1000] * 1000, world_size=32, ulysses_sp=8, max_token_len_per_gpu=32768)
    assert uniform == (budget // 1000) * 4

    tail = mod.largest_single_micro_batch_size(
        sample_lengths=[1000] * 990 + [200_000] * 10,
        world_size=32, ulysses_sp=8, max_token_len_per_gpu=32768)
    assert tail == 4, (
        f"ten 200k-token rows mean one sample per dp rank is all that fits the "
        f"{budget:,}-token budget, so the answer is dp_size=4 and not the {uniform} a "
        f"mean-based estimate would report; got {tail}"
    )

    huge = mod.largest_single_micro_batch_size(
        sample_lengths=[budget + 1], world_size=32, ulysses_sp=8, max_token_len_per_gpu=32768)
    assert huge == 0, "a sample larger than the whole budget cannot be packed at any batch size"


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
