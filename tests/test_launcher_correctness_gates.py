"""The gates in `30_run_sft_verl.sh` that protect correctness rather than memory.

Packed training (`data.pad_mode=no_padding`) puts several conversations in one sequence
and relies on `cu_seqlens` to keep the gated-delta-net recurrence from running across
the boundaries. verl passes cu_seqlens, but decides whether the kernel accepts it with

    def _call_accepts_kwarg(fn, name):            # verl/models/transformers/qwen3_5.py:41
        params = signature(fn).parameters
        return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())

and transformers' pure-torch `torch_chunk_gated_delta_rule` carries a `**kwargs` it
never reads. So the check is a permanent false positive: without FLA's real kernel the
argument is silently dropped, documents bleed into each other, and the loss curve looks
completely normal. That is why the gate is a hard failure with a named escape hatch
rather than a warning.

Two further facts the gate encodes, both measured against real installs:
  * the PyPI `flash-linear-attention` wheel and sdist ship no `fla/ops` at all, so
    `pip install flash-linear-attention` does not satisfy the requirement;
  * transformers 5.15.0 removed `self.chunk_gated_delta_rule` from
    `Qwen3_5GatedDeltaNet.__init__`, which verl 0.9.0 calls unconditionally
    (`qwen3_5.py:167`), so the window is >=5.11,<5.15.

Textual assertions on purpose: no unit test can run a 32-GPU job, and the failure mode
is someone tidying a gate away.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT  # noqa: E402

LAUNCHER = ROOT / "scripts" / "30_run_sft_verl.sh"
SETUP = ROOT / "scripts" / "01b_setup_env_verl.sh"
TEXT = LAUNCHER.read_text(encoding="utf-8")
SETUP_TEXT = SETUP.read_text(encoding="utf-8")


def test_the_packed_varlen_gate_runs_before_the_launch_even_without_sp():
    gate_at = TEXT.index("packed-varlen correctness gate")
    torchrun_at = TEXT.index("\ntorchrun")
    assert gate_at < torchrun_at
    window = TEXT[gate_at:torchrun_at]
    assert "fla.ops.gated_delta_rule" in window, "the gate does not check for FLA's ops"
    assert "cu_seqlens" in window, "the gate does not check the packing argument itself"
    assert "ALLOW_UNSAFE_PACKING" in window, "no named escape hatch"


def test_the_transformers_window_is_checked_by_measurement_not_by_version_string():
    window = TEXT[TEXT.index("packed-varlen correctness gate"):TEXT.index("\ntorchrun")]
    assert "self.chunk_gated_delta_rule" in window, (
        "the gate trusts a version number instead of checking the attribute verl "
        "actually calls; a backport or a patched build would be misjudged"
    )


def test_ulysses_sp_checks_the_cp_apis_and_head_divisibility():
    window = TEXT[TEXT.index("packed-varlen correctness gate"):TEXT.index("\ntorchrun")]
    assert "fla.ops.cp.comm" in window and "fla.ops.cp.context" in window, (
        "ULYSSES_SP>1 needs FLA's cross-rank recurrent-state passing; without it verl "
        "raises NotImplementedError deep inside the first forward"
    )
    assert "num_attention_heads" in window


def test_the_sp_warning_no_longer_claims_the_gdn_path_is_unimplemented():
    """verl 0.9.0 DOES implement Ulysses SP for the gated-delta-net layers.

    Keeping the old "NOT validated on this architecture" text would keep the operator
    away from the one knob that shards activations, which is what a long-sequence OOM
    needs. The honest statement is "implemented, with preconditions" -- and the
    preconditions are the gate above.
    """
    assert "NOT validated on" not in TEXT, (
        "the SP warning still claims the gated-delta-net path is unvalidated; verl "
        "0.9.0 implements it (qwen3_5.py:75-226, monkey_patch.py:497-548)"
    )
    # The note is split across echo lines, so match the claim, not the line.
    assert "params+grads+Adam footprint is unchanged" in TEXT, (
        "the SP note does not say what SP cannot fix, so it will be tried against a "
        "static-footprint OOM"
    )
    assert "not for a static-footprint OOM" in TEXT


def test_flash_attn_is_gated_because_remove_padding_forces_flash_attention_2():
    window = TEXT[TEXT.index("packed-varlen correctness gate"):TEXT.index("\ntorchrun")]
    assert "flash_attn" in window, (
        "nothing checks flash_attn, yet verl sets attn_implementation=flash_attention_2 "
        "whenever model.use_remove_padding is True (its default), so the job dies at "
        "model load after every rank has read the weights"
    )
    assert "use_remove_padding=False" in window, "no documented way to run without FA2"


def test_the_environment_script_installs_fla_from_git_not_from_pypi():
    assert "git+https://github.com/fla-org/flash-linear-attention" in SETUP_TEXT, (
        "the setup script installs the PyPI wheel, which ships fla/layers and "
        "fla/models but no fla/ops -- `import fla.ops.gated_delta_rule` then fails"
    )
    assert "slower, correct" not in SETUP_TEXT, (
        "the setup script still claims the pure-PyTorch fallback is correct; it drops "
        "cu_seqlens, so packed documents share a recurrent state"
    )


def test_the_environment_script_pins_the_transformers_window():
    assert '"transformers>=5.11,<5.15"' in SETUP_TEXT, (
        "transformers is unpinned above 5.15, where Qwen3_5GatedDeltaNet no longer "
        "sets self.chunk_gated_delta_rule and verl's first forward raises AttributeError"
    )


def test_the_environment_script_installs_verls_undeclared_runtime_deps():
    for pkg in ("pillow", "uvicorn", "fastapi"):
        assert pkg in SETUP_TEXT, (
            f"{pkg} is missing; verl 0.9.0 does not declare it but the SFT path imports "
            f"it unconditionally, so training dies inside torchrun"
        )


def test_the_entrypoint_import_check_is_fatal_not_a_warning():
    at = SETUP_TEXT.index("verl.trainer.sft_trainer")
    window = SETUP_TEXT[at - 400:at + 800]
    assert "sys.exit" in window, (
        "the entrypoint import check is still advisory; that is exactly how three "
        "missing runtime deps reached the cluster with a setup script that 'succeeded'"
    )


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
