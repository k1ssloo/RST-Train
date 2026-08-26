"""`06_eval.py` must hand harbor a `model_info`, or it scores nothing and says it passed.

THE FAILURE THIS PREVENTS
    Terminus-2 reaches the model through LiteLLM, which looks up the context window and
    per-token price in its own model map. `hosted_vllm/<served-name>` is never in that
    map -- the name is whatever `--served-model-name` we chose -- so harbor 0.21.0
    refuses to start the agent:

        ValueError: hosted_vllm models require model_info

    Measured: every task fails this way, before the agent issues one command. And
    because it is a *harness-infrastructure* failure, the taxonomy correctly excludes
    it from the pass-rate denominator -- so the run completes, reports no failures
    against the model, and the pass rate is computed over an empty denominator. The
    script's whole purpose is to make that unmisreadable, so the flag is not optional.

WHY max_input_tokens IS DERIVED AND NOT HARDCODED
    It has to agree with what sglang was actually launched with (`--context-length`).
    Hardcode it too high and LiteLLM cheerfully sends a prompt the server then rejects,
    and the task fails for a reason that looks nothing like "the context is too small".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, load_script  # noqa: E402

ev = load_script("06_eval")


class _Args:
    """Only the fields agent_kwargs_argv reads."""

    def __init__(self, context_length=65536, max_output_tokens=4096, harbor_bin="harbor"):
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.harbor_bin = harbor_bin


def _forwarded(argv):
    """The parsed model_info dict out of the produced argv."""
    assert argv[0] == "--agent-kwarg", argv
    assert argv[1].startswith("model_info="), argv
    return json.loads(argv[1][len("model_info="):])


# ------------------------------------------------------------------ the payload


def test_the_four_keys_litellm_needs_are_all_present():
    ev.harbor_run_flags.cache_clear()
    original = ev.harbor_run_flags
    ev.harbor_run_flags = lambda _bin: frozenset({"--agent-kwarg"})
    try:
        argv, record = ev.agent_kwargs_argv(_Args())
        info = _forwarded(argv)
        assert set(info) == {"max_input_tokens", "max_output_tokens",
                             "input_cost_per_token", "output_cost_per_token"}
        assert record["forwarded"] is True
    finally:
        ev.harbor_run_flags = original


def test_max_input_tokens_is_the_context_minus_the_output_reserve():
    """The number must be reachable by the server, not aspirational."""
    original = ev.harbor_run_flags
    ev.harbor_run_flags = lambda _bin: frozenset({"--agent-kwarg"})
    try:
        info = _forwarded(ev.agent_kwargs_argv(_Args(65536, 4096))[0])
        assert info["max_input_tokens"] == 61440
        assert info["max_output_tokens"] == 4096
        assert info["max_input_tokens"] + info["max_output_tokens"] == 65536, (
            "input + output must not exceed what sglang was launched with")
        # and it tracks a changed context length rather than staying at a constant
        info = _forwarded(ev.agent_kwargs_argv(_Args(32768, 2048))[0])
        assert info["max_input_tokens"] == 30720
    finally:
        ev.harbor_run_flags = original


def test_costs_are_zero_but_present():
    # Local weights have no price; LiteLLM still computes spend and needs the keys.
    original = ev.harbor_run_flags
    ev.harbor_run_flags = lambda _bin: frozenset({"--agent-kwarg"})
    try:
        info = _forwarded(ev.agent_kwargs_argv(_Args())[0])
        assert info["input_cost_per_token"] == 0
        assert info["output_cost_per_token"] == 0
    finally:
        ev.harbor_run_flags = original


def test_a_context_smaller_than_the_reserve_still_yields_a_positive_window():
    # Nonsense config, but it must not emit max_input_tokens <= 0 and have LiteLLM
    # reject every request for a reason that reads as a model problem.
    original = ev.harbor_run_flags
    ev.harbor_run_flags = lambda _bin: frozenset({"--agent-kwarg"})
    try:
        info = _forwarded(ev.agent_kwargs_argv(_Args(1024, 4096))[0])
        assert info["max_input_tokens"] >= 1
    finally:
        ev.harbor_run_flags = original


# ------------------------------------------------------------------ the honesty gate


def test_a_harbor_without_the_flag_is_reported_loudly_and_not_silently_skipped():
    """The case a reader most needs to know about.

    If the flag cannot be forwarded, every task may fail in the harness and the pass
    rate is computed over nothing. Recording that in `protocol.agent_model_info` is
    what makes a zero-denominator run diagnosable from results.json alone.
    """
    original = ev.harbor_run_flags
    ev.harbor_run_flags = lambda _bin: frozenset({"--temperature"})   # no --agent-kwarg
    try:
        argv, record = ev.agent_kwargs_argv(_Args())
        assert argv == []
        assert record["forwarded"] is False
        assert "NOT FORWARDED" in record["control"]
        assert "empty denominator" in record["control"]
        # the intended payload is still recorded, so what was missing is knowable
        assert record["model_info"]["max_input_tokens"] == 61440
    finally:
        ev.harbor_run_flags = original


# ------------------------------------------------------------------ wiring


def test_the_flag_is_actually_passed_to_the_harbor_subprocess():
    # A correct payload that never reaches argv is the same as no payload.
    source = (ROOT / "scripts" / "06_eval.py").read_text(encoding="utf-8")
    assert "argv += args.agent_kwargs_argv" in source, (
        "model_info was built but never added to the harbor command line")
    assert "args.agent_kwargs_argv, args.agent_kwargs_record = agent_kwargs_argv(args)" in source


def test_the_decision_is_recorded_in_results_json():
    source = (ROOT / "scripts" / "06_eval.py").read_text(encoding="utf-8")
    assert '"agent_model_info": args.agent_kwargs_record,' in source


def test_max_output_tokens_is_settable_from_the_cli():
    source = (ROOT / "scripts" / "06_eval.py").read_text(encoding="utf-8")
    assert '"--max-output-tokens"' in source


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
