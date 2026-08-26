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


# ------------------------------------------------------------------ serving the model
#
# Three things that each stopped sglang from ever answering a request. All were found by
# running it on a GPU shared with another user's job, which is the normal case on a
# shared box and the one the hardcoded defaults did not survive.


def test_ninja_is_reachable_by_the_sglang_subprocess():
    """The failure that looks like a healthy start.

    sglang JIT-compiles some kernels by shelling out to `ninja`, which pip puts in the
    interpreter's own bin/ -- not on PATH, because nothing activates a venv. It dies
    with `FileNotFoundError: 'ninja'` AFTER logging the KV cache size and
    max_total_num_tokens, so the log's last useful line reads like success.
    """
    source = (ROOT / "scripts" / "06_eval.py").read_text(encoding="utf-8")
    assert "serve_env" in source and "Path(sys.executable).parent" in source
    assert "env=serve_env" in source, "the PATH fix was built but not passed to the child"


def test_the_memory_fraction_is_not_hardcoded():
    """0.85 assumes the card is ours.

    mem_fraction_static is measured against the memory available to sglang, so on a card
    where another process already holds most of the HBM the default is what you want --
    but it has to be movable, because the failure otherwise is a CUDA OOM that reads
    like a model problem rather than a co-tenancy problem.
    """
    source = (ROOT / "scripts" / "06_eval.py").read_text(encoding="utf-8")
    assert '"--mem-fraction-static", str(args.mem_fraction_static),' in source
    assert '"--mem-fraction-static", type=float, default=0.85' in source


def test_the_in_flight_request_cap_is_available_and_optional():
    """Why this one matters more than the memory fraction.

    Each in-flight request reserves its own gated-delta-net state -- 49 MiB/request at
    4B -- and sglang sizes that pool from the CUDA graph's max batch (256), i.e. ~12.6
    GiB gone before any KV cache. The resulting error names --mem-fraction-static and
    never mentions the request count, so it sends you at the wrong knob.

    Default 0 must leave sglang's own behaviour untouched: this is a co-tenancy escape
    hatch, not a new default.
    """
    source = (ROOT / "scripts" / "06_eval.py").read_text(encoding="utf-8")
    assert 'if args.max_running_requests:' in source
    assert '"--max-running-requests", str(args.max_running_requests)' in source
    assert '"--max-running-requests", type=int, default=0' in source


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
