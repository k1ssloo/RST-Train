"""`rst_common/probe_core.py` -- the rules behind the release probe (E0/E1/E2).

Hypotheses under test (see notes/RSI_ALGORITHM_CANDIDATES_ZH.md and TODO.md):
  H1  RST verifiers lean on existence-only / literal checks -> `classify_style`
  H2  RST oracles are single-instance scripts -> `plan_mutation`, `patch_dockerfile`,
      `artifact_delta` (what the oracle wrote, so a replay can be attempted)
  H3  the release holds cheap successes -> `ctrf_test_names` / `pin_trajectory`,
      command classes, the cheap rule
Every rule is pinned here on synthetic inputs so the census numbers can be argued
about at the level of "this function counts as X because...".
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import ROOT, load_repo_module, need  # noqa: E402

pc = load_repo_module("rst_common.probe_core")


# ------------------------------------------------------------------ styles

def test_each_style_has_a_canonical_example():
    cases = {
        "existence_or_nonempty_only": "def test_a():\n    assert os.path.exists('/app/r.txt')\n",
        "literal_compare_only": "def test_b():\n    assert read_value() == 'PASS'\n",
        "read_then_literal_compare": "def test_c():\n    d = json.load(open('/app/r.json'))\n    assert d['count'] == 37\n",
        "recompute_from_env": ("def test_d():\n    rows = open('/app/log').read().splitlines()\n"
                               "    assert sum(1 for r in rows if 'ERR' in r) == int(open('/app/n').read())\n"),
        "rerun_tool_or_script": "def test_e():\n    out = subprocess.run(['tool'], capture_output=True)\n    assert out.returncode == 0\n",
        "other": "def test_f():\n    assert compute() is not None\n",
    }
    for style, src in cases.items():
        got = pc.census_test_file(src)["functions"][0]["style"]
        assert got == style, f"{src!r} classified as {got}, expected {style}"


def test_a_read_that_lives_in_a_helper_counts_when_helpers_are_included():
    src = ("def _profile():\n    return open('/app/ENVIRONMENT').read().strip()\n\n"
           "def test_x():\n    assert _profile() == 'modern'\n")
    with_helpers = pc.census_test_file(src, include_helpers=True)["functions"][0]["style"]
    alone = pc.census_test_file(src, include_helpers=False)["functions"][0]["style"]
    assert with_helpers == "read_then_literal_compare" and alone == "literal_compare_only", (
        "the helper carries the READ; both views are reported by E0 so the delta is visible")


def test_a_literal_on_either_side_of_in_is_a_constant_comparison():
    """The first rules matched `x == "lit"` but not `"lit" in content`, which put 2,307
    pool functions that compare an artifact against a constant into `other`."""
    for src in ('def test_a():\n    c = open("/app/r.md").read()\n    assert "2.0.91" in c\n',
                'def test_b():\n    c = open("/app/r.md").read()\n    assert "ERROR" not in c\n',
                'def test_c():\n    c = open("/app/r.md").read()\n    assert c.strip() != "TODO"\n'):
        assert pc.census_test_file(src)["functions"][0]["style"] == "read_then_literal_compare", src


def test_reading_a_file_then_asserting_on_it_is_never_existence_only():
    """An `exists()` guard followed by a content assertion is a content check; the
    2026-09-03 rules called 4,161 pool functions (12.6%) existence-only for that."""
    src = ('def test_a():\n    p = "/app/r.md"\n    assert os.path.isfile(p)\n'
           '    with open(p) as f:\n        content = f.read()\n    assert "extra-lists" in content.lower()\n')
    assert pc.census_test_file(src)["functions"][0]["style"] == "read_then_literal_compare"
    bare = 'def test_b():\n    assert os.path.isfile("/app/r.md")\n'
    assert pc.census_test_file(bare)["functions"][0]["style"] == "existence_or_nonempty_only"


def test_class_based_tests_get_ctrf_style_qualnames():
    src = "class TestX:\n    def test_y(self):\n        assert Path('/app/a').exists()\n"
    c = pc.census_test_file(src)
    assert c["test_qualnames"] == ["TestX::test_y"]
    assert c["dominant_style"] == "existence_or_nonempty_only" and c["existence_only_task"]


def test_oracle_inspection_flags_the_mount_path_but_not_an_agent_output_named_solution():
    src = ("def test_a():\n    s = Path('/app/solution/solve.sh')\n    assert s.is_file()\n"
           "    assert 'abc' not in s.read_text()\n")
    c = pc.census_test_file(src)
    assert c["inspects_oracle_script"] and c["asserts_oracle_exists"]
    other = pc.census_test_file("def test_b():\n    assert Path('/app/solution.py').exists()\n")
    assert not other["inspects_oracle_script"], "/app/solution.py is an agent file, not the oracle"


def test_pass_literal_and_unparseable_are_flagged_not_hidden():
    c = pc.census_test_file('def test_a():\n    """doc"""\n    pass\n')
    assert c["pass_literal"] and c["functions"][0]["style"] == "other"
    bad = pc.census_test_file("def test_a(:\n")
    assert bad["parse_ok"] is False and bad["n_tests"] == 0 and bad["dominant_style"] == "none"


def test_dominant_style_needs_a_strict_majority():
    src = ("def test_a():\n    assert Path('/a').exists()\n"
           "def test_b():\n    assert Path('/b').exists()\n"
           "def test_c():\n    assert subprocess.run(['x']).returncode == 0\n")
    assert pc.census_test_file(src)["dominant_style"] == "existence_or_nonempty_only"
    two = "def test_a():\n    assert Path('/a').exists()\n" "def test_c():\n    assert subprocess.run(['x']).returncode == 0\n"
    assert pc.census_test_file(two)["dominant_style"] == "mixed"


# ---------------------------------------------------------------- ctrf / pin

def _ctrf(*names):
    return {"results": {"tests": [{"name": n, "status": "passed"} for n in names]}}


def test_ctrf_names_drop_the_file_prefix_and_params_but_keep_the_class():
    names = pc.ctrf_test_names(_ctrf("test_state.py::test_a", "test_state.py::TestX::test_y[1-2]"))
    assert names == frozenset({"test_a", "TestX::test_y"})
    assert pc.ctrf_test_names(None) == frozenset()


def test_pin_is_unique_ambiguous_unmatched_or_no_ctrf():
    cands = {"t1": frozenset({"test_a", "test_b"}), "t2": frozenset({"test_a"}), "t3": frozenset({"test_a"})}
    assert pc.pin_trajectory(frozenset({"test_a", "test_b"}), cands).kind == "unique"
    amb = pc.pin_trajectory(frozenset({"test_a"}), cands)
    assert amb.kind == "ambiguous" and amb.n_candidates == 2
    assert pc.pin_trajectory(frozenset({"test_zzz"}), cands).kind == "unmatched"
    assert pc.pin_trajectory(frozenset(), cands).kind == "no_ctrf"


def test_workdir_breaks_a_tie_and_says_so():
    cands = {"t2": frozenset({"test_a"}), "t3": frozenset({"test_a"})}
    pin = pc.pin_trajectory(frozenset({"test_a"}), cands, cwd_hint="/app",
                            candidate_workdirs={"t2": "/app", "t3": "/workspace"})
    assert pin.task_id == "t2" and pin.tie_break == "workdir"
    assert pc.terminal_cwd("Current Terminal Screen:\nroot@fa40b1a3-91e9-4ef4:/app# ") == "/app"
    assert pc.terminal_cwd("root@abc:~/exercises# ") == "/root/exercises"


# ------------------------------------------------------------------ Dockerfile

DOCKERFILE = """FROM ubuntu:22.04
RUN mkdir -p /app && \\
    printf 'ZOOKEEPER\\nRANGER\\n' > /etc/lists/required.txt && \\
    echo 'PROFILE=modern' > /app/ENVIRONMENT
RUN cat <<'EOF' > /app/config.yaml
threshold: 5
name: alpha
EOF
COPY data/input.csv /app/input.csv
COPY <<EOF /app/note.txt
hello 7
EOF
WORKDIR /app
USER worker
CMD ["bash"]
"""


def test_fixtures_are_recovered_from_echo_printf_heredoc_and_copy():
    fixtures = {f.path: f for f in pc.dockerfile_fixtures(DOCKERFILE, {"data/input.csv": b"id,value\n1,10\n"})}
    assert fixtures["/etc/lists/required.txt"].content == "ZOOKEEPER\nRANGER\n"
    assert fixtures["/app/ENVIRONMENT"].content == "PROFILE=modern\n"
    assert fixtures["/app/config.yaml"].content == "threshold: 5\nname: alpha\n"
    assert fixtures["/app/input.csv"].content == "id,value\n1,10\n" and fixtures["/app/input.csv"].source_kind == "copy_file"
    assert fixtures["/app/note.txt"].content == "hello 7\n" and fixtures["/app/note.txt"].source_kind == "copy_heredoc"
    assert pc.dockerfile_workdir(DOCKERFILE) == "/app"


def test_a_variable_payload_is_a_fixture_without_recoverable_content():
    fx = pc.dockerfile_fixtures("RUN echo \"$HOSTNAME\" > /app/host\n")
    assert fx and fx[0].path == "/app/host" and fx[0].content is None


def test_binary_copy_sources_do_not_crash_the_planner():
    fx = pc.dockerfile_fixtures("COPY blob.bin /app/blob.bin\n", {"blob.bin": b"\xff\xfe\x00"})
    assert fx[0].content is None
    m, reason = pc.plan_mutation("COPY blob.bin /app/blob.bin\n", "cat /app/blob.bin", "", {"blob.bin": b"\xff\xfe"})
    assert m is None and reason == "undecodable"


# -------------------------------------------------------------------- mutation

def test_planner_prefers_the_fixture_both_sides_read_and_a_numeric_token():
    solve = "cat /app/config.yaml\ncat /app/ENVIRONMENT\n"
    tests = "def test_a():\n    assert open('/app/config.yaml').read()\n"
    m, reason = pc.plan_mutation(DOCKERFILE, solve, tests, {"data/input.csv": b"id,value\n1,10\n"})
    assert reason is None and m.path == "/app/config.yaml", "config.yaml is read by both; ENVIRONMENT by solve only"
    assert (m.before, m.after, m.kind, m.ref_strength) == ("5", "6", "numeric_in_domain", "both")
    assert m.new_content == "threshold: 6\nname: alpha\n"


def test_planner_refuses_a_fixture_the_oracle_writes_and_names_the_reason():
    solve = "echo LOGSEARCH >> /etc/lists/required.txt\n"
    m, reason = pc.plan_mutation("RUN printf 'A\\nB\\n' > /etc/lists/required.txt\n", solve, "", {})
    assert m is None and reason == "fixture_written_by_solve"
    m, reason = pc.plan_mutation("RUN echo x > /app/unused\n", "ls /app", "", {})
    assert m is None and reason == "fixture_not_referenced"
    m, reason = pc.plan_mutation("RUN echo x > /app/unused\n", "cat /app/unused", "", {}, compose=True)
    assert reason == "compose_task"
    m, reason = pc.plan_mutation("FROM x\nRUN apt-get update\n", "ls", "", {})
    assert reason == "no_fixture_written"


def test_mutate_content_never_touches_keys_or_paths():
    new, line, before, after, kind = pc.mutate_content("PORT=8080\nPATH=/usr/bin\n")
    assert (before, after, kind, line) == ("8080", "8081", "numeric_in_domain", 1)
    new, line, before, after, kind = pc.mutate_content("PROFILE=modern\n")
    assert (before, after, kind) == ("modern", "modernx", "alpha_value")
    new, line, before, after, kind = pc.mutate_content("alpha beta\n")
    assert (before, after, kind) == ("beta", "betax", "alpha_field")
    assert pc.mutate_content("# only a comment\n") is None
    assert pc.mutate_content("/usr/local/bin\n") is None


def test_the_same_task_always_gets_the_same_plan():
    solve = "cat /app/config.yaml\n"
    a, _ = pc.plan_mutation(DOCKERFILE, solve, "", {}, seed=1)
    b, _ = pc.plan_mutation(DOCKERFILE, solve, "", {}, seed=1)
    assert a == b


def test_patch_appends_one_layer_before_cmd_and_restores_the_user():
    m, _ = pc.plan_mutation(DOCKERFILE, "cat /app/config.yaml", "", {})
    patched = pc.patch_dockerfile(DOCKERFILE, m)
    lines = patched.splitlines()
    cmd_idx = next(i for i, ln in enumerate(lines) if ln.startswith("CMD"))
    run_idx = next(i for i, ln in enumerate(lines) if ln.startswith("RUN mkdir -p ") and "base64 -d" in ln)
    assert run_idx < cmd_idx, "the mutation layer must precede CMD"
    assert lines[run_idx - 1] == "USER root" and lines[run_idx + 1] == "USER worker"
    payload = lines[run_idx].split("echo '", 1)[1].split("'", 1)[0]
    assert base64.b64decode(payload).decode() == m.new_content
    assert patched.count("rst-probe: mutate") == 1
    assert DOCKERFILE.splitlines()[:5] == lines[:5], "every original layer is byte-identical, so the cache holds"


def test_patch_without_user_directive_adds_none():
    df = "FROM ubuntu\nRUN echo 'k=1' > /app/f\n"
    m, _ = pc.plan_mutation(df, "cat /app/f", "", {})
    patched = pc.patch_dockerfile(df, m)
    assert "USER" not in patched and patched.endswith("\n")


# -------------------------------------------------------------- artifact delta

def test_artifact_delta_separates_created_from_modified_and_ignores_untouched():
    before = "/app/in.txt\t10\t1.0\n/app/keep\t5\t2.0\n"
    after = "/app/in.txt\t10\t1.0\n/app/keep\t6\t2.0\n/app/out.json\t3\t9.0\n"
    created, modified = pc.artifact_delta(before, after)
    assert created == ["/app/out.json"] and modified == ["/app/keep"]
    assert pc.artifact_delta("", "") == ([], [])


# ----------------------------------------------------------- instruction paths

def test_instruction_paths_strip_punctuation_and_skip_the_private_dirs():
    text = ("Write the report to /app/report.json. Evidence goes in /app/out/evidence.csv, "
            "and never touch /tests/test.sh or /logs/verifier/reward.txt (see /app/report.json).")
    assert pc.instruction_paths(text) == ["/app/report.json", "/app/out/evidence.csv"]
    assert pc.plausible_content("/app/x.json") == "{}\n" and pc.plausible_content("/app/y.txt") == "ok\n"


# ------------------------------------------------------------ command classes

def test_write_only_means_every_command_writes_and_none_reads():
    assert pc.is_write_only(["echo '{\"endpoint\":\"/api/pay\"}' > /app/report.json", "mkdir -p /app/out"])
    assert not pc.is_write_only(["cat /app/logs/app.log", "echo x > /app/report.json"])
    assert pc.command_class("cat /app/logs/app.log") == "read"
    assert pc.command_class("cat > /app/r.txt <<'EOF'\nhi\nEOF") == "write"
    assert pc.command_class("python3 analyze.py") == "other"
    assert not pc.is_write_only([])


def test_tamper_and_interactive_flags():
    assert pc.tamper_flags(["sed -i 's/0/1/' /tests/test_state.py"]) == ["/tests", "test_state.py"]
    assert pc.tamper_flags(["ls /app"]) == []
    assert pc.interactive_flags(["C-c", "vim", "ls"]) == ["key:C-c", "bare:vim"]


def test_cheap_rule_is_absolute_or_relative():
    assert pc.cheap_rule(3, 100) and pc.cheap_rule(8, 100, max_ratio=0.1) and not pc.cheap_rule(20, 100)
    assert pc.oracle_command_count("#!/bin/bash\nset -e\n# c\nls\nif x; then\n  echo a\nfi\n") == 3


# ------------------------------------------------------------- 2x2 outcome

def _arms(nop=0.0, oracle=1.0, touch=0.0, om=1.0, am=0.0):
    return {"nop": (nop, True), "oracle": (oracle, True), "touch": (touch, True),
            "oracle_mut": (om, True), "artifacts_mut": (am, True)}


def test_every_2x2_cell_and_every_uninterpretable_reason():
    assert pc.classify_task_outcome(_arms(om=1, am=0), True) == "parametric_ok"
    assert pc.classify_task_outcome(_arms(om=1, am=1), True) == "verifier_fixture_insensitive"
    assert pc.classify_task_outcome(_arms(om=0, am=0), True) == "oracle_brittle"
    assert pc.classify_task_outcome(_arms(om=0, am=1), True) == "oracle_brittle_and_verifier_constant"
    assert pc.classify_task_outcome(_arms(nop=1), True) == "uninterpretable:nop_passes"
    assert pc.classify_task_outcome(_arms(oracle=0), True) == "uninterpretable:oracle_fails_baseline"
    assert pc.classify_task_outcome(_arms(), False) == "uninterpretable:mutation_not_applied"
    arms = _arms()
    arms["artifacts_mut"] = (None, False)
    assert pc.classify_task_outcome(arms, True) == "uninterpretable:infra:artifacts_mut", (
        "an infra failure is unmeasured, never a 0")
    del arms["touch"]
    assert pc.classify_task_outcome(arms, True).startswith("uninterpretable:missing:")


# ------------------------------------------------------------- import boundary

def test_probe_agents_import_only_stdlib_harbor_and_probe_core():
    import ast
    src = (ROOT / "rst_common" / "probe_agents.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    allowed_roots = {"json", "shlex", "shutil", "time", "pathlib", "harbor", "rst_common", "__future__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed_roots, alias.name
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root in allowed_roots, node.module
            if root == "rst_common":
                assert node.module == "rst_common" and [a.name for a in node.names] == ["probe_core"], (
                    "agents may import probe_core only; harbor's venv has no pandas")
    assert "import pandas" not in src and "import pyarrow" not in src


def test_probe_agents_import_inside_an_interpreter_that_has_harbor():
    need("harbor")
    agents = load_repo_module("rst_common.probe_agents")
    for name in ("SnapshotOracleAgent", "ReplayArtifactsAgent", "TouchPathsAgent", "ReplayKeystrokesAgent"):
        cls = getattr(agents, name)
        assert cls.import_path() == f"rst_common.probe_agents:{name}"


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
