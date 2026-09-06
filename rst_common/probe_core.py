"""Pure definitions behind the RST release probe: E0 census, E1 sandbox, E2 audit.

WHY ONE STDLIB-ONLY MODULE
    Three consumers need the same rules and none of them can share an interpreter:
    the census and audit scripts run in `.venv` (pandas), the custom Harbor agents run
    inside harbor's own venv (`~/.venvs/AgentGen`, which has no pandas), and the tests
    run anywhere. A rule that lives in two places is two rules, and every number the
    probe reports is only worth something if the same rule produced it everywhere.
    So everything here is `re`/`ast`/`json`, importable from all three.

WHAT THE RULES ARE, AND WHAT THEY MEASURED
    * `classify_style` sorts one verifier test function into one of six styles by
      regex over its source (plus the module-level helpers it calls).
      `STYLE_RULES_VERSION` names the rule set, because these numbers moved once
      already and the move was the rules, not the data. A first pass on 2026-09-03
      over the 5,140 sweet-pool tasks reported 26.7% of functions as
      existence-or-nonempty only. Two defects in those rules, both since fixed and
      both measured on the same 32,633 functions:
        - it labelled a function existence-only whenever an `exists()` matched,
          even when the function went on to read the file and assert on its
          contents. 4,161 functions (12.6%) were miscounted that way; the class
          now requires an existence check AND no read.
        - `LIT_RE` matched `x == "lit"` but not `"lit" in content`, so 2,307 of
          those 4,161 -- real comparisons against a constant -- fell into `other`.
      The corrected reading of the same pool is nearer 13% existence-only, and the
      class that matters for the probe is "reads the artifact, compares it to a
      constant", because that is precisely what the `artifacts_mut` arm attacks.
    * `pin_trajectory` maps a released trajectory to the exact task variant it ran
      on by the SET of verifier test names in its `ctrf.json`. Measured on
      `trajectories-00000.tar`: 95.1% unique, 4.9% ambiguous, 0 unmatched. The
      obvious alternative -- matching the instruction text in the prompt -- misses
      91.5% because the released `instruction.md` drifted after the trials ran.
    * `plan_mutation` picks ONE fixture the Dockerfile bakes and the oracle or the
      verifier reads but the oracle does not write, and changes one data token.
      Calibration over the pool: ~59% of tasks have such a fixture. Everything
      else is counted under a named NA reason rather than silently skipped.
    * `instruction_paths` is the budget-1 adversary's whole strategy: create the
      absolute paths the instruction names. 889 of 2,387 such paths in a 1,000-task
      sample already exist in the image (they are inputs), so callers must skip
      paths that exist and touch only the missing ones.

WHAT THIS MODULE NEVER DOES
    Touch the filesystem, run a container, or import pandas. Callers do the I/O.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import shlex
import warnings
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

STYLE_RULES_VERSION = "2026-09-04"

# --------------------------------------------------------------------------- styles

STYLES = (
    "rerun_tool_or_script",
    "recompute_from_env",
    "read_then_literal_compare",
    "literal_compare_only",
    "existence_or_nonempty_only",
    "other",
)

# The style regexes. Each is one claim about what a verifier check looks like; the
# docstring above records which of them have already been wrong and by how much.
EXIST_RE = re.compile(
    r"\.exists\(\)|\.is_file\(\)|\.is_dir\(\)|os\.path\.(exists|isfile|isdir|lexists)"
    r"|\.stat\(\)\.st_size|os\.path\.getsize\(|len\([^)]*\)\s*>\s*0|strip\(\)\s*!=\s*['\"]{2}"
)
# A comparison against a constant baked into the verifier. Both `x == "lit"` and
# `"lit" in content` count: measured on the 5,140-task pool, 2,307 of the 4,185
# functions that read a file and then assert on its bytes use the second form, and
# an earlier version of this regex matched only the first -- so they fell into
# `other` and the "verifier compares to a constant" class was undercounted by 7
# points of all functions. That class is exactly what the artifacts_mut arm attacks.
LIT_RE = re.compile(
    r"==\s*(['\"][^'\"]{1,60}['\"]|-?\d+(\.\d+)?)"
    r"|!=\s*(['\"][^'\"]{1,60}['\"]|-?\d+(\.\d+)?)"
    r"|(?:not\s+)?in\s*\(?\s*['\"][^'\"]{1,60}['\"]"
    r"|['\"][^'\"]{1,80}['\"]\s+(?:not\s+)?in\b"
)
READ_RE = re.compile(
    r"open\(|read_text\(|read_bytes\(|json\.load|csv\.|yaml\.|toml|configparser"
    r"|\.read\(\)|\.readlines?\(|glob\(|iterdir|listdir|sqlite3\.connect|zipfile\.|tarfile\."
)
RECOMPUTE_RE = re.compile(
    r"for\s+\w+\s+in|sum\(|len\(\s*\[|Counter|sorted\(|max\(|min\(|set\(|split\("
    r"|re\.(search|findall|match|sub)|hashlib|sha256|md5"
)
SUBPROC_RE = re.compile(r"subprocess|os\.system|check_output|run\(\[|os\.popen|pexpect")
MTIME_RE = re.compile(r"st_mtime|getmtime|st_ctime|os\.utime|mtime")
# Harbor mounts the reference solution at /solution/solve.sh; "/app/solution.py" is an
# agent output and must not count, hence the trailing slash.
ORACLE_INSPECT_RE = re.compile(r"/solution/|solve\.sh")
HISTORY_RE = re.compile(r"bash_history|\.history\b|history\.log|tmux")
RANDOMNESS_RE = re.compile(
    r"\brandom\.|\bsecrets\.|uuid4?\(|\$RANDOM\b|\bshuf\b|/dev/urandom|openssl rand"
    r"|time\.time\(\)|datetime\.now\(|date \+%s"
)
# A body that asserts nothing: docstring, `pass`, `assert True`, comments.
_PASS_LITERAL_RE = re.compile(r"^\s*(pass|assert\s+True|\.\.\.)\s*$")


def classify_style(text: str) -> str:
    """One of `STYLES` for the source of one test function (+ its helpers).

    Precedence is deliberate: a function that re-runs the task's tool is a rerun
    check even if it also compares a literal; a function that reads a file and
    computes something over it is a recompute check even if a literal appears.
    """
    reads = bool(READ_RE.search(text))
    if SUBPROC_RE.search(text):
        return "rerun_tool_or_script"
    if reads and RECOMPUTE_RE.search(text):
        return "recompute_from_env"
    if reads and LIT_RE.search(text):
        return "read_then_literal_compare"
    if LIT_RE.search(text):
        return "literal_compare_only"
    if EXIST_RE.search(text) and not reads:
        return "existence_or_nonempty_only"
    return "other"


def function_flags(text: str) -> dict[str, bool]:
    """Per-function flags that are orthogonal to the style class."""
    inspects = bool(ORACLE_INSPECT_RE.search(text))
    asserts_exists = False
    if inspects:
        # `sol = Path("/app/solution/solve.sh")` ... `assert sol.is_file()` is the shape
        # the pool actually has, so track names assigned from the oracle path.
        oracle_vars: set[str] = set()
        for line in text.splitlines():
            m = re.match(r"\s*(\w+)\s*=\s*.*(?:/solution/|solve\.sh)", line)
            if m:
                oracle_vars.add(m.group(1))
            if "assert" in line and EXIST_RE.search(line) and (
                ORACLE_INSPECT_RE.search(line) or any(re.search(r"\b" + re.escape(v) + r"\b", line) for v in oracle_vars)
            ):
                asserts_exists = True
                break
    body_lines = [
        ln for ln in text.splitlines()[1:]
        if ln.strip() and not ln.strip().startswith("#")
    ]
    # Drop a leading docstring (single or triple quoted, possibly multi-line).
    joined = "\n".join(body_lines)
    joined = re.sub(r'^\s*("""|\'\'\')[\s\S]*?\1\s*', "", joined, count=1)
    remaining = [ln for ln in joined.splitlines() if ln.strip()]
    pass_literal = not remaining or all(_PASS_LITERAL_RE.match(ln) for ln in remaining)
    return {
        "uses_mtime": bool(MTIME_RE.search(text)),
        "inspects_oracle": inspects,
        "asserts_oracle_exists": asserts_exists,
        "reads_history": bool(HISTORY_RE.search(text)),
        "pass_literal": pass_literal,
    }


@dataclass(frozen=True)
class TestFn:
    qualname: str          # "test_x" or "TestClass::test_x"
    source: str
    helpers: tuple[str, ...]
    lineno: int


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            fn = child.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "self":
                names.add(fn.attr)
    return names


def split_test_functions(src: str) -> tuple[list[TestFn], dict[str, str], bool]:
    """AST-split a verifier module into its `test*` functions.

    Returns `(functions, helper_sources, parse_ok)`. Class-level tests get the
    `Class::name` qualname because that is how `ctrf.json` names them. Module-level
    (and same-class) helpers are returned by name so the classifier can see a READ
    that lives one call away -- `_read_env_profile()`-style helpers are common.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return [], {}, False
    helpers: dict[str, str] = {}
    functions: list[TestFn] = []

    def source_of(node: ast.AST) -> str:
        seg = ast.get_source_segment(src, node)
        return seg if seg is not None else ""

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("test"):
            helpers[node.name] = source_of(node)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            functions.append(TestFn(node.name, source_of(node), tuple(sorted(_called_names(node))), node.lineno))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("test"):
                        functions.append(TestFn(f"{node.name}::{item.name}", source_of(item),
                                                tuple(sorted(_called_names(item))), item.lineno))
                    else:
                        helpers.setdefault(item.name, source_of(item))
    return functions, helpers, True


def census_test_file(src: str, include_helpers: bool = True) -> dict:
    """Everything E0 records about one `test_state.py`.

    `include_helpers=False` classifies each function on its own source only -- the
    rule the 2026-09-03 quick census used. With helpers, a test whose READ lives in
    a `_read_config()` helper is a recompute check rather than a literal one; on the
    pool that moves ~13 points of functions from existence/literal into
    recompute/rerun. E0 reports both so the delta is visible, not buried.
    """
    functions, helpers, parse_ok = split_test_functions(src)
    rows = []
    for fn in functions:
        text = fn.source
        if include_helpers:
            text += "\n" + "\n".join(helpers[h] for h in fn.helpers if h in helpers)
        row = {"qualname": fn.qualname, "style": classify_style(text), "lineno": fn.lineno}
        row.update(function_flags(text))
        rows.append(row)
    counts = {s: 0 for s in STYLES}
    for row in rows:
        counts[row["style"]] += 1
    n = len(rows)
    if n == 0:
        dominant = "none"
    else:
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], STYLES.index(kv[0])))
        top_style, top_n = ranked[0]
        if top_n * 2 > n:
            dominant = top_style
        else:
            dominant = "mixed"
    return {
        "parse_ok": parse_ok,
        "n_tests": n,
        "test_qualnames": [r["qualname"] for r in rows],
        "style_counts": counts,
        "dominant_style": dominant,
        "existence_only_task": n > 0 and counts["existence_or_nonempty_only"] == n,
        "no_recompute_or_rerun": counts["recompute_from_env"] + counts["rerun_tool_or_script"] == 0,
        "inspects_oracle_script": any(r["inspects_oracle"] for r in rows) or bool(ORACLE_INSPECT_RE.search(src)),
        "asserts_oracle_exists": any(r["asserts_oracle_exists"] for r in rows),
        "reads_history": any(r["reads_history"] for r in rows),
        "pass_literal": any(r["pass_literal"] for r in rows),
        "uses_mtime": any(r["uses_mtime"] for r in rows),
        "uses_random": bool(RANDOMNESS_RE.search(src)),
        "functions": rows,
    }


# ----------------------------------------------------------------------- ctrf / pin

_PARAM_RE = re.compile(r"\[.*\]$")


def ctrf_test_names(ctrf: dict | None) -> frozenset[str]:
    """`results.tests[].name` -> {"test_x", "TestC::test_y"}: file prefix and params stripped."""
    if not isinstance(ctrf, dict):
        return frozenset()
    tests = ((ctrf.get("results") or {}).get("tests")) or []
    names: set[str] = set()
    for t in tests:
        name = str((t or {}).get("name") or "")
        if not name:
            continue
        name = _PARAM_RE.sub("", name)
        parts = name.split("::")
        if parts and parts[0].endswith(".py"):
            parts = parts[1:]
        if parts:
            names.add("::".join(parts))
    return frozenset(names)


def ctrf_summary(ctrf: dict | None) -> dict:
    tests = ((ctrf or {}).get("results") or {}).get("tests") or []
    summary = ((ctrf or {}).get("results") or {}).get("summary") or {}
    total = int(summary.get("tests") or len(tests) or 0)
    passed = int(summary.get("passed") or sum(1 for t in tests if (t or {}).get("status") == "passed"))
    return {"total": total, "passed": passed,
            "failed_names": [str(t.get("name")) for t in tests if (t or {}).get("status") != "passed"]}


@dataclass(frozen=True)
class Pin:
    task_id: str | None
    kind: str              # unique | ambiguous | unmatched | no_ctrf
    n_candidates: int
    tie_break: str | None = None


_CWD_RE = re.compile(r"root@[0-9a-f-]+:(\S+?)#")


def terminal_cwd(prompt: str) -> str | None:
    """The cwd in Terminus-2's `root@<host>:<cwd># ` prompt as recorded in step 0, or None."""
    hits = _CWD_RE.findall(prompt or "")
    if not hits:
        return None
    cwd = hits[-1]
    return "/root" if cwd == "~" else cwd.replace("~", "/root", 1) if cwd.startswith("~/") else cwd


def pin_trajectory(names: frozenset[str], candidates: dict[str, frozenset[str]],
                   cwd_hint: str | None = None,
                   candidate_workdirs: dict[str, str | None] | None = None) -> Pin:
    """Which task variant produced a trajectory whose verifier ran exactly `names`."""
    if not names:
        return Pin(None, "no_ctrf", 0)
    hits = sorted(tid for tid, tnames in candidates.items() if tnames == names)
    if len(hits) == 1:
        return Pin(hits[0], "unique", 1)
    if not hits:
        return Pin(None, "unmatched", 0)
    if cwd_hint and candidate_workdirs:
        narrowed = [tid for tid in hits if (candidate_workdirs.get(tid) or "") == cwd_hint]
        if len(narrowed) == 1:
            return Pin(narrowed[0], "unique", len(hits), tie_break="workdir")
    return Pin(None, "ambiguous", len(hits))


# ------------------------------------------------------------------- Dockerfile

_CONTINUATION_RE = re.compile(r"\\\n")


def dockerfile_instructions(text: str) -> list[tuple[int, str, str]]:
    """`(line_no, INSTRUCTION, argument)` with `\\`-continuations joined.

    Heredoc bodies (`<<EOF` ... `EOF`) are kept inside the argument of the
    instruction that opened them, separated by newlines, so a caller can read
    them back.
    """
    out: list[tuple[int, str, str]] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        start = i + 1
        if not raw.strip() or raw.lstrip().startswith("#"):
            i += 1
            continue
        buf = raw
        while buf.rstrip().endswith("\\") and i + 1 < n:
            i += 1
            buf = buf.rstrip()[:-1] + "\n" + lines[i]
        m = re.match(r"^\s*([A-Za-z]+)\s+([\s\S]*)$", buf)
        i += 1
        if not m:
            continue
        instr, arg = m.group(1).upper(), m.group(2)
        # BuildKit / shell heredocs: collect until the terminator line.
        for hm in re.finditer(r"<<-?\s*(['\"]?)(\w+)\1", arg):
            term = hm.group(2)
            body: list[str] = []
            while i < n and lines[i].strip() != term:
                body.append(lines[i])
                i += 1
            if i < n:
                i += 1  # skip the terminator
            arg = arg + "\n" + "\n".join(body) + f"\n{term}"
        out.append((start, instr, arg))
    return out


def dockerfile_workdir(text: str) -> str | None:
    wd = None
    for _, instr, arg in dockerfile_instructions(text):
        if instr == "WORKDIR":
            wd = arg.strip().split()[0] if arg.strip() else wd
    return wd


def _split_shell_commands(cmd: str) -> list[str]:
    """Split on `&&`, `;`, `||` and newlines outside quotes. Heredoc bodies stay attached."""
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(cmd):
        ch = cmd[i]
        if quote:
            buf.append(ch)
            if ch == quote and (i == 0 or cmd[i - 1] != "\\"):
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif cmd.startswith("&&", i) or cmd.startswith("||", i):
            parts.append("".join(buf))
            buf = []
            i += 1
        elif ch in (";", "\n"):
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _unescape_printf(payload: str, is_printf: bool) -> str:
    if not is_printf:
        return payload
    return (payload.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\"))


_REDIRECT_RE = re.compile(r"(?<![<>])>{1,2}\s*(['\"]?)(/[^\s'\"]+)\1")
_ECHO_RE = re.compile(r"^\s*(echo|printf)\s+(-e\s+|-n\s+|-en\s+|-ne\s+)?(?P<q>['\"])(?P<body>[\s\S]*?)(?P=q)\s*>{1,2}\s*['\"]?(?P<path>/[^\s'\"]+)")
_HEREDOC_RE = re.compile(r"cat\s*<<-?\s*(['\"]?)(\w+)\1\s*>{1,2}\s*['\"]?(?P<path>/[^\s'\"]+)['\"]?\s*\n(?P<body>[\s\S]*?)\n\2\s*$")
_COPY_HEREDOC_RE = re.compile(r"^<<-?\s*(['\"]?)(\w+)\1\s+(?P<path>/\S+)\s*\n(?P<body>[\s\S]*?)\n\2\s*$")


@dataclass(frozen=True)
class Fixture:
    path: str
    source_kind: str       # run_echo | run_printf | run_heredoc | copy_heredoc | copy_file
    line_no: int
    content: str | None    # None when the Dockerfile writes it but the bytes are not recoverable


def dockerfile_fixtures(text: str, env_files: dict[str, str | bytes] | None = None) -> list[Fixture]:
    """Files the Dockerfile itself writes into the image, with their content when knowable.

    Handles `RUN echo '...' > P`, `RUN printf '...' > P`, `RUN cat <<EOF > P`,
    BuildKit `COPY <<EOF P`, and `COPY src P` where `src` is a file under
    `environment/` (`env_files` maps its relative path to bytes). A `RUN` with a
    variable in the payload is recorded with `content=None`: still a fixture, not
    mutable by this planner.
    """
    env_files = env_files or {}
    fixtures: list[Fixture] = []
    workdir = "/"
    for line_no, instr, arg in dockerfile_instructions(text):
        if instr == "WORKDIR":
            workdir = arg.strip().split()[0] if arg.strip() else workdir
            continue
        if instr == "COPY" or instr == "ADD":
            m = _COPY_HEREDOC_RE.match(arg.strip())
            if m:
                fixtures.append(Fixture(m.group("path"), "copy_heredoc", line_no, m.group("body") + "\n"))
                continue
            toks = [t for t in shlex.split(arg.split("\n", 1)[0], posix=True) if not t.startswith("--")]
            if len(toks) < 2:
                continue
            *srcs, dst = toks
            for src in srcs:
                src_rel = src.lstrip("./")
                if src_rel in env_files:
                    dst_path = dst if not dst.endswith("/") else dst + PurePosixPath(src_rel).name
                    if not dst_path.startswith("/"):
                        dst_path = str(PurePosixPath(workdir) / dst_path)
                    fixtures.append(Fixture(dst_path, "copy_file", line_no, _maybe_text(env_files[src_rel])))
                else:
                    prefix = src_rel.rstrip("/") + "/"
                    for rel, payload in env_files.items():
                        if rel.startswith(prefix):
                            dst_path = str(PurePosixPath(dst) / rel[len(prefix):])
                            if not dst_path.startswith("/"):
                                dst_path = str(PurePosixPath(workdir) / dst_path)
                            fixtures.append(Fixture(dst_path, "copy_file", line_no, _maybe_text(payload)))
            continue
        if instr != "RUN":
            continue
        hm = _HEREDOC_RE.search(arg)
        if hm:
            fixtures.append(Fixture(hm.group("path"), "run_heredoc", line_no, hm.group("body") + "\n"))
        for cmd in _split_shell_commands(arg):
            em = _ECHO_RE.match(cmd)
            if em:
                body = em.group("body")
                if "$" in body or "`" in body:
                    fixtures.append(Fixture(em.group("path"), "run_" + em.group(1), line_no, None))
                else:
                    is_printf = em.group(1) == "printf" or bool(em.group(2) and "e" in em.group(2))
                    content = _unescape_printf(body, is_printf)
                    if em.group(1) == "echo" and not (em.group(2) and "n" in em.group(2)):
                        content += "\n"
                    fixtures.append(Fixture(em.group("path"), "run_" + em.group(1), line_no, content))
                continue
            if hm and hm.group("path") in cmd:
                continue
            for rm in _REDIRECT_RE.finditer(cmd):
                path = rm.group(2)
                if path.startswith("/dev/") or path.startswith("/proc/"):
                    continue
                if not any(f.path == path and f.line_no == line_no for f in fixtures):
                    fixtures.append(Fixture(path, "run_redirect", line_no, None))
    # last write wins for content; keep first line_no
    by_path: dict[str, Fixture] = {}
    for f in fixtures:
        prev = by_path.get(f.path)
        if prev is None:
            by_path[f.path] = f
        else:
            by_path[f.path] = Fixture(f.path, f.source_kind, prev.line_no, f.content)
    return list(by_path.values())


def _maybe_text(content) -> str | None:
    if isinstance(content, bytes):
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(content, str):
        if "\x00" in content:
            return None
        return content
    return None


# --------------------------------------------------------------- oracle / solve.sh

_WRITE_TARGET_RE = re.compile(
    r"(?:>{1,2}\s*|tee\s+(?:-a\s+)?|sed\s+-i\S*\s+(?:[^\n;&|]*\s)?|cp\s+(?:[^\n;&|]*\s)|mv\s+(?:[^\n;&|]*\s)"
    r"|rm\s+(?:-\w+\s+)*|truncate\s+(?:-s\s*\S+\s+)?|install\s+(?:[^\n;&|]*\s))['\"]?(?P<path>/[^\s'\"]+)"
)


def solve_written_paths(solve_sh: str) -> set[str]:
    """Absolute paths solve.sh writes, appends to, edits in place, moves onto, or removes."""
    return {m.group("path") for m in _WRITE_TARGET_RE.finditer(solve_sh or "")}


def references_path(text: str, path: str) -> bool:
    """True when `text` mentions the path or its basename (RST oracles often `find` it)."""
    if not text:
        return False
    if path in text:
        return True
    base = PurePosixPath(path).name
    return bool(base) and bool(re.search(r"(?<![\w.-])" + re.escape(base) + r"(?![\w-])", text))


_CONTROL_TOKENS = {"fi", "done", "esac", "}", "{", "then", "else", "do", "elif", ")", "(", "in"}


def oracle_command_count(solve_sh: str) -> int:
    """Non-blank, non-comment lines minus pure control tokens: the E2 normaliser."""
    n = 0
    for line in (solve_sh or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.rstrip(";") in _CONTROL_TOKENS or s.startswith("set -") or s.startswith("set +"):
            continue
        n += 1
    return n


_NETWORK_RE = re.compile(
    r"\b(curl|wget)\b|pip3?\s+install|apt(-get)?\s+(install|update)|git\s+clone|npm\s+(install|ci)\b"
    r"|go\s+(get|install)\b|cargo\s+install|gem\s+install|conda\s+install|yum\s+install|apk\s+add"
    r"|https?://"
)


def oracle_needs_network(solve_sh: str) -> bool:
    text = "\n".join(ln for ln in (solve_sh or "").splitlines() if not ln.strip().startswith("#"))
    return bool(_NETWORK_RE.search(text))


def toml_allow_internet(task_toml: str) -> str:
    m = re.search(r"^\s*allow_internet\s*=\s*(true|false)", task_toml or "", re.M | re.I)
    return m.group(1).lower() if m else "absent"


def toml_agent_timeout(task_toml: str) -> float | None:
    """`[agent] timeout_sec`, or None."""
    section = None
    for line in (task_toml or "").splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1].strip()
            continue
        if section == "agent":
            m = re.match(r"timeout_sec\s*=\s*([0-9.]+)", s)
            if m:
                return float(m.group(1))
    return None


def test_runner(test_sh: str) -> str:
    t = test_sh or ""
    if re.search(r"^\s*uvx\b", t, re.M):
        return "uvx"
    if re.search(r"python3?\s+-m\s+pytest", t):
        return "python -m pytest"
    if re.search(r"^\s*pytest\b", t, re.M):
        return "pytest"
    return "other"


# --------------------------------------------------------------------- mutation

_NUM_RE = re.compile(r"(?<![\w./-])-?\d+(?:\.\d+)?(?![\w./-])")
_KV_RE = re.compile(r"^(\s*[\"']?[A-Za-z_][\w.-]*[\"']?\s*[=:]\s*)(.*)$")
_ALPHA_RE = re.compile(r"(?<![\w/.-])[A-Za-z]{2,}(?![\w/.-])")


@dataclass(frozen=True)
class Mutation:
    path: str
    source_kind: str
    line_no: int          # 1-based line in the fixture content
    before: str
    after: str
    kind: str             # numeric_in_domain | alpha_value | alpha_field
    new_content: str
    dockerfile_line: int
    ref_strength: str = "unknown"   # both | tests_only | solve_only -- who reads the fixture

    def to_json(self) -> dict:
        return asdict(self)


KIND_RANK = {"numeric_in_domain": 0, "alpha_value": 1, "alpha_field": 2}
REF_RANK = {"both": 0, "tests_only": 1, "solve_only": 2}


def mutate_content(content: str) -> tuple[str, int, str, str, str] | None:
    """One token change: `(new_content, line_no, before, after, kind)` or None.

    Numeric data tokens win (`37 -> 38`), then an alphabetic value after `=`/`:`
    (`modern -> modernx`), then the last alphanumeric token of the first data line.
    Keys, paths and the first token of `KEY=VALUE` are never touched, so the mutation
    changes what the task's answer should be rather than whether the task parses.
    """
    lines = content.splitlines()
    data_idx = [i for i, ln in enumerate(lines) if ln.strip() and not ln.lstrip().startswith("#")]
    if not data_idx:
        return None

    def value_span(ln: str) -> int:
        m = _KV_RE.match(ln)
        return len(m.group(1)) if m else 0

    for i in data_idx:
        ln = lines[i]
        start = value_span(ln)
        for m in _NUM_RE.finditer(ln, start):
            tok = m.group(0)
            if "." in tok:
                head, tail = tok.split(".", 1)
                new = f"{int(head) + 1}.{tail}"
            else:
                new = str(int(tok) + 1)
            new_line = ln[:m.start()] + new + ln[m.end():]
            new_lines = list(lines)
            new_lines[i] = new_line
            return _rejoin(content, new_lines), i + 1, tok, new, "numeric_in_domain"
    for i in data_idx:
        ln = lines[i]
        start = value_span(ln)
        if start == 0:
            continue
        m = _ALPHA_RE.search(ln, start)
        if m and m.group(0).lower() not in ("true", "false", "null", "none", "yes", "no"):
            new = m.group(0) + "x"
            new_lines = list(lines)
            new_lines[i] = ln[:m.start()] + new + ln[m.end():]
            return _rejoin(content, new_lines), i + 1, m.group(0), new, "alpha_value"
    i = data_idx[0]
    ln = lines[i]
    toks = list(re.finditer(r"[A-Za-z][A-Za-z0-9_]*", ln))
    toks = [t for t in toks if "/" not in ln[max(0, t.start() - 1):t.end() + 1]]
    long_toks = [t for t in toks if len(t.group(0)) >= 2]
    toks = long_toks or toks
    if toks:
        m = toks[-1]
        new = m.group(0) + "x"
        new_lines = list(lines)
        new_lines[i] = ln[:m.start()] + new + ln[m.end():]
        return _rejoin(content, new_lines), i + 1, m.group(0), new, "alpha_field"
    return None


def _rejoin(original: str, lines: list[str]) -> str:
    text = "\n".join(lines)
    if original.endswith("\n"):
        text += "\n"
    return text


NA_REASONS = (
    "compose_task",
    "no_fixture_written",
    "fixture_not_referenced",
    "fixture_written_by_solve",
    "undecodable",
    "no_mutable_token",
)


def plan_mutation(dockerfile: str, solve_sh: str, test_py: str,
                  env_files: dict[str, str | bytes] | None = None,
                  seed: int = 0, compose: bool = False,
                  max_bytes: int = 64 * 1024) -> tuple[Mutation | None, str | None]:
    """Choose the fixture and the token. Returns `(mutation, None)` or `(None, na_reason)`.

    Rank: referenced by both solve.sh and the verifier > verifier only > solve only;
    then a numeric in-domain change > an alphabetic value > an alphabetic field;
    then smaller content. The seed only breaks exact ties, so the plan is a function
    of the task, not of the run. `ref_strength` travels with the plan because a
    `(oracle_mut=1, artifacts_mut=1)` outcome means "verifier ignores the fixture"
    only when the fixture is actually read -- on a solve-only reference it may just
    mean the mutation did not matter.
    """
    if compose:
        return None, "compose_task"
    fixtures = dockerfile_fixtures(dockerfile, env_files)
    if not fixtures:
        return None, "no_fixture_written"
    written = solve_written_paths(solve_sh)
    reasons: list[str] = []
    ranked: list[tuple[tuple, Mutation]] = []
    for f in fixtures:
        in_solve = references_path(solve_sh, f.path)
        in_tests = references_path(test_py, f.path)
        if not (in_solve or in_tests):
            reasons.append("fixture_not_referenced")
            continue
        if f.path in written:
            reasons.append("fixture_written_by_solve")
            continue
        if f.content is None or len(f.content.encode("utf-8", "replace")) > max_bytes:
            reasons.append("undecodable")
            continue
        result = mutate_content(f.content)
        if result is None:
            reasons.append("no_mutable_token")
            continue
        new_content, line_no, before, after, kind = result
        strength = "both" if (in_solve and in_tests) else "tests_only" if in_tests else "solve_only"
        rank = (REF_RANK[strength], KIND_RANK[kind], len(f.content),
                hashlib.sha256(f"{seed}:{f.path}".encode()).hexdigest())
        ranked.append((rank, Mutation(f.path, f.source_kind, line_no, before, after, kind,
                                      new_content, f.line_no, strength)))
    if ranked:
        return min(ranked, key=lambda x: x[0])[1], None
    for reason in NA_REASONS:
        if reason in reasons:
            return None, reason
    return None, "no_fixture_written"


def patch_dockerfile(dockerfile: str, mutation: Mutation) -> str:
    """Append ONE layer that rewrites the fixture. Placed before the final CMD/ENTRYPOINT.

    One appended layer keeps every cached layer of the original image, so the
    mutated image costs one `RUN` instead of a full apt rebuild. base64 sidesteps
    shell quoting of arbitrary fixture bytes; `>` on an existing file keeps its mode.
    If the Dockerfile switched to a non-root USER, the write runs as root and the
    USER is restored afterwards.
    """
    payload = base64.b64encode(mutation.new_content.encode("utf-8")).decode("ascii")
    quoted_path = shlex.quote(mutation.path)
    layer = [
        f"# rst-probe: mutate {mutation.path} line {mutation.line_no}: {mutation.before!r} -> {mutation.after!r}",
    ]
    lines = dockerfile.splitlines()
    last_user = None
    for _, instr, arg in dockerfile_instructions(dockerfile):
        if instr == "USER":
            last_user = arg.strip()
    if last_user and last_user not in ("root", "0"):
        layer.append("USER root")
    layer.append(f"RUN mkdir -p {shlex.quote(str(PurePosixPath(mutation.path).parent))} && "
                 f"echo '{payload}' | base64 -d > {quoted_path}")
    if last_user and last_user not in ("root", "0"):
        layer.append(f"USER {last_user}")
    insert_at = len(lines)
    for idx in range(len(lines) - 1, -1, -1):
        s = lines[idx].strip().upper()
        if s.startswith("CMD ") or s.startswith("ENTRYPOINT ") or s.startswith("CMD[") or s.startswith("ENTRYPOINT["):
            insert_at = idx
            break
    new_lines = lines[:insert_at] + layer + lines[insert_at:]
    return "\n".join(new_lines) + "\n"


# ------------------------------------------------------------- artifact delta

def parse_listing(text: str) -> dict[str, tuple[int, float]]:
    """`find -printf '%p\\t%s\\t%T@\\n'` output -> {path: (size, mtime)}."""
    out: dict[str, tuple[int, float]] = {}
    for line in (text or "").splitlines():
        parts = line.rsplit("\t", 2)
        if len(parts) != 3:
            continue
        path, size, mtime = parts
        try:
            out[path] = (int(size), float(mtime))
        except ValueError:
            continue
    return out


def artifact_delta(before: str, after: str) -> tuple[list[str], list[str]]:
    """`(created, modified)` paths, sorted. A file is modified when size or mtime changed."""
    b = parse_listing(before)
    a = parse_listing(after)
    created = sorted(p for p in a if p not in b)
    modified = sorted(p for p in a if p in b and a[p] != b[p])
    return created, modified


# ----------------------------------------------------------- instruction paths

_INSTRUCTION_PATH_RE = re.compile(
    r"(?<![\w/])/(?:app|root|home|tmp|etc|opt|srv|var|data|workspace|repo|project|usr/local|mnt)"
    r"(?:/[\w.@+-]+)+"
)
_EXCLUDED_PREFIXES = ("/tests", "/logs", "/solution")


def instruction_paths(text: str) -> list[str]:
    """Absolute paths an instruction names, deduplicated, trailing punctuation stripped."""
    out: list[str] = []
    for m in _INSTRUCTION_PATH_RE.finditer(text or ""):
        p = m.group(0).rstrip(".,;:)]}'\"`")
        while p.endswith(".") or p.endswith(","):
            p = p[:-1]
        if any(p.startswith(x) for x in _EXCLUDED_PREFIXES):
            continue
        if p.endswith("/"):
            p = p[:-1]
        if p and p not in out:
            out.append(p)
    return out


def plausible_content(path: str) -> str:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".json":
        return "{}\n"
    if suffix in (".yaml", ".yml"):
        return "[]\n"
    if suffix == ".csv":
        return "name,value\n"
    return "ok\n"


# ------------------------------------------------------------- command classes

_WRITE_CMD_RE = re.compile(
    r"^\s*(?:sudo\s+)?(?:echo|printf|tee|touch|mkdir|cp|mv|ln|install|dd|truncate|chmod|chown)\b"
    r"|^\s*(?:sudo\s+)?cat\s*(?:<<|>)|>\s*/|>>\s*/"
)
_READ_CMD_RE = re.compile(
    r"^\s*(?:sudo\s+)?(?:cat|head|tail|less|more|grep|egrep|rg|awk|cut|sort|uniq|wc|find|ls|stat|file|jq|xxd"
    r"|od|strings|diff|tree|readlink|realpath|column|nl|tac|sed)\b"
)
_TAMPER_RE = re.compile(r"/tests\b|test_state\.py|/logs/verifier|reward\.(?:txt|json)|/solution\b|solve\.sh|\bchattr\b")
_SPECIAL_KEYS = {"C-c", "C-d", "C-z", "C-l", "Escape", "Enter", "Up", "Down", "Left", "Right", "Tab", "q", "Q"}
_BARE_INTERACTIVE_RE = re.compile(r"^\s*(?:vim?|vi|nano|emacs|less|more|top|htop|python3?|node|irb|psql|mysql|sqlite3|bash|sh|ssh\s+\S+)\s*$")


def command_class(keystroke: str) -> str:
    """write | read | other, for one keystroke block as the agent sent it."""
    s = keystroke.strip()
    if not s:
        return "other"
    first = s.splitlines()[0]
    if _WRITE_CMD_RE.search(first) or _WRITE_CMD_RE.search(s):
        # `cat file` is a read; `cat > file`, `cat <<EOF` are writes -- already handled by the regex order.
        if re.match(r"^\s*(?:sudo\s+)?cat\s+(?!<<|>)", first) and ">" not in first and "<<" not in first:
            return "read"
        return "write"
    if _READ_CMD_RE.search(first):
        return "read"
    return "other"


def is_write_only(keystrokes: list[str]) -> bool:
    """Every non-empty keystroke writes and none reads: the pure "type the answer" shape."""
    classes = [command_class(k) for k in keystrokes if k.strip()]
    return bool(classes) and all(c == "write" for c in classes)


def tamper_flags(keystrokes: list[str]) -> list[str]:
    return sorted({m.group(0) for k in keystrokes for m in _TAMPER_RE.finditer(k)})


def interactive_flags(keystrokes: list[str]) -> list[str]:
    flags: list[str] = []
    for k in keystrokes:
        s = k.strip()
        if s in _SPECIAL_KEYS:
            flags.append(f"key:{s}")
        elif _BARE_INTERACTIVE_RE.match(s):
            flags.append(f"bare:{s.split()[0]}")
    return flags


def cheap_rule(n_commands: int, oracle_n_cmds: int, max_cmds: int = 3, max_ratio: float = 0.1) -> bool:
    return n_commands <= max_cmds or (n_commands / max(1, oracle_n_cmds)) <= max_ratio


# ------------------------------------------------------------ outcome classes

ARMS = ("nop", "oracle", "touch", "oracle_mut", "artifacts_mut")


def classify_task_outcome(arms: dict[str, tuple[float | None, bool]], mutation_applied: bool) -> str:
    """The 2x2 cell for one task from `{arm: (reward, scorable)}`.

    Only interpretable when the controls hold: `nop` scored 0 (no vacuous pass),
    `oracle` scored 1 (the local harness reproduces RST's acceptance), the
    mutation actually landed, and every arm is scorable (an infra failure is
    "unmeasured", never a 0).
    """
    for arm in ARMS:
        if arm not in arms:
            return f"uninterpretable:missing:{arm}"
        reward, scorable = arms[arm]
        if not scorable or reward is None:
            return f"uninterpretable:infra:{arm}"
    if arms["nop"][0] >= 1.0:
        return "uninterpretable:nop_passes"
    if arms["oracle"][0] < 1.0:
        return "uninterpretable:oracle_fails_baseline"
    if not mutation_applied:
        return "uninterpretable:mutation_not_applied"
    om = arms["oracle_mut"][0] >= 1.0
    am = arms["artifacts_mut"][0] >= 1.0
    if om and not am:
        return "parametric_ok"
    if om and am:
        return "verifier_fixture_insensitive"
    if not om and not am:
        return "oracle_brittle"
    return "oracle_brittle_and_verifier_constant"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)
