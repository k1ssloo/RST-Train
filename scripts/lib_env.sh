# Shared environment entry. SOURCE this file, never execute it.
#
# WHY THIS FILE EXISTS
#   scripts/01b_setup_env_verl.sh runs `micromamba activate rstverl` in ITS OWN
#   process. A child script cannot change its parent's PATH, so `bash
#   scripts/20_run_all.sh` used to build the environment and then run every
#   subsequent python stage under whatever interpreter the login shell happened to
#   have. Nothing in the repo activated it, and the setup script's closing
#   "ENV READY: micromamba activate rstverl" is a line of text for a human, not an
#   instruction a launcher follows.
#
#   The failure did not land at the env stage. It landed two stages later as a
#   ModuleNotFoundError from 03_build_sft_data.py, which reads like a code bug and
#   is not one.
#
#   So: the setup scripts now record how to re-enter the env they built
#   ($BASE_FOLDER/env-<name>.sh), and rst_enter_env sources that and then PROVES it
#   worked. Verification is the point. Guessing where micromamba installed itself is
#   guesswork; `find_spec("torch")` is not.
#
# USAGE
#   source "$REPO_DIR/scripts/lib_env.sh"
#   rst_bootstrap_python          # before the env exists: guarantee *a* python3
#   rst_enter_env                 # after it exists: enter it or fail with instructions

# The four imports every python stage in this repo needs. Deliberately checked with
# find_spec rather than a real import: locating them costs milliseconds, importing
# torch costs seconds and would run on every stage.
RST_ENV_REQUIRE="${RST_ENV_REQUIRE:-torch transformers pandas pyarrow}"

rst_has_modules() {  # rst_has_modules <module...>  -> 0 if every one is importable
  python - "$@" <<'PY' >/dev/null 2>&1
import importlib.util as u
import sys

missing = []
for name in sys.argv[1:]:
    try:
        if u.find_spec(name) is None:
            missing.append(name)
    except Exception:
        missing.append(name)
sys.exit(1 if missing else 0)
PY
}

rst_missing_modules() {  # print the ones that are absent, for error messages
  python - "$@" <<'PY' 2>/dev/null
import importlib.util as u
import sys

out = []
for name in sys.argv[1:]:
    try:
        if u.find_spec(name) is None:
            out.append(name)
    except Exception:
        out.append(name)
print(" ".join(out))
PY
}

rst_env_ok() {  # is the current `python` the training env?
  # shellcheck disable=SC2086
  rst_has_modules $RST_ENV_REQUIRE
}

# Ubuntu 22.04 and most slim images ship `python3` with no `python` alias
# (python-is-python3 is not installed by default). Every script here calls `python`,
# and the pre-env stages -- model_registry.py and 00_preflight.sh, both stdlib-only --
# run before any env exists. So make `python` resolve to something before then. The
# shim goes in a directory of its own so that entering the real env later takes
# precedence over it rather than fighting it.
rst_bootstrap_python() {
  command -v python >/dev/null 2>&1 && return 0
  if ! command -v python3 >/dev/null 2>&1; then
    echo "no python and no python3 on PATH. This repo needs python3 >= 3.10 just to" >&2
    echo "read configs/models.json before the training env exists. Install one" >&2
    echo "(apt-get install -y python3, or load a module) and re-run." >&2
    return 1
  fi
  local shim="${BASE_FOLDER:-$PWD}/.bootstrap-bin"
  mkdir -p "$shim"
  ln -sfn "$(command -v python3)" "$shim/python"
  export PATH="$shim:$PATH"
  echo "=== no \`python\` on PATH; shimmed $shim/python -> $(command -v python3)"
  echo "    (pre-env stages only; entering the training env overrides this)"
}

# Enter the training environment. Order of attempts is cheapest-and-most-certain
# first, and every attempt is followed by the same verification.
rst_enter_env() {  # rst_enter_env [env_name]
  local want="${1:-${ENV_NAME:-rstverl}}"
  local stub="${BASE_FOLDER:-}/env-$want.sh"

  # 0. Already in it? Covers the normal case (20_run_all.sh activated, then invoked
  #    30_run_sft_verl.sh as a child) and the operator who activated it by hand, and
  #    a prebaked image that has the stack installed system-wide with no env at all.
  if rst_env_ok; then
    echo "=== env: $(command -v python)  [$RST_ENV_REQUIRE all present]"
    return 0
  fi

  # micromamba's and conda's shell hooks reference their own internal variables and
  # are not written to survive `set -u`. Ours is, so drop it for the duration and put
  # it back exactly as it was.
  local restore_u
  restore_u="$(shopt -po nounset 2>/dev/null || true)"
  set +u

  # 1. The stub the setup script wrote. It knows the answer because it was inside the
  #    env when it recorded it -- sys.prefix, not a guess.
  if [[ -f "$stub" ]]; then
    echo "=== entering the env recorded by the setup script: $stub"
    # shellcheck disable=SC1090
    source "$stub" || true
    if rst_env_ok; then
      eval "$restore_u"
      echo "=== env: $(command -v python)"
      return 0
    fi
    echo "=== $stub did not produce a usable env; trying the package managers" >&2
  fi

  # 2. No stub: an env built by hand, or built before this file existed. Look for the
  #    manager in the places its own installer uses.
  local mm=""
  local cand
  for cand in "${MAMBA_EXE:-}" micromamba "$HOME/.local/bin/micromamba" \
              "$HOME/bin/micromamba" "$HOME/micromamba/bin/micromamba"; do
    [[ -n "$cand" ]] || continue
    if command -v "$cand" >/dev/null 2>&1; then mm="$cand"; break; fi
  done
  if [[ -n "$mm" ]]; then
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "$("$mm" shell hook --shell bash 2>/dev/null)" || true
    micromamba activate "$want" >/dev/null 2>&1 || true
    if rst_env_ok; then
      eval "$restore_u"
      echo "=== env: $(command -v python)  (micromamba $want)"
      return 0
    fi
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "$want" >/dev/null 2>&1 || true
    if rst_env_ok; then
      eval "$restore_u"
      echo "=== env: $(command -v python)  (conda $want)"
      return 0
    fi
  fi

  eval "$restore_u"
  local missing
  # shellcheck disable=SC2086
  missing="$(rst_missing_modules $RST_ENV_REQUIRE)"
  cat >&2 <<EOF

=== CANNOT ENTER THE TRAINING ENVIRONMENT ($want)
    python  : $(command -v python 2>/dev/null || echo "(none on PATH)")
    missing : ${missing:-"(python itself failed to run)"}
    stub    : $stub $([[ -f "$stub" ]] && echo "(present, but activating it did not help)" || echo "(absent)")

    This is an environment problem, not a code problem. One of these is true:

    a) the env was never built     -> bash scripts/01b_setup_env_verl.sh
                                      (BACKEND=slime? scripts/01_setup_env.sh)
    b) it was built somewhere else -> enter it yourself, then re-run with
                                      SKIP_STAGES="env":
                                        micromamba activate $want
                                        SKIP_STAGES="env" bash scripts/20_run_all.sh
    c) it was built by an older copy of these scripts, so no stub exists -> either
       (b), or re-run the setup script once to write $stub. It is idempotent:
       micromamba create on an existing env is a no-op and pip re-resolves in place.

    If the env is genuinely fine and the CHECK is wrong for it -- a different backend
    needs a different set -- narrow the requirement instead of disabling it:
      RST_ENV_REQUIRE="torch transformers" bash scripts/20_run_all.sh

    Do NOT work around this by pip-installing torch into the system interpreter. The
    torch CUDA build is chosen from the driver version in 01b_setup_env_verl.sh, and
    a mismatched one fails later at cuda init looking unrelated.
EOF
  return 1
}

# Record how to re-enter the env we are currently inside. Called at the end of a
# setup script, from within the activated env -- sys.prefix is then the ground truth.
rst_write_env_stub() {  # rst_write_env_stub <env_name> <out_path>
  local name="$1" out="$2" prefix mamba_exe
  prefix="$(python -c 'import sys; print(sys.prefix)')" || return 1
  mamba_exe="${MAMBA_EXE:-}"
  [[ -x "$mamba_exe" ]] || mamba_exe="$(command -v micromamba 2>/dev/null || true)"
  [[ -x "$mamba_exe" ]] || mamba_exe="$HOME/.local/bin/micromamba"
  mkdir -p "$(dirname "$out")"
  cat > "$out" <<EOF
# Written by $(basename "${BASH_SOURCE[1]:-a setup script}") at $(date -Is).
# SOURCE this to enter the training env; do not execute it:
#
#     source $out
#
# It exists because \`micromamba activate\` inside a child process cannot change its
# parent's PATH -- see scripts/lib_env.sh. The launchers source it automatically.
export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-${MAMBA_ROOT_PREFIX:-$HOME/micromamba}}"
RST_ENV_NAME="$name"
RST_ENV_PREFIX="$prefix"
RST_MAMBA_EXE="$mamba_exe"

_rst_restore_u="\$(shopt -po nounset 2>/dev/null || true)"
set +u
if [[ -x "\$RST_MAMBA_EXE" ]]; then
  eval "\$("\$RST_MAMBA_EXE" shell hook --shell bash 2>/dev/null)" || true
  micromamba activate "\$RST_ENV_NAME" >/dev/null 2>&1 || true
fi
# Belt and braces. The hook needs a shell rc that may not be sourced in a
# non-interactive job; this does not. \$RST_ENV_PREFIX/bin first on PATH is the thing
# that actually has to be true, and it was read from sys.prefix inside the env.
if [[ -x "\$RST_ENV_PREFIX/bin/python" \\
      && "\$(command -v python 2>/dev/null)" != "\$RST_ENV_PREFIX/bin/python" ]]; then
  export PATH="\$RST_ENV_PREFIX/bin:\$PATH"
  export CONDA_PREFIX="\$RST_ENV_PREFIX"
fi
eval "\$_rst_restore_u"; unset _rst_restore_u
EOF
  echo "=== wrote $out  (source it to re-enter '$name'; the launchers do this for you)"
}
