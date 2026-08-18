"""Code that must be identical in eval and in RL.

Anything in here defines a measurement, not a convenience. The rule of thumb for
what belongs: if two copies of it could drift and make two numbers
non-comparable, it goes here.

`scripts/*.py` cannot import this by package name on their own -- they are run as
`python scripts/06_eval.py`, so `sys.path[0]` is `scripts/`. They add the repo
root explicitly:

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

`rl/` and `verl_backend/` are imported with the repo root already on PYTHONPATH
(see 12_run_grpo.sh), so a plain `from rst_common.harbor import ...` works there.
"""
