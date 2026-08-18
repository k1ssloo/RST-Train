# Agentic GRPO on RST terminal tasks — code + rollout prerequisites

Companion to `PLAN.md` (SFT). Read that first. Priority is still SFT-first; this
exists so the RL work is real code with measured prerequisites instead of an
outline.

## Status — be honest about what is and isn't verified

| piece | state |
|---|---|
| Task selection + materialization (`scripts/10_build_rl_taskset.py`) | ✅ **run locally**: 5,140 tasks materialized, 999 groups, 15 s, 378 MB |
| Verifier-leak guard | ✅ **run**: 0 leaks across 5,140 build contexts |
| GRPO rollout function (`rl/generate.py`) | ⚠️ written against the real slime API; **never executed** |
| Image prebuild (`scripts/11_prebuild_images.py`) | ⚠️ written; **no image has been built yet** |
| GRPO launcher (`scripts/12_run_grpo.sh`) | ⚠️ written; needs cluster |
| **DPO fallback** (`DPO_PLAN.md`, `scripts/33_run_dpo.sh`) | ✅ **run end to end** on 0.8B: 2,673 pairs built, step-0 loss = log 2 exactly. Needs **no container** |

Everything on this page needs a sandbox. If there is nowhere to run task containers,
`DPO_PLAN.md` is the path that still trains something from this data — off-policy, on
logged trajectories, no container and no privilege. It is a fallback, not a
substitute: it reweights behaviour already in other policies' trajectories and cannot
discover a strategy none of them used. `20_run_all.sh` picks it automatically
(`RUN_DPO=auto`) exactly when `RUN_RL=1` and the sandbox check fails.

`rl/generate.py` is written against APIs I read in slime's source
(`slime/agent/adapters/common.py`, `slime/utils/types.py`,
`examples/coding_agent_rl/generate.py`), not against docs. But nothing in it has
run. Treat every claim below marked ⚠️ as a hypothesis with a named test.

## Architecture

```
                     host process                          GPU (colocated)
   Harbor(Terminus-2) ──OpenAI /v1/chat/completions──▶ OpenAIAdapter ──▶ SGLang
        │                Authorization: Bearer <session_id>  │              │
        │                                            (captures exact       │
        ▼                                             token ids+logprobs)  │
   Docker container                                          │             │
   • runs shell commands only                                ▼             ▼
   • NO network, never talks to the model            list[Sample] ──▶ slime GRPO
   • verifier injected AFTER the agent exits                          (Megatron)
```

Three design choices, each with a reason:

1. **Reuse Harbor + Terminus-2 rather than reimplement the agent loop.** The SFT
   data *is* Terminus-2 output; a reimplementation would drift from the prompt
   format we trained on and quietly invalidate the comparison. Harbor already
   does the tmux keystroke protocol, the container lifecycle, and the verifier.
2. **`OpenAIAdapter` in the middle, purely for token capture.** slime's adapters
   "render the chat template, call SGLang with `input_ids` and
   `return_logprob=True`" and hand back sampled ids — deliberately avoiding
   re-tokenization. Re-tokenizing response text would corrupt the importance
   ratio. The adapter resolves the session from `Authorization: Bearer <sid>`
   (`common.py::_request_session_id`), so we pass the session id as Harbor's API
   key; that is what keeps concurrent rollouts separated on one adapter port.
3. **GRPO, not the paper's PPO.** A 27.8 B critic adds ~55.6 GB bf16 params and
   ~334 GB fp32 Adam state — roughly doubling optimizer memory on a cluster
   already half the paper's size. slime ships GRPO for this exact model with the
   paper's own RL settings (`kl-coef 0.00`, `entropy-coef 0.00`, `eps-clip 0.2`).

## Three correctness rules that are easy to get wrong

1. **Never apply the SFT JSON normalization in RL.** The SFT pipeline rewrites
   fenced ```` ```json ```` blocks (57 % of turns). Doing that on a rollout would
   train on tokens the policy did not emit. In RL the adapter's ids pass through
   untouched.
2. **An infrastructure failure is not a reward of 0.** A Docker build failure or
   DNS timeout must *abort* the sample (`remove_sample=True`), not teach the
   policy its actions were bad. `rl/generate.py` classifies against a narrow
   marker list and aborts; a plain reward-0 never enters that path.
3. **Reward only from the task's own verifier**, read from Harbor's
   `result.json:verifier_result.rewards.reward` after the agent exits.

## The rollout prerequisites — with measured numbers

### 1. Somewhere to run task containers — not necessarily *this* machine

**99 % of the 37,484 task Dockerfiles run `apt-get`/`pip`/`npm install` at build
time.** These are untrusted third-party build scripts. On the local path, both
`11_prebuild_images.py` and `rl/generate.py` **refuse to start** unless a
non-default `DOCKER_HOST` is set.

```bash
dockerd-rootless-setuptool.sh install     # or rootless podman, which serves the same API
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
```

**The hard requirement is a container runtime *somewhere*, not on the training pod.**
Measured on a real cluster: a pod under AppArmor's `docker-default` profile cannot
call `mount(2)` at all (the profile carries a literal `deny mount,`), so rootless
podman fails while unpacking image layers however it is configured, and no package or
storage driver fixes it. `bash scripts/00b_setup_sandbox.sh --diagnose` says which
case you are in and names the single fix.

When that is the case, point Harbor elsewhere instead of blocking on ops:

```bash
export RST_HARBOR_ENV=daytona DAYTONA_API_KEY=...    # or e2b / modal / remote daemon / k8s
source scripts/00b_setup_sandbox.sh                  # picks one from available credentials
```

Those backends build the task's own Dockerfile provider-side, so this pod needs zero
container privilege. Terminus-2 is a host-side agent, so the agent loop and every
model call still run here against the local shim — nothing inbound to the pod, and
`--network none` inside the task container still holds. `BACKENDS.md` has the full
table plus the two behavioural differences: proxy handling inverts, and §2 below is
skipped rather than failed.

The agent container must have **no network**. Harbor runs on the host and is what
talks to the adapter; the container only executes shell commands.

### 2. Prebuilt images — but only ~5,140, not 37,484

*(Local and remote-daemon paths only. With an off-machine backend the provider owns
the image cache, so `11_prebuild_images.py` detects that, writes
`prebuild_report.json` with `"skipped": true` and a reason, and exits 0 — the first
rollout per distinct image pays the build once, provider-side.)*

Lazy building during training turns every rollout into a network-bound build and
every registry hiccup into a stalled trainer. Prebuild the selected pool:

```bash
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
python scripts/11_prebuild_images.py --taskset $BASE_FOLDER/rl-sweet --sample 40   # size probe FIRST
python scripts/11_prebuild_images.py --taskset $BASE_FOLDER/rl-sweet --workers 8
```

Measured pool composition (5,140 tasks): **68 distinct base images**
(`ubuntu:22.04` 2,677 / `ubuntu:20.04` 356 / `python:3.11-slim` 320 /
`centos:7` 200 / `ubuntu:18.04` 189 / …), **4,430 single-image tasks and 710
docker-compose multi-service tasks** (13.8 % — these need `docker compose` and
more RAM per rollout), 19 with multiple Dockerfiles.

⚠️ **Disk is unmeasured.** Layers are shared across tasks with the same base, so
a naive per-image sum badly over-counts. Run the `--sample 40` probe and compare
`docker system df` before/after to get the real incremental cost before
committing disk. Do not guess.

### 3. Task selection is a first-class efficiency lever

GRPO's advantage is computed *within* a group of `n_samples_per_prompt` rollouts.
If all 8 rollouts score the same, the advantage is identically zero — the group
costs 8 sandboxes and contributes nothing. The paper's reward sat at 0.11 → 0.14,
i.e. most of its groups were all-fail.

The trajectory release lets us avoid that. Measured over 231,092 clean
trajectories (2,246 of 12,010 groups have any data at all):

| band | groups | tier |
|---|---|---|
| all-fail (0 %) | 897 | `hard` — zero advantage, pure cost |
| 0–10 % | 252 | `hard` |
| 10–35 % | 469 | **`sweet`** |
| 35–65 % | 394 | **`sweet`** |
| 65–90 % | 144 | **`sweet`** |
| > 90 % | 90 | `easy` — little left to learn |
| no data | 9,764 groups | `unknown` (28,059 tasks) |

Tier counts at task level: **sweet 5,140 / hard 4,089 / easy 196 / unknown
28,059**. Start on `sweet` (mean pass rate 0.393), then fold in `unknown` once
the loop is stable — that is where the bulk of the pool lives.

```bash
python scripts/10_build_rl_taskset.py \
  --tasks-root $BASE_FOLDER/rst-tasks --traj-root $BASE_FOLDER/rst-trajectories \
  --out $BASE_FOLDER/rl-sweet --tier sweet --materialize
```

### 4. Sandbox concurrency, not GPUs, is the bottleneck

`12_run_grpo.sh` derives `RST_MAX_SANDBOXES` from cores and RAM
(`min(cores/4, RAM_GB/8)`), budgeting ~2 cores / ~4 GB per concurrent rollout and
never taking more than a quarter of the cores — the CPU-offloaded optimizer needs
the rest. compose tasks cost more; lower it if you see build thrash.

Throughput sanity: from the SFT data, a trajectory is ~12 assistant turns and the
reference rollouts took 3–14 min wall-clock. With `rollout-batch-size 8 ×
n-samples-per-prompt 8 = 64` rollouts per step, a step needs 64 sandbox-runs;
at 16 concurrent that is ~4 waves ≈ 20–60 min **per GRPO step**. RL here is
sandbox-bound by one to two orders of magnitude. Plan days, not hours.

### 5. `ADAPTER_PUBLIC_HOST` must be reachable

Harbor dials the adapter over TCP. `rl/generate.py` refuses to start if unset.
Default `ADAPTER_PORT=18101`.

### 6. Harbor 0.21.0 installed in the training env

`pip install harbor==0.21.0` (needs Python ≥ 3.12; the `slime` env is 3.12). CLI
shape verified against `terminalevo/runner/harbor.py`:
`harbor run --path <SINGLE_TASK_DIR> --agent terminus-2 --model <m> --env docker
--n-attempts 1 --n-concurrent 1 --max-retries 0 --jobs-dir <d> --job-name <n> --quiet`.
Also apply the tracked tmux patch (`patches/harbor-0.21.0-tmux-stdout.patch` in
the TerminalEvo repo): some tmux builds report an oversized `send-keys` on stdout
rather than stderr, which otherwise reads as a spurious failure.

### 7. Initialize from the SFT checkpoint, not the base model

The paper cold-started PPO from the base model. With one-eighth the RL budget,
starting from your SFT checkpoint is strictly better: `INIT_CKPT` defaults to the
SFT run. You still need its `_torch_dist` conversion.

### 8. Disable EAGLE speculative decoding at first

slime's shipped script enables it; it depends on the MTP head that a text-only
Megatron round trip drops. Commented out in `12_run_grpo.sh`. Re-enable only for
throughput after correctness is established.

## Order of operations

1. `10_build_rl_taskset.py --tier sweet --materialize` — ✅ already validated locally.
2. Rootless daemon up; `11_prebuild_images.py --sample 40`. **GATE:** measure real
   disk via `docker system df`, and confirm the build success rate. Decide pool
   size from data.
3. Full prebuild. **GATE:** failed builds are excluded from the pool, not
   discovered mid-run.
4. **Single-rollout smoke test, no training.** Serve the SFT checkpoint under
   plain SGLang, run one `harbor run --path <task>` against it, confirm a reward
   comes back. This validates Harbor + Terminus-2 + Docker + the verifier with
   zero slime involvement.
5. **Adapter smoke test.** Same thing but pointed at `OpenAIAdapter`, with
   `HOSTED_VLLM_API_KEY=<session_id>`. **GATE:** `finish_session` returns
   non-empty `list[Sample]` and the token ids round-trip to the text Harbor saw.
   This is the highest-risk unverified step in the whole RL path.
6. `12_run_grpo.sh` with `--num-rollout 2` on ~8 tasks. **GATE:** non-zero
   advantage on at least some groups; abort-rate from infrastructure < 10 %.
7. Full run.

## RL risk register

| risk | severity | mitigation / test |
|---|---|---|
| ⚠️ Harbor's LiteLLM client may not forward the API key as a plain `Bearer` (adapter resolves sessions from it) | **high** | step 5. Fallback: the adapter also accepts `metadata.session_id` / `user` in the body, or run one adapter port per concurrent slot |
| ⚠️ Token capture correctness (re-tokenization would break the ratio) | **high** | step 5: assert decoded ids == Harbor's recorded assistant text |
| Degenerate all-same-reward groups waste the whole budget | high | tier selection (§3); log the degenerate-group fraction every step and drop tasks that stay degenerate |
| Infra failures mislabeled as reward 0 | high | narrow marker list + abort path in `rl/generate.py`; watch the abort-reason histogram |
| Sandbox throughput dominates | high | accepted and budgeted (§4); consider a separate CPU node pool for sandboxes |
| **The pod cannot run containers at all** (AppArmor `docker-default` denies `mount(2)`; observed, not hypothetical) | **high** | `00b_setup_sandbox.sh --diagnose` identifies it in one command. Three independent unblocks: the one-flag ops ask `--security-opt apparmor=unconfined`; `RST_HARBOR_ENV=daytona\|e2b\|modal\|<remote daemon>\|k8s`, which needs no local privilege; and — if neither lands — the DPO fallback (`DPO_PLAN.md`), which needs no container at all but is off-policy and leaves the checkpoint agentically unevaluated. Pursue the first two; SFT is unaffected either way |
| Provider concurrency quota exceeded on an off-machine backend | med | `RST_MAX_SANDBOXES` defaults to a conservative 8 there instead of a cores/RAM formula; 429s surface in the abort-reason histogram as infra failures |
| 710 compose tasks are heavier / flakier | med | they are tagged in the manifest; drop them for the first run if the abort rate is high |
| Docker disk exhaustion mid-run | med | `--sample` probe first; `docker system prune` policy between runs |
| CP correctness on gated-delta-net layers | high | same open question as SFT; see `PLAN.md` §5. Resolve during SFT, inherit the answer here |

## Cleared, so nobody re-investigates

**46 tasks contain `environment/tests/`, and this is NOT a verifier leak.** That
directory holds the *project's own* fixtures (a PHPUnit suite, Ansible
playbooks, JSON fixtures) — workspace content the agent is meant to work on. The
private RST verifier is the distinct task-root `tests/test_state.py` +
`tests/test.sh`. A precise check (verifier filenames, or byte-identical content,
anywhere in the build context) found **0 leaks in 5,140 tasks**. The guard is in
`10_build_rl_taskset.py` and excludes offenders automatically if the release ever
changes — so it flags on content, not on the directory name.
