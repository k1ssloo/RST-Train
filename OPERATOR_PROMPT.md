You are operating a 4-node × 8×A100 cluster (32 GPUs) to fine-tune a Qwen3.5 model
on synthesized terminal-agent trajectories, then benchmark it and write a report.
Five model sizes are supported (0.8B … 35B-A3B); see "Pick a model first" below.

You are starting from an EMPTY directory. Bootstrap first:

​    export BASE_FOLDER=<shared-scratch-dir>      # >= 400 GB free, ideally shared across nodes
​    mkdir -p "$BASE_FOLDER" && cd "$BASE_FOLDER"
​    git clone https://github.com/k1ssloo/RST-Train.git
​    cd RST-Train

Then READ THESE FILES BEFORE RUNNING ANYTHING:

  - README.md    — status table: what is already verified vs. never executed
  - BACKENDS.md  — READ THIS EARLY. verl + FSDP is the PRIMARY backend, not slime.
  - PLAN.md      — the SFT spec: measured facts, hardware decision tables, risk register
  - RL_PLAN.md   — the RL phase: prerequisites, gates, unverified assumptions
  - DPO_PLAN.md  — only if RL turns out to be blocked: what to train when no
                   container will run here. Skip it while GRPO is still viable.

Everything in PLAN.md marked "measured" was verified against the real data and the
real upstream source trees. Everything marked UNVERIFIED has not been run and is
your job to validate. Do not assume anything beyond that.

## Mission

For qwen3.5-27b AND qwen3.5-9b: an SFT checkpoint, then an RL checkpoint, each
benchmarked on Terminal-Bench-Hard and Terminal-Bench 2 against the same base model,
plus a markdown report per stage with YOUR analysis of anything abnormal. Training,
evaluation and reporting are chained automatically; you supervise, fix, and diagnose.

Reference numbers below apply to Qwen3.5-27B ONLY (paper Tables 3–4, pass rate %,
tb-hard / tb2). No other size has published numbers:
  Qwen3.5-27B base   22.67 / 41.20
  paper SFT round 1  23.00 / 42.32   <- the fair target for a single SFT pass
  paper SFT round 3  28.33 / 47.94   <- three cumulative synthesis rounds, not one
Beating round 1 is the goal. Do not expect round 3.

## Pick a model first

​    python scripts/model_registry.py --list

Five models are supported; `MODEL_KEY` selects one and everything else
(parallelism, loss mask, spec file, vision handling, serving TP) follows from
configs/models.json, which VALIDATES the config and refuses impossible ones.

  qwen3.5-0.8b      0.87B   ~5 min/epoch    2+ GPUs   pipeline smoke test
  qwen3.5-4b        4.66B  ~25 min/epoch    8+ GPUs   iteration workhorse
  qwen3.5-9b        9.65B  ~50 min/epoch    8+ GPUs   primary low-cost result
  qwen3.5-27b      27.78B ~150 min/epoch   32  GPUs   the paper's model
  qwen3.5-35b-a3b  35.95B  ~40 min/epoch    8+ GPUs   MoE, ~3B active

### FIRST BATCH — what you are authorized to run now

Run **qwen3.5-27b and qwen3.5-9b: SFT, then RL, then eval** for both. These two are
marked first_batch=true in the registry. Nothing else is authorized yet.

​    MODEL_KEY=qwen3.5-27b RUN_RL=1 bash scripts/20_run_all.sh
​    MODEL_KEY=qwen3.5-9b  RUN_RL=1 bash scripts/20_run_all.sh

Qwen3.5-0.8B has already been run locally on one GPU against the real
pre-tokenized data (`scripts/16_smoke_forward_backward.py`), and it found real bugs
that are now fixed. What it established, which you can rely on:

  * the model loads and the gated-delta-net path executes (18 linear + 6 full
    attention layers at 0.8B)
  * the loss_mask -> labels conversion is correct: loss 0.45 vs log(vocab)=12.42,
    and an all-masked control yields NaN rather than the same number, which is what
    proves the mask reaches the loss
  * 27-29% of tokens carry a label, matching the dataset manifest
  * Liger's loss matches the unfused loss (0.4505 vs 0.4495)
  * MEASURED memory: at 32,329 tokens the UNFUSED cross-entropy OOMs on an 80GB
    card even at 0.75B params, attempting a single 29.85 GiB allocation
    (= seq x 248,320 vocab x 4 bytes, fp32). With Liger the same run peaks at
    13.14 GiB. That term is independent of model size, so it is worse at 27.8B.

Still unexercised anywhere: multi-node FSDP, the checkpoint export path, and the RL
bridge. Run `16_smoke_forward_backward.py` yourself on your hardware before the
cluster run -- it takes minutes and its memory numbers are what you should size
`configs/models.json` from.

Optional but strongly advised first: one throwaway qwen3.5-0.8b SFT run
(`MODEL_KEY=qwen3.5-0.8b`, ~5 min/epoch, 2 GPUs). Same architecture as 27B, so it
clears the two unverified high-risk steps -- HF<->Megatron conversion tolerating the
ViT/MTP tensors, and CP correctness on the gated-delta-net layers -- for almost
nothing. Failing those on 0.8B costs minutes; failing them after booking 32 GPUs for
27B costs a day. Do not report 0.8B numbers as a result; it is a smoke test.

### After the first batch: continue or wait

`20_run_all.sh` writes `$BASE_FOLDER/verdict.json` with `in_range: true|false`.
"In range" means the report produced **zero FAIL findings**. WARNs do not block --
a WARN is a caveat for a human to weigh; a FAIL means a number is wrong or
untrustworthy.

  * **In range** -> proceed on your own initiative: continue to the next model
    (4b, 35b-a3b) and keep training. You do not need to ask.
  * **Out of range** -> STOP that model, write the analysis into the report, and
    WAIT for a human. Do not start another model's long run on top of an
    unexplained failure, and do not "fix" it by loosening a check.

The RL stage enforces this itself: it refuses to start if the SFT verdict has FAILs,
and refuses entirely for models not in the first batch. That is deliberate -- RL
costs days of sandbox time, so it must not be built on a checkpoint you cannot
trust.

All five share one byte-identical tokenizer AND one training-time chat-template
render, so the SAME dataset and --loss-mask-type qwen3_5 apply to all of them. Do
not rebuild the data per model.

TWO MODEL-SPECIFIC THINGS THAT WILL BITE YOU:

    1. qwen3.5-0.8b must be SERVED with a thinking-on chat template. Its own template
       defaults thinking off, so its generation prompt already closes the think block
       while training targets open with "\n</think>\n\n". 20_run_all.sh fetches the
       right template automatically via the registry -- do not remove that step, and
       if you serve 0.8B by hand, pass --chat-template.
    2. qwen3.5-35b-a3b's expert-parallel rows are UNVALIDATED on A100. slime's shipped
       launcher uses TP2/EP8 on 8 GPUs; ours scale that to 32. If DeepEP misbehaves on
       SM80, drop --moe-enable-deepep and use --moe-token-dispatcher-type alltoall.
       Report what you had to change.

Only qwen3.5-27b has published reference numbers. For any other model the report
SKIPS (does not pass) the regression-vs-base and reference-reproduction checks, and
says so. Do not invent a target for 4B/9B/35B.

## Backend: use verl + FSDP. Do NOT start with Megatron.

​    MODEL_KEY=qwen3.5-9b bash scripts/30_run_sft_verl.sh

Megatron on A100 requires swapping cuDNN to satisfy the TransformerEngine/apex
build, which a shared cluster will not let you do. So slime + Megatron
(scripts/05_run_sft.sh) is the SECONDARY path here: keep it for reference, try it
only if verl fails and you have somehow obtained the ability to change cuDNN.

Two things on the verl path are mandatory, not optional. Both are explained in
BACKENDS.md; neither is a style preference:

    1. `model.use_liger=True`. This model's vocab is 248,320, so materialized logits
       for a 32K sequence are 16.3 GiB in bf16 (~32.6 GiB if the loss upcasts to
       fp32) -- larger than the activations. Liger's fused cross-entropy is what makes
       the run fit at all. Without it you OOM and it will look like a parallelism
       problem.
    2. Pre-tokenized data via `scripts/15_export_pretokenized.py`. verl's built-in
       multi-turn dataset tokenizes message-by-message; measured on 200 real rows,
       200/200 disagree with the whole-conversation render, because the Qwen3.5
       template puts an empty <think> block on the LAST assistant turn and
       turn-by-turn building makes every turn "last" (21 blocks instead of 1 in a
       21-turn conversation). Do NOT set `ignore_input_ids_mismatch: True` to silence
       that -- it silences the check, not the bug, and you would train on tokens
       serving never emits. `30_run_sft_verl.sh` builds the pre-tokenized file for you
       and refuses to start if its trained-token fraction leaves the measured band.

verl's FSDP engine has no context parallelism, so there is no CP correctness
question to resolve on this path -- one fewer unknown. (For what it is worth,
torchtitan, PyTorch's own reference implementation of this architecture, also lists
CP as TODO. It is genuinely unsettled for gated-delta-net layers.)

## No Docker? That is expected, and it is solved

The cluster will not give you Docker daemon access. You do not need it.

​    source scripts/00b_setup_sandbox.sh        # finds/starts a runtime, exports DOCKER_HOST
​    bash   scripts/00b_setup_sandbox.sh --check

Rootless **podman** serves the same Docker API that Harbor speaks, so pointing
DOCKER_HOST at podman's socket makes Harbor work with NO code change. This was
verified end to end on a box with no docker.sock permission: build a real task image
(26.6 s, including apt-get/git clone/pip), `run -d --network none`, `exec`, tmux 3.2a
new-session/send-keys/capture-pane, and no network inside the container.

Scope, precisely: **SFT needs no container runtime at all.** Only eval and RL do. So
if the runtime is unavailable you can still train -- but you cannot measure whether
training helped, and you must say the eval was impossible rather than call a
checkpoint good.

FOUR THINGS THAT WILL BITE YOU, all found by actually building the real task images:

    1. `--format docker` is required: 16% of task Dockerfiles use `SHELL`, which
       podman's default OCI format ignores and then errors on.
    2. **podman >= 4.4 is required**: 31% use Dockerfile heredocs (`RUN <<EOF`).
       Ubuntu 22.04 ships 3.4.4, which fails them with "Unknown instruction: IF" --
       looks like a broken task, is actually a stale toolchain, and would silently
       cost you a third of the pool. If yours is old, install a STATIC rootless podman
       (32 MB, no root, no package manager):
         curl -sSL -o /tmp/podman.tgz https://github.com/mgoltzsche/podman-static/releases/latest/download/podman-linux-amd64.tar.gz
         mkdir -p $HOME/.local && tar xzf /tmp/podman.tgz -C $HOME/.local --strip-components=1
         export PATH=$HOME/.local/bin:$PATH
       `11_prebuild_images.py` aborts on this rather than building two thirds quietly.
    3. A `git clone` inside a build can fail "hardlink different from source" under
       rootless kernel overlay. Retry in a vfs store under a separate `--root`;
       `11_prebuild_images.py` does that automatically.
    4. podman < 4 has no `compose` subcommand, so the 710 docker-compose multi-service
       tasks (13.8% of the pool) need podman >= 4.x too, or must be excluded. If you
       exclude them, say so -- it changes which tasks the numbers cover.

If podman is not installed at all, ask for the `podman` + `uidmap` packages. That is
normally an easier request than Docker access: no daemon, no root, no group
membership. But check the next section FIRST -- installed-and-working podman that
still cannot start a container is a different problem with a different ask.

### When podman is installed, working, and STILL cannot run a container

This happened on a real cluster, so read it before asking for anything. Symptom:
podman is fine (`podman info` clean, image blobs pull through the proxy), and then
unpacking a layer fails with

    ApplyLayer ... remount /, flags: 0x44000: permission denied

`0x44000` is `MS_REC|MS_PRIVATE` -- containers/storage isolating mount propagation,
not tar extraction failing. Probe it directly rather than guessing:

    bash scripts/00b_setup_sandbox.sh --diagnose

That runs `00c_probe_sandbox.py`, which calls `unshare(2)` and `mount(2)` itself and
reads `/proc/self/attr/current`. The signature to look for: `unshare -U` and
`unshare -Ur` SUCCEED while every `mount` returns **EACCES** even inside your own
user namespace where you hold full capabilities, and the AppArmor profile is
`docker-default (enforce)`.

That is an LSM denial, and there is **no workaround from inside the pod**. AppArmor
profiles are inherited across `unshare()` and `execve()`; `docker-default` contains a
literal `deny mount,` (it permits only `umount`) and grants no `change_profile` rule,
so you cannot transition out of it. fuse-overlayfs, vfs, bubblewrap, a newer podman:
all dead, because all of them mount. EACCES rather than EPERM is how you tell this
apart from a missing capability.

**Do not ask for the `podman` + `uidmap` packages in this case.** They are already
installed and working; the packages were never the problem. The minimal ask is one
flag on the pod spec:

    --security-opt apparmor=unconfined        # docker/podman
    # k8s >= 1.30:  securityContext.appArmorProfile.type: Unconfined
    # k8s <  1.30:  container.apparmor.security.beta.kubernetes.io/<name>: unconfined

You do **not** need `--cap-add SYS_ADMIN` (the user namespace already grants caps
inside it; AppArmor is overriding them, and it would override SYS_ADMIN too), nor
`/dev/fuse` (the vfs storage driver needs no FUSE), nor `--privileged`, nor any
seccomp change (`Seccomp: 0` already). One flag is far likelier to be approved than
four, so ask for exactly the one.

### You do not have to wait for ops

Harbor 0.21.0 ships ~25 environment backends, and several run the container
somewhere else entirely, needing **zero** container privilege here:

| backend | what it needs | builds the task Dockerfile? |
|---|---|---|
| `daytona`, `e2b`, `modal` | outbound HTTPS + an API key | yes, provider-side |
| remote Docker daemon | reachability + the task tree at the same absolute path there | yes |
| `gke`/`ack`/`openshift` | a namespace + RBAC to create sibling pods | yes |
| `hf-sandbox` | a **prebuilt** image; refuses a Dockerfile | no |
| `singularity` | a docker image or `.sif`, not a Dockerfile | no |

Terminus-2 is a **host-side** agent: it drives the container through
`environment.exec(...)`, so with a managed provider the agent loop and every model
call still run in this pod against the local vLLM/SGLang shim. No reverse tunnel, no
inbound access to the pod, and `--network none` semantics inside the task container
are preserved. `00b_setup_sandbox.sh` picks one of these automatically when it finds
the credentials, and exports `RST_HARBOR_ENV` / `RST_HARBOR_ENV_KWARGS`:

    export RST_HARBOR_ENV=daytona DAYTONA_API_KEY=...   # then eval and RL just work

Two consequences to plan for. `11_prebuild_images.py` cannot warm a cache it does not
own, so it detects this and exits 0 having built nothing -- the first rollout per
distinct image pays the provider-side build once. And `RST_MAX_SANDBOXES` becomes a
provider **quota** number rather than a local-resource one; exceeding it returns 429s,
which the rollout code classifies as infrastructure failures and drops.

### Either way, the checkpoint does not go unmeasured

SFT needs no container at all, so it is never blocked. If eval is, run the
container-free fallback. It is a weaker signal and says so in its own output, but
"unmeasured" is not an acceptable deliverable:

    python scripts/06b_eval_offline.py \
        --model-path $BASE_FOLDER/out-hf-full \
        --base-model $BASE_FOLDER/$MODEL_DIR_NAME \
        --holdout    $DATA_DIR/rst_sft_holdout.parquet \
        --tokenizer  $BASE_FOLDER/$MODEL_DIR_NAME \
        --out        $BASE_FOLDER/eval/offline

Held-out loss and next-token accuracy over the supervised spans (with a base-vs-tuned
delta -- the only thing that answers "did SFT do anything"), plus greedy
action-protocol agreement: parse rate, first-keystroke match, command-list match. Pass
the result to `14_make_report.py --offline-eval`. The report still records a **FAIL**
on benchmark coverage, deliberately: it must never let "we could not measure it" read
as "it looks good".

### And the RL phase still has something to train: DPO

Eval is not the only thing a missing container blocks -- GRPO is blocked outright,
because every rollout needs one. The fallback for the TRAINING half is DPO on the
logged trajectories, which needs no container, no network and no privilege: the
release already contains successes and failures on the same tasks, scored by those
tasks' own verifiers.

    bash scripts/33_run_dpo.sh          # or RUN_DPO=1 bash scripts/20_run_all.sh

`20_run_all.sh` also picks it up on its own (`RUN_DPO=auto`) exactly when `RUN_RL=1`
and the sandbox check fails, so you do not have to notice. Read `DPO_PLAN.md` before
quoting any of its numbers -- two things there are easy to get wrong in a report:
DPO is **off-policy** (it cannot discover a strategy no logged trajectory used, so it
is never "our RL result"), and `holdout_reward_accuracy` is **likelihood ranking**
where 0.5 means no preference, not a pass rate. A DPO checkpoint with no agentic eval
is still an untested checkpoint.

## The one-command path

​    export MODEL_KEY=qwen3.5-0.8b            # start here, then 27b
​    export MASTER_ADDR=<head-node-ip>
​    export HOSTFILE="$BASE_FOLDER/hostfile"          # one node IP per line, head first
​    export RST_DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock   # dedicated daemon, see below
​    export WANDB_KEY=<key>                           # omit -> offline mode automatically
​    bash scripts/20_run_all.sh 2>&1 | tee "$BASE_FOLDER/run_all.log"

That runs: preflight → env → download → build data → convert checkpoint → train →
export → eval (yours + the reference checkpoint) → report. Every stage writes a
marker in `$BASE_FOLDER/.stage/` and is skipped if already done, so re-running
resumes. Delete a marker to force a stage; `SKIP_STAGES="env download"` to skip by
name.

**Run stages 1–5 individually the first time** (see "Order" below). The wrapper is
for resuming and for the long unattended stretch, not for hiding the risky steps.

## Getting the data: download, don't rebuild

The training data is already built and validated. Prefer downloading it:

​    hf download NiuNiu0110/RST-SFT-Qwen3.5-27B --repo-type dataset \
​      --local-dir "$BASE_FOLDER/sft-hf"
​    mkdir -p "$BASE_FOLDER/sft-v1-cap10"
​    cp "$BASE_FOLDER/sft-hf/data/cap10/train.parquet"   "$BASE_FOLDER/sft-v1-cap10/rst_sft_train.parquet"
​    cp "$BASE_FOLDER/sft-hf/data/cap10/holdout.parquet" "$BASE_FOLDER/sft-v1-cap10/rst_sft_holdout.parquet"
​    cp "$BASE_FOLDER/sft-hf/manifest_cap10.json"        "$BASE_FOLDER/sft-v1-cap10/manifest.json"

`cap10` = 10,778 examples (the paper's exact count); `cap8` = 8,886 (ablation).
Then `SKIP_STAGES="data"` so the wrapper does not rebuild it.

If you do rebuild (`scripts/03_build_sft_data.py`), you MUST then run
`scripts/03b_validate_sft_data.py`. It must print `contract failures : 0` and
`user-turn leakage : 0`. Anything else is a stop condition — the training target
would be wrong and no amount of training fixes that.

## You may modify the code to get training running

This repository was written without access to your cluster, and several steps have
never been executed. **You are authorized to edit any file in it** -- scripts,
configs, the registry, even the rollout code -- to make training start and keep
running. Do not sit blocked waiting for a human on a plumbing problem. Expected and
entirely fine to fix yourself:

  * launcher/env bugs, wrong paths, missing exports, quoting, arg-order problems
  * NCCL / Ray / Docker / filesystem / permission issues
  * OOM: apply the fallback ladder, or edit configs/models.json parallelism rows
  * a slime or Megatron flag that changed name or no longer exists
  * a pinned wheel that will not build for your CUDA/driver combination
  * anything in configs/models.json that is simply wrong for your hardware

Rules when you do:

    1. Fix the cause, not the symptom. If a launcher passes a flag slime removed,
       update the flag -- do not delete the feature it enabled.
    2. **Never change these to make a run start.** Each one silently invalidates the
       result, so a run that "succeeds" after touching them is worse than no run:
         - `--loss-mask-type qwen3_5` (anything else trains on terminal output)
         - the data gate in 20_run_all.sh (contract failures / user-turn leakage == 0)
         - `max_tokens_per_gpu * cp >= max_seq_len` and the other registry asserts
         - infrastructure-vs-model-failure separation in 06_eval.py and rl/generate.py
         - the reference-checkpoint eval, when the model has one
           If you believe one of these is genuinely wrong, say so, explain why, and WAIT.
    3. Record every edit in notes/DEVIATIONS.md: what you changed, why, and what
       evidence led you there. `git diff` is not a substitute for the reason.
    4. Commit locally as you go so the diff is reviewable. Do not push.
    5. If a fix changes what the numbers mean -- shorter sequences, fewer eval runs,
       a different LR, dropped examples -- say so explicitly in the report. A silent
       scope reduction reads as a clean result and is the worst outcome here.

## Adapt to the machine you actually have

The versions in this repo were chosen on a different machine. Yours will differ in
driver, CUDA, kernel, glibc, podman, GPU generation and available packages. When
something fails, **first identify what your system actually is, then adapt** — do
not retry the same command hoping for a different result, and do not import a
version from these docs that your system cannot run.

Establish the facts before changing anything:

​    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
​    python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability())"
​    python -c "import transformers;print(transformers.__version__)"
​    podman --version; uname -r; ldd --version | head -1
​    bash scripts/00_preflight.sh

Then adapt along these axes. Each has a hard floor that is NOT negotiable and a
choice above it that is yours:

  * torch CUDA build <- driver version. Floor: torch must report
    `cuda.is_available() == True`. A wheel newer than the driver fails at init with
    an unhelpful message. `01b_setup_env_verl.sh` maps driver>=570 -> cu128,

    >=550 -> cu126, >=525 -> cu121; if that mapping is wrong for your driver, change
    >the mapping.

  * kernels <- compute capability. Floor: A100 is sm80, so bf16 only, FA2 not FA3,
    no FP8, no FlashQLA, GDN via fla-Triton or the pure-PyTorch fallback. If
    flash-attn will not build, `sdpa` is correct and slower -- take it and say so.

  * transformers >= 5.15.0. This is a FLOOR, not a preference: older versions do not
    know `qwen3_5` at all.

  * podman >= 4.4 for task images (31% use Dockerfile heredocs). Below that,
    install the static build; do not work around it by skipping tasks silently.

  * sequence length and parallelism <- measured memory, not the tables here. Run
    `scripts/16_smoke_forward_backward.py --seq-len N` on ONE GPU first: it reports
    real peak memory and exits nonzero on a failed check. Edit
    `configs/models.json` from what you measure.

  * data layout <- whatever the trainer you end up using actually wants. The
    training target is already fixed and verified in `input_ids` + `loss_mask`
    (scripts/15_export_pretokenized.py), so reshaping it into another container --
    packed, padded, `labels` with -100, a different column naming, a webdataset --
    is a mechanical transform you may do freely. What you must NOT do is recompute
    the mask with a different implementation, or let a trainer re-tokenize from
    `messages`, because that reintroduces exactly the bug the pre-tokenization
    exists to remove.

Rules while adapting:

    1. Change one variable at a time and record what you observed. "Upgraded three
       things and it works now" is not a result anyone can act on later.
    2. Prefer the smallest, most local change: a flag over a version bump, a version
       bump over a code edit, a code edit over a redesign.
    3. Write every adaptation into notes/DEVIATIONS.md with the system fact that
       forced it (e.g. "driver 535 -> cu121 wheel, cu128 failed at cuda init").
    4. If an adaptation changes what the numbers mean -- shorter sequences, fewer eval
       runs, dropped tasks, sdpa instead of FA2 -- state it in the report. That is the
       difference between an adapted experiment and an unreported one.

## Hard rules — these are the exceptions to the above

1. `--loss-mask-type qwen3_5`. NEVER the default `qwen`. The default mis-segments
   this chat template and trains on terminal output and the harness prompt.
2. `--qwen-gdn-backend fla`. A100 is SM80; FlashQLA requires SM90+. Do not install
   FlashQLA, and do not "fix" a GDN error by switching to flashqla.
3. No FP8 anywhere. A100 has no FP8 tensor cores. Never run convert_hf_to_fp8.py.
   3b. If the Megatron/TE/apex stack simply will not build, you have a documented
   fallback: verl + FSDP via `scripts/30_run_sft_verl.sh`. Read BACKENDS.md first.
   Two things there are mandatory, not optional: `model.use_liger=True` (without a
   fused cross-entropy the 248,320-row logits alone are 16-33 GB at 32K and you will
   OOM), and pre-tokenized data via `scripts/15_export_pretokenized.py` (verl's
   built-in multi-turn dataset mismatches the Qwen3.5 render on 100% of our rows,
   producing one empty <think> block per assistant turn instead of one per
   conversation). Do not set `ignore_input_ids_mismatch: True` to make it pass.
   Switching backends is a deviation: record it and say so in the report.
4. Version policy differs by backend, deliberately:
   * slime path (`01_setup_env.sh`): DO NOT change the pins (torch 2.11.0+cu129,
     flash_attn 2.8.3, transformer_engine 2.16.1, flash-linear-attention 0.4.2,
     SGLang v0.5.15.post1, Megatron 1dcf0daf). That FA2/TE/Megatron combination is
     genuinely fragile; if a pin will not install, report it rather than bumping it.
   * verl path (`01b_setup_env_verl.sh`): the opposite. It DETECTS the driver and
     compute capability and picks the torch CUDA build accordingly, then verifies by
     importing. You are expected to adjust it if detection is wrong on your machine.
     Two invariants there: never install FlashQLA (sm90+ only) and never enable FP8
     on A100, and never let a dependency silently replace the driver-matched torch —
     the script guards that, because `pip install liger-kernel` did exactly that
     during local validation (swapped cu128 for PyPI's cu130).
5. `messages[0]` has role `user`, not `system`. That is how Terminus-2 delivers the
   harness prompt, so it keeps training and serving identical.
6. `max-tokens-per-gpu × CP ≥ max-seq-len` must hold or the longest sequence cannot
   be placed. scripts/model_registry.py asserts this (plus tp·pp·cp·dp == gpus and
   tp within one node) and exits non-zero. Do not weaken those asserts; fix
   configs/models.json instead, and say what you changed.
7. After converting back to HF you MUST run scripts/07_restore_vision.py for any
   model with a vision tower (all five here do). A text-only Megatron round trip
   drops `model.visual.*` and `mtp.*`, and the checkpoint will not then load as a
   ConditionalGeneration model. 20_run_all.sh does this conditionally on the
   registry's has_vision.
8. Never size memory from the trajectory parquet's `input_tokens` column — it is a
   sum over turns, roughly 2× the true sequence length.
9. Evaluation needs a DEDICATED (ideally rootless) Docker daemon. Benchmark tasks
   build untrusted third-party Dockerfiles; do not build them on the host's default
   daemon.

## Environment facts you must discover, not assume

GPU memory per card (80GB vs 40GB), NVLink, InfiniBand vs Ethernet, and whether
there is a shared filesystem are ALL UNKNOWN to whoever wrote the plan. Do not ask
a human. Run:

​    bash scripts/00_preflight.sh --hostfile "$HOSTFILE"

It prints the parallelism row to use. 05_run_sft.sh auto-detects the same things;
override with MEM_CLASS=80GB|40GB|40GB-alt. Log which row you used and why.

## Order. Each step is a gate.

Compute is cheap (~1.5–3 h per epoch); integration is what fails. Steps 1–3 cost
almost nothing and de-risk the expensive ones. Do not skip ahead.

STEP 1 — Preflight. GATE: host RAM must have headroom for Adam CPU offload (~334 GB
  fp32 state, sharded across DP). If not, drop `--optimizer-cpu-offload`, raise CP,
  and say so.

STEP 2 — Environment + download. Use the backend-matched script:
    verl (default):  bash scripts/01b_setup_env_verl.sh
    slime (only if you can change cuDNN): bash scripts/01_setup_env.sh
  They are NOT interchangeable and must not share one env: 01_setup_env.sh builds
  apex + TransformerEngine + Megatron, which is the part that needs the cuDNN swap.
  01b installs none of that. `20_run_all.sh` picks the right one from $BACKEND.
  Then `bash scripts/02_download.sh`. GATE:
  02_download.sh sha256-verifies every dataset shard against the published manifest
  and exits non-zero on mismatch. Do not proceed past a mismatch.

STEP 3 — Checkpoint conversion (`04_convert_ckpt.sh to_dist`). HIGHEST-RISK
  UNVERIFIED STEP: the text-only slime spec must tolerate the ViT
  (`model.visual.*`) and MTP (`mtp.*`) tensors in the HF checkpoint. CPU/RAM-bound
  (~120 GB RAM, 20–40 min), so run it BEFORE booking all 32 GPUs. GATE: completes,
  output ~52 GiB. If it fails on unexpected keys, report the exact keys and stop.

STEP 4 — Single-node smoke. `ACTOR_NUM_NODES=1`, TP4/PP2/CP1/DP1, ~200 examples.
  GATE: loss decreases over ~20 steps, no NaN. Proves slime + Megatron + the GDN
  Triton kernels + the loss mask actually step.

STEP 5 — Context-parallel correctness. UNVERIFIED and important: the 48
  gated-delta-net layers carry recurrent state across the sequence, so CP is not
  obviously sound for them. slime ships CP4 as the default for this exact model,
  which is suggestive, not proof. Train the SAME slice for 20 steps at CP1 and CP2
  and compare loss curves. GATE: curves agree within noise. If they diverge, CP is
  broken here — fall back to CP1, which needs MEM_CLASS=80GB with
  `--max-tokens-per-gpu 32768`. Report the comparison either way.

STEP 6 — Full run + automatic eval + report: `bash scripts/20_run_all.sh`.
  Defaults: cap10 (10,778 ex), 1 epoch, GBS 128, LR 3e-6→3e-7 cosine, seq 32768 →
  ~82 optimizer steps. Watch in wandb: loss over ~82 points, trained tokens/step
  ≈ 32.6M/82 ≈ 397K, grad norm, per-GPU memory. On OOM apply the fallback ladder in
  PLAN.md §1 IN ORDER (① halve max-tokens-per-gpu ② CP 2→4 ③ PP 2→4 ④ only as a
  last resort reduce max-seq-len, which drops the long-horizon examples the data
  exists to teach).

## Evaluation (automatic, but understand it)

`scripts/06_eval.py` serves the checkpoint under SGLang and drives Harbor with
Terminus-2 on Docker, 3 independent runs, reporting mean ± std.

- **tb-hard** 100 tasks, verifiers ship → scored
- **tb2** 89 tasks, verifiers ship → scored
- **lhtb** 46 tasks, verifiers are WITHHELD upstream (0/46 ship `tests/`) →
  reported as `unscorable`. **Do not produce an LHTB number.** If you find a way to
  score it, say exactly how; otherwise leaving it blank is the correct answer.

Infrastructure failures are excluded from the pass-rate denominator and reported as
their own rate. A Docker build failure is not a wrong answer. Never "fix" a low
score by counting infra failures as model failures, or vice versa.

**The reference eval is what makes your number interpretable.** 20_run_all.sh also
scores the authors' released `Zhongzhi1228/Qwen3.5-27B-SFT` through the SAME
harness. If it does not land near 28.33 / 47.94, THE HARNESS IS WRONG, not your
training — fix the harness before interpreting anything about your own checkpoint.
Do not skip this to save time.

## The report — your main deliverable

`scripts/14_make_report.py` runs automatically and writes `$BASE_FOLDER/REPORT.md`:
a results table against the paper's numbers, a findings table from mechanical
checks (loss mask, GDN backend, parallelism arithmetic, sequence placement, data
provenance, NaN/grad-norm/step-count, infra-failure rate, regression vs base,
reference-harness validity), and an empty **Analysis** section.

**You must fill in the Analysis section.** The tool reports what it can check; it
does not diagnose. For every 🔴 FAIL and 🟡 WARN:

- Name the evidence — log line, file path, command output. If you are guessing,
  write "hypothesis:" and say what would confirm it. Never present speculation as
  fact.
- Say explicitly whether you are claiming a **correctness** problem or an
  **infrastructure** problem. Do not describe a correctness bug as flakiness.
- If a check is a false positive, say why its threshold is wrong in this case
  rather than silently ignoring it.
- State plainly whether the headline numbers are trustworthy.

Then re-run the generator so the report on disk includes your analysis, and report
the final verdict. The generator exits 2 when any FAIL finding exists — a non-zero
exit with a written explanation is a perfectly good outcome; a clean exit you
achieved by disabling a check is not.

## When something fails

- A GATE failure is stop-and-report. Never disable a correctness check to make a
  step pass.
- An infrastructure failure (NCCL timeout, OOM, image pull, disk) is yours to retry
  and fix; apply the documented fallback ladder rather than inventing a new config.
- Distinguish the two in everything you write.
- Log deviations to `notes/DEVIATIONS.md` as you go, and per-step progress to
  `notes/RUN_LOG.md`.

## RL — in scope for 27B and 9B, read RL_PLAN.md first

Agentic GRPO: Harbor + Terminus-2 drive a Docker sandbox per rollout, slime's
OpenAIAdapter captures the exact sampled tokens, reward comes from each task's own
verifier. `RUN_RL=1` chains it after SFT automatically, gated on the SFT verdict.

Prerequisites you must satisfy first (RL_PLAN.md has the measured numbers):

  * a DEDICATED rootless Docker daemon (RST_DOCKER_HOST); the scripts refuse to run
    on the default daemon, because task Dockerfiles are untrusted build scripts
  * prebuilt task images: `python scripts/11_prebuild_images.py --taskset
    $BASE_FOLDER/rl-sweet --sample 40` FIRST to measure real disk via
    `docker system df`, then the full build. 99% of task Dockerfiles install
    packages at build time, so lazy building makes every rollout network-bound.
  * `ADAPTER_PUBLIC_HOST` reachable from the Harbor process
  * `pip install harbor==0.21.0` plus the tmux patch noted in RL_PLAN.md

TWO UNVERIFIED, LOAD-BEARING ASSUMPTIONS. Test both before a long run; RL_PLAN.md
gives the exact smoke test for each:

    1. that Harbor's LiteLLM client forwards the session id as an `Authorization:
       Bearer` header (that is how the adapter separates concurrent rollouts)
    2. that token capture round-trips -- decoded ids must equal the assistant text
       Harbor recorded. If they do not, the importance ratio is wrong and the run is
       silently off-policy.

Expect RL to be SANDBOX-bound, not GPU-bound: roughly 20-60 minutes per GRPO step.
Budget days. Watch the fraction of groups with zero reward variance -- those cost a
full set of sandboxes and contribute nothing to the gradient; if it stays high, the
task tier selection needs revisiting rather than the learning rate.

If the sandbox never materializes, do not leave the RL phase empty and do not fake it
with GRPO on a machine that cannot roll out. Run the DPO fallback instead
(`DPO_PLAN.md`, `scripts/33_run_dpo.sh`) and report it as what it is: off-policy
preference training on logged trajectories, agentically unevaluated.

## Report back

Final summary must contain:

    1. detected hardware and the parallelism row used;
    2. the STEP 5 CP1-vs-CP2 comparison result;
    3. final loss, step count, wall-clock, peak per-GPU memory;
    4. eval table for BOTH your checkpoint and the reference, side by side;
    5. every deviation from PLAN.md, and every code edit you made, with reasons;
       5b. the `in_range` verdict per model, and for anything out of range, your analysis
       of the cause and what you need from a human;
    6. anything in PLAN.md that turned out to be WRONG — that matters more than a
       clean run, because the plan's author could not test these steps.

State plainly what you verified versus what you assumed. Never report success for a
step you skipped.