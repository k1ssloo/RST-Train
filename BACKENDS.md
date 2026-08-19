# Training backends

**Primary path: verl + FSDP** (`scripts/30_run_sft_verl.sh`).
**Secondary: slime + Megatron** (`scripts/05_run_sft.sh`, `12_run_grpo.sh`).

The order is dictated by the cluster, not by preference: getting Megatron built on
A100 requires swapping cuDNN to satisfy TransformerEngine/apex, which a shared
cluster will not permit. The slime path is kept because it is the paper's own and
because its OpenAI adapter is genuinely better for Harbor-driven RL — but it is not
the one to start with here.

A side benefit of the FSDP path: it has no context parallelism, so the open question
about CP correctness on the 48 gated-delta-net layers simply does not arise.

## What leaving Megatron actually costs

`slime/utils/arguments.py` declares `--train-backend` with `choices=["megatron"]`.
There is one option. So leaving Megatron means leaving slime, and slime is where
three things live:

1. `--loss-mask-type qwen3_5` — the verified multi-turn mask
2. the launchers and the Ray/parallelism plumbing
3. `slime.agent.adapters.OpenAIAdapter` — an OpenAI-compatible HTTP endpoint that
   records exact sampled token ids, which is what lets an **external** harness like
   Harbor drive the policy without re-tokenization

(1) is solved: `scripts/15_export_pretokenized.py` bakes the verified mask into the
data as `input_ids` + `loss_mask`, so it is backend-agnostic. (3) is the real cost —
see below.

## verl + FSDP

`verl-project/verl` (23 k★, active). Its `verl/workers/engine/` has six backends:
`fsdp`, `megatron`, `torchtitan`, `automodel`, `mindspeed`, `veomni`. Its Megatron
registry already lists `Qwen3_5ForConditionalGeneration`, and the FSDP path goes
through HF Transformers, which has `models/qwen3_5` as of 5.15.0.

```bash
MODEL_KEY=qwen3.5-9b bash scripts/30_run_sft_verl.sh
```

### Do NOT use verl's built-in multi-turn SFT dataset for this model

`verl/trainer/config/sft_trainer_engine.yaml` carries its own warning:

> `MultiTurnSFTDataset` apply_chat_template to each turn separately and concat
> `input_ids` as a whole sequence, which may not equal to apply_chat_template to
> whole messages at once. For example, Qwen Thinking series models add
> `<think></think>` tags to last turn … Set `ignore_input_ids_mismatch` to `True`
> to ignore input_ids mismatch.

We measured this on 200 real rows of our training set:

```
rows checked : 200
identical    : 0
MISMATCHED   : 200  (100 %)

example: whole = 14,236 tokens vs turn-by-turn = 14,258 tokens  (+22)
  <think> blocks in whole render      : 1
  <think> blocks in turn-by-turn      : 21
  assistant turns                     : 21
```

The mechanism: the Qwen3.5 template injects an empty `<think>\n\n</think>\n\n`
before the **last** assistant turn. Build the sequence turn-by-turn and every
assistant turn is "last" at some point, so a 21-turn conversation gets 21 think
blocks instead of 1. `ignore_input_ids_mismatch: True` silences the assertion; it
does not fix the sequence, and you would then train on tokens serving never emits.

**Therefore:** feed pre-tokenized data through verl's `data.custom_cls` hook.
`verl_backend/rst_sft_dataset.py` does this, and `30_run_sft_verl.sh` refuses to
start if the export dropped rows for a contract failure or if the trained-token
fraction leaves the 0.25–0.45 band measured for this dataset.

### Memory: no context parallelism, so fused CE is mandatory

verl's FSDP engine shards parameters, not the sequence. One GPU must hold a whole
32 K sequence's activations. The binding term is not the activations — it is the
**logits**, because this model's vocab is 248,320:

| term | 27.8 B, 32×80 GB, seq 32 K |
|---|---|
| params + grads + Adam, sharded | 444.8 GB / 32 = **13.9 GB/GPU** |
| activations (full recompute, 64 layers) | **21.5 GB/GPU** |
| logits | **~29.9 GB measured** (the loss upcasts to fp32; independent of model size) |
| working set | ~2 GB |

So a fused linear cross-entropy — one that chunks over the sequence and never
materializes full logits — is load-bearing, not an optimization. With one, ~37 GB/GPU
→ comfortable on 80 GB. Without one you OOM. On 40 GB cards this does not fit; use
optimizer CPU offload and/or 16 K sequences, and say so in the report because it
changes what the run means.

### On the verl path, `model.use_liger=True` is NOT that fused CE

Stated separately because it cost a run. Liger-Kernel does ship `qwen3_5.py` /
`qwen3_5_moe.py` with an FLCE, and Liger's own default for that kernel is on — but
verl's FSDP engine applies Liger with the flag forced off:

```python
# verl/workers/engine/fsdp/transformer_impl.py
# Apply Liger kernel; disable fused_linear_cross_entropy (conflicts with verl's forward patching)
_apply_liger_kernel_to_instance(model=module, fused_linear_cross_entropy=False, swiglu=True)
```

So `use_liger` buys swiglu + rms_norm only, and the run says so in one line — this is
what to grep for in a suspicious run:

```
Applying Liger kernels to model instance with model type: qwen3_5 with kwargs:
{'fused_linear_cross_entropy': False, 'swiglu': True}
```

verl fuses the CE through its own switch instead:

| flag | effect |
|---|---|
| `model.use_fused_kernels=True` | patch the model's `forward` at all |
| `model.fused_kernel_options.impl_backend=torch` | `verl/models/transformers/qwen3_5.py::forward_with_torch_backend` — `FusedLinearForPPO`, chunked |
| `…=triton` | `forward_with_triton_backend` — a triton `linear_cross_entropy`; faster, not exercised on SM80 here |

Both return `log_probs`, and `verl.workers.utils.losses.sft_loss` reads
`model_output["log_probs"]`, so the loss is the same function either way — only the
memory differs. With `use_fused_kernels=False` the fallback
`forward_with_normal_backend` computes `self.lm_head(hidden_states)` in full, and
**the run then reports ~78 GB/GPU of activations beside ~2 GB of sharded parameters,
which reads like a parallelism misconfiguration and is not one.**
`scripts/30_run_sft_verl.sh` sets both flags and gates on those config keys existing
in the installed verl *before* it launches 32 processes (`FUSED_KERNELS=0` opts out).

### Measured on real data, one GPU, Qwen3.5-0.8B (0.75 B params)

`scripts/16_smoke_forward_backward.py` runs a real forward/backward over the
pre-tokenized rows and reports peak memory. Results on an H100-80GB (sm90) with
torch 2.13+cu130, `attn_implementation=sdpa`, gradient checkpointing on:

| sequence length | peak, unfused CE | peak, Liger fused CE | outcome |
|---|---|---|---|
| 4,096 | 14.98 GiB | 5.52 GiB | −63 % |
| 8,192 | 28.43 GiB | 6.75 GiB | −76 % |
| ~16,000 | 48.34 GiB | 8.57 GiB | −82 % |
| **32,329** | **OUT OF MEMORY** | **13.14 GiB** | Liger is the difference between running and not |

At 32,329 tokens the unfused cross-entropy tried to allocate a single
**29.85 GiB** tensor (`32,048,676,864` bytes) on top of 48.11 GiB already held, and
died. Note what that number is: `≈ seq × vocab × 4 bytes` = 32,268 × 248,320 × 4.
The loss upcasts logits to fp32, and **this term does not depend on model size** —
FSDP shards parameters, not logits. So on the 27.8 B model the same 29.85 GiB
appears, with *more* activation memory beside it.

That is why a fused CE is stated as mandatory rather than recommended. It is not a
throughput optimization; without one a 32 K sequence does not fit on an 80 GB card
even for a 0.75 B model.

Liger's loss also matches the unfused loss where both run (0.4505 vs 0.4495 at 4 K;
0.3745 vs 0.3742 at 16 K), so the saving costs no accuracy.

One caveat on reading that table: `16_smoke_forward_backward.py` patches Liger into a
plain HF model, so the "Liger fused CE" column measures *Liger's* FLCE. verl disables
exactly that kernel (above) and substitutes its own, so the column proves the size of
the logit term and that a fused CE removes it — it does not prove that
`model.use_liger=True` removes it under verl. Only `model.use_fused_kernels=True`
does that on the verl path.

`fla` is *not* required on the FSDP path: transformers' qwen3_5 declares
`@use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")`, so there
is a pure-PyTorch fallback. Installing `fla` is a speed choice.

### RL: verl has no OpenAI adapter, so the bridge is ours

verl's AgentLoop reaches the policy through
`server_manager.generate(request_id, prompt_ids=[...], sampling_params=...) -> TokenOutput`
— token ids in, token ids out, **in-process**. It is not an HTTP endpoint. Harbor
speaks OpenAI `/v1/chat/completions`, so something must sit in between. slime ships
exactly that; verl does not. This is the main concrete cost of the verl path.

`verl_backend/harbor_agent_loop.py` implements the bridge: an aiohttp shim that
accepts Harbor's requests, applies the chat template, calls `server_manager.generate`,
returns OpenAI-shaped JSON, and **records the sampled ids per turn**. Sessions are
keyed off the `Authorization: Bearer` token, which is how Harbor's LiteLLM client
passes its API key.

We do not re-tokenize Harbor's response text. That would let the ids we train on
differ from the ids that were sampled, making the importance ratio wrong and the run
silently off-policy — no metric would flag it.

`Session.assemble()` is the delicate part, because Terminus-2 re-sends the whole
conversation every turn. It walks turns, appends only the newly-revealed prompt
prefix (`mask 0` — harness text and terminal observations) then the sampled response
(`mask 1`), and **raises** if a prompt diverges from recorded history (context
compaction), rather than emitting a sequence that never existed. Tested:

```
2-turn re-send        : no duplication, mask=1 only on sampled tokens   OK
single turn                                                            OK
history rewritten     : raises instead of corrupting                   OK
20-turn growth        : trained tokens == sum of response lengths       OK
```

**Everything in `harbor_agent_loop.py` is UNVERIFIED against a running verl.** The
interface matches the real API (`AgentLoopBase.__init__`, `@register`,
`run(sampling_params, **kwargs) -> AgentLoopOutput`, `AgentLoopOutput.response_mask`),
but it has never been executed. The two smoke tests in `RL_PLAN.md` apply here too,
plus one more: assert that `Session.assemble()`'s `mask==1` spans decode to exactly
the assistant text Harbor recorded.

## Rejected, and why

| candidate | verdict |
|---|---|
| **torchtitan** | Has a real `models/qwen3_5/` (FSDP/HSDP, TP+SP with head-sharded TP on the GatedDeltaNet projections, EP, PP), verified numerical parity vs HF (~3e-7 KL, 100 % top-1/top-5 match, bit-identical logits across FSDP/EP/TP). But **no RL**, and its README lists **"Add Context Parallel (CP) support"** under TODO. Best choice if you only want SFT and value correctness evidence. |
| **HF Transformers + FSDP2 / DeepSpeed ZeRO-3** | Works (transformers 5.15 has qwen3_5 with a pure-PyTorch GDN fallback), needs Liger fused CE for the same reason as above. Most flexible, but you write the trainer, the multi-turn masking and the RL loop yourself. |
| **LLaMA-Factory** | Already ships `qwen3_5` **and** `qwen3_5_nothink` templates; easiest SFT path. SFT/DPO only — no agentic RL, so it cannot carry the second half of this project. |
| **Axolotl** | Same tier as LLaMA-Factory; qwen3_5 support not confirmed. |
| **MiniMax-M2 as a model** | 240 B (447.8 GiB). Full-parameter SFT needs ~2.9 TB of optimizer state. Not sensible on 32 A100s. |

## A cross-check worth keeping in mind

torchtitan is PyTorch's own reference implementation of this architecture, it has
TP+SP, EP and PP working with bit-identical logits — and it still lists **context
parallelism as TODO**. Linear-attention layers carry recurrent state across the
sequence, so CP over them is not a free lunch. That is independent corroboration
for the CP1-vs-CP2 comparison required in `OPERATOR_PROMPT.md` before trusting
Megatron's CP4 default on the 48 gated-delta-net layers. It is also a reason the
verl/FSDP path (no CP at all) is a legitimate fallback rather than a downgrade.


---

## Container runtime, without Docker

Neither backend needs a container runtime for SFT. Evaluation and RL do, because
Harbor builds each task's Dockerfile and drives tmux inside the container.

**Rootless podman is the answer**, because it serves the same Docker API Harbor
speaks. `source scripts/00b_setup_sandbox.sh` finds or starts it and exports
`DOCKER_HOST`; Harbor then needs no patch. Verified end to end on a box with no
`docker.sock` permission:

| step | result |
|---|---|
| `podman build` of a real RST task Dockerfile | **26.6 s**, incl. `apt-get install`, `git clone`, `pip install` |
| docker CLI → podman socket | client 29.2.1 / server 3.4.4 / **API 1.40** |
| `run -d --network none` then `exec` | OK |
| tmux 3.2a `new-session` / `send-keys` / `capture-pane` | OK |
| network inside the container | none, confirmed |

Rootless podman is a *stronger* answer to the original security requirement, not a
compromise: task Dockerfiles are untrusted third-party build scripts, and here there
is no privileged daemon to hand them to at all.

### Four failure modes that only appear once you build the real images

Found by building the actual task pool, not by reading docs:

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | `SHELL is not supported for OCI image format` | 16 % of task Dockerfiles use `SHELL`; podman defaults to OCI | `--format docker` (automatic) |
| 2 | `Unknown instruction: IF` | 31 % use heredocs (`RUN <<EOF`); needs podman ≥ 4.4, Ubuntu 22.04 ships 3.4.4 | static rootless podman, 32 MB, no root: `podman-static` v5.8.4+ |
| 3 | `fatal: hardlink different from source` | `git clone` inside a rootless kernel-overlay build | retry in a vfs store under a separate `--root` (automatic) |
| 4 | `unknown shorthand flag: 'f'` | podman < 4 has no `compose` subcommand; 710 tasks (13.8 %) are multi-service | podman ≥ 4.x, or exclude those tasks and say so |

(2) is the dangerous one: it looks like a broken task rather than a stale toolchain,
so it would quietly remove a third of the pool. `11_prebuild_images.py` therefore
aborts on it instead of building the other two thirds.

**Correction.** An earlier version of this file said Harbor cannot drive
Apptainer/Singularity without a new backend. That is wrong: Harbor 0.21.0 ships
`environments/singularity/`. The real limitation is narrower — it wants a docker
image reference or a `.sif`, not a task `Dockerfile`, so it does not fit the RST task
format without a build step of your own. Same for `hf-sandbox`, which requires a
prebuilt `[environment].docker_image` and refuses a Dockerfile.

## When this machine cannot run a container at all

Rootless podman needs `mount(2)`. A pod under AppArmor's `docker-default` profile
cannot call it — the profile contains a literal `deny mount,` and permits only
`umount`. Measured signature, on a pod where podman 5.8.4 / crun 1.28 / netavark
were all installed and `podman info` was clean:

| probe | result |
|---|---|
| `unshare -U`, `unshare -Ur` | ✅ succeed |
| `mount -t tmpfs` | ❌ **EACCES** |
| `mount --bind` | ❌ **EACCES** |
| `mount --make-rprivate /` | ❌ **EACCES** |
| `/proc/self/attr/current` | `docker-default (enforce)` |
| `Seccomp` in `/proc/self/status` | `0` — not seccomp |

It fails inside a *self-created* user namespace where the process holds full
capabilities, and EACCES (not EPERM) is the LSM signature. There is no in-pod
workaround: AppArmor profiles are inherited across `unshare()` and `execve()`, and
`docker-default` grants no `change_profile` rule, so nothing can transition out of
it. fuse-overlayfs, vfs, bubblewrap and every other approach mount, so all are dead.
`bash scripts/00b_setup_sandbox.sh --diagnose` runs these probes and prints the
verdict; the minimal ops ask is **one flag**, `--security-opt apparmor=unconfined`
(not SYS_ADMIN, not `/dev/fuse`, not privileged — see `OPERATOR_PROMPT.md`).

### Backends that run the container elsewhere

These need no container privilege on this machine, so they unblock eval and RL
without waiting for ops. `00b_setup_sandbox.sh` selects one from the credentials it
finds and exports `RST_HARBOR_ENV` / `RST_HARBOR_ENV_KWARGS`; `06_eval.py`,
`rl/generate.py` and `verl_backend/harbor_agent_loop.py` all honour it.

| backend | needs | how the task image gets built |
|---|---|---|
| `daytona` | outbound HTTPS + `DAYTONA_API_KEY` | declarative build → content-hash snapshot `harbor__<env_hash>__snapshot` |
| `e2b` | outbound HTTPS + `E2B_API_KEY` | `Template().from_dockerfile` → `AsyncTemplate.build` |
| `modal` | outbound HTTPS + a Modal token | `Image.from_dockerfile` |
| remote Docker daemon | reachable `DOCKER_HOST` **and the task tree at the same absolute path on the daemon host** (Harbor bind-mounts it; build contexts stream over the socket and need no sync) | locally-authored, built there |
| `gke`/`ack`/`openshift` | namespace + RBAC to create sibling pods | in-cluster |

Daytona is what the RST paper itself used. Terminus-2 is a **host-side** agent — it
drives the container via `environment.exec(...)` — so the agent loop and every model
call stay in this pod, talking to the local vLLM/SGLang shim. Nothing inbound to the
pod is required, and `--network none` inside the task container still holds.

Two asymmetries this introduces, both handled in code:

- **Proxy.** `--env docker` must have `HTTP(S)_PROXY` *stripped* (the vLLM shim is
  local); an off-machine backend must *keep* it, with the local endpoints added to
  `NO_PROXY`. Ray's `runtime_env` replaces rather than extends the worker
  environment, so `12_run_grpo.sh` names the proxy vars explicitly in
  `RUNTIME_ENV_JSON`.
- **Prebuilding.** `11_prebuild_images.py` cannot warm a cache it does not own. It
  detects the off-machine case, writes `prebuild_report.json` with `"skipped": true`
  and a reason, and exits 0 — a warm-cache step that is structurally impossible is
  not a failure. The first rollout per distinct image pays the provider-side build
  once. `RST_MAX_SANDBOXES` also stops being a local-resource number and becomes the
  provider's concurrency quota; exceed it and the 429s are (correctly) classified as
  infrastructure failures and dropped.

### What they cost for this workload

Published rates, read 2026-08-18. Priced against the sandbox shape
`12_run_grpo.sh` actually asks for — **2 vCPU / 4 GiB per rollout**, 64 rollouts per
GRPO step (`rollout-batch-size 8 × n-samples-per-prompt 8`), 3–14 min per rollout
measured from the reference trajectories (§4), so 3.2–14.9 sandbox-hours per step
with 8.5 at the midpoint.

| | rate for 2 vCPU + 4 GiB | per GRPO step (mid) | 200 steps | free credit | concurrency |
|---|---|---|---|---|---|
| **E2B** | $0.0504/vCPU-h + $0.0162/GiB-h = **$0.166/h** | **$1.41** ($0.53–2.47) | ~$283 | $100 one-time (Hobby) | 20 (Hobby) / 100 (Pro $150/mo) |
| **Modal** *Sandbox* | $0.1419/core-h + $0.0240/GiB-h = **$0.238/h** | **$2.03** ($0.76–3.55) | ~$406 | $30/mo (Starter) / $100/mo (Team $250/mo) | 100 containers (Starter) |
| **Modal** *standard Function* | $0.0472/core-h + $0.0080/GiB-h = **$0.079/h** | **$0.68** | ~$135 | same | same |
| **Daytona** | **$0.0858/vCPU-h ⇒ $0.172/h, UNVERIFIED**, plus unpublished RAM + disk | ≥$1.46 | ≥$293 | $200; up to $50k startup credits | pooled, see below |

E2B: `$0.000014/vCPU/s` and `$0.0000045/GiB/s`; Hobby caps a session at 1 h, which is
above our longest measured rollout but will kill a hung one — acceptable, since
`rl/generate.py` classifies that as an infrastructure abort rather than reward 0.

Modal bills a **physical core = 2 vCPU**, so our 2-vCPU rollout is one core. Note the
two rate cards: `$0.00003942/core/s` for Sandboxes vs `$0.0000131/core/s` for standard
Functions — the same compute is **3× cheaper outside the Sandbox product**, so if the
Harbor `modal` backend can be pointed at a Function-shaped runner that is the single
largest saving available anywhere in this table. Do not pin a region (1.5–1.75×) or
ask for non-preemptible (3×). Volumes are $0.09/GiB/mo with 1 TiB/mo free, which
covers the image cache outright.

Daytona is what the RST paper used, and it is the one whose bill you cannot predict
from public pages. Three things are documented and matter more than the rate:

- **Billing is on *reserved* resources, not consumed** — a mostly-idle rollout costs
  the same as a busy one, so oversizing the sandbox is pure waste.
- **A *stopped* sandbox still bills disk**; only `archived` (containers) is free, and
  **a deleted sandbox's snapshots keep billing for storage**. This is the real trap
  for us: Harbor's daytona backend builds one content-hash snapshot per distinct task
  environment (`harbor__<env_hash>__snapshot`), and a run that walks 1,600 distinct
  tasks leaves 1,600 snapshots of multi-GiB task images behind at an unpublished
  per-GiB rate. Delete them between runs; do not merely stop sandboxes.
- **Resources are a pooled tier, not a sandbox count.** Tier 1 is 10 vCPU / 30 GiB
  disk, which at 2 vCPU per rollout is **5 concurrent sandboxes, not 16** — Tier 2
  (100 vCPU / 200 GiB / 300 GiB) is the first tier that fits `RST_MAX_SANDBOXES=16`.
  The docs' own two tables disagree on Tier 1 RAM (10 vs 20 GiB).

The `$0.0858/vCPU/h` figure sits next to an OS/Windows selector on the pricing page,
and no RAM or disk rate is published anywhere, so treat any Daytona total as a floor
and get a quote before committing a long run. The docs also warn charges can lag
usage by up to 48 h, so a runaway will not show up in the dashboard promptly.

Two cost items apply to every provider. The first rollout per distinct task image pays
a provider-side build (99 % of these Dockerfiles `apt`/`pip install`), which is tens of
dollars over a full run, not hundreds — but it is billed compute, and it is why §2's
prebuild step being skipped is a cost note as well as a correctness note. And none of
this applies to the DPO fallback (`DPO_PLAN.md`): it needs no container at all, so its
marginal cloud cost is zero.
