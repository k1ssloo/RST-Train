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
| logits, bf16 | **16.3 GB** — and ~32.6 GB if the loss upcasts to fp32 |
| working set | ~2 GB |

So `model.use_liger=True` is load-bearing, not an optimization: Liger-Kernel ships
`qwen3_5.py` / `qwen3_5_moe.py` with a fused linear cross-entropy that chunks over
the sequence and never materializes full logits. With it, ~37 GB/GPU → comfortable
on 80 GB. Without it you OOM. On 40 GB cards this does not fit; use optimizer CPU
offload and/or 16 K sequences, and say so in the report because it changes what the
run means.

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

**Apptainer/Singularity does not work unchanged** — it does not serve the Docker
API, so Harbor cannot drive it without a new environment backend, and that backend
is not written here.
