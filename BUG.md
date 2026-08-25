# BUG.md — defects found reviewing the 4×8 A100 SFT failure

Written after the 2026-08-19 27B run died in `loss.backward()` on all four nodes. Scope: the
whole SFT and DPO path, not just the OOM. Every entry says how it was established, because
"read the source" and "measured it" are different kinds of claim and only one of them
survives a version bump.

Evidence used throughout:

| source | what it is |
|---|---|
| `logs/train_rank1.log` (HF: `khazic/rst-qwen3.5-27b-sft`) | the failing run, node 1 |
| torch `2.13.0+cu130` source, this checkout's `.venv-gpu` | the FSDP2 claims, line by line |
| verl `main` (`volcengine/verl`) | the verl claims — **the cluster runs 0.9.0, so line numbers drift; function names and logic were checked, not offsets** |
| one local H100 80GB | `scripts/35_probe_fsdp2_grad_accum.py` and two purpose-built probes |
| `data/sft-v1-cap10/pretokenized_train.parquet` | 10,578 rows, 99,939,485 tokens, 32.42 % trained, longest 32,329 |

Model, for the arithmetic below: 27.78 B params, 64 layers (48 gated-delta-net + 16 full
attention, `full_attention_interval=4`), `hidden_size` 5120, **`vocab_size` 248,320**,
`tie_word_embeddings=false`, plus a `vision_config` (27-block ViT) and
`mtp_num_hidden_layers=1`.

---

## BUG-1 — FSDP2 retains a full **unsharded** fp32 gradient across micro-batches

*SFT · fatal · fixed in `1c0c25d` · **measured***

`verl/workers/engine/fsdp/transformer_impl.py::FSDPEngine._gradient_sync_context` calls
`set_requires_gradient_sync(False)` on every non-final micro-batch. FSDP2 answers that at
`_fsdp_param_group.py:607-612`:

```python
if not self.reduce_grads:
    if self.reshard_after_backward:
        self.reshard()
    for fsdp_param in self.fsdp_params:
        fsdp_param.to_accumulated_grad_if_needed()
    return                                  # <-- no reduce-scatter
```

and `to_accumulated_grad_if_needed` (`_fsdp_param.py:802-813`) stores
`unsharded_grad.to(self.reduce_dtype)` — a **whole-parameter** fp32 tensor, not a shard.
verl's mixed-precision policy is `param_dtype=bf16, reduce_dtype=fp32`, so the early-return
guard (`grad.dtype == reduce_dtype`) never fires.

**Cost:** `27.78e9 × 4 B = 103.5 GiB/GPU`, allocated ~1.43 GiB at a time as backward walks
layer 64 → layer 1. Proportional to **parameters**, so more GPUs, a smaller token budget,
less data, `engine.optimizer_offload` and `ULYSSES_SP` all leave it untouched.

**Why the log reads the way it does.** Not one impossible allocation — memory accumulating:

- `75.36 GiB` held of a `79.33 GiB` card, request only `552 MiB`
- traceback inside `loss.backward()` → `_backward_prefetch` → `unshard` → `all_gather`
- `NumelOut/NumelIn = 383,437,056 / 11,982,408 = 32.0` exactly ⇒ **the shard degree really
  was 32**, so this is not the earlier rendezvous bug
- `383,437,056` is one decoder layer (GDN 115 M + MLP 267 M), i.e. FSDP wraps per layer

**Fix:** `verl_backend/fsdp2_grad_accum.py` neutralizes `_gradient_sync_context`, so FSDP2
reduce-scatters every micro-batch and accumulates into the fp32 **sharded** gradient
(`_fsdp_collectives.py:744-749`). A sum of reduce-scatters is the reduce-scatter of a sum:
no numerical trade, only bandwidth (one extra reduce-scatter per micro-batch).

**Measured**, `scripts/35_probe_fsdp2_grad_accum.py` on one H100:

```
verl default : 8 unsharded_accumulated_grad tensor(s), 33.6M/33.6M params, dtype fp32
patched      : 0 tensor(s); sharded grads already populated
max relative difference over 8 gradient tensors: 0.000e+00
```

Applied at import time from `verl_backend/rst_sft_dataset.py`, which verl loads through
`data.custom_cls.path` in every rank, so a cluster-side wrapper cannot bypass it. **Grep the
log for `[rst-fsdp2]`.**

> The log published on HF **predates this fix**: it contains `[rst] guarded FSDPParam…`, an
> earlier patch that only stopped an `AttributeError` on the same code path and left the
> 103.5 GiB in place. It is evidence the bug exists, not evidence the fix failed.

---

## BUG-2 — multi-rank DPO dies before its first gradient

*DPO · fatal · fixed · **reproduced***

`dpo_common.masked_logprob_sum` deliberately never calls `model(...)` — one
`model(input_ids)` would materialize `seq × 248,320` logits. It calls
`model.get_decoder()(input_ids=...)` and then `model.get_output_embeddings()(states)` once
per `--logit-chunk` slice.

`19_train_dpo.shard_model` used to `fully_shard` each decoder layer **and the root model**.
FSDP2 unshards a group from the hook on the module handed to `fully_shard`, so the root
group — `embed_tokens`, the final norm, `lm_head`, the ViT — was sharded and then never
unsharded, because `model.__call__` never runs.

Reproduced on one H100 with the exact layout (`world_size=1` still wraps params as DTensor,
so it needs no second GPU):

```
A. model(input_ids)              [normal path]  : loss=0.3408  params_with_grad=6/6
B. decoder(...) then head(...)   [the DPO path] : RuntimeError: aten.embedding.default got
   mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before
   calling distributed operators!
```

It fires in **gate 3's calibration forward**, before a single gradient — i.e. on every
multi-rank DPO run, immediately. `scripts/33_run_dpo.sh:436` launches
`torchrun --nnodes $NNODES --nproc_per_node $NGPUS`, so the 27B run would have hit it.

**Why review missed it:** `shard_model` returns early at `world_size == 1`, and the only DPO
run so far was single-GPU (README: "run end to end on 0.8B/H100"). The ✅ was accurate about
what was run and silent about what was not.

**Fix:** make the two modules that are *called* the FSDP entry points —

```python
for layer in layers:
    fully_shard(layer, mp_policy=policy)
fully_shard(head, mp_policy=policy, reshard_after_forward=False)
fully_shard(decoder, mp_policy=policy)
```

`reshard_after_forward=False` on the head is load-bearing: it is re-entered once per logit
chunk, and resharding would all-gather a `248,320 × 5120` matrix each time. Anything outside
`decoder`+`lm_head` (the ViT on a VLM checkpoint) is left unsharded and
`requires_grad_(False)` — DPO passes no `pixel_values`, so it can never earn a gradient, and
freezing also stops AdamW decaying it (torch's AdamW skips `grad is None`, **not** a zero
grad). The count is printed, not silent.

Verified with the real `shard_model` + the real `masked_logprob_sum` on a VLM-shaped model:

```
[dpo] 1 param tensor(s) live outside decoder+lm_head (e.g. visual.weight); frozen
RESULT: logp=-78.1446 n=16  trainable_with_grad=6/6  frozen=['visual.weight']
clip_grad_norm_ over DTensor params = 39.3010 (finite=True)
```

Regression-locked by `tests/test_dpo_sharding.py`, which asserts the *layout* so it runs
without CUDA.

---

## BUG-3 — `ULYSSES_SP=8` with an undivided token budget buys nothing and costs memory

*SFT · high · fixed · **derived from verl source, arithmetic checked against the data***

Three facts compose badly:

1. `verl/workers/engine/utils.py::prepare_micro_batches` sets
   `max_token_len = max_token_len_per_gpu * sp_size`;
2. `dp_size = world_size // sp`;
3. `scripts/30_run_sft_verl.sh` passed `data.max_token_len_per_gpu="$MAX_TOKENS_PER_GPU"`
   (32768, sized by the registry as "one GPU holds a whole sequence") **regardless of
   `ULYSSES_SP`**.

So the micro-batch count is a function of `max_token_len_per_gpu` alone, and **per-GPU token
count == `max_token_len_per_gpu` at every `sp`**. Measured on the real data:

| `mtpg` | `sp` | group budget | tok/GPU | micro-batches (typ/worst) |
|---|---|---|---|---|
| 32768 | 1 | 32,768 | 32768 | 2 / 4 |
| 32768 | **8** | 262,144 | **32768** | **2 / 4** ← identical |
| 8192 | 8 | 65,536 | 8192 | 5 / 16 |
| 4096 | 8 | 32,768 | **4096** | 10 / 31 |

**The only thing SP buys is the ability to put `max_token_len_per_gpu` *below*
`max_seq_len`** — which `sp=1` can never do, because bin packing cannot split one sample and
the longest row is 32,329 tokens. At `mtpg ≥ max_seq_len`, `sp=1` gives identical packing and
identical per-GPU footprint, so `sp>1` is strictly dominated while paying:

- an all-to-all per full-attention layer (16 layers);
- FLA `cp_context` recurrent-state passing **and** a `conv1d` halo exchange per
  gated-delta-net layer (48 layers);
- an `inputs_embeds` tensor `sp` times larger, **twice**. This config has a `vision_config`,
  so verl takes the VLM branch and only *pads* `input_ids` (`ulysses_pad`) — the slice
  happens later inside the patched `Qwen3_5TextModel.forward`. `_get_input_embeds` therefore
  embeds all 262,144 tokens (`262144 × 5120 × 2 B = 2.56 GiB` vs 320 MiB at `sp=1`), and
  verl adds `0.0 * image_embeds.mean()` to keep the ViT in the graph, producing a second
  full-length tensor. ≈ 5 GiB/GPU of pure waste at `sp=8`.

Meanwhile the decoder-activation line it *should* have reduced stayed at
`64 × 32768 × 5120 × 2 B = 21.5 GiB`.

**Fix:** `ULYSSES_SP` is verl's context parallelism, so the registry now divides the budget
by it exactly as it folds Megatron's `cp` into it (`model_registry.py --ulysses-sp`), the
invariant became `max_tokens_per_gpu × (cp × ulysses_sp) ≥ max_seq_len`, and
`30_run_sft_verl.sh` passes the operator's `ULYSSES_SP` in and **refuses** `SP>1` with
`mtpg ≥ max_seq_len` (`ALLOW_OVERSIZED_SP_BUDGET=1` overrides). `ULYSSES_SP=8` now yields
`MAX_TOKENS_PER_GPU=4096`, group budget 32,768 ≥ 32,329, and `21.5 GiB → 2.7 GiB` of
checkpointed activations.

Two guards came with it: `sp` must divide the world size (else verl cannot build the
`(dp, sp)` mesh), and `sp > gpus_per_node` is warned about (the all-to-all and the GDN state
exchange then cross the inter-node fabric every layer).

Loss-neutral: `sft_loss` normalizes by `batch_num_tokens`, the global trained-token count
(`transformer_impl.py:706-710`), and with BUG-1 fixed the accumulation is exact. Changing
`mtpg` moves memory and throughput, not the objective.

---

## BUG-4 — a supervised first token would leak across a packed document boundary

*SFT data · medium · fixed · latent (no current row hits it) · **verl source***

`verl/workers/utils/losses.py::sft_loss` aligns the mask to the log-probs with

```python
loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)
```

`torch.roll` is **cyclic** and this runs on the *flattened, packed* micro-batch, not per
sample. So sample A's last position inherits sample B's **first** mask value, and the
batch's final position inherits the batch's first. The alignment is correct — position *i*
predicts token *i+1*, and `mask[i+1]` says whether that token is supervised — but its
safety rests entirely on `loss_mask[0] == 0` for every row.

It holds for everything `15_export_pretokenized.py` produces (a conversation opens on
`<|im_start|>`, never a target), and nothing asserted it. If it ever broke, the symptom
would be a few tokens of cross-document supervision per micro-batch — invisible in the loss
curve.

**Fix:** asserted in three places, cheapest first — the exporter drops such a row (`--strict`
aborts), `RSTPretokenizedSFTDataset.__init__` refuses the table, and the launcher's parquet
gate refuses before `torchrun`.

---

## BUG-5 — the micro-batch gate certified configurations that cannot be placed

*gate · medium · fixed*

`estimate_micro_batches` returned `ceil(total_tokens / budget)`, which does not care that a
single **sample** may exceed the budget — bin packing cannot split one. `mtpg=4096, sp=1`
against a 32,329-token row therefore came back as a valid 10-micro-batch plan; verl would
have discovered it inside `rearrange_micro_batches`, mid-step, after 32 ranks read the
weights. It now raises, and `_cli` prints a `FATAL:` line and exits 2 instead of leaking a
traceback out of what is used as `... || exit 2`.

---

## BUG-6 — the offload gate claimed "GPU static ~0" for the wrong knob

*gate · low · fixed*

Two keys one word apart:

- `engine.offload_policy=true` — FSDP2 `CPUOffloadPolicy`: sharded **params, grads and**
  optimizer state on the host. GPU static ≈ 0. This is what `OFFLOAD_OPTIM=1` sets.
- `engine.optimizer_offload=True` — Adam state only; fp32 master + fp32 grad
  (8 B/param ÷ shard, **6.5 GiB/GPU** here) stay resident. **This is what the failing run
  actually passed.**

The gate printed "GPU static ~0" and `exit 0` for both, and `OFFLOAD_OPTIM`'s name points at
the weaker one. Now it names the key it set and prints the resident figure for the other.

---

## BUG-7 — `save_hf` cast integer buffers to bf16

*DPO · medium · fixed*

`state = {k: v.to(torch.bfloat16) for k, v in state.items()}` hit every tensor. An integer
or bool buffer in the state dict is silently corrupted, and the checkpoint then *loads* and
misbehaves rather than failing. Guarded with `v.is_floating_point()`.

---

## BUG-8 — DPO gate 1 pinned the weights but not which pairs were scored

*DPO · medium · fixed*

Reference logprobs are constants of a `(checkpoint, dataset, mask, dtype, chunking)` tuple.
Gate 1 compared only `checkpoint_fingerprint`. `18_dpo_ref_logprobs.py` skips pairs over its
own `--max-seq-len`, so a different value in `19` made **gate 2 (coverage)** fail with a
message pointing at an unfinished reference pass rather than at the mismatch. Now
`max_seq_len` is a hard gate-1 failure and `logit_chunk` a recorded warning (it only
regroups an fp32 sum, ~1e-8/token, so it costs gate 3 its "exact" verdict and nothing else).

---

## BUG-9 — DPO exhausts host RAM at load with no message

*DPO · medium · mitigated (warning) · **arithmetic***

`load_model` calls `from_pretrained` without `device_map`, so **every rank** builds a full
CPU copy before FSDP takes its shard. 27.78 B in fp32 × 8 local ranks ≈ **888 GiB of host
RAM**. The OOM killer answers by SIGKILLing one process, and the other 31 ranks hang in NCCL
until the watchdog fires ~10 minutes later with no mention of memory anywhere.

`check_host_ram()` now measures the checkpoint, multiplies by `LOCAL_WORLD_SIZE` and the
dtype inflation, compares against `/proc/meminfo`, and warns before the first read. It is a
warning, not a refusal — the real fix is a rank-by-rank or meta-device load, which is not
done.

---

## BUG-10 — the primary launcher never looked at the interconnect

*ops · low · fixed (note only)*

`05_run_sft.sh` and `12_run_grpo.sh` detect an empty `/sys/class/infiniband` and set
`NCCL_IB_DISABLE=1`; `30_run_sft_verl.sh` — the **primary** path — did neither, so a run can
be network-bound with nothing in its log saying so. The failing run had
`NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0` from the operator's own `env-rstverl.sh`, i.e.
**every FSDP2 all-gather crossed nodes over TCP**: 64 unshards of ~766 MB (bf16) per
forward, again per recompute, and BUG-1's fix adds one reduce-scatter per micro-batch on
top. The launcher now says so and names HSDP (`FSDP_SIZE=8` + `OFFLOAD_OPTIM=1`) as the
shape to try. See OPEN-1.

---

## BUG-11 — the container-free eval crashed, so the finished 4B checkpoint is unmeasured

*eval · high · fixed · **observed in the cluster's own log***

The 4B SFT run completed all 82 steps (loss 0.3167 → 0.2132, no divergence) and its report
still records the checkpoint as unmeasured: *"NO benchmark produced a score … Not even the
container-free fallback ran"*. The sandbox being blocked is expected — that is why
`06b_eval_offline.py` exists. The fallback dying too is not:

```
File ".../scripts/06b_eval_offline.py", line 243, in score_rows
RuntimeError: Expected all tensors to be on the same device, but got target is on cuda:7,
              different from other tensors on cuda:1 (wrapper_CUDA_nll_loss_forward)
```

`load_model` loads with `device_map="auto"`, so the head does not have to share a device
with the last decoder block — and when the head is **tied** to the input embedding,
accelerate keeps it near the *first* device while `hidden` comes off the *last*. Hence
logits on `cuda:1`, targets on `cuda:7`. The chunked scoring loop is the only place in this
repo where a tensor built from the *input ids* meets a tensor produced by the *head*, which
is why nothing else in the run noticed. (The 27B config has `tie_word_embeddings=false`, so
its head lands wherever accelerate puts it — the hazard is the same, only less certain.)

**Fix:** `gold = targets[block].to(logits.device)`. The gold tensor is `chunk × 8` bytes, so
it is the cheap side to move, and the `logits.argmax(-1) == gold` comparison one line below
— which would have raised next — is carried by the same move.

Regression-locked by `tests/test_offline_eval_scoring.py`. A sharded load cannot be
reproduced on a single-GPU box, so the device-follow is asserted against the source with the
traceback quoted; the same file checks the chunked arithmetic on CPU against an independent
full-logits computation (position i is predicted from hidden i−1, position 0 is never a
target however the mask is set, and `--chunk` must not change the number).

Rerunning it needs the checkpoint only, not a retrain:

```bash
python scripts/06b_eval_offline.py --model-path $BASE_FOLDER/runs/4b/out-hf-full \
    --holdout $BASE_FOLDER/runs/4b/sft-v1-cap10/rst_sft_holdout.parquet \
    --tokenizer <tokenizer> --out $BASE_FOLDER/runs/4b/eval/offline
```

The report generator's companion complaint — *"loss curve: no loss values scraped from
logs"* — is separate and not a trainer bug: verl's stdout went to `run_all.log`, not into
the run directory the report scans. Point `--run-dir` at the directory holding that stdout.

---

## BUG-12 — a missing checkpoint path was reported as a malformed Hub repo id

*eval · low · fixed · **observed in the cluster's own log***

After the 27B training stage failed, `out-hf-full` was never written, and the eval stage
printed three copies of

```
AutoModelForCausalLM: OSError: Repo id must be in the form 'repo_name' or
'namespace/repo_name': '/llm-align/liuchonghan/rst/out-hf-full'
```

— transformers falling through to the Hub resolver, once per auto class. Read literally it
accuses the operator of passing a malformed name, and it says nothing about the actual
state: an absolute path that does not exist. `load_model` now checks that first and says
so, before importing torch (which cannot make a missing directory exist and costs seconds
plus a CUDA context).

---

## BUG-13 — a failed DPO reference shard was reported without saying which one, or how

*dpo · medium · fixed · **observed in the cluster's own log***

The 27B `dpo` stage failed 25 s in with

```
a reference shard failed; see /llm-align/liuchonghan/rst/logs/dpo_ref_shard*.log
```

and every one of the sixteen logs that message points at ends in a complete

```
[done] 153 rows -> .../ref_logps_shard<n>.parquet
determinism probe: max |delta| = 0 nats
```

so the evidence the operator is sent to says the work succeeded. The cause is the wait
loop in `scripts/33_run_dpo.sh`:

```bash
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
```

`rc` is one bit. It keeps neither the shard id nor the exit status, and the pid it did
have is gone by the time anything is printed — so the failing process cannot be
identified even in principle, and a status of `137` (signalled / OOM-killed) is
indistinguishable from `1` (python raised).

That distinction is the whole diagnosis here. A non-zero status from a shard whose log
ends in `[done]` means the scoring finished, the parquet is on disk, and the status came
from *after* the last flush — interpreter teardown, a CUDA/NCCL destructor, or a signal.
That reads "re-run me, it costs a directory listing", because `18_dpo_ref_logprobs.py`
is idempotent and keeps every scored row. An empty log is the opposite: python never
printed, so it died before it started. The old message could not tell the operator which
of those had happened.

The launcher now keeps `shard_ids` beside `pids` and captures each status with
`wait "${pids[$i]}" || code=$?`, then hands the `<shard>=<code>` list to
`rst_explain_shard_failures` (`scripts/lib_env.sh`), which names every failed shard with
its status and reads that shard's own log to say which of the three cases it is —
including the log tail when the log stops short of `[done]`.

Pinned by `tests/test_dpo_shard_failure_report.py`, including a source assertion that
the collapsing wait loop does not come back.

**Still open on the cluster side:** *why* those shards returned non-zero after printing
`[done]` is not known — the old message destroyed the evidence. Re-running
`bash scripts/33_run_dpo.sh` is now the diagnostic rather than a rebuild: it resumes
every already-scored row and prints the shard and the status.

## BUG-14 — an NCCL watchdog timeout was answered with three hypotheses, all wrong
*dpo · medium · fixed · **observed in the cluster's own log***

The 4B DPO attempt at `2026-08-20T00:11` got all the way through the gates —
`reference coverage: 2,673/2,673 pairs over 8 shards`, `[gates] fingerprint ok`,
`[data] dropping 16 pairs to align with 8 ranks x 4 accum`, `2,432 pairs -> 76 steps` —
then produced no step at all for ten minutes and died:

```
[rank1] ... [Rank 1] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=9687, ...) ran for 600027 milliseconds
```

with six ranks reporting `SeqNum=1, OpType=BROADCAST` and a **different** `NumelIn`
each — 9687 / 14042 / 3328 / 5678 / 8365 / 6697 — while ranks 0 and 2 were already at
`SeqNum=2, OpType=ALLREDUCE` with 15,047,680 and 7,767,040. Then SIGABRT on all eight
ranks, `ChildFailedError`, `=== FAILED dpo` at 00:24:24. (Its 23:57 predecessor was a
different failure: a BUG-2 recurrence, `aten.embedding.default got mixed torch.Tensor
and DTensor`, raised at the cluster's `19_train_dpo.py:428` — a stale checkout without
our fix, which sits at line 560 here.)

The defect is not the timeout, which is a cluster-side condition; it is that
`33_run_dpo.sh` responded to it by printing its GATE 1 / GATE 2 / GATE 3 advice —
fingerprint mismatch, reference coverage, calibration tolerance — **none of which can
produce a watchdog timeout**, since all three are checked and passed before the first
collective. Worse, the trainer's output only streamed to the console, so the one
diagnostic detail in ~500 lines of C++ watchdog frames was not on disk to be read.

Those per-rank sizes are the entire diagnosis, and they read exactly two ways:

- **sizes (or sequence numbers) differ per rank** — the collective's size depends on that
  rank's own data, so the ranks are not running the same sequence of collectives. A
  divergence in the trainer (a per-row tensor implicitly replicated as a DTensor, an
  uneven micro-batch count). Re-running reproduces it; only a code change fixes it.
- **sizes identical everywhere** — the collective was well formed and one rank never
  arrived. The host OOM killer first, because `19_train_dpo.py` builds the whole model on
  CPU in *every* rank before FSDP shards it (params × bytes × local_ranks of RAM per
  node) and a SIGKILL there leaves the survivors in NCCL with no mention of memory
  anywhere; then a rank that raised its own traceback; then a short rendezvous.

**What changed.** `33_run_dpo.sh` now tees the trainer to
`$BASE_FOLDER/logs/dpo_train.log` (`pipefail` is already set and `tee` exits 0, so `RC`
is still torchrun's status), and its failure block calls `rst_explain_nccl_timeout`
(`scripts/lib_env.sh`) after the gate advice. That helper prints nothing unless the log
actually contains a watchdog timeout; when it does, it lists every rank with its own
collective and size, picks the correct one of the two readings above, surfaces any
`[rank*]: *Error` that was raised *before* the timeout as the real first failure, and
states the re-run cost (19 is not resumable, but the reference parquets from 18 are
kept). Pinned by `tests/test_nccl_timeout_report.py`.

**Not a defect in our sharding.** `shard_model` keeps `embed_tokens` inside the decoder's
FSDP group, the 9B trains clean with the same script, and this same 4B run completed 76
steps on a later manual attempt — so the BROADCAST divergence was not reproducible and is
not attributed to this repo's code.

## BUG-15 — a DPO run that changed nothing reported an improvement and no warnings
*dpo · medium · fixed · **observed in both finished runs***

The two DPO runs that reached the end wrote this into
`dpo/dpo_training_summary.json` and shipped it to the Hub:

| | holdout accuracy | holdout `reward_margin` | `clip_active_fraction` | `warnings` |
|---|---|---|---|---|
| 4B | 0.5 → **0.5938** | 6e-05 | 0.0 | `[]` |
| 9B | 0.5 → **0.5391** | 1e-05 | 0.0 | `[]` |

The accuracy is the number anyone will quote. It is also a **sign test**: `rank_score`
credits a pair whenever the margin clears `TIE_EPS = 1e-6`. The margin it is testing the
sign of is `beta × (logp_π − logp_ref)`, and the reference logprobs were scored in bf16
— good to roughly 1e-3 nats/token, the same figure `how_to_read_this` already quotes for
the step-0 residual. At `beta 0.1` that puts the arithmetic floor of the whole quantity
at 1e-4, and both runs came in **below** it. The 9B's `holdout_ties` going 64 → 1 is the
same fact from the other side: every exact tie was broken, by 1e-05 nats.

The gap is precise, and it was in our own comment. `TIE_EPS`'s note says a trained
margin is 1e-2..1, "six orders of magnitude away, so this band cannot swallow real
signal" — true of `TIE_EPS`, but nothing ever looked at the three decades in between,
where a margin counts as a preference while sitting far below anything that comment
would call trained. Both runs landed exactly there, and the only existing warning covers
the *opposite* regime (`clip_active_fraction > 0.9`, i.e. the clip and not `--lr` setting
the step size), so at `clip 0.0` nothing fired at all.

**What changed.** `dpo_common.noise_floor_warning()` compares the final holdout margin
against `beta × REF_LOGP_NOISE_NATS` and, when it falls below, appends a warning to
`runtime_warnings` — so it is both printed and archived in the summary's `warnings`,
which is the field an empty list let these two runs through. It quotes the accuracy and
the margin together, the tie collapse, the last training step's margin, and says what to
do: treat the run as a plumbing success and a training no-op, and raise `--beta`/`--lr`
or give it more steps or pairs. The floor scales with beta because the reward does; a
genuinely trained margin, a real negative margin, and a run with no holdout eval all stay
silent. Pinned by `tests/test_dpo_noise_floor.py` against the two observed summaries.

**Not a numerical bug.** The gates are all exact (`step0_loss` = log 2 to the bit,
`calibration_passed`, `dtype_match`). The pipeline is correct; at `lr 5e-7`, `beta 0.1`,
one epoch over 2,432 pairs, length-normalized, it simply does not move the model. That is
a training-regime decision for the operator to make — the defect was reporting it as an
improvement.

---

## BUG-16 — the operator prompt demanded the one transformers version that cannot train

**Found in** the cluster's own `notes/DEVIATIONS.md`, uploaded to
`khazic/rst-qwen3.5-27b-ota-sft`. **Fixed in** `OPERATOR_PROMPT.md`, pinned by
`tests/test_launcher_correctness_gates.py::test_the_operator_prompt_states_the_same_window_the_scripts_enforce`.

`OPERATOR_PROMPT.md` — the first and, for a start-from-an-empty-folder run, the *only*
file the cluster LLM reads before touching the scripts — listed under "Hard floors,
non-negotiable":

> `transformers >= 5.15.0` (older versions do not know `qwen3_5` at all)

Both halves are wrong, and the operator's notes disprove each independently. D5.1: their
system interpreter runs transformers 5.8.0 and `"qwen3_5" in CONFIG_MAPPING_NAMES` is
True, so the parenthetical justification is false. D18: 5.15.0 is the version that
*breaks* — it removed `self.chunk_gated_delta_rule` from `Qwen3_5GatedDeltaNet.__init__`
in favour of the `kernels` package's `_kernel_funcs` indirection, which verl 0.9.0 reads
unconditionally (`qwen3_5.py:167`), so the first forward dies with an `AttributeError`
and installing `kernels` does not restore it.

So our own code had already reversed the prompt without the prompt noticing:
`01b_setup_env_verl.sh:112` installs `"transformers>=5.11,<5.15"` and
`30_run_sft_verl.sh:440` FATALs above the window, by probing for the attribute rather
than parsing a version string. An operator who trusted the prompt over the scripts would
install 5.15.0, then be refused by the launcher it told them to run — or, worse, satisfy
the prompt with a stale environment and hit the AttributeError inside torchrun.

**What changed.** The prompt now states the window with its upper bound, names 5.15 as
the breaking version and why, points at the two scripts that enforce it, and says not to
widen it to silence a pip resolver warning. The new test asserts the prompt contains the
exact pin string the setup script installs and contains no `>= 5.15`-shaped claim, so the
two cannot drift apart again silently. `PLAN.md`'s mentions of 5.15.0 are provenance
statements about where `models/qwen3_5` lives and stay as they are.

**Class of defect, not a typo.** Every other gate in this repo is enforced by code that
runs. This one lived only in prose, in the document that overrides everything else by
being read first, and it survived precisely because nothing compared it to the scripts.

---

## BUG-17 — the report could not find the loss curve of any verl run

**Found in** `khazic/rst-qwen3.5-4b-ota-sft/reports/REPORT.md` (4B OTA-SFT, 110 steps,
finished). **Fixed in** `scripts/14_make_report.py` + `scripts/20_run_all.sh`, pinned by
`tests/test_report_loss_curve.py`.

Every verl REPORT.md so far, including the one for the only OTA run that finished, carried

> 🟡 WARN | training | loss curve | no loss values scraped from logs; the curve could not
> be checked. Point `--run-dir` at the directory holding the trainer stdout.

while 110 perfectly scrapeable steps sat in `$BASE_FOLDER/logs/run.log`. So the one check
that would notice a diverging or NaN loss has never run, on any run, and the report asked
a human to supply a path the launcher itself had chosen. Three causes, all ours:

1. `20_run_all.sh` passed `--run-dir "$BASE_FOLDER/$RUN_NAME"` — the trainer's **output**
   directory. Under verl/FSDP that holds `global_step_*/` and `latest_checkpointed_iteration.txt`
   and nothing else; `parse_training_log`'s `*.log` / `logs/*.log` / `**/run.log` globs
   correctly found zero files. This was inherited from the slime/Megatron path, where the
   trainer did write its stdout under the run dir.
2. The learning-rate pattern matched only `learning_rate`. verl prints `train/lr:2.24e-06`,
   so `lr` was missing from every scraped step even where the loss was found.
3. `run.log` is a single appended log for **all** stages. Scraping it whole would have
   concatenated the SFT curve with the DPO trainer's, whose loss is log 2 to the bit by
   construction — 0.144 followed by 0.693147, which the "loss decreased" check reads as a
   run that blew up at the end. Fixing (1) without fixing this would have replaced a
   missing check with a false one.

**What changed.** `14_make_report.py` takes `--train-log PATH` (repeatable) and
`--train-stage NAME`; `stage_section()` slices the requested `=== STAGE <name>` block out
of a `20_run_all.sh` log and returns a log without those markers whole, so slime stdout
still scrapes. `20_run_all.sh` grew `collect_train_logs`, which passes whichever of
`$RST_RUN_LOG`, `$BASE_FOLDER/run_all.log`, `$BASE_FOLDER/logs/run.log` exist — the
operator chooses where the stdout goes (`| tee …`), so the path is discovered rather than
assumed, and a candidate that does not exist is simply not passed. The SFT report asks for
the `train` section and the GRPO report for `rl`. Verified against the real 4B log: 110
steps, loss 0.3019 → 0.1442, `lr` 1e-06 → 3e-07, and DPO's 0.693147 outside the curve.

**Why it stayed invisible.** The WARN was honest and specific, and it named a remedy — so
it read like a note about someone's incomplete invocation rather than a defect in the
invocation the launcher writes itself.

---

## BUG-18 — a relaunch resumed the checkpoint and restarted the lr schedule 7.8× above its floor

**Found in** `khazic/rst-qwen3.5-4b-tmax-sft` (two wandb runs, both exitcode 0).
**Fixed in** `scripts/resume_guard.py` + `scripts/30_run_sft_verl.sh` +
`scripts/14_make_report.py`, pinned by `tests/test_resume_schedule_gate.py`. **Measured**
from the run's own `wandb` files.

The tmax SFT model was produced by two launches of the same `RUN_NAME`:

| wandb run | `total_epochs` | steps | loss | `train/lr` |
|---|---|---|---|---|
| `bic28b9c` | 1 | 1 → 42 | 0.2624 → **0.19885** | 3.0e-06 → **3.000e-07** (the `min_lr_ratio` floor) |
| `l6op97sl` | 3 | 43 → 84 | 0.19539 → **0.19650** | **2.3546e-06** → 1.005e-06 |

`trainer.resume_mode` defaults to `auto`, so the second launch found `global_step_42` in
`trainer.default_local_dir` and continued the step counter — while `total_training_steps`
was re-derived from the *new* epoch count (42 → 126) and the cosine rebuilt over that
total. The resumed step therefore landed part-way **up** the new curve: 7.8× above the
floor the checkpoint had been annealed to. The 42 extra steps moved the loss
0.1954 → 0.1965, i.e. nothing, which is what a second half-anneal at 8× the final lr buys.

**Confirmed from inside the checkpoints**, not just from wandb. Each
`global_step_*/extra_state_world_size_8_rank_0.pt` carries the scheduler's own
`lr_scheduler` state (`base_lrs`, `last_epoch`, `_last_lr`), readable without torch. Every
step from the second launch onward sits on a 126-step cosine (3 × 42, `min_lr_ratio=0.1`,
`lr_warmup_steps_ratio=0.03`) to every printed digit, while step 42 sits on the floor of
the 42-step one:

| step | `_last_lr` in the checkpoint | the 126-step cosine at that step |
|---|---|---|
| 42 | **3.0000e-07** | 2.3838e-06 |
| 60 | 1.8048e-06 | 1.8048e-06 |
| 80 | 1.1294e-06 | 1.1294e-06 |
| 100 | 5.8689e-07 | 5.8689e-07 |
| 120 | 3.1582e-07 | 3.1582e-07 |

So the discontinuity is exactly one curve being swapped for another across the resume
boundary: the optimizer state came from a run annealed to 3.00e-07 and was handed a
schedule that says 2.35e-06 at the very next step.

Neither run warned. verl does not compare the schedule it is about to apply with the one
the checkpoint came from, and our launcher recorded nothing to compare against. The only
trace was the `train/lr` column of a log nobody diffs, and the report has no lr check.

**What changed.**

* `scripts/resume_guard.py` — importable, so the logic is unit-tested rather than trusted.
  `schedule_fingerprint()` extracts the curve-defining overrides (`trainer.total_epochs`,
  `optim.lr`, `lr_scheduler_type`, `min_lr_ratio`, `lr_warmup_steps_ratio`, `weight_decay`,
  `betas`, `data.train_batch_size`, `data.train_files`, `model.path`) plus the world size,
  and writes them as `rst_launch.json` beside the checkpoints. On a launch that would
  resume, a differing fingerprint is exit 2 naming each changed knob, why it reshapes the
  curve, and the four ways forward (new `RUN_NAME`, restore the knobs,
  `trainer.resume_mode=disable`, or `ALLOW_SCHEDULE_CHANGE_ON_RESUME=1`). Absent keys are
  differences too — dropping `min_lr_ratio=0.1` does not leave the floor where it was.
  `latest_checkpoint_step()` walks `global_step_*/` and ignores
  `latest_checkpointed_iteration.txt`, because `resume_mode: auto` does and because the
  27B run left a stale one saying 82 with no such directory.
* `30_run_sft_verl.sh` runs the guard on node rank 0 with `"${VERL_ARGS[@]}"` — the array
  actually passed to the trainer — immediately before `torchrun`.
* `14_make_report.py::find_lr_restart()` catches it after the fact from the log alone: find
  the lr peak (warmup rises legitimately), then flag any rise above 1.5× after it. A cosine
  is non-increasing from its peak. This only fires when both launches are in the scraped
  log, which is what the BUG-17 fix passes.

An honest resume — same schedule, restarted after a node failure — is still allowed, and a
resume that pre-dates the gate is allowed with a note that the earlier curve is unverified;
refusing there would strand exactly the runs the gate exists for.

**Why it stayed invisible.** Both launches succeeded, the loss went down overall, and the
final loss (0.1965) is a fine-looking number. "More epochs" is the most ordinary knob to
change, and the reward for changing it was two half-trained anneals reported as one run.

---

# Open — not fixed, needs the cluster

### OPEN-1 · 4-node FSDP2 over TCP is likely throughput-bound
The run died in the first backward, so there is no tokens/s measurement at all. Confirm
whether the cluster really has no RDMA (`ls /sys/class/infiniband`, `ibv_devinfo`). If it
does not, full 32-way sharding is the wrong topology: `engine.fsdp_size=8` (HSDP) keeps
all-gathers inside a node over NVLink and leaves one inter-node gradient all-reduce per
step. That raises the static term to `16 B/param ÷ 8` and needs `OFFLOAD_OPTIM=1`
(`offload_policy`) to fit. Decide from a measurement, not from this paragraph.

### OPEN-2 · gated-delta-net + Ulysses SP correctness is still unvalidated
verl's implementation is real and careful — `_build_fla_cp_context` passes recurrent state
between ranks through `fla.ops.cp.context.build_cp_context`, `_ConvPrefixExchange` is a
custom autograd Function doing the `causal-conv1d` halo exchange, and
`_packed_chunk_gated_delta_rule` raises `NotImplementedError` rather than degrading if the
kernel will not take `cp_context`. It is **not** the cause of the OOM. But nothing here or
upstream has shown it produces the same loss as `sp=1`, and the launcher already labels it
`UNVALIDATED`. The cheap check: same data, same seed, 20 steps at `sp=1` vs `sp=8`, compare
the loss curves. Do it before quoting a number from an SP run.

### OPEN-3 · `mtp.*` and `model.visual.*` are in the 27.78 B but are not trained
`07_restore_vision.py` already treats both as copy-from-original. Two consequences nobody
has measured: the MTP head is never called in forward (which is why the pre-`1c0c25d` patch
needed to guard a missing `_unsharded_param`), and verl's dummy ViT forward gives the vision
tower a **zero** gradient rather than `None`, so AdamW applies `weight_decay=0.1` to it every
step — ~3e-5 total over 82 steps, negligible but not bit-identical. Freezing both before
`fully_shard` would return their share of the fp32 master/grad shards. Worth a one-line
`named_parameters()` dump at load so the 27.78 B in `configs/models.json` is checked against
the checkpoint once instead of trusted forever.

### OPEN-4 · `logs/train_rank0.log` was never published
verl prints `log_gpu_memory_usage` ("After FSDP, memory allocated (GB): …") on **global rank
0 only**, and the HF repo has `train_rank1/2/3.log`. That line is the one direct read of the
static footprint and the whole reason `34_diagnose_oom.py` exists. Keep rank 0's stdout.

### OPEN-5 · check what the cluster can actually pull before telling it to pull
BUG-1's fix (`1c0c25d`) **is** on `origin/main`. Everything in this file after it — BUG-2
through BUG-12 — is what has to reach the cluster next.

The 27B attempt at 2026-08-19 18:24 (+08:00) confirms none of it has arrived. Two
independent tells in its own `logs/run_all.log`: no `[rst-fsdp2]` line anywhere (so BUG-1 is
live — 8/8 ranks died at `73.60 GiB` allocated asking for `640 MiB`, inside
`liger_kernel/ops/swiglu.py::swiglu_forward` during checkpoint recompute), and
`RST_MAX_TOKENS_PER_GPU=32768` beside `RST_ULYSSES=8`, which the BUG-3 launcher now refuses
outright. `After FSDP … 3.19 GiB` puts the shard degree at 32, so the topology is right and
the failure is entirely the retained gradient. `RST_OPTIMIZER_OFFLOAD=1` was set to fight it
and cannot (BUG-1, BUG-6).

Worth recording because it nearly went into this file as a finding: a `refs/remotes/origin/main`
that has never been fetched in a given clone is not evidence about the remote. This checkout's
cached ref sat at `f819661`, so `git branch -vv` reported "ahead 7" and the obvious reading was
that the OOM fix had never been pushed. `git fetch` said `origin/main = 1c0c25d`. Confirm with a
fetch, not with the cached ref, before concluding that a cluster is running stale code.

---

# Checked and found correct — do not re-litigate

Recorded so the next review does not spend a day on these.

**Ulysses does not divide the effective learning rate.** `sft_loss` multiplies by
`dp_size` (= `world // sp` = 4) while FSDP2 reduce-scatters with `ReduceOp.AVG` over all 32
ranks (`_fsdp_collectives.py:842-846`) — which looks like a factor of 1/8. It is cancelled
by `verl/utils/ulysses.py:229-230`, where `Gather.backward` multiplies the gradient by
`sp_world_size`. Net factor 1.

**GQA survives `sp=8`.** `num_key_value_heads=4` does not divide 8, but
`monkey_patch.py:120-128` repeats KV heads by `max(sp // nheads_k, 1)` first, and
`nheads_k=4, sp=8, repeats=2` is its own documented example. `num_attention_heads=24` is
divisible by 8, which the launcher already gates.

**The fused cross-entropy is real.** `FusedLinearForPPO`
(`verl/utils/experimental/torch_functional.py`) is a genuine `autograd.Function`: chunked
forward, chunked recompute in backward, peak logits `512 × 248,320 × 4 B ≈ 485 MiB`. And
`transformer_impl.py:289-297` does hardcode `fused_linear_cross_entropy=False` for Liger, so
the launcher's long comment about `use_fused_kernels` vs `use_liger` is accurate.

**The chat template renders one `<think>` block per conversation, not per turn.** The jinja
condition is `loop.index0 > ns.last_query_index`, and `03_build_sft_data.py` emits terminal
output as bare `user` messages (not `<tool_response>`-wrapped), so `last_query_index` is the
final user turn and only the last assistant turn gets the block. That matches the
"1 vs 21" measurement behind the pre-tokenized export, `qwen3_5_mask`'s `<think>\n` skip is
right, and train/serve stay consistent under `enable_thinking=False`.

**DPO's split-backward gradient is exact.** With
`z = (π_c − π_r) − (ref_c − ref_r)`, the code's
`base = −β(σ(−βz)(1−ε) − σ(βz)ε)` is `dL/dz` for sigmoid-DPO, `2(z − 1/(2β))` is IPO's, the
`±1` signs are right, and `coefficient /= n` under `--length-normalize` is exactly
`dz/d(Σ log p)`. No approximation.

**`fully_shard` before `.to(device)` is fine.** Tested both orderings on the H100; params
land on `cuda:0` and gradients populate either way. Not a bug.
