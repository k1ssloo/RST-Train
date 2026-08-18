# RST → Qwen3.5-27B: SFT (then RL) on 4×8 A100

**Audience: the operator LLM on the cluster.** Everything here is either measured
on the real release or read out of the real source trees. Where I could not
verify something, it is marked **UNVERIFIED** — treat those as the risk list, not
as facts. Priority per the human: **reproduce SFT strictly first**, RL second.

---

## 0. The five facts that determine this plan

1. **`Qwen3.5-27B` is not a plain dense transformer.** It is
   `Qwen3_5ForConditionalGeneration`: 27.78 B params, bf16 = 51.7 GiB, **64 text
   layers = 48 gated-delta-net ("linear_attention") + 16 full-attention GQA**
   (`full_attention_interval: 4`), hidden 5120, ffn 17408, 24 q-heads × head_dim
   256, 4 KV heads, vocab 248 320, `max_position_embeddings` 262 144, rope_theta
   1e7, mrope. **Plus** a 27-block ViT (`model.visual.*`) and a 1-layer MTP head
   (`mtp.*`). Dense MLP — *not* MoE.
   → Consequences: A100 has no FP8 (stay bf16); the GDN layers need Triton/FLA
   kernels; the ViT/MTP tensors must be restored after a Megatron round trip.

2. **The paper's 262,145 context is a launcher ceiling, not a data property.**
   Measured over the real trajectories, the SFT sequence length is
   **p50 8.0 K, p90 17.0 K, p99 28.2 K, max 32.3 K**. 98.1 % fit in 32 K.
   → We train at `--max-seq-len 32768`. This is the single biggest cost saving.
   (The parquet's `input_tokens` column is a *sum over turns* — context is
   re-sent every turn — so it is ~2× the real sequence length. Don't size from it.)

3. **slime ships this exact configuration.** `THUDM/slime`
   `scripts/run-qwen3.5-27B.sh` defaults to `ACTOR_NUM_NODES=4`,
   `ACTOR_NUM_GPUS_PER_NODE=8` → **32 GPUs, your cluster** — with TP4/PP2/CP4 and
   GRPO at `kl-coef 0.00 / entropy-coef 0.00 / eps-clip 0.2`, which matches the
   paper's stated RL settings. `scripts/models/qwen3.5-27B.sh` gives exact
   `MODEL_ARGS`. Appendix F: the launcher "provides the same training path for
   four-node runs."

4. **`--loss-mask-type qwen3_5` is the paper's "Qwen3.5-specific loss masking",
   and it is not the default.** The default is `qwen`, which mis-segments this
   template and would train on terminal output and the harness prompt. Verified
   locally: with `qwen3_5`, **32.98 % of tokens are trained, 0 user-turn leakage**.

5. **63 % of assistant outputs are malformed and the fix has a second-order
   effect.** 62.6 % of turns are wrapped in ```` ```json ```` fences; 0.1 % are
   unparseable. Critically, when a turn was fenced, the *next* observation begins
   `Previous response had warnings: - Extra text detected before JSON object`.
   If you normalize the assistant turn but leave that preamble, you train the
   model to accept "you had warnings" feedback for clean output. Our pipeline
   strips it — **50,169 observations required this repair**.

Software support is real and first-party (checked against the actual source
trees, not docs): transformers 5.15.0 has `models/qwen3_5`; vLLM 0.27.1 registers
`Qwen3_5ForConditionalGeneration` (+`Qwen3_5MTP`); SGLang 0.5.17 has
`qwen3_5.py`/`qwen3_5_text.py`/`qwen3_5_mtp.py`; verl 0.9.0's Megatron registry
includes `Qwen3_5ForConditionalGeneration`.

---

## 1. Hardware plan

| | |
|---|---|
| Train | 4 nodes × 8 A100 = 32 GPUs, SM80, bf16 only |
| Framework | slime (Ray + Megatron-LM backend) + SGLang for eval/rollout |
| Debug | 1 shared H100 (SM90) — data + conversion + 1-GPU smoke only |

> **Backend note.** verl + FSDP is the primary path (`scripts/01b_setup_env_verl.sh`,
> `scripts/30_run_sft_verl.sh`) because building Megatron on A100 needs a cuDNN swap
> a shared cluster will not permit. See `BACKENDS.md`. The slime/Megatron notes below
> remain accurate for that secondary path.

**A100-specific deltas from slime's upstream `build_conda.sh` (both handled in
`scripts/01_setup_env.sh`):**

- **Skip FlashQLA.** Upstream installs `git+https://github.com/QwenLM/FlashQLA`,
  the fast GDN backend — its own comment says **"requires SM90+"**. On A100 use
  `--qwen-gdn-backend fla` (Triton, SM80-ok). `fla` *is* the argparse default, so
  the risk is only that someone copies the SM90 flag over.
- **No FP8 anywhere.** Never use `tools/convert_hf_to_fp8.py`.
- Keep every other pin exactly (`torch 2.11.0+cu129`, `flash_attn 2.8.3`,
  `transformer_engine 2.16.1`, `flash-linear-attention 0.4.2`, SGLang
  `v0.5.15.post1`, Megatron `1dcf0daf`). The FA2/TE/Megatron combination is
  version-sensitive; "upgrading" it is the most likely way to lose a day.

### Parallelism (registry-driven)

CP is what makes a 32 K sequence fit: **per-GPU sequence tokens = seq / CP**, so
`max-tokens-per-gpu × CP ≥ max-seq-len` is a hard constraint (`05_run_sft.sh`
asserts it). Weights per GPU = 27.8 B / (TP·PP) — CP does *not* shard weights.

All parallelism now lives in **`configs/models.json`**, resolved and *validated* by
`scripts/model_registry.py` (see §8 for the full model list):

```bash
python scripts/model_registry.py --list
python scripts/model_registry.py --key qwen3.5-27b --mem-class 80GB --gpus 32
```

For `qwen3.5-27b`:

| `MEM_CLASS` | TP | PP | CP | DP | `--max-tokens-per-gpu` | weights/GPU |
|---|---|---|---|---|---|---|
| **80GB** | 4 | 2 | 2 | 2 | 16384 | 6.95 GB |
| **40GB** | 8 | 1 | 4 | 1 | 8192 | 6.95 GB |

The registry enforces `tp·pp·cp·dp == gpus`, `tp ≤ gpus_per_node`, and
`max_tokens_per_gpu·cp ≥ max_seq_len`, and exits non-zero with an explanation
rather than letting an impossible config reach the trainer.

> Corrected: an earlier revision of this table listed 40GB as TP8/PP2/CP2 with
> `max_tokens_per_gpu 8192`. That gives `8192·2 = 16384 < 32768`, so the longest
> sequence in the dataset could not be placed and the launcher's own assertion
> would have rejected it. The row is now TP8/PP1/CP4.

**Fallback ladder** on OOM or GDN/CP trouble: ① halve `--max-tokens-per-gpu`
② CP 2→4 (DP down) ③ PP 2→4 ④ *last resort* `--max-seq-len 16384`. Prefer ①–③:
④ drops ~10 % of examples, biased toward exactly the long-horizon trajectories
this dataset exists to teach.

**Host RAM.** `--optimizer-cpu-offload` is what makes 27.8 B fit — fp32 Adam
states are ~334 GB, offloaded and sharded across DP. Verify headroom with
`00_preflight.sh` **before** launching; at DP1 one node carries the whole thing.

---

## 2. Data pipeline — done and verified locally

Already executed on this box against the full 23 GB release. Outputs in
`data/sft-v1/` with `manifest.json`.

```
327,189 trajectories
  └─ gate: status=completed ∧ has_trajectory ∧ ¬has_exception
           ∧ reward=1.0 ∧ task_present_in_task_dataset
     → 60,932 clean successes / 1,338 task groups     ← reproduces your audit exactly
  └─ group cap 8, round-robin over the 4 model sources
     → 9,479 selected
  └─ reconstruct ATIF-v1.7 → messages; normalize JSON; repair warning preamble
     → 9,133   (−226 unparseable, −120 missing required keys)
  └─ dedup (exact + per-group command signature)
     → 9,080   (−53)
  └─ tokenize, assert slime contract, drop >32 K
     → 8,886   (−194 too long, −0 contract failures)
        = 8,686 train + 200 holdout, 1,327/1,338 groups covered
```

**Measured:** 82.4 M total tokens, **27.25 M trained tokens** (32.98 %), mean
9,482 tok, 12.0 assistant turns/example, 57.4 % of turns rewritten.
Model mix: iter `qwen35-27b-iter0000161-hf` 5,608 / `Qwen3.5-27B` 1,421 /
`Qwen3.6-27B-base` 1,383 / `gpt-oss-120b` 474.

Conversation shape (matters — it must match serving):

```
messages[0]  role=user       ← the full Terminus-2 harness prompt + task + initial screen
messages[1]  role=assistant  ← canonical JSON {analysis, plan, commands[, task_complete]}
messages[2]  role=user       ← terminal observation
...                             (trailing observation with no reply is dropped)
```

`messages[0]` is **`user`, not `system`** — because that is how Terminus-2
actually delivers it (`steps[0].source == "user"`). Keeping it as `user` makes
train and serve identical. Do not "improve" this to `system`.

Two things to know about the target text:
- The chat template injects `<think>\n\n</think>\n\n` **before the final
  assistant turn only** (`enable_thinking=False` is ignored by this template).
  slime's mask skips the `<think>\n` prefix and trains `\n</think>\n\n` + content
  + `<|im_end|>`. That is consistent with how SGLang will serve it.
- Each message may carry **`step_loss_mask: 0`** to exclude that single assistant
  turn from the loss while keeping it as context. Unused so far; it is the right
  lever if you later want to drop, say, low-quality first turns.

### Reproduce / re-tune

```bash
python scripts/03_build_sft_data.py \
  --traj-root $BASE_FOLDER/rst-trajectories \
  --tokenizer $BASE_FOLDER/Qwen3.5-27B \
  --out-dir   $BASE_FOLDER/sft-v1-cap10 \
  --per-group 10 --max-seq-len 32768 --workers 20     # ~5 min on 20 cores

python scripts/03b_validate_sft_data.py \
  --parquet $BASE_FOLDER/sft-v1-cap10/rst_sft_train.parquet \
  --tokenizer $BASE_FOLDER/Qwen3.5-27B --sample 400
# must print: contract failures 0, user-turn leakage 0, trained ≈33%
```

Knobs: `--per-group` (1→1,338 … 8→9,479 … 16→17,443 selected); `--models` to
rebalance the source mix. Group-capping is essential: per-group successes are
median 28, max 284, so uncapped training would be dominated by a few lineages.

### `--per-group 10` reproduces the paper's example count exactly

Built as `data/sft-v1-cap10/`, and it lands on **10,778 final examples — the
paper's stated SFT count, to the example**:

```
60,932 eligible → cap 10 → 11,582 selected → 11,090 reconstructed
       → 11,010 after dedup → −232 over 32 K → 10,778   (= 10,578 train + 200 holdout)
```

99.9 M total tokens, 32.6 % trained (≈32.6 M trained tokens), 1,329 groups,
p50 8,014 / p90 16,920 / p99 28,265. Contract failures 0, leakage 0.

I did not tune toward this number — it fell out of the gate + cap-10 + dedup +
32 K pipeline. Treat it as **strong corroboration that the paper used a
per-group cap of ~10 with essentially this filtering**, not as proof (it could
still be coincidence). **Use `data/sft-v1-cap10/` as the primary run** and keep
`data/sft-v1/` (cap 8, 8,886 ex) as the ablation. At GBS 128 the cap-10 set gives
**82 optimizer steps/epoch** instead of 67.

---

## 3. SFT run

```bash
export BASE_FOLDER=/shared/rst SLIME_DIR=$BASE_FOLDER/slime
export MASTER_ADDR=<head-ip> HOSTFILE=$BASE_FOLDER/hostfile
export ACTOR_NUM_NODES=4 ACTOR_NUM_GPUS_PER_NODE=8   # the GPU count comes from here only
export MEM_CLASS=auto WANDB_KEY=<key>        # omit key → offline
bash scripts/00_preflight.sh --hostfile $HOSTFILE   # do this first, read it
bash scripts/01b_setup_env_verl.sh   # verl (primary); 01_setup_env.sh only for slime/Megatron
source $BASE_FOLDER/env-rstverl.sh   # ^ activated in its own process only; this enters it
bash scripts/02_download.sh
bash scripts/04_convert_ckpt.sh to_dist      # 20–40 min, CPU-bound, ~120 GB RAM
bash scripts/05_run_sft.sh
```

Hyperparameters — paper (Appendix F) unless noted:

| | value | source |
|---|---|---|
| epochs | 1 | paper |
| global batch size | 128 | paper |
| LR | 3e-6 → 3e-7, cosine | paper |
| warmup | 0.03 | *ours* (paper silent; slime template uses 0.1) |
| optimizer | Adam β 0.9/0.98, wd 0.1, CPU offload | paper + slime |
| max seq len | 32768 | *ours*, measured (paper 262,145) |
| recompute | full / uniform / 1 layer | slime |
| attention | `flash` (FA2) + `--qwen-gdn-backend fla` | A100 |

**8,686 / 128 = 67 optimizer steps for 1 epoch.** That is a very short run — at
GBS 128 the warmup fraction 0.03 is only ~2 steps. Expect ~**1.5–3 h/epoch**
(≈1.8e19 FLOPs at 20–35 % MFU; the Triton GDN kernels are the uncertainty). If
1 epoch underfits, 2–3 epochs costs almost nothing; the paper's single epoch was
over 10,778 examples, so consider `--per-group 10` before adding epochs.

**Watch in wandb:** loss curve over only 67 points (log every step — `save-interval 20`
gives 3 checkpoints); trained-token count per step should be ≈27.25 M/67 ≈ 407 K;
grad norm; and per-GPU memory. wandb offline is automatic when `WANDB_KEY` is
unset — sync later with `wandb sync $BASE_FOLDER/wandb/offline-*`.

### After training

```bash
bash scripts/04_convert_ckpt.sh to_hf $BASE_FOLDER/qwen35-27b-rst-sft-v1/iter_XXXX $BASE_FOLDER/out-hf
python scripts/07_restore_vision.py --trained $BASE_FOLDER/out-hf \
  --original $BASE_FOLDER/Qwen3.5-27B --out $BASE_FOLDER/out-hf-full
python scripts/06_eval.py --model-path $BASE_FOLDER/out-hf-full --tp 4 \
  --benchmarks tb-hard,tb2 --runs 3 --out $BASE_FOLDER/eval/mine
```

The restore step is not optional: a text-only round trip loses `model.visual.*`
and `mtp.*`, and the checkpoint then won't load as `Qwen3_5ForConditionalGeneration`.

Or just run `bash scripts/20_run_all.sh`, which chains train → export → eval →
report and writes `$BASE_FOLDER/REPORT.md`. **Only tb-hard (100 tasks) and tb2 (89
tasks) are locally scorable — LHTB's verifiers are withheld upstream (0/46 tasks
ship `tests/`), so it is reported as `unscorable` rather than as a number.**

Targets (paper Tables 3–4, pass-rate %): base **41.20 / 22.67 / 18.10** →
SFT round 3 **47.94 / 28.33 / 22.44** on TB2 / TB-Hard / LHTB. Note those are
*three cumulative synthesis rounds*; a single SFT pass on one release should be
compared to **round 1: 42.32 / 23.00 / 21.32**. Download
`Zhongzhi1228/Qwen3.5-27B-SFT` (`DOWNLOAD_REFERENCE=1`) and evaluate it through
your own harness first — if it doesn't reproduce ≈47.9 on TB2, your harness is
wrong, not your training.

---

## 4. Order of operations (risk-first)

The compute is cheap; the integration is not. Do these in order, and do the
cheap validations on the H100 before burning cluster time.

1. **On the H100 now (no cluster):** data pipeline ✅ done; mask contract ✅ done.
2. **On the H100:** `04_convert_ckpt.sh to_dist`. It is CPU/RAM-bound (52 GiB in,
   221 GB RAM available here) and is the **highest-risk unverified step** — the
   text-only spec must tolerate the `model.visual.*` / `mtp.*` keys. Failing here
   costs nothing; failing here after you've booked 32 GPUs costs a day.
3. **On the H100:** serve `Qwen/Qwen3.5-27B` under SGLang with TP1 and run one
   Terminus-2 Harbor task against Docker. This validates the eval path
   end-to-end at small scale.
4. **Cluster, 1 node / 8 GPUs:** `ACTOR_NUM_NODES=1` with `TP4 PP2 CP1 DP1` and
   a 200-example slice. Confirms slime + Megatron + GDN kernels + the loss mask
   actually step.
5. **Cluster, 4 nodes:** full run.

---

## 5. Risk register

| risk | severity | mitigation |
|---|---|---|
| **UNVERIFIED:** Megatron CP correctness on gated-delta-net layers | high | slime ships CP4 as the *default* for this exact model, which is the strongest evidence available — but linear attention carries recurrent state across the sequence, so CP is not obviously sound. Validate at step 4: train 20 steps at CP1 and CP2 on the same slice and compare loss curves. If they diverge, use CP1 + `MEM_CLASS=80GB` + `max-tokens-per-gpu 32768`. |
| **UNVERIFIED:** HF→torch_dist tolerates ViT/MTP keys | high | step 2, free on the H100 |
| TP8 without NVLink | med | `00_preflight.sh` reports `HAS_NVLINK`; use `40GB-alt` (TP4/CP4) instead |
| Ethernet-only interconnect | med | `05_run_sft.sh` sets `NCCL_IB_DISABLE=1` + `NCCL_SOCKET_IFNAME` when `/sys/class/infiniband` is empty. Keep TP intra-node. |
| No shared FS | med | `00_preflight.sh --hostfile` probes it; if local-only, replicate `$BASE_FOLDER/Qwen3.5-27B*` and the parquet to every node |
| Host RAM too small for Adam offload | med | preflight; else drop `--optimizer-cpu-offload` and raise CP |
| 67 steps is a very short schedule | low | `--per-group 10` (10.8 K, paper-matched) and/or `NUM_EPOCH=2` |
| Source-mix skew (63 % from one iter model) | low | `--models` allowlist; or lower `--per-group` |
| Only 46/500 instructions map to a public task | low | **do not claim exact-environment replay.** These are reward-verified trajectories. Fine for SFT; means RL must use the *task* dataset, not the trajectory dataset. |

---

## 6. RL — outline only (deprioritized, but the plan is coherent)

Paper (§5.5): **agentic PPO** from the *base* model (cold actor at step 0), critic
value head warm-loaded from an earlier terminal-agent critic with two critic-only
warm-up steps, **KL penalty and entropy bonus disabled, PPO clip ε = 0.2**, GAE,
per-batch advantage whitening, reward from each task's built-in verifier with
"customized reward shaping", 37,484-task pool reshuffled per epoch, rewrite
variants kept without dedup. Reward rose ~0.11 → >0.14, peaking around steps
55–60; mean turns 19–20 → >30. Sandbox: Daytona via Harbor with Terminus-2.

Three things I'd change for 32 GPUs, and why:

1. **GRPO, not PPO.** A 27.8 B critic doubles parameter + optimizer memory
   (~890 GB of state). slime's shipped `run-qwen3.5-27B.sh` is already GRPO with
   the paper's exact `kl-coef 0.00 / entropy-coef 0.00 / eps-clip 0.2`, colocated
   (`--colocate`) with SGLang rollout at `--rollout-num-gpus-per-engine 2`. Start
   from that file, swap the data source.
2. **Docker, not Daytona** — you have Docker on the training nodes. The real cost
   is building/caching 37,484 task images; shard by task and pre-warm a registry.
   Rollout will be sandbox-bound, not GPU-bound.
3. **Do offline preference learning first.** 1,279 task groups have *both* clean
   successes and clean failures — a ready-made DPO/offline-GRPO set needing **no
   sandbox at all**. It is the cheapest way to get a second data point on whether
   the data helps, and it de-risks the RL harness work.

**Disable EAGLE speculative decoding initially** (`--sglang-speculative-algorithm
EAGLE` in the shipped script). It depends on the MTP head, which our text-only
round trip drops; get correctness first, then re-enable for rollout throughput.

---

## 7. Provenance

- Paper: arXiv **2608.05466v3**, Appendix F "Training and Evaluation
  Implementation" (SFT: 64 GPUs, 8×8, SLIME as a Ray job, TP4/PP2/CP2, Adam +
  cosine + optimizer CPU offload + flash attention, 1 epoch / 10,778 examples /
  GBS 128 / ctx 262,145 / LR 3e-6→3e-7; 122B-A10B uses TP2/PP8/CP2/EP4).
  §5.5 for RL, Tables 3–4 for scores. Local copy: `probe/paper.pdf`.
  Note an internal inconsistency: §5.5/Table 4 report a +20.0 % TB2 relative gain
  while the Conclusion says 11.82 %.
- Code read directly, not from docs: `THUDM/slime`
  (`scripts/run-qwen3.5-27B.sh`, `scripts/run-qwen3.5-35B-A3B-sft.sh`,
  `scripts/models/qwen3.5-27B.sh`, `build_conda.sh`, `slime/utils/mask_utils.py`,
  `slime/utils/arguments.py`), vLLM `v0.27.1` registry, transformers `v5.15.0`
  model list, verl `v0.9.0` mcore registry.
- Data: `Zhongzhi1228/Recursive-Task-Synthesis-Trajectories` (66 tars, 22.4 GiB,
  all sha256-verified), `…/Recursive-Task-Synthesis` (8 tars, 3.55 GiB),
  `…/Terminal-Bench-Hard` (100 tasks), `…/Recursive-Task-Synthesis-Quality-1K`.
- **Not** used: the project website repo — it has the viewer/rubric/API adapter
  but none of the synthesis pipeline, operator selection, Daytona orchestration,
  or the SFT/PPO launchers. slime upstream is the real substitute, and it is a
  better one than the website repo would have been.


---

## 8. Other models

`configs/models.json` holds every supported model; `scripts/model_registry.py
--list` prints it. Select one with `MODEL_KEY=` — everything else (parallelism,
loss mask, spec file, vision handling, serving TP) follows automatically.

| key | params | bf16 | layers | min GPUs | ~min/epoch | role |
|---|---|---|---|---|---|---|
| `qwen3.5-0.8b` | 0.87 B | 1.6 GiB | 24 | 2 | ~5 | pipeline smoke test |
| `qwen3.5-4b` | 4.66 B | 8.7 GiB | 32 | 8 | ~25 | iteration workhorse |
| `qwen3.5-9b` | 9.65 B | 18.0 GiB | 32 | 8 | ~50 | primary low-cost result |
| `qwen3.5-27b` | 27.78 B | 51.7 GiB | 64 | 32 | ~150 | the paper's model |
| `qwen3.5-35b-a3b` | 35.95 B / ~3 B active | 67.0 GiB | 40 | 8 | ~40 | MoE; best capability/hour |

**Why these and not others.** slime ships 39 model specs, but only **four**
loss-mask implementations (`qwen` = ChatML, `qwen3`, `qwen3_5`, `distill_qwen`).
Everything above uses `qwen3_5`, so it needs no new mask code. GLM / Llama / MiMo /
Moonlight would each need a new mask generator plus validation. MiniMax-M2 is
240 B — full-parameter SFT would need ~2.9 TB of optimizer state, which is not
sensible on 32 A100s.

**One dataset serves all five.** Measured: the tokenizer is byte-identical across
them (sha256 `4b565170da5bed0e`) *and* so is the training-time chat-template
render. The published `cap10`/`cap8` datasets and `--loss-mask-type qwen3_5` apply
unchanged — no re-tokenization, no re-validation.

**One caveat, on `qwen3.5-0.8b` only.** Its chat template defaults thinking *off*,
so its **generation** prompt already closes the think block
(`<think>\n\n</think>\n\n`) while 4B/9B/27B/35B end at `<think>\n`. Training
targets begin with `\n</think>\n\n`, which lines up with the latter. So 0.8B must
be *served* with a thinking-on template or the block is doubled and eval silently
degrades. The registry sets `serve_chat_template_repo` and `20_run_all.sh` fetches
it automatically.

**Unvalidated:** the MoE (EP) rows for `qwen3.5-35b-a3b` have not been run on
A100. slime's shipped SFT launcher uses TP2/EP8 on 8 GPUs; our rows scale that to
32 and are a starting point. If DeepEP misbehaves on SM80, drop
`--moe-enable-deepep` and use `--moe-token-dispatcher-type alltoall`.

**No reference numbers below 27B.** The paper published scores only for
Qwen3.5-27B and 122B-A10B. `14_make_report.py` therefore *skips* (does not pass)
the regression-vs-base and reference-reproduction checks for other models, and the
report says so. Validate the harness once with 27B, then reuse it.
