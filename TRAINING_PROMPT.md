# Kickoff prompt: Nemotron + TMax SFT, and the OTA 27B rerun

Copy everything below the line into the operator LLM's first message.

---

Four jobs this pass, in this order. `OPERATOR_PROMPT.md` still governs everything not
restated here — same repo, same launcher, same gates, same "never change these" list.

**Datasets live under `NiuNiu0110`, models under `khazic`.** Two different accounts; do
not push weights to the dataset account or vice versa.

---

## Job 1 — rerun the OTA 27B, which never trained

`khazic/rst-qwen3.5-27b-ota-sft` has notes and **zero checkpoints, zero logs**. The 9B
and 4B OTA runs finished 110 steps on `nnodes 1 / nproc_per_node 8`; the 27B was
launched the same way and was **refused before torchrun**, correctly: at shard degree 8
the static footprint is `27.78e9 × 16 B ÷ 8` = **51.7 GiB/GPU = 65 %** of an 80 GB card,
over the 55 % gate. Nothing is broken in the code — the 4-node allocation had ended and
8 GPUs cannot hold this model's optimizer state.

So it needs the full cluster, not a retry:

```bash
export MODEL_KEY=qwen3.5-27b
export ACTOR_NUM_NODES=4 ACTOR_NUM_GPUS_PER_NODE=8   # 32 GPUs -> shard 32 -> 12.9 GiB/GPU
export MASTER_ADDR=<head-ip>                         # NOT 127.0.0.1; NODE_RANK differs per node
export DATA_DIR=$BASE_FOLDER/ota RUN_NAME=qwen3.5-27b-ota-sft
bash scripts/20_run_all.sh
```

Before booking the nodes, confirm the shape is real — 30 s, and it prints the effective
shard degree instead of letting you infer it from an OOM:

```bash
torchrun --nnodes 4 --nproc_per_node 8 --node_rank $NODE_RANK --master_addr $MASTER_ADDR \
         scripts/34_diagnose_oom.py --runtime
```

Also confirm `grep '\[rst-fsdp2\]'` appears in the rank logs. Without that line the
second OOM constant (103.5 GiB/GPU of unsharded fp32 gradient) is unpatched and it will
die in `loss.backward()` no matter how many GPUs you have.

If 4 nodes are genuinely unavailable, **do not force it** with `OFFLOAD_OPTIM=1` and
call it done — say so and stop. 8 GPUs at 27B is not a smaller run, it is a different one.

---

## Job 2 — SFT on the Nemotron Terminal-Corpus (the big one)

[`NiuNiu0110/Nemotron-Terminal-SFT-terminus`](https://huggingface.co/datasets/NiuNiu0110/Nemotron-Terminal-SFT-terminus)
— 360,057 trajectories, **3.88 B tokens**, NVIDIA's corpus converted to this repo's
contract. Upstream's own numbers are why this is Job 2 and not Job 3: it took Qwen3-32B
from 3.4 % to **27.4 %** on Terminal-Bench 2.0, beating 480B Qwen3-Coder.

### Pick a config. Do not train the whole thing by reflex.

The corpus is **~39x** either sibling SFT set. At GBS 128:

| config | train rows | steps/epoch | download | what it is |
|---|---|---|---|---|
| `skill_based_mixed` | 5,671 | 44 | 0.09 GiB | taxonomy tasks, mixed difficulty |
| `skill_based_easy` | 44,692 | 349 | 0.67 GiB | taxonomy tasks, easy |
| `skill_based_medium` | 88,912 | 694 | 1.97 GiB | taxonomy tasks, medium |
| `dataset_adapters` | 220,381 | 1,721 | 3.71 GiB | math/code/SWE *transformed* into terminal tasks |
| `default` (all) | 359,656 | **2,809** | 6.44 GiB | everything |

**Start with the three `skill_based_*` configs together — 139,275 rows, 1,088
steps/epoch.** That is the natively-terminal material, and it is already 10x the RST
cap10 run you have numbers for. `dataset_adapters` is 61 % of the corpus but is
transformed math/code, so it is the part most likely to buy less per token for terminal
skill; add it in a second pass if the skill-based run moves the benchmark.

Work out the wall-clock from your own measured tokens/s before launching — at 3.88 B
tokens the whole corpus is a multi-day job on 32 GPUs and nobody here has measured it.
If your budget will not cover an epoch, say which config you chose and why.

```bash
hf download NiuNiu0110/Nemotron-Terminal-SFT-terminus --repo-type dataset \
    --local-dir $BASE_FOLDER/nemo-hf \
    --include 'data/train_skill_based_*.parquet' 'data/holdout.parquet' 'manifest.json'
mkdir -p $BASE_FOLDER/nemo
python - <<'PY'
import glob, pandas as pd, os
base = os.environ["BASE_FOLDER"]
parts = sorted(glob.glob(f"{base}/nemo-hf/data/train_skill_based_*.parquet"))
pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) \
  .to_parquet(f"{base}/nemo/rst_sft_train.parquet", index=False)
print("train rows:", sum(len(pd.read_parquet(p)) for p in parts))
PY
cp $BASE_FOLDER/nemo-hf/data/holdout.parquet $BASE_FOLDER/nemo/rst_sft_holdout.parquet
cp $BASE_FOLDER/nemo-hf/manifest.json        $BASE_FOLDER/nemo/manifest.json
```

Then, per size, with `SKIP_STAGES="data"`:

```bash
for KEY in qwen3.5-4b qwen3.5-9b qwen3.5-27b; do
  MODEL_KEY=$KEY DATA_DIR=$BASE_FOLDER/nemo RUN_NAME=${KEY}-nemo-sft \
    SKIP_STAGES="data" bash scripts/20_run_all.sh
done
```

4B and 9B fit on 1 node. **27B needs 4 nodes**, for the reason in Job 1. The launcher
builds `pretokenized_train.parquet` itself on first use — that step is not published
because it would be ~4x the size of what it is derived from, and it will take a while
at this row count.

### Two things about this data that will look like bugs and are not

**1. Only the final assistant turn's reasoning is supervised.** Every turn upstream
carries a `<think>` block (47.1 % of assistant tokens), and they are all kept in
`messages` — but observations here are plain `user` turns, so the Qwen3.5 template's
`last_query_index` lands on the last observation and reasoning renders for the final
turn alone. **That is what inference looks like:** the terminus-2 harness re-renders
history every turn under the same rule, so the model never sees its own earlier
reasoning either. Do not "fix" this by restructuring the data mid-run.

**2. Trained fraction is 43.6 %, not ~32 %.** Same cause — a real CoT on the final turn
plus the JSON action on every turn. Verified: 0 contract failures, 0 supervised first
tokens, 0 rows over 32,768. Do not touch the mask.

---

## Job 3 — SFT on the TMax trajectories, 4B then 9B then 27B

[`NiuNiu0110/TMax-Agent-SFT-terminus`](https://huggingface.co/datasets/NiuNiu0110/TMax-Agent-SFT-terminus)
— 5,645 trajectories, 42.9 M tokens, teacher Qwen3.6-27B.

```bash
hf download NiuNiu0110/TMax-Agent-SFT-terminus --repo-type dataset --local-dir $BASE_FOLDER/tmax-hf
mkdir -p $BASE_FOLDER/tmax
cp $BASE_FOLDER/tmax-hf/data/pretokenized/train.parquet $BASE_FOLDER/tmax/pretokenized_train.parquet
cp $BASE_FOLDER/tmax-hf/data/messages/train.parquet     $BASE_FOLDER/tmax/rst_sft_train.parquet
cp $BASE_FOLDER/tmax-hf/data/messages/holdout.parquet   $BASE_FOLDER/tmax/rst_sft_holdout.parquet
cp $BASE_FOLDER/tmax-hf/manifest.json                   $BASE_FOLDER/tmax/manifest.json

for KEY in qwen3.5-4b qwen3.5-9b qwen3.5-27b; do
  MODEL_KEY=$KEY DATA_DIR=$BASE_FOLDER/tmax RUN_NAME=${KEY}-tmax-sft \
    SKIP_STAGES="data" bash scripts/20_run_all.sh
done
```

**It is small: 5,445 rows = 42 steps/epoch at GBS 128.** One epoch is not a run. Do
**3 epochs (~127 steps)** and report the loss at each epoch boundary; if holdout loss
turns up after epoch 2, say so and keep the epoch-2 checkpoint.

TMax differs from Nemotron in a way that matters for interpretation: it trains a
`<think>` CoT plus a **native `bash` tool call** on *every* turn, not the
`{analysis, plan, commands}` JSON. Its trained fraction is 45.3 %. `messages[0]` is a
`system` turn carrying the tool schema and observations arrive as `user` turns wrapped
in `<tool_response>` — a byte-exact flattening of the native tool-calling render
(asserted per row, 5,795/5,795). Do not re-render it as `tools=` + `role="tool"`.

---

## Job 4 — termigen: NOT this pass

[`NiuNiu0110/Termigen-RL-Taskset`](https://huggingface.co/datasets/NiuNiu0110/Termigen-RL-Taskset)
(private) — 3,541 GRPO tasks. **`allenai/open-instruct-termigen` has zero assistant
turns**; all 3,556 rows are `(system, user)` plus a `ground_truth` and an `env_config`.
It is an RLVR task pool, so there is nothing to SFT and no pairs to build. Do not try.

It is ready for the *later* GRPO pass, with two caveats already recorded in its card:
it has **no measured pass rates** (`tier: "unknown"`), so an unknown fraction of it is
all-fail or all-pass and therefore zero-gradient at full sandbox cost — tier it with a
cheap sampling pass first. And it is **one Docker image per task**, 3,541 distinct tags,
so pre-pulling is a real job. 15 tasks were already excluded for shipping their own
verifier inside the Docker build context (10 byte-identical copies).

---

## No DPO on TMax or Nemotron

Both are success-only, so there is no within-task contrast to pair. For TMax this was
measured rather than assumed: of 2,020 tasks only **291 (14.4 %)** have both a success
and a failure trajectory, and `1,136 + 1,175 − 291 = 2,020` exactly — a TMax task is
almost always either always solved or never solved, so the contrast measures difficulty,
not quality. Best case is 444 pairs over 291 prompts against the existing stage's 2,673.

The **RST DPO stage is unaffected** and still runs on its own 2,673 pairs where
`OPERATOR_PROMPT.md` says it does. If you think DPO should run on either new corpus,
argue it in the report; do not just run it.

---

## Naming — keep these exact

| | |
|---|---|
| run dir / `RUN_NAME` | `qwen3.5-{4b,9b,27b}-{nemo,tmax}-sft`, `qwen3.5-27b-ota-sft` |
| input datasets (read-only) | `NiuNiu0110/{Nemotron-Terminal,TMax-Agent}-SFT-terminus` |
| HF model repos | `khazic/rst-qwen3.5-{4b,9b,27b}-{nemo,tmax}-sft` |
| the OTA rerun pushes to | `khazic/rst-qwen3.5-27b-ota-sft` (overwrite it; it holds no weights) |

Leave alone: `khazic/rst-qwen3.5-{4b,9b,27b}-sft` (RST cap10),
`khazic/rst-qwen3.5-{4b,9b}-ota-sft` (OpenThoughts, already trained).

Push each checkpoint's `notes/` **and** its logs. The OTA 27B failure was only
diagnosable because the 9B run had logs to compare against, and it cost a day to
establish something a single retained rank log would have said immediately.

---

## Deliverable

One report covering all seven runs (1 OTA + 3 Nemotron + 3 TMax): GPU shape used,
final and per-epoch loss, step count, wall-clock, peak per-GPU memory,
`in_range` / `checkpoint_trustworthy` per run, and every deviation. For Nemotron, state
**which config you trained and how many epochs** — a subset run reported as "Nemotron
SFT" without that is unreadable.

Reference eval targets exist for **27B only**; do not invent one for 4B/9B. State
plainly that the three corpora train **different assistant formats** — RST/OTA and
Nemotron use the JSON action object, TMax uses native tool calls — so their eval numbers
are not directly comparable to each other.
