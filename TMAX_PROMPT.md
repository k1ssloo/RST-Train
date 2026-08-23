# Kickoff prompt: TMax SFT (4B/9B/27B) + the OTA 27B rerun

Copy everything below the line into the operator LLM's first message.

---

Two jobs this pass, in this order. `OPERATOR_PROMPT.md` still governs everything not
restated here — same repo, same launcher, same gates, same "never change these" list.

## Job 1 — rerun the OTA 27B, which never trained

`khazic/rst-qwen3.5-27b-ota-sft` has notes and **zero checkpoints, zero logs**. The 9B and
4B OTA runs finished 110 steps on `nnodes 1 / nproc_per_node 8`; the 27B was launched the
same way and was **refused before torchrun**, correctly: at shard degree 8 the static
footprint is `27.78e9 × 16 B ÷ 8` = **51.7 GiB/GPU = 65 %** of an 80 GB card, over the
55 % gate. Nothing is broken in the code — the 4-node allocation had ended and 8 GPUs
cannot hold this model's optimizer state.

So it needs the full cluster, not a retry:

```bash
export MODEL_KEY=qwen3.5-27b
export ACTOR_NUM_NODES=4 ACTOR_NUM_GPUS_PER_NODE=8      # 32 GPUs -> shard 32 -> 12.9 GiB/GPU
export MASTER_ADDR=<head-ip>                            # NOT 127.0.0.1, and NODE_RANK must differ per node
export DATA_DIR=$BASE_FOLDER/ota RUN_NAME=qwen3.5-27b-ota-sft
bash scripts/20_run_all.sh
```

Before booking the nodes, confirm the shape is real — 30 s, and it prints the effective
shard degree instead of letting you infer it from an OOM:

```bash
torchrun --nnodes 4 --nproc_per_node 8 --node_rank $NODE_RANK --master_addr $MASTER_ADDR \
         scripts/34_diagnose_oom.py --runtime
```

Also confirm `grep '\[rst-fsdp2\]'` appears in the rank logs. Without that line the second
OOM constant (103.5 GiB/GPU of unsharded fp32 gradient) is unpatched and it will die in
`loss.backward()` no matter how many GPUs you have.

If 4 nodes are genuinely unavailable, **do not force it** with `OFFLOAD_OPTIM=1` and call
it done — say so and stop. 8 GPUs at 27B is not a smaller run, it is a different one.

## Job 2 — SFT on the TMax trajectories, 4B then 9B then 27B

New dataset, published and public, no HF token needed:

```bash
hf download NiuNiu0110/TMax-Agent-SFT-terminus --repo-type dataset --local-dir $BASE_FOLDER/tmax-hf
mkdir -p $BASE_FOLDER/tmax
cp $BASE_FOLDER/tmax-hf/data/pretokenized/train.parquet $BASE_FOLDER/tmax/pretokenized_train.parquet
cp $BASE_FOLDER/tmax-hf/data/messages/train.parquet     $BASE_FOLDER/tmax/rst_sft_train.parquet
cp $BASE_FOLDER/tmax-hf/data/messages/holdout.parquet   $BASE_FOLDER/tmax/rst_sft_holdout.parquet
cp $BASE_FOLDER/tmax-hf/manifest.json                   $BASE_FOLDER/tmax/manifest.json
```

Then, per size, with `SKIP_STAGES="data"`:

```bash
for KEY in qwen3.5-4b qwen3.5-9b qwen3.5-27b; do
  MODEL_KEY=$KEY DATA_DIR=$BASE_FOLDER/tmax RUN_NAME=${KEY}-tmax-sft \
    SKIP_STAGES="data" bash scripts/20_run_all.sh
done
```

4B and 9B fit on 1 node. **27B needs 4 nodes**, for the reason in Job 1.

### Two things about this data that will look like bugs and are not

1. **Trained fraction is 45.3 %, not ~32 %.** RST and OTA train a JSON action object;
   TMax trains a real `<think>` CoT plus a native `bash` tool call, and 71 % of its
   assistant tokens are reasoning. 45 % is the expected number — verified 0 observation
   leakage across all 22,305,351 non-assistant tokens. Do not "fix" the mask.

2. **`messages[0]` is a `system` turn** carrying the tool schema, and observations arrive
   as `user` turns wrapped in `<tool_response>`. That is a byte-exact flattening of the
   native tool-calling render (asserted per row, 5,795/5,795), which is what lets
   `--loss-mask-type qwen3_5` and `15_export_pretokenized.py` apply unchanged. Do not
   re-render it as `tools=` + `role="tool"`; you would get the same tokens at best.

### It is small — plan the epochs, not just the steps

5,445 train rows = **42 steps/epoch at global batch 128**. One epoch is not a run. Do
**3 epochs (~127 steps)** and report the loss at each epoch boundary; if the holdout loss
turns up after epoch 2, say so and keep the epoch-2 checkpoint.

## No DPO on TMax. Do not build one.

Measured, not assumed: of 2,020 TMax tasks only **291 (14.4 %)** have both a success and a
failure trajectory, and `1,136 + 1,175 − 291 = 2,020` exactly — a TMax task is almost
always either always solved or never solved, so the success/failure contrast measures task
difficulty, not response quality. Best case is 444 pairs over 291 distinct prompts against
the existing stage's 2,673. If you think DPO should run here anyway, argue it in the
report; do not just run it.

The **RST DPO stage is unaffected** and still runs on its own 2,673 pairs where
`OPERATOR_PROMPT.md` says it does.

## Naming — keep these exact

**Datasets live under `NiuNiu0110`, models under `khazic`.** Two different accounts; do not
push weights to the dataset account or vice versa.

| | |
|---|---|
| run dir / `RUN_NAME` | `qwen3.5-{4b,9b,27b}-tmax-sft`, `qwen3.5-27b-ota-sft` |
| input dataset (read-only) | `NiuNiu0110/TMax-Agent-SFT-terminus` |
| HF model repo | `khazic/rst-qwen3.5-{4b,9b,27b}-tmax-sft` |
| the OTA rerun pushes to | `khazic/rst-qwen3.5-27b-ota-sft` (overwrite it; it holds no weights) |

Existing repos to leave alone: `khazic/rst-qwen3.5-{4b,9b,27b}-sft` (RST cap10),
`khazic/rst-qwen3.5-{4b,9b}-ota-sft` (OpenThoughts, already trained).

Push each checkpoint's `notes/` **and** its logs. The OTA 27B failure was only diagnosable
because the 9B run had logs to compare against, and it cost a day to establish something a
single retained rank log would have said immediately.

## Deliverable

One report covering all four runs: 4 GPU shapes used, final + per-epoch loss, step count,
wall-clock, peak per-GPU memory, `in_range` / `checkpoint_trustworthy` per run, and every
deviation. Reference eval targets exist for **27B only** — do not invent a target for
4B/9B. State plainly that TMax models are trained on a different assistant format from the
RST/OTA models, so their eval numbers are not directly comparable to those.
