# Forwardable note: pre-tokenized SFT data

Short addendum to `OPERATOR_PROMPT.md`. Everything here is already in the repo; this
just makes sure it is not missed.

---

```text
ADDENDUM — pre-tokenized SFT data (verl + FSDP path)

The verl path does NOT train on the `messages` parquet. It trains on a pre-tokenized
file, `pretokenized_train.parquet` (columns: `input_ids`, `loss_mask`), which has the
verified Qwen3.5 loss mask already applied.

You do not depend on the Hub for this. Two equivalent ways to get it:

  (a) Download it — config `cap10_pretokenized`, ~75 MB (smaller than the messages
      version; token ids compress better than JSON):

        hf download NiuNiu0110/RST-SFT-Qwen3.5-27B --repo-type dataset \
          --local-dir "$BASE_FOLDER/sft-hf"
        cp "$BASE_FOLDER/sft-hf/data/cap10_pretokenized/train.parquet" \
           "$BASE_FOLDER/sft-v1-cap10/pretokenized_train.parquet"

  (b) Do nothing. `scripts/30_run_sft_verl.sh` builds it automatically when the file
      is absent, from the messages parquet plus the tokenizer in the downloaded model
      directory (~5 min). Manually:

        python scripts/15_export_pretokenized.py \
          --parquet   "$BASE_FOLDER/sft-v1-cap10/rst_sft_train.parquet" \
          --tokenizer "$BASE_FOLDER/<model-dir>" \
          --out       "$BASE_FOLDER/sft-v1-cap10/pretokenized_train.parquet"

REGENERATE (b), do not download, in any of these cases:
  * the Hub is unreachable, or you distrust the copy
  * you changed --max-seq-len
  * you are training a model whose tokenizer is NOT the Qwen3.5 family's
Download is only safe across Qwen3.5-0.8B/4B/9B/27B/35B-A3B because those five share
one byte-identical tokenizer AND one identical training-time chat-template render.

WHY THIS FILE EXISTS AT ALL — do not "simplify" it away.
verl's built-in MultiTurnSFTDataset templates each message separately and
concatenates. On this data that disagrees with the whole-conversation render on
200/200 sampled rows: the Qwen3.5 template injects an empty `<think>\n\n</think>\n\n`
before the LAST assistant turn, so building turn-by-turn makes every turn "last" and
a 21-turn conversation gets 21 think blocks instead of 1. verl's documented escape
hatch `ignore_input_ids_mismatch: True` silences the assertion, not the bug — you
would train on a token sequence serving never produces. Do not set it.

You may freely RESHAPE this data for whatever trainer you end up using — packing,
padding, `labels` with -100, different column names, webdataset, anything. To build
`labels`: `labels[i] = input_ids[i]` where `loss_mask[i]==1` else `-100`, and do NOT
shift (HF models shift internally).
What you must NOT do is recompute the mask with a second implementation, or let a
trainer re-tokenize from `messages`. That reintroduces exactly the bug above.

THE LAUNCHER CHECKS THE FILE, whatever its provenance — downloaded, copied or built.
It validates the parquet itself, not a sidecar manifest, and aborts on:
  * missing `input_ids`/`loss_mask` columns (i.e. a `messages` parquet by mistake)
  * len(input_ids) != len(loss_mask) on any row
  * any row with zero trained tokens
  * any row over max_seq_len
  * trained-token fraction outside 0.25–0.45  (expected: 32.42%)
Read that last one carefully if it fires: near 100% means the mask is ABSENT and you
would be training on the harness prompt and terminal output; near 0% means it masks
everything. Neither is a threshold to widen — investigate before spending GPU time.
```
