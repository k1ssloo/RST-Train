# Training prompt：SETA / Terminal-Lego DeepSeek / Opus × Qwen3.5

将分隔线以下内容复制给集群侧训练 agent。本轮数据已上传，尚未启动 GPU 训练。
数据范围为 SETA、Terminal-Lego DeepSeek 和 Opus 三组。前两组已有验证过的 SFT
产物；Opus 已上传原始轨迹，本地适配器已完成全量转换验证。集群执行者按
第 3.1 节转换，再完成其三个尺寸的训练。

---

你负责使用 RST-Train 完成 **9 个独立 SFT 任务**：分别用 SETA、
Terminal-Lego DeepSeek 和 Terminal-Lego Opus 数据训练 **Qwen3.5-4B、9B、27B**，导出可加载的
Hugging Face 权重，完成评估并提交报告。

**checkpoint 固定每 200 个 optimizer/global steps 保存一次，即
`trainer.save_freq=200`。训练结束必须额外保存最终 checkpoint。**

先阅读 `README.md`、`BACKENDS.md`、`BUG.md`、`OPERATOR_PROMPT.md`。
沿用其中的环境、FSDP2、融合交叉熵和权重导出检查；本 prompt 定义本轮任务范围、
数据、命名与保存频率，优先于旧 prompt 中的 RST/OTA/Nemotron/TMax 训练队列。
本轮只运行 SFT 及其评估，不启动 DPO、GRPO 或教师轨迹生成。

## 1. 数据与固定版本

数据账号是 **`NiuNiu0110`**。以下三个仓库均为 **private**，通过安全注入的
`HF_TOKEN` 或已有 HF 登录凭据读取；不得把 token 写进脚本、日志或模型卡。

| 数据 | HF dataset repo | train / holdout | 协议 |
|---|---|---:|---|
| SETA | `NiuNiu0110/SETA-SFT-native-tools` | 976 / 100 | SETA 原生工具调用 |
| Terminal-Lego DeepSeek | `NiuNiu0110/Terminal-Lego-DeepSeek-SFT-terminus` | 11,938 / 200 | Terminus-2 JSON |
| Terminal-Lego Opus | `NiuNiu0110/Terminal-Lego-Opus-Trajectories-unscored` | 8,318 条原始记录；转换后确定切分数量 | 转为 Terminus-2 SFT；保留未评分标记 |

固定数据 revision，禁止自动漂移到后续 `main`：

```text
SETA      605f104decc273e8064e51752a43c6000774a4be
DeepSeek  292330c713254548893fbaef435cfd7fb6199096
Opus      1c5578aff4cad23df70ceb46851c128aeafb2fd5
```

两个 SFT 仓库都有 `default`（messages）和 `pretokenized`（input_ids + loss_mask）
两个 config，以及 train/holdout 两个 split。Opus 仓库提供 `default` config 的
`unscored` split，必须先完成第 3.1 节的转换。三个尺寸复用各自数据组的同一份
预分词文件与固定切分；不混合三组数据，不把 holdout 加入训练，不逐轮重新渲染
聊天模板。SETA/DeepSeek 使用已发布切分，Opus 只在首次转换时建立切分。

SETA 的训练数据有 **9,836,475 tokens / 4,919,395 supervised tokens**；
DeepSeek 有 **105,263,578 / 42,581,331**。最长序列分别为 26,391 和
32,766 tokens。Opus 的保留行数和 token 统计以转换结果为准；三组统一使用
`MAX_SEQ_LEN=32768`、`data.truncation=error`。

SETA/DeepSeek 按上游 reward=1 筛选；本地没有重放验证器。**SFT 不要求 reward，
Opus 必须作为独立的未评分 SFT 数据组训练。** 它的元数据仅有
`oracle_passed_task`、`difficulty`，不能据此推断模型轨迹成功；保留未知评分，
不伪造 reward=1 或 reward=0。原始仓库的未评分/待转换状态不免除本轮三个 Opus
训练任务。TerminalWorld 和 Terminal-Lego 环境包不属于本轮 SFT 数据。

## 2. 九个任务与权重命名

沿用上一轮约定：**数据在 `NiuNiu0110`，训练后的模型权重在 `khazic`**。
数据读取凭据不等同于模型写入凭据；上传权重前核对凭据对 `khazic` 的写权限。

| 数据 | 官方初始化模型 | RUN_NAME | HF 模型仓库 |
|---|---|---|---|
| SETA | `Qwen/Qwen3.5-4B` | `qwen3.5-4b-seta-sft-v1` | `khazic/rst-qwen3.5-4b-seta-sft-v1` |
| SETA | `Qwen/Qwen3.5-9B` | `qwen3.5-9b-seta-sft-v1` | `khazic/rst-qwen3.5-9b-seta-sft-v1` |
| SETA | `Qwen/Qwen3.5-27B` | `qwen3.5-27b-seta-sft-v1` | `khazic/rst-qwen3.5-27b-seta-sft-v1` |
| DeepSeek | `Qwen/Qwen3.5-4B` | `qwen3.5-4b-terminal-lego-deepseek-sft-v1` | `khazic/rst-qwen3.5-4b-terminal-lego-deepseek-sft-v1` |
| DeepSeek | `Qwen/Qwen3.5-9B` | `qwen3.5-9b-terminal-lego-deepseek-sft-v1` | `khazic/rst-qwen3.5-9b-terminal-lego-deepseek-sft-v1` |
| DeepSeek | `Qwen/Qwen3.5-27B` | `qwen3.5-27b-terminal-lego-deepseek-sft-v1` | `khazic/rst-qwen3.5-27b-terminal-lego-deepseek-sft-v1` |
| Opus | `Qwen/Qwen3.5-4B` | `qwen3.5-4b-terminal-lego-opus-unscored-sft-v1` | `khazic/rst-qwen3.5-4b-terminal-lego-opus-unscored-sft-v1` |
| Opus | `Qwen/Qwen3.5-9B` | `qwen3.5-9b-terminal-lego-opus-unscored-sft-v1` | `khazic/rst-qwen3.5-9b-terminal-lego-opus-unscored-sft-v1` |
| Opus | `Qwen/Qwen3.5-27B` | `qwen3.5-27b-terminal-lego-opus-unscored-sft-v1` | `khazic/rst-qwen3.5-27b-terminal-lego-opus-unscored-sft-v1` |

每个任务都从对应官方模型独立初始化。禁止跨 SETA、DeepSeek、Opus 继续微调，
或从旧 RST/OTA/TMax/Nemotron 权重继续训练后仍使用上表命名。
同一任务的故障恢复才允许沿用其目录和 optimizer/scheduler 状态；改变数据或
训练日程时使用新的 `v2` 名称，并记录差异。

## 3. 下载与数据检查

使用所有训练节点可访问的共享存储。`BASE_FOLDER` 是本轮独立工作目录；模型
缓存按尺寸共享，每个任务的日志、checkpoint 和评估目录由 `RUN_NAME` 隔离。

```bash
export BASE_FOLDER="<shared-scratch>/terminal-trajectories-v1"
mkdir -p "$BASE_FOLDER"
git clone https://github.com/k1ssloo/RST-Train.git "$BASE_FOLDER/RST-Train"
export REPO_DIR="$BASE_FOLDER/RST-Train"
cd "$REPO_DIR"

hf download NiuNiu0110/SETA-SFT-native-tools --repo-type dataset \
  --revision 605f104decc273e8064e51752a43c6000774a4be \
  --local-dir "$BASE_FOLDER/datasets/seta-hf"
hf download NiuNiu0110/Terminal-Lego-DeepSeek-SFT-terminus --repo-type dataset \
  --revision 292330c713254548893fbaef435cfd7fb6199096 \
  --local-dir "$BASE_FOLDER/datasets/terminal-lego-deepseek-hf"
hf download NiuNiu0110/Terminal-Lego-Opus-Trajectories-unscored --repo-type dataset \
  --revision 1c5578aff4cad23df70ceb46851c128aeafb2fd5 \
  --local-dir "$BASE_FOLDER/datasets/terminal-lego-opus-unscored-hf"
(cd "$BASE_FOLDER/datasets/terminal-lego-opus-unscored-hf" && sha256sum -c SHA256SUMS)

for DATA_KEY in seta terminal-lego-deepseek; do
  DATA_HF="$BASE_FOLDER/datasets/${DATA_KEY}-hf"
  (cd "$DATA_HF" && sha256sum -c SHA256SUMS)
  mkdir -p "$BASE_FOLDER/data/$DATA_KEY"
  cp "$DATA_HF/data/pretokenized/train.parquet" \
     "$BASE_FOLDER/data/$DATA_KEY/pretokenized_train.parquet"
  cp "$DATA_HF/data/pretokenized/holdout.parquet" \
     "$BASE_FOLDER/data/$DATA_KEY/pretokenized_holdout.parquet"
  cp "$DATA_HF/data/messages/train.parquet" \
     "$BASE_FOLDER/data/$DATA_KEY/rst_sft_train.parquet"
  cp "$DATA_HF/data/messages/holdout.parquet" \
     "$BASE_FOLDER/data/$DATA_KEY/rst_sft_holdout.parquet"
  cp "$DATA_HF/release_manifest.json" "$BASE_FOLDER/data/$DATA_KEY/"
done
```

SETA 的受监督比例为 **50.01%**，超出了旧启动器按 RST 数据写死的 25–45%
范围。这不是修改 loss mask 的理由。本轮新增 `SFT_DATA_MANIFEST` 检查：
实际训练文件的 SHA-256、行数、token 数、监督 token 数必须与已验证发布一致。
缺少 manifest 时仍执行旧范围检查。

如果集群 checkout 尚无该改动，从固定版本的数据仓库应用附带补丁：

```bash
if [[ ! -f scripts/sft_data_gate.py ]]; then
  git apply --check "$BASE_FOLDER/datasets/seta-hf/code/sft-data-manifest.patch"
  git apply "$BASE_FOLDER/datasets/seta-hf/code/sft-data-manifest.patch"
fi
rg 'SFT_DATA_MANIFEST|validate_mask_fraction' scripts/30_run_sft_verl.sh
python tests/run_tests.py test_sft_data_gate
bash -n scripts/30_run_sft_verl.sh
```

补丁冲突需要检查当前代码后移植相同验证逻辑；不要覆盖较新的启动器或删除检查。
下载的 manifest 描述 Hub 内的原始路径；复制后的文件只改名字，内容哈希应保持
一致。保存 `SHA256SUMS`、`release_manifest.json` 和完整数据 revision 到运行记录。

### 3.1 Opus：先转换，再训练三个尺寸（必做）

使用已下载的 `data/terminal-lego-opus-4-6-8k.json`，不调用教师模型生成新轨迹。
源文件应有 8,318 条记录，SHA-256 为
`925da53ae7686e22320c4ac9e506b43ad1c8a55979f2f8bf500283a686a24c10`。
8,318 是原始记录数，不能当作转换后的训练条数。
按下述固定参数，本地实测保留 **8,066 train + 200 holdout**；预分词零新增丢弃。
训练集为 **41,451,885 tokens / 14,723,292 supervised tokens**；在集群复现时
核对这些计数，差异须记录原因。所有保留记录仍为未评分，GPU 训练尚未执行。

`scripts/03j_build_terminal_lego_sft.py` 已提供 `--release opus-8k --allow-unscored`
转换模式。默认仍要求 `reward=1`，该开关只支持 Opus；显式提供的失败或非法
reward 仍会被排除。运行现有适配器，无须另写转换脚本。它执行以下检查：

- 缺少 reward 的 Opus 记录可进入结构校验；保留 `reward=null` 或省略该字段，
  显式记录 `reward_available=false`、`reward_policy=unscored`，不得由 oracle
  标记或助手的 `task_complete` 声明推断成功。
- 来源按固定 dataset revision、源文件 SHA-256、原始行号追溯，并保留
  `oracle_passed_task`、`difficulty`。未提供的 trial/path 字段保留缺失；
  不伪造 DeepSeek 来源路径，也不把缺失路径当作所有记录共享的任务组。
- 复用 RST 的整条轨迹格式、角色顺序、完整性、助手 JSON/命令结构和控制标记
  检查，规范化 `human/gpt` 为 `user/assistant`。按任务正文哈希分组，排除初始
  随机终端屏幕；检查重复轨迹、命令签名和 train/holdout 的任务/提示重叠。
- 使用已核对的 Qwen3.5 分词器，超过 32,768 tokens 的记录整条排除并记原因。
  以 seed 1228、目标 holdout 200 条按任务组一次切分，记录实际数量；三个尺寸
  复用该切分。未评分不构成排除理由，结构性排除必须逐项统计。
- `tests/test_terminal_lego_convert.py` 覆盖正常及失败路径，包括缺失 reward/
  来源字段、无效格式、分组泄漏、评分不被伪造及 DeepSeek 成功筛选行为。

输出目录须为新目录或空目录；已有源文件按固定 SHA-256 和大小校验。
完成转换后复制 messages 文件为训练约定名称，再严格导出预分词文件：

```bash
export OPUS_DATA_DIR="$BASE_FOLDER/data/terminal-lego-opus-unscored"
python tests/run_tests.py test_terminal_lego_convert
python scripts/03j_build_terminal_lego_sft.py \
  --release opus-8k --allow-unscored \
  --source "$BASE_FOLDER/datasets/terminal-lego-opus-unscored-hf/data/terminal-lego-opus-4-6-8k.json" \
  --tokenizer "$BASE_FOLDER/Qwen3.5-27B" \
  --out-dir "$OPUS_DATA_DIR" --max-seq-len 32768 --holdout 200 --seed 1228
for OPUS_SPLIT in train holdout; do
  cp "$OPUS_DATA_DIR/terminal_lego_sft_${OPUS_SPLIT}.parquet" \
     "$OPUS_DATA_DIR/rst_sft_${OPUS_SPLIT}.parquet"
  python scripts/15_export_pretokenized.py \
    --parquet "$OPUS_DATA_DIR/rst_sft_${OPUS_SPLIT}.parquet" \
    --tokenizer "$BASE_FOLDER/Qwen3.5-27B" \
    --out "$OPUS_DATA_DIR/pretokenized_${OPUS_SPLIT}.parquet" \
    --max-seq-len 32768 --strict
done
```

上述命令在第 4 节环境和分词器就绪后运行。全量检查 token/mask 等长、二值掩码、
首 token 屏蔽、非空助手监督及提示/终端观察不被监督。完成检查后生成新的
`release_manifest.json`，采用 `rst-sft-release-v1` 格式，在
`artifacts["data/pretokenized/train.parquet"]` 中填入实际 `sha256`、`rows`、
`total_tokens`、`trained_tokens`；同时保存 holdout 哈希、切分、排除记录、
分词器指纹及完整验证结果。原始归档中的 `training_ready=false` manifest 不能
直接用于 SFT 启动检查，也不能复制 DeepSeek 的统计来替代 Opus 的实测值。

完成转换后，**必须依次执行 Opus 的 4B、9B、27B 三个 SFT 任务**，使用
`DATA_KEY=terminal-lego-opus-unscored`。原始上传数据保持原样；转换产物与模型卡
明确标记为未评分轨迹的 SFT，不能称为已验证成功轨迹训练。

## 4. 模型版本与资源

本次逐文件比较了 `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、
`vocab.json`、`merges.txt`。以下三个 revision 的五个文件均与导出分词器完全一致，
详情见数据仓库 `tokenizer_compatibility.json`。按这些版本下载模型：

```bash
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir "$BASE_FOLDER/Qwen3.5-4B"
hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir "$BASE_FOLDER/Qwen3.5-9B"
hf download Qwen/Qwen3.5-27B \
  --revision fc05daec18b0a78c049392ed2e771dde82bdf654 \
  --local-dir "$BASE_FOLDER/Qwen3.5-27B"
```

在正式运行前按 `scripts/00_preflight.sh` 和 `scripts/01b_setup_env_verl.sh` 建立
训练环境，运行 CPU tests 和独立 smoke test。保留兼容的 transformers/verl/FLA
版本、融合交叉熵、gradient checkpointing，以及日志中的 `[rst-fsdp2]` 证据。

按既有 **A100 80GB** 集群规划：4B/9B 各使用 1 节点 × 8 GPU；27B 使用
4 节点 × 8 GPU，`FSDP_SIZE=-1`，默认 `ULYSSES_SP=1`。实际硬件不同必须重新
核算并记录。27B 不可在只有 8 张卡时假定满足原方案；完整 32 卡 rendezvous
和 shard degree 必须实测。Ulysses 或 CPU offload 的变更应先验证正确性与吞吐。

估算磁盘时包含基础模型、每 200 步保存的模型分片、optimizer、extra 状态及
HF 导出副本；根据实际 checkpoint 大小与保留数量分配空间，不沿用旧单任务估算。

## 5. 训练命令与 checkpoint 频率

首轮统一采用 **3 epochs、global batch size 128、learning rate 3e-6**，cosine
衰减到 3e-7、warmup ratio 0.03；这是首轮实验配置，尚未经这三组数据调优。
现有 verl 0.9.0 的 train loader 使用 `drop_last=True`，预计每 epoch 为
`floor(train_rows / 128)` 个优化器步：

| 数据 | 预计 steps/epoch | 3 epochs 总步数 | 应有 checkpoint |
|---|---:|---:|---|
| SETA | 7 | 21 | 最终 `global_step_21` |
| DeepSeek | 93 | 279 | `global_step_200`、最终 `global_step_279` |
| Opus | `floor(N_train / 128)` | `K = 3 × floor(N_train / 128)` | 每 200 步及最终 `global_step_K`；按转换实测数量确定 |

以实际 dataloader/训练器日志为准，报告各 epoch 实际消费样本数与丢弃尾批。
**保存间隔 200 指优化器步，不是 micro-batch、样本数或 epoch。** 未满 200 步时
只保存最终 checkpoint 是预期行为；不要为了产生 `step_200` 擅自增加 epochs。
verl 0.9.0 在 `is_last_step` 分支额外保存最终状态，部署版本也必须确认此行为。

直接调用 `30_run_sft_verl.sh`，它支持末尾 Hydra overrides。`20_run_all.sh`
没有相同参数转发，并默认串接其他阶段，本轮不要用它来替代下面的命令。

每次从上表选择一个任务。以下为 4B SETA 的完整训练调用；4B/9B 的单节点
任务替换 `MODEL_KEY`、`DATA_KEY`，并据此形成对应 `RUN_NAME`。
`DATA_KEY` 分别为 `seta`、`terminal-lego-deepseek`、`terminal-lego-opus-unscored`；
Opus 的转换与校验完成后使用完全相同的训练调用：

```bash
set -euo pipefail
export MODEL_KEY=qwen3.5-4b DATA_KEY=seta
export RUN_NAME="${MODEL_KEY}-${DATA_KEY}-sft-v1"
export DATA_DIR="$BASE_FOLDER/data/$DATA_KEY"
export SFT_DATA_MANIFEST="$DATA_DIR/release_manifest.json"
export NNODES=1 NGPUS=8 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export FSDP_SIZE=-1 ULYSSES_SP=1 MAX_SEQ_LEN=32768 SAVE_HF_MODEL=1
export WANDB_MODE=offline  # 有已配置的实验跟踪账户时可使用 online
mkdir -p "$BASE_FOLDER/$RUN_NAME/logs"

bash scripts/30_run_sft_verl.sh \
  trainer.total_epochs=3 \
  trainer.save_freq=200 \
  trainer.seed=1228 \
  trainer.resume_mode=auto \
  data.val_files=null \
  2>&1 | tee "$BASE_FOLDER/$RUN_NAME/logs/train_rank${NODE_RANK}.log"
```

启动器的 registry 会赋值 `NUM_EPOCH`，所以仅 `export NUM_EPOCH=3` 不可靠；
上面的 `trainer.total_epochs=3` 和 `trainer.save_freq=200` 都必须实际传入。
检查最终 Hydra 配置，不得被默认 `save_freq=20` 或外层配置覆盖。

对于 27B，在分配到的 **每个节点各调用一次**相同命令，运行前把
`MODEL_KEY=qwen3.5-27b`、`NNODES=4 NGPUS=8`、`MASTER_ADDR=<rank-0-host>`
和各节点唯一的 `NODE_RANK=0/1/2/3` 设置好；随后重新计算 `RUN_NAME` 及日志路径。
各节点使用相同 `MASTER_PORT` 和共享数据/输出目录。不能只在主节点执行一次
就认为其余节点已启动。用 `scripts/34_diagnose_oom.py --runtime` 的分布式诊断
确认 world size/sharding 后再开始正式训练。

不同数据组和模型尺寸的运行目录必须不同；不要复用旧 `.stage`、checkpoint 或
LR schedule。故障重启保留同一任务的日程，按 `resume_guard.py` 的检查恢复。

## 6. 评估与协议区别

**每个已保存 checkpoint 都用本组完整 holdout 做离线评估，并与同尺寸官方基础
模型比较。** SETA 100 条、DeepSeek 200 条及 Opus 实际留出的全部记录参与；使用现有脚本的
`--max-rows 0 --max-seq-len 32768`，记录实际评分行数、监督 token 数与排除数。

上面的训练命令刻意用 `data.val_files=null`：verl 0.9.0 的验证 loader 也沿用
GBS 128 并 `drop_last=True`，SETA 100 条会变成零个验证 batch，DeepSeek 也
无法覆盖全量 holdout。使用独立的完整离线评估避免这个问题。

完成保存后选择实际存在的步数，先合并/恢复，再评分。例如：

```bash
export STEP=21  # SETA 首轮预计值；DeepSeek 分别处理 200 和最终 279，以实测为准
export EXPORT_DIR="$BASE_FOLDER/$RUN_NAME/exports/global_step_${STEP}"
bash scripts/08_prepare_eval_ckpt.sh \
  --model-key "$MODEL_KEY" \
  --ckpt "$BASE_FOLDER/$RUN_NAME/global_step_${STEP}" \
  --out "$EXPORT_DIR"

# BASE_MODEL_PATH 按本任务选择对应的官方本地目录。
export BASE_MODEL_PATH="$BASE_FOLDER/Qwen3.5-4B"
python scripts/06b_eval_offline.py \
  --model-path "$EXPORT_DIR" --base-model "$BASE_MODEL_PATH" \
  --holdout "$DATA_DIR/pretokenized_holdout.parquet" \
  --out "$BASE_FOLDER/$RUN_NAME/eval/step-${STEP}" \
  --max-rows 0 --max-seq-len 32768 --max-actions 0
```

报告 held-out NLL、perplexity、next-token accuracy 及相对基础模型的变化。
DeepSeek 和转换后的 Opus 可另外做 Terminus-2 action probe 和真实终端评测；SETA 使用六种原生
工具，需要兼容 adapter 才能做交互评估。不要用单一 bash/Terminus-2 parser
把 SETA 的其他原生工具调用记为模型失败，也不要直接横比两种协议的成功率。
Opus holdout 仍然未经逐条奖励验证；其 NLL 衡量对这些轨迹的拟合程度，
不证明参考动作成功。报告中单独列出，实际任务成功率由真实交互评估测量。

数据只完成了内部任务分组检查，没有完成与 Terminal-Bench 等基准的全面重叠
排查；真实 benchmark 前检查任务来源与重叠，报告排除项。离线 NLL 改善不是
终端任务通过率，没有运行的指标标为未测量。

## 7. 导出、发布与交付

`SAVE_HF_MODEL=1` 请求同时保存 HF 模型、分布式模型、optimizer 和 extra。
仍须检查实际保存文件：只有 config/tokenizer 的 `huggingface/` 目录不是模型。
保留所有必需分片并运行 `08_prepare_eval_ckpt.sh` 的文本权重变化、vision/MTP
恢复和 load/generate 检查，再将完整 HF 导出放入上表对应模型仓库。

周期性 checkpoint 用 `global_step_<实际步数>` 目录保存，含完整恢复状态。
导出的模型 revision/tag 使用 `step-000200` 等六位步数；最后一步另打 `final`
标签。仓库 `main` 指向按本组 holdout NLL 选定的最佳已保存 checkpoint，模型卡
同时记录 best/final 的实际步数，不能宣称选中了未保存的中间 epoch。
默认创建私有模型仓库；已存在同名仓库先核对运行身份，避免覆盖其他实验。

模型卡和 `notes/` 至少记录：基础模型与 revision、数据 repo/revision/config、
训练行数/token 数、`save_freq=200`、实际保存步数、超参数、GPU 拓扑、代码版本、
数据哈希、协议、已完成评估和已知限制。保留 rank 0 及其余节点日志。
SETA 数据沿用 Apache-2.0；DeepSeek 和 Opus 上游轨迹未声明许可证，不得自行标成
Apache-2.0。缺少 `khazic` 写凭据时保留已验证导出并报告路径，不把权重误传到
数据账号或其他既有仓库。

交付一份九行汇总表：run、最终/最佳 HF revision、GPU 配置、epochs/global steps、
实际 checkpoint 列表、训练 loss、完整 holdout 指标、墙钟时间、峰值显存、恢复或
配置偏离情况。说明每项 GPU/交互评估是否真正执行。本 prompt 的本地准备阶段
已验证 SETA/DeepSeek SFT、Opus 下载及本地未评分转换和预分词、原始数据上传及 CPU 代码，
没有任何可引用的 GPU 训练结果。附上执行阶段生成的 Opus 转换 manifest、实际 train/holdout 数量、
token 统计、排除原因和未评分说明；不得将仅下载 Opus 作为其三个训练任务的完成。
