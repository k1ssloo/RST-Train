# Llama 3.2 / Phi-4-mini：历史数据 SFT + DPO 执行指令

你是远端训练服务器上的执行 LLM。请在 `RST-Train` 中完成以下实验，持续推进到
训练、导出、评估和结果归档。先检查服务器现状，再执行；不要只返回计划或启动命令。
某个模型、数据权限或评测后端阻塞时，记录具体原因并继续其他独立任务。

## 1. 交付范围

对 **Llama-3.2-3B-Instruct** 和 **Phi-4-mini-instruct**，分别在下面七组历史主配置上
独立训练，共 **14 次 SFT**。每次 SFT 从对应官方 Instruct checkpoint 初始化，
不能把七个数据集串成连续微调，也不能混合后只交付一个模型。

延续用户确认的 **SFT + DPO**：另做 **2 次 RST DPO**，分别从两个模型自己的
`rst-cap10` SFT 最终 checkpoint 初始化。共计 **16 个正式训练任务**；smoke 不计入。

“之前所有数据集”按历史主配置解释。RST cap8 消融、Nemotron 全量
`dataset_adapters` 对照不在本轮默认矩阵中；如用户后续明确追加，使用独立 run。
Termigen、SWE-Gym、TerminalWorld 等只有任务环境的池子，以及尚未验收的
TerminalEvo 产物，不是本轮 SFT 轨迹集。

先读 `MODEL_SUPPORT.md`，再查本文件引用的脚本。旧的 `PLAN.md`、
`OPERATOR_PROMPT.md`、`TRAINING_PROMPT.md` 和
`TERMINAL_TRAJECTORIES_TRAINING_PROMPT.md` 用于追溯实验，当前模型与命令以本文件为准。

## 2. 固定代码、模型和数据版本

在已有 `RST-Train` checkout 检查 `git status`、分支和 origin；干净时使用
`git pull --ff-only`，记录 `git rev-parse HEAD`。有未提交修改或分叉时保留现场，
使用独立 checkout 执行，不做 `reset --hard` 或覆盖已有实验。

| MODEL_KEY | 官方 HF repo | 固定 revision | 本地 MODEL_DIR_NAME |
|---|---|---|---|
| `llama3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | `Llama-3.2-3B-Instruct` |
| `phi4-mini` | `microsoft/Phi-4-mini-instruct` | `cfbefacb99257ffa30c83adab238a50856ac3083` | `Phi-4-mini-instruct` |

`BASE_FOLDER` 选服务器上有足够空间的持久目录，例如 `/shared/rst`。
模型完整下载到 `$BASE_FOLDER/$MODEL_DIR_NAME`，包含全部权重、config、tokenizer、
chat template 和 generation config；只有 tokenizer 的目录不能训练。
使用 `hf download <repo> --revision <sha> --local-dir <dir>`，或等价的
`huggingface_hub.snapshot_download`。Llama 需要当前 HF 账户具备 gated model 访问权。
若无权限，记录 Llama 阻塞并先做 Phi，不更换模型或借用不明权重。

### 2.1 在预分词前固定 padding（BUG-28）

官方 Llama-3.2 tokenizer 未配置 `pad_token`，verl 会在加载时补为 EOS，
使 `special_tokens_map` 与普通加载不同。保留严格指纹校验；按方案 A 在本地
checkpoint 中保存统一配置。先完成本节，再启动数据导出或训练，避免同一模型目录
被运行中的进程继续使用旧配置。使用完整的独立 `--local-dir` 副本，不直接修改 HF
共享缓存或已归档 checkpoint。基础模型 revision 与本地 padding 补丁分别记账。

在**实际训练的 Python/verl 环境**先检查 Phi，再准备 Llama：

```bash
mkdir -p "$BASE_FOLDER/tokenizer-preparation"
python scripts/14_prepare_tokenizer.py \
  --model "$BASE_FOLDER/Phi-4-mini-instruct" --verify-verl \
  --report "$BASE_FOLDER/tokenizer-preparation/phi4-mini.json"

python scripts/14_prepare_tokenizer.py \
  --model "$BASE_FOLDER/Llama-3.2-3B-Instruct" \
  --pad-token '<|finetune_right_pad_id|>' --apply --verify-verl \
  --report "$BASE_FOLDER/tokenizer-preparation/llama3.2-3b.json"
```

脚本只选用**现有词表中的单个 token**，校验 ID 在模型词表范围内，不增加词表或
resize embeddings。Llama 的预期 pad ID 是 `128004`，以该 checkpoint 的实际映射
为准；不匹配时核查模型来源，不手填 ID。现有 EOS、chat template 和编码规则不变。
脚本同步 tokenizer/model/generation 的 padding 元数据，保存原文件备份与前后哈希，
并检查保存、重载和实际 verl 加载后的指纹。没有 `--apply` 时只检查；需修改返回 1，
验证失败返回 2，不允许忽略退出码继续排队。重复执行已准备好的配置不改文件。

本地核验的官方 Phi revision `cfbefac…` 已有
`pad_token=eos_token=<|endoftext|>`、ID `199999`。若远端也通过普通/verl 加载与已有
parquet 的指纹检查，保留其配置和数据，无须随 Llama 重导。若不同，核对实际 revision
和本地改动后再处理。仓库按位置生成 attention/loss mask，共用 EOS/pad ID 本身不会
屏蔽真实 EOS；不要改成按 token ID 一律去除 EOS。

Llama 配置固定后，按第 3 节从 messages **重导七组数据的 train 与 holdout**，包括
尚未完成的 Nemotron。下文为 Llama 使用新的 `pad-v1/` 子目录；保留旧文件与 manifest
以便追溯。Phi 可沿用通过检查的现有目录。不要只改旧 parquet 的指纹来通过校验。
已生成的 Llama DPO pairs/reference cache 若绑定旧指纹，也必须重建并重新打分。
SFT 导出的 HF checkpoint 应继承准备后的 tokenizer；DPO 前对实际 `POLICY` 再运行
只读检查。`30_run_sft_verl.sh` 现在会在启动训练前比较普通与实际 verl loader。
这些步骤不改变保存计划：仍每 **200 个 optimizer/global steps** 保存，并额外保存 final。

### 2.2 固定数据版本

数据固定在以下版本。路径均相对于 HF dataset repo，数量是重新分词前的历史数量。

| DATA_KEY | HF dataset repo | messages 文件 | train / holdout | epochs |
|---|---|---|---:|---:|
| `rst-cap10` | `NiuNiu0110/RST-SFT-Qwen3.5-27B` | `data/cap10/{train,holdout}.parquet` | 10,578 / 200 | 1 |
| `ota` | `NiuNiu0110/OpenThoughts-Agent-v1-SFT-terminus` | `data/messages/{train,holdout}.parquet` | 14,112 / 200 | 1 |
| `nemo` | `NiuNiu0110/Nemotron-Terminal-SFT-terminus` | 合并 `data/train_skill_based_{easy,medium,mixed}.parquet`；holdout 为 `data/holdout.parquet` | 139,275 / 401 | 1 |
| `tmax` | `NiuNiu0110/TMax-Agent-SFT-terminus` | `data/messages/{train,holdout}.parquet` | 5,445 / 200 | 3 |
| `seta` | `NiuNiu0110/SETA-SFT-native-tools` | `data/messages/{train,holdout}.parquet` | 976 / 100 | 3 |
| `terminal-lego-deepseek` | `NiuNiu0110/Terminal-Lego-DeepSeek-SFT-terminus` | `data/messages/{train,holdout}.parquet` | 11,938 / 200 | 3 |
| `terminal-lego-opus-unscored` | `NiuNiu0110/Terminal-Lego-Opus-SFT-unscored` | `data/messages/{train,holdout}.parquet` | 8,066 / 200 | 3 |

| DATA_KEY | 固定 dataset revision |
|---|---|
| `rst-cap10` | `448bce17549bb9c57844a1f515aac8b105b4e677` |
| `ota` | `f11b9105126c805a3af7f5125736affef3a74479` |
| `nemo` | `edc67a3eb6487f0dec8551e06072e4bcd9835894` |
| `tmax` | `7d74fd38a3bc437bbd2c377bd8a1effba64d79ba` |
| `seta` | `605f104decc273e8064e51752a43c6000774a4be` |
| `terminal-lego-deepseek` | `292330c713254548893fbaef435cfd7fb6199096` |
| `terminal-lego-opus-unscored` | `06096e5ca61f7df7f333b7f2c21285743febb2b4` |

SETA 和两个 Lego 数据仓库是私有的；使用服务器已有的安全注入凭据。GitHub 凭据不用于
HF。不要将任何 token 写进代码、Markdown、命令行参数、模型卡或日志。
缺少数据访问权时保留该任务的 blocked 状态，不私自改成别的数据源。

下载时指定 `--repo-type dataset --revision <sha>`，仅取 messages 和发布 metadata，
不要下载 Qwen 的 pretokenized 文件作为训练输入。核验 revision、文件 SHA-256、行数、
轨迹 ID 和任务组划分；保存原始发布 manifest。Nemotron 只合并三个 skill_based 文件
一次，不再拼接 `default` 或 dataset_adapters，holdout 不进入合并。

## 3. 数据重编码与审计

将每组 messages 规范放置到：

```text
$BASE_FOLDER/data/<DATA_KEY>/rst_sft_train.parquet
$BASE_FOLDER/data/<DATA_KEY>/rst_sft_holdout.parquet
$BASE_FOLDER/data/<DATA_KEY>/source_metadata/
$BASE_FOLDER/data/<DATA_KEY>/llama3.2-3b/pad-v1/pretokenized_{train,holdout}.parquet
$BASE_FOLDER/data/<DATA_KEY>/phi4-mini/pretokenized_{train,holdout}.parquet
```

文件名统一只是为复用启动器，原始 messages、身份字段和奖励 metadata 不变。
分别给两个模型用自己的原生模板重新分词，绝不复用 Qwen 的 token IDs、loss masks
或 DPO reference 分数。由 `config.json` 自动选择 `llama3` / `phi3` mask。

```bash
# MODEL_KEY、MODEL_PATH、DATA_KEY 已按上述矩阵设置。
export DATA_DIR="$BASE_FOLDER/data/$DATA_KEY"
export PRETOK_DIR="$DATA_DIR/$MODEL_KEY"
if [[ "$MODEL_KEY" == llama3.2-3b ]]; then
  export PRETOK_DIR="$PRETOK_DIR/pad-v1"
fi
for split in train holdout; do
  python scripts/15_export_pretokenized.py \
    --parquet "$DATA_DIR/rst_sft_${split}.parquet" \
    --tokenizer "$MODEL_PATH" --loss-mask-type auto \
    --out "$PRETOK_DIR/pretokenized_${split}.parquet" \
    --max-seq-len 32768 --strict
done
```

本轮 **MAX_SEQ_LEN=32768**，与历史实验一致。registry 的 8192 是新家族的保守默认，
不是本轮生产配置。超长整行丢弃并报告；不截断轨迹，不悄悄降到 8K。
`--strict` 不保证零丢弃：还必须核验生成的 `pretokenized_*_manifest.json`，
只接受明确计数的 `too_long`，其他丢弃原因须查明并修复。
任一 split 为空、监督为零、掩码错位或模板指纹不符时，该任务不能训练。

已发布的 SETA/TMax messages 是 **role/content 文本**，含序列化的 `<think>`、
`<tool_call>`、`<function=...>`、`<tool_response>`。完整保留这些文本；不要转换成
Terminus-2 JSON，也不要当成独立 native `tool_calls` 字段再次序列化。
其他数据中的 reasoning 文本同样保留。Llama/Phi 与 Qwen 对这些文本的模板处理和
token 数可能不同，记录实际变化，不套用 Qwen 的 25%–45% 监督比例。

如果服务器上的文件含独立 `tool_calls`、reasoning 字段或非文本 content，先核对是否
下载错版本；若确需转换，显式转换并测试后报告，不能绕过导出器丢掉字段。
原始 Opus 的 `reward=null/unknown`、未评分标记和来源必须保留在原始文件、
数据审计及模型卡中；预分词表不保存所有 metadata，保留可追溯的 ID 映射。
不要把 Opus 写成成功轨迹或伪造 reward。

保存每个模型/数据集的行数、总 token、监督 token、长度分布、超长丢弃 ID 和原因、
train/holdout 任务组交集、文件哈希与内嵌 tokenizer 指纹。
从短、中、长轨迹抽查 mask：assistant 为目标，system/user/环境输出仅为上下文，
首 token 不监督，`step_loss_mask` 生效。Llama 的日期参数使用
`rst_common.tokenization.render_kwargs("llama3")`，训练与推理一致。

**不要把原 Qwen `release_manifest.json` 设为新文件的 `SFT_DATA_MANIFEST`**：
重新分词后哈希、长度和行数可能不同。原 metadata 放在 `source_metadata/`；
新导出 manifest 和 parquet 内嵌指纹才描述当前训练输入。

## 4. 环境与 GPU 验证

使用支持目标架构的 Transformers、PyTorch CUDA 和 verl/FSDP2；记录实际版本、
GPU 型号/显存、驱动、拓扑和空闲资源。CPU 兼容性检查使用过 Transformers 5.15.0，
这不代表任意 CUDA/verl 组合均验证过。为本轮使用独立环境；不要替换已有 Qwen 环境的
Torch/FLA，也不要启动 Megatron/slime 或在线 RL。

可先运行 `python -m pytest tests/ -q`，明确报告 optional dependency 的 skip。
当前本地记录为 484 passed、1 skipped（缺 Harbor），没有全尺寸 GPU 训练结论。
Phi 官方 tokenizer 对七组历史数据各抽查 10 条均通过；Nemotron 抽的是 holdout，
不是全语料验收。Llama 官方 tokenizer 仍需服务器在有权限时验证。
Phi padding 检查另见 `reports/tokenizer_padding_check_20260909.json`；该报告只运行了
下载的上游 verl tokenizer helper，不代表已验证远端安装的 verl/FSDP 训练环境。

每个模型先在隔离目录完成真实 forward/backward、一次参数更新、保存、重载和多卡
短训练。可以先用 `scripts/16_smoke_forward_backward.py --model <dir>
--parquet <pretok> --seq-len 4096 --rows 2 --out <smoke.json>` 检查数值；
该脚本会裁剪 smoke 样本，只能作为短序列检查。随后使用正式的 padded verl 路径，
对接近 32768 tokens 的真实样本测显存和吞吐，再准入生产。
smoke 的样本/输出目录及短训练 schedule 必须与正式 run 分开。

默认 `ULYSSES_SP=1`、padding、原生 HF forward、关闭 Liger/fused kernels/remove-padding，
启用 gradient checkpointing。长序列 logits 和激活仍可能很大，不能按 3B 权重大小
推断单卡必然能训练。OOM 时先核对分片、micro-batch 和 attention 实现，再验证
更多已分配 GPU 或 CPU offload；不要以截短正式数据来掩盖问题。

只使用调度器分配或已确认空闲的 GPU；不要 `pkill` 他人进程，不停止已有 rollout、
TerminalEvo 或代理服务。单节点 8×A100 80GB 可作为待测配置，实际数量以分配和 smoke
为准；CPU 数据准备可同时推进，通过验证的独立 run 才按资源排队。

## 5. 14 次独立 SFT

超参数：GBS **128**、LR **3e-6**、cosine floor **3e-7**、warmup **0.03**、
seed **1228**，epochs 见矩阵。各数据集的历史 epoch 数不同，“训练一遍”指完成一次
该配置的实验，不把 TMax/SETA/Lego 都改成 1 epoch。

每 **200 个 optimizer/global steps** 保存 checkpoint，并额外保存最终状态。
未满 200 步仅有 final 是正常结果，不擅自延长训练。未发生新丢弃时，按
verl 0.9.0 的 `drop_last=True`，预期总步数如下；以实际 loader 和日志校正。

| DATA_KEY | epochs | 预计总 optimizer steps |
|---|---:|---:|
| `rst-cap10` | 1 | 82 |
| `ota` | 1 | 110 |
| `nemo` | 1 | 1,088 |
| `tmax` | 3 | 126 |
| `seta` | 3 | 21 |
| `terminal-lego-deepseek` | 3 | 279 |
| `terminal-lego-opus-unscored` | 3 | 189 |

每个模型、每组数据分别调用如下模板。先将 `RST_TRAIN_GPUS` 设为该作业实测可用的卡数。
多节点时修改 NNODES、NODE_RANK、MASTER_ADDR，并在每节点启动一次；共享同一数据/输出。

```bash
set -euo pipefail
# 本例是 Llama + RST cap10；根据矩阵逐项替换 MODEL_KEY、MODEL_PATH、DATA_KEY、EPOCHS。
export MODEL_KEY=llama3.2-3b
export MODEL_PATH="$BASE_FOLDER/Llama-3.2-3B-Instruct"
export DATA_KEY=rst-cap10
EPOCHS=1
export DATA_DIR="$BASE_FOLDER/data/$DATA_KEY"
export PRETOK_DIR="$DATA_DIR/$MODEL_KEY"
if [[ "$MODEL_KEY" == llama3.2-3b ]]; then
  export PRETOK_DIR="$PRETOK_DIR/pad-v1"
fi
export PRETOK="$PRETOK_DIR/pretokenized_train.parquet"
export RUN_NAME="${MODEL_KEY}-${DATA_KEY}-sft-v1"
export NNODES=1 NGPUS="${RST_TRAIN_GPUS:?set allocated GPU count}"
export NODE_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500  # 并发作业须分配独立端口
export FSDP_SIZE=-1 ULYSSES_SP=1 MAX_SEQ_LEN=32768 SAVE_HF_MODEL=1
export FUSED_KERNELS=0 WANDB_MODE=offline
unset SFT_DATA_MANIFEST
mkdir -p "$BASE_FOLDER/$RUN_NAME/logs"

bash scripts/30_run_sft_verl.sh \
  trainer.total_epochs="$EPOCHS" trainer.save_freq=200 trainer.seed=1228 \
  trainer.resume_mode=auto data.val_files=null \
  data.train_batch_size=128 optim.lr=3e-6 \
  optim.lr_scheduler_type=cosine optim.min_lr_ratio=0.1 \
  optim.lr_warmup_steps_ratio=0.03 \
  2>&1 | tee "$BASE_FOLDER/$RUN_NAME/logs/train_rank${NODE_RANK}.log"
```

Phi 使用 `MODEL_KEY=phi4-mini` 和 `$BASE_FOLDER/Phi-4-mini-instruct`。
确认最终 Hydra 配置包含上述覆盖值。registry 会赋值 `NUM_EPOCH`、`LR`、
`GLOBAL_BATCH_SIZE`，只 export 同名变量不可靠。
使用 `30_run_sft_verl.sh` 的尾部 overrides；`20_run_all.sh` 不转发这些配置且会
启动其他历史阶段，不用于本轮队列。

`resume_guard.py` 检查同一 run 的 schedule；断点续训保持原 schedule。更改 epochs、
数据或目标模型时新建 run，不能删保护文件或从已完成 1 epoch 的目录续到 3 epochs。
记录每个 epoch 实际样本/token、尾批丢弃、loss、LR、梯度、峰值显存和用时。

## 6. 导出和评估

`SAVE_HF_MODEL=1` 请求同时保存 HF 模型与恢复训练所需状态；实际验证权重分片齐全，
不要把只有 config/tokenizer 的 `huggingface/` 目录当完整模型。
若需要从 FSDP 分片导出，使用仓库真实存在的脚本：

```bash
# STEP 从当前 run 的实际保存结果确定。
bash scripts/08_prepare_eval_ckpt.sh \
  --model-key "$MODEL_KEY" \
  --ckpt "$BASE_FOLDER/$RUN_NAME/global_step_${STEP}" \
  --out "$BASE_FOLDER/exports/$RUN_NAME/step-${STEP}"
```

该脚本完成分片检查、合并、文本权重变化检查和加载生成测试。
它通过 `$BASE_FOLDER/$MODEL_DIR_NAME` 找基础模型，因此按第 2 节放置 checkpoint。
不存在 `07b_merge_fsdp.sh` 等旧提案名称，不要引用不存在的命令。
脚本末尾打印的旧评测示例不作为本轮输入规范。

所有正式保存点与同家族官方基础模型，使用**同一份目标模型的预分词 holdout**比较：

```bash
python scripts/06b_eval_offline.py \
  --model-path "$BASE_FOLDER/exports/$RUN_NAME/step-${STEP}" \
  --base-model "$MODEL_PATH" \
  --holdout "$PRETOK_DIR/pretokenized_holdout.parquet" \
  --out "$BASE_FOLDER/eval/$RUN_NAME/step-${STEP}" \
  --max-rows 0 --max-seq-len 32768 --max-actions 0
```

评测前也核验 parquet 的 tokenizer/template 指纹与模型一致。
`06b` 的 messages 分支仍使用 Qwen mask，**本轮必须输入 pretokenized holdout**。
`--max-actions 0` 关闭默认 Terminus action probe，避免把 SETA/TMax 的工具协议错判为失败。
holdout 不进训练；不只取前 128 条。报告全量保留 holdout 的 NLL、监督 token 数及
超长排除数。NLL 只在同模型同数据下比较，不把不同 tokenizer 的 NLL 直接排名。

资源和 harness 可用时，再用同一终端评测任务比较 base/SFT/DPO，报告实际任务成功率；
SETA/TMax 先确认工具协议适配。没有沙箱或协议适配时明确标为未做 agentic eval，
继续其他训练。不得用离线 NLL/偏好准确率替代真实终端成功率。

## 7. 2 次 RST DPO

仅使用有同任务成功/失败配对证据的 RST 原始轨迹。已发布
`NiuNiu0110/RST-DPO-Qwen3.5-27B` 是 Qwen token 数据，不能用于这两个模型。
Nemotron/TMax 的成功轨迹和 reward unknown 的 Opus 不自动生成 DPO 对。

原始源为 `Zhongzhi1228/Recursive-Task-Synthesis-Trajectories`，固定 revision
`2ec62e6cbbc9e6b014d016ffc31bf266049326fa`。需要时下载 `data/*.tar`、`metadata/*`
到 `$BASE_FOLDER/rst-trajectories` 并按 `metadata/shard_manifest.jsonl` 验证。
可以复用已验证的约 23 GB 原始源或匹配来源签名的重建文本 cache；不能伪造 cache
签名，也不能复用 Qwen 偏好 token。不要为取轨迹调用会启动其他准备阶段的总入口。

`scripts/17_build_dpo_data.py` 使用目标 tokenizer 重新构建，沿用 `--per-side 14`、
seed 1228 和原有配对/分组逻辑。核验 train/holdout 任务组互斥、共同 prompt、
非相同 chosen/rejected、两侧非零监督。分词后 pairs 数可能变化，不强制冒报 2,673。

以下为独占已分配节点、GPU 从 0 连续编号时的调用模板；每个模型单独运行：

```bash
# MODEL_KEY、NNODES、NGPUS、MASTER_ADDR、NODE_RANK 按该作业设置。
export POLICY="$BASE_FOLDER/exports/${MODEL_KEY}-rst-cap10-sft-v1/final"
export TOKENIZER="$POLICY"
export TRAJ_ROOT="$BASE_FOLDER/rst-trajectories"
export PAIRS_DIR="$BASE_FOLDER/dpo-data/$MODEL_KEY/rst-v1"
export REF_DIR="$PAIRS_DIR/ref-rst-cap10-sft-final"
export OUT_DIR="$BASE_FOLDER/${MODEL_KEY}-rst-dpo-v1"
export DPO_FETCH_HF=0 PER_SIDE=14 MAX_SEQ_LEN=32768 DPO_SEQ_LEN=32768
export DPO_LR=5e-7 DPO_BETA=0.1 DPO_EPOCHS=1 DPO_LENGTH_NORM=1
export DPO_GRAD_ACCUM=4 PARAM_DTYPE=fp32 DPO_LOGIT_CHUNK=512
export MASTER_PORT=29501
mkdir -p "$OUT_DIR/logs"

bash scripts/33_run_dpo.sh --seed 1228 --save-every 200 \
  2>&1 | tee "$OUT_DIR/logs/run.log"
```

先把已验证的 RST SFT 最终 HF 导出放到上述 `final` 路径，或将 POLICY 改为实际目录。
reference 与初始 policy 必须是同一个 checkpoint。记录有效 batch：
`NNODES × NGPUS × DPO_GRAD_ACCUM` 对；如果保持旧实验 batch 需调整 accum，记录原因。
默认 `--length-normalize`，不关闭 step-0 calibration，不放宽容差。

注意 `33` 的 reference 子进程会设置 `CUDA_VISIBLE_DEVICES=0..NGPUS-1`，且 reference/
训练日志写入 `$BASE_FOLDER/logs/dpo_*`。两个 DPO 顺序运行并在每次结束后归档这些日志。
若只分配了非连续/非零起始 GPU，分别调用 `17`、`18`、`19` 并显式绑定已分配设备，
保持全局 shard 编号、reference dtype、chunk size 和训练配置一致；不能假设 `33`
会保留外层的 GPU 子集映射。多节点由每节点参与，使用共享 pairs/reference 目录。

reference 阶段可恢复；`19` 当前不恢复 optimizer/schedule，中断的 DPO 不能伪称续训，
保留旧产物后在新输出目录从同一 SFT POLICY 重跑。`--save-every 200` 保存阶段 HF 权重，
不是 optimizer 断点；最终输出是 `$OUT_DIR/hf`。
验证 checkpoint 指纹、reference 完整覆盖、step-0 loss≈log(2)，记录 holdout preference
accuracy、ties、margin、裁剪比例和实际配对数，再用 RST 的同一预分词 holdout 评估。

## 8. 发布与验收

沿用历史约定：数据在 `NiuNiu0110`，模型在 `khazic`。使用服务器已配置且有写权限的
HF 账户上传经过加载验证的模型、tokenizer、config、模型卡和脱敏日志，不覆盖旧 Qwen。

- SFT：`khazic/rst-<MODEL_KEY>-<DATA_KEY>-sft-v1`，共 14 个独立模型 repo。
- DPO：`khazic/rst-<MODEL_KEY>-rst-dpo-v1`，共 2 个 repo。
- checkpoint 使用 `step-000200` 等 tag，并为最终状态记录 `final` 指向的准确 revision。
  best 仅能从真实保存并评估过的 checkpoint 中选择。

遵循对应模型许可证，Llama 保留 Llama 3.2 条款，不把所有产物统一标 Apache-2.0。
缺少写权限时，继续本地训练、验证和归档，记录待上传目录，不误传到另一个账号。

在 `$BASE_FOLDER/reports/llama_phi_training_matrix.{json,md}` 持续维护 16 行状态：
`pending / running / completed / blocked / failed`。每行记录模型和数据 revision、
代码 SHA、独立初始化来源、行数/token/丢弃、epochs/实际 steps、超参数、GPU 拓扑、
峰值显存、耗时、各保存点路径、HF revision、holdout 指标，以及阻塞/失败原因。
保存原始 release metadata 与新导出 manifest 的差异，不使用旧 Qwen 统计冒充新结果。

**completed 必须有完成训练的日志、实际权重、重载验证和评估记录。**
下载完成、生成启动脚本、GPU smoke、仅出现 checkpoint 目录，均不等于完成训练。
最终交付 16 行矩阵及可访问产物；明确区分已训练、未上传、未做 agentic eval 和真正阻塞。
