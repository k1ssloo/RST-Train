# Terminal-Lego 现成轨迹下载与 SFT

已下载官方生成的 **23,152 条原始记录、709.1 MB**，并分别转换为 Qwen3.5 SFT：
DeepSeek **11,938 train + 200 holdout**；Opus 未评分 **8,066 train + 200 holdout**。
没有调用教师模型生成新轨迹，也没有执行 GPU 训练。

## Hugging Face 发布

以下均为 `NiuNiu0110` 下的私有仓库，已核对远端文件哈希和认证下载：

- [Terminal-Lego-DeepSeek-SFT-terminus](https://huggingface.co/datasets/NiuNiu0110/Terminal-Lego-DeepSeek-SFT-terminus)：
  `292330c713254548893fbaef435cfd7fb6199096`，含 messages/pretokenized 与 train/holdout。
- [Terminal-Lego-Opus-Trajectories-unscored](https://huggingface.co/datasets/NiuNiu0110/Terminal-Lego-Opus-Trajectories-unscored)：
  `1c5578aff4cad23df70ceb46851c128aeafb2fd5`，保留原始 JSON，未推断 reward。

本轮 DeepSeek、Opus 与 SETA 的 4B/9B/27B 独立训练计划、权重名称和每 200 步保存要求见
[`TERMINAL_TRAJECTORIES_TRAINING_PROMPT.md`](TERMINAL_TRAJECTORIES_TRAINING_PROMPT.md)。
Opus 已纳入九任务计划；`03j` 现支持显式未评分转换，命令见下文。
集群执行者可直接转换并训练三个尺寸，无须另写适配器。

## 下载来源

| 发布版本 | 固定 commit | 实际记录数 | 成功标记 |
|---|---|---:|---|
| [Terminal-Lego-Traj-8k：Opus 4.6](https://huggingface.co/datasets/Lego-X/Terminal-Lego-Traj-8k/tree/02f33fb252b15308f309e7d10472bb781eeb1ecf) | `02f33fb2…` | 8,318 | 无逐条模型 reward |
| [Terminal-Lego-Traj-Deepseek-V3-2-15k](https://huggingface.co/datasets/Lego-X/Terminal-Lego-Traj-Deepseek-V3-2-15k/tree/922c2fcb019854911fc05eb1a208d2685ac6e006) | `922c2fcb…` | 14,834 | 12,400 条 reward=1；2,434 条 reward=0 |

两个文件均通过固定版本的 LFS SHA-256 和大小校验。模型名称来自发布文件名；
原始行没有独立的模型配置字段。两个版本可能覆盖相同任务，23,152 是对话记录
总数，不是去重后的任务数，文件名中的 8k/15k 也不能替代实际计数。

原始文件位于：

```text
data/terminal-lego-trajectories/
  source-opus-8k/terminal-lego-opus-4-6-8k.json
  source-deepseek-15k/terminal-lego-deepseek-v3-2-15k.json
```

各来源目录还保存 `source.json`（版本、哈希、下载地址）、`audit.json`（实测
字段与计数）和第一条完整样本。

## SFT 筛选结果

Opus 全部 8,318 条记录的元数据只有 `oracle_passed_task`、`difficulty`。这
不能证明每条模型轨迹通过了验证器；助手自己的 `task_complete=true` 也不是
grader reward。HF 上保留原始下载；使用 `--allow-unscored` 单独转换为未评分 SFT。
每条保留 `reward=null`、`reward_available=false`、`reward_policy=unscored` 及
原始 `oracle_passed_task`、`difficulty`；未提供的 trial/path/run 保留 null。
该开关仅支持 Opus，显式提供的 reward=0 或非法 reward 仍被排除，默认模式仍
要求 reward=1。未评分模式继续检查完整性、动作格式、控制标记、去重及长度。

2026-09-08 对固定 Opus 源文件的实测结果：

```text
8,318 条原始轨迹
  − 14 条助手动作缺少必需字段
  −  8 条助手回答无法解析
  −  3 条含聊天模板控制标记
  − 12 条最后未声明任务完成
  = 8,281 条完整候选；共同去重检查未再删除记录
  − 15 条超过 32,768 tokens
  = 8,266 条未评分 SFT：8,066 train + 200 holdout
```

共 **42,418,933 tokens**，最长 **31,781 tokens**；8,266 条评分全部保留未知。
共享规范化器改写了 2,266 个助手回合，并修复 1,562 段已过时的格式警告前缀。

DeepSeek 按以下顺序过滤：

```text
14,834 条原始轨迹
  − 2,434 条 reward=0
  −    63 条含无法解析的助手回答
  −     2 条含聊天模板控制标记
  −    76 条最后未声明任务完成
  −     1 条命令结构无效
  = 12,258 条完整候选；共同去重检查未再删除记录
  −   120 条超过 32,768 tokens
  = 12,138 条 SFT：11,938 train + 200 holdout
```

完整数据共 **107,024,374 tokens**，最长 **32,766 tokens**。不完整或过长记录
整条排除，不截断。保留的 `messages` 使用现有 Terminus-2 协议，包含完整提示、
助手的 `analysis/plan/commands` 和终端观察。沿用 RST JSON 规范化，实际修正了
一条因格式已修复而过时的后续警告。

发布 reward 作为上游证据保存，本次没有重放这些轨迹来重新验证评分。

## 任务编号不能直接关联环境池

DeepSeek 文件有 **368 个裸 `task_编号` 对应不同任务正文**。例如 `task_00000`
在两个批次中分别是 PyTorch 张量生成和 Linux 文本文件查找，也不同于本地
环境池同编号的递归通配符查找任务。不能仅凭编号合并轨迹、套用环境排除表
或构造 DPO 成败对。

转换器按初始提示中 `Task Description` 正文的 SHA-256 建立任务组，分组时
排除初始终端屏幕里的随机容器 ID；训练消息仍保留完整屏幕。按任务组切分，并
额外检查非空来源路径及 Opus 任务标识不跨 train/holdout；缺失路径不充当任务组。
未声称完成跨数据集的语义去重。

## 复现命令

在 `RST-Train/` 下执行。已有下载会校验后复用；哈希不一致时停止，不覆盖
已有源文件。转换输出目录必须为空，复跑时换新目录。

```bash
# 下载并保留 Opus 原始轨迹。
.venv/bin/python scripts/03j_build_terminal_lego_sft.py \
  --release opus-8k --download --download-only

# 显式转换未评分 Opus；绝不推断模型 reward。
.venv/bin/python scripts/03j_build_terminal_lego_sft.py \
  --release opus-8k --allow-unscored \
  --tokenizer data/Qwen3.5-27B-tokenizer \
  --out-dir data/terminal-lego-trajectories/opus-unscored-sft-v1 \
  --max-seq-len 32768 --holdout 200 --seed 1228

# 下载 DeepSeek 原始轨迹并转换；仅下载时同样可用 --download-only。
.venv/bin/python scripts/03j_build_terminal_lego_sft.py \
  --release deepseek-15k --download \
  --tokenizer data/Qwen3.5-27B-tokenizer \
  --out-dir data/terminal-lego-trajectories/deepseek-sft-v1

# 为任一转换目录导出 token 与助手损失掩码。
# DeepSeek 使用 data/terminal-lego-trajectories/deepseek-sft-v1。
LEGO_SFT_DIR=data/terminal-lego-trajectories/opus-unscored-sft-v1
for lego_split in train holdout; do
  .venv/bin/python scripts/15_export_pretokenized.py \
    --parquet "$LEGO_SFT_DIR/terminal_lego_sft_${lego_split}.parquet" \
    --tokenizer data/Qwen3.5-27B-tokenizer \
    --out "$LEGO_SFT_DIR/pretokenized_${lego_split}.parquet" \
    --strict
done
```

两个转换目录分别包含：

- `terminal_lego_sft_train.parquet`、`terminal_lego_sft_holdout.parquet`：messages
  数据；保留 `source_row`、`source_task_path`、`source_trial_id` 以定位原始记录。
- `pretokenized_train.parquet`、`pretokenized_holdout.parquet`：`input_ids` 和
  `loss_mask`；提示及终端输出不参与训练损失。
- `manifest.json`：过滤计数、任务组、token 统计、输入版本和输出哈希；新增
  `allow_unscored`、`reward_available_examples`、`unscored_examples`。
  `reward_policy` 为 `source_reward_one` 或 `unscored`，详细规则见
  `reward_policy_description`。DeepSeek 的筛选与切分不变；新增列使 parquet 哈希改变。
- `rejected_rows.jsonl`：原始行号及排除原因；各预分词产物另有独立 manifest。

## 验证

DeepSeek 两份预分词产物均以 `--strict` 导出，**12,138/12,138 条保留，零新增丢弃**。
训练部分有 42,581,331 个受监督 tokens；连同 holdout 共 43,272,763 个。
所有落盘行均通过来源对应、token/mask 等长、二值掩码及首 token 屏蔽检查。
原始行号收支与分组核对见 `sft_validation.json`，预分词文件哈希及检查结果见
`pretokenized_validation.json`；下载汇总为上一级的 `download_manifest.json`。

Opus 同样以 `--strict` 导出，**8,266/8,266 条保留，零新增丢弃**。
训练集有 **41,451,885 tokens / 14,723,292 supervised tokens（35.52%）**；
holdout 有 **967,048 / 330,785**。本地 `opus-unscored-sft-v1/validation.json`
记录原始行收支、未知评分和元数据保留、任务切分、token/mask 检查及文件哈希；
`release_manifest.json` 已通过训练启动器的 `SFT_DATA_MANIFEST` 数据检查。
这些 Opus 转换产物保存在本地，HF 上述固定 revision 仍只含原始轨迹。

适配器现有 22 项回归测试在 pytest 和独立 runner 下均通过；包含真实 Qwen3.5 分词器
的助手监督、任务描述与终端观察屏蔽检查。最终 CPU suite：**436 passed，
19 skipped**，跳过项为 18 项缺 torch、1 项缺 Harbor。

```bash
.venv/bin/python -m pytest tests/ -q -rs
.venv/bin/python tests/run_tests.py test_terminal_lego_convert
```
