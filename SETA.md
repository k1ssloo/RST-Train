# SETA 轨迹接入与处理结果

已将 CAMEL-AI 官方 Kimi thinking 轨迹转换为 RST-Train 可读取的 Qwen3.5 SFT
数据：**976 条训练、100 条留出**，同时生成 `messages` 和 `input_ids + loss_mask`
两种 parquet。这里完成的是数据准备与校验，尚未进行 GPU 训练或终端评测。

已上传至私有数据仓库
[`NiuNiu0110/SETA-SFT-native-tools`](https://huggingface.co/datasets/NiuNiu0110/SETA-SFT-native-tools)，
revision `605f104decc273e8064e51752a43c6000774a4be`；远端文件哈希及认证下载已验证。
4B/9B/27B 独立训练命令、权重命名和每 200 步保存要求见
[`TERMINAL_TRAJECTORIES_TRAINING_PROMPT.md`](TERMINAL_TRAJECTORIES_TRAINING_PROMPT.md)。

## 数据来源

使用 [camel-ai/seta-sft-kimi-k2.5-thinking](https://huggingface.co/datasets/camel-ai/seta-sft-kimi-k2.5-thinking/tree/0090d97428087925452d74b9e8e3435f0837f36a)，
固定 revision 为 `0090d97428087925452d74b9e8e3435f0837f36a`，Apache-2.0。
文件 `data/train-00000-of-00001.parquet` 为 299,197,651 bytes，SHA-256：

```text
16b5af40bec49ad2338dc1d2268bf3ba67f4cbb8a76e82e598e9257d3398a9a4
```

轨迹由 Kimi K2.5 通过 TITO agent 在 `seta-env-v2` 上生成。每条包含原始请求、
最终回答、思考、工具调用及返回。现成 token 和 mask 使用 **Qwen3-8B**；本仓库
重新从 `raw_conv_json` 构建，不能直接复制这些 token 到 Qwen3.5 训练中。

实际文件有 **1,768 条、1,768 个不同 task_id，且任务提示文本无重复**。
数据卡正文中的 1,488 条和 1,112 条 reward=1 是旧统计；本文件实际有
**1,212 条 reward=1**。上游 reward 表示验证测试通过数/总数，本文未重新执行验证器。

官方还提供 [nothink 版本](https://huggingface.co/datasets/camel-ai/seta-sft-kimi-k2.5-nothink)，
数据卡说明它来自相同 rollouts，移除渲染文本中的思考并加入 `/no_think`。
本次处理 thinking 版本，未把两者当成两批独立轨迹合并。官方代码目前位于
[camel-ai/seta](https://github.com/camel-ai/seta)，数据卡中的 `terminal_agent` 链接已失效。

## 筛选与转换

`scripts/03i_build_seta_sft.py` 按以下顺序筛选；表中排除数互不重叠：

| 阶段 | 排除 | 剩余 |
|---|---:|---:|
| 原始发布 | — | 1,768 |
| reward 不等于 1 | 556 | 1,212 |
| 最后一次响应不是正常 `stop` | 38 | 1,174 |
| 调用参数解析或工具返回配对检查 | 98 | 1,076 |
| 模板、长度、mask 与去重检查 | 0 | 1,076 |

98 条中，58 条工具参数无法严格解析为 JSON 对象，40 条工具返回无法一一配对。
未正常结束的 38 条包含 36 条仍在调用工具、1 条长度截断和 1 条引擎过载。
这些样本保留在原始文件中，排除原因记录在 `rejected_rows.jsonl`；不猜测修复代码。

处理流程：

1. 按官方方法还原 `request.messages + response.choices[0].message`，解析 JSON
   字符串形式的工具参数，保留原始思考和可见回答。
2. 使用 Qwen3.5 工具调用格式。连续工具返回合并为一个包含多个
   `<tool_response>` 的 `user` 回合，与原生模板保持一致，并保留之前回合的思考。
3. 每条检查原生工具消息与扁平 `{role, content}` 消息的渲染逐字一致，使用共同的
   长度、去重和 loss-mask 检查；不截断轨迹。去重签名包含工具名及全部参数，
   覆盖写文件、等待进程等没有 `command` 字段的操作。
4. 按 task_id 分组划分留出集，seed=1228；任务 ID 及相同任务提示均不得跨 split。
   只监督助手输出；系统提示、任务文本、工具返回、助手头和 `<think>` 开头不参与损失。

全部保留样本合计 **10,895,396 tokens**，其中 **5,458,934** 个训练目标 tokens
（50.1%）；中位长度 8,942.5，p90=17,641.5，最长 26,391，均在 32,768 上限内。

## 使用范围

这些文件符合 RST-Train 的 SFT 数据结构，可由现有预分词导出器和 verl 数据集类读取。
交互协议保留 SETA 的 `shell_exec`、`shell_write_content_to_file`、`shell_view`、
`shell_wait`、`shell_write_to_process`、`shell_kill_process` 六种工具。
当前 Harbor/Terminus-2 评测使用 `analysis/plan/commands` JSON；要评测 SETA 工具协议，
还需适配相应 agent/toolkit，不能把数据格式兼容当作执行协议已经兼容。

所有 1,768 条日志的 `request` 都只有 `messages`，没有 `tools` 定义；上游渲染文本
同样没有工具 schema。本次保留原始上下文，没有根据新版 toolkit 补写定义。
工具实现可参考固定源码
[terminal_toolkit.py](https://github.com/camel-ai/seta/blob/e4715b01174e6c9503fc46120d81dd692ced75e6/seta_env/toolkits/terminal_toolkit.py)，
但它不是这些历史请求所用 schema 的证明。

本版本每个任务只有一条轨迹，不能直接构造同任务成功/失败的 DPO 对。
留出集用于 SFT loss 检查；尚未做与其他训练来源或评测集的跨数据集去污染。

## 本地文件与重现

在 `RST-Train/` 下运行。原始文件及其 `source.json` 校验信息已经缓存于
`data/seta-sft/source-thinking/`；研究元数据与官方源码摘取位于 `data/seta-sft/research/`。
转换器拒绝覆盖非空输出目录，重跑请换一个版本目录。

```bash
.venv/bin/python scripts/03i_build_seta_sft.py \
  --source data/seta-sft/source-thinking/train.parquet \
  --source-info data/seta-sft/source-thinking/source.json \
  --tokenizer data/Qwen3.5-27B-tokenizer \
  --out-dir data/seta-sft/sft-v2 --max-seq-len 32768 --holdout 100

for split in train holdout; do
  .venv/bin/python scripts/15_export_pretokenized.py \
    --parquet "data/seta-sft/sft-v2/seta_sft_${split}.parquet" \
    --tokenizer data/Qwen3.5-27B-tokenizer \
    --out "data/seta-sft/sft-v2/pretokenized_${split}.parquet" --strict
  .venv/bin/python scripts/03b_validate_sft_data.py \
    --parquet "data/seta-sft/sft-v2/seta_sft_${split}.parquet" \
    --tokenizer data/Qwen3.5-27B-tokenizer --sample 100000 --show 0
done
```

本次实际产物在 `data/seta-sft/sft-v1/`：

| 文件 | 内容 |
|---|---|
| `seta_sft_train.parquet` / `seta_sft_holdout.parquet` | 976 / 100 条消息及来源信息 |
| `pretokenized_train.parquet` / `pretokenized_holdout.parquet` | 对应 token 与 mask，导出未额外丢样本 |
| `manifest.json` | 固定版本、文件哈希、筛选计数、tokenizer 指纹及 split 统计 |
| `pretokenized_*_manifest.json` | 分词导出计数和目标 token 数 |
| `rejected_rows.jsonl` | 每条被筛除轨迹的 ID 与原因 |
| `verification.json` | 全量独立 mask 对比与提示词监督检查 |

## 验证

对全部 1,076 条产物使用 `03b_validate_sft_data.py` 的独立实现重新分词并计算 mask，
逐条与预分词文件比对：**0 token/mask 不一致、0 提示词 token 被监督、0 跨 split
任务重叠**。`verification.json` 记录了两种格式的文件哈希及各 split 的检查结果。

新增 `tests/test_seta_convert.py` 的 13 项测试覆盖参数解析、截断、重复或缺失返回、
思考保留、真实 Qwen3.5 模板和 mask、文件校验、去重以及空留出集 schema。
pytest 和 standalone runner 均通过。完整本地测试：**389 passed，19 skipped**；
跳过项为 18 项缺少 torch、1 项当前数据环境缺少 Harbor。未运行 GPU 训练。

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python tests/test_seta_convert.py
```
