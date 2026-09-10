# 其他模型的本地 SFT / DPO 验证（2026-09-10）

覆盖注册表中排除 Llama、Qwen、Phi 后的全部四个 checkpoint：SmolLM3-3B、OLMo-3-7B-Instruct、Gemma 4 E2B/E4B。
四者均通过官方完整权重的短序列训练检查；本次发现并修复 OLMo3 的 FSDP2 位置编码精度问题（[BUG-30](../BUG.md)）。

## 覆盖范围

- 26 个消息文件，共 **437,581 行/模型**，覆盖 train、holdout、原始数据和当前 TerminalEvo 审计池。逐文件累计，包含不同数据版本间的重叠，不是去重样本数。
- 每条运行生产代码的完整对话渲染、直接分词一致性、assistant mask、首 token、词表边界检查；统计 8K/32K 超长，未截断。Gemma 两款官方 tokenizer/template/选项指纹相同，共享分词结果；权重分别检查。
- 现有三版 DPO 的 4,019 对全部从原消息重新编码，保留原配对与 train/holdout 分组；其他 SFT 集合没有现成偏好对。Opus 保留 unscored 属性。
- 任务资产、metadata、仅含任务提示的 RL taskset、发布/下载副本和旧 TerminalEvo 审计快照未作为独立训练语料重复计数。选取的文件路径、SHA-256、逐 split 结果见 [JSON](cross_family_validation_20260910.json)。

## 完整轨迹可用数量

每格为 **8K / 32K** 下的可用完整轨迹数；包含各自已有的 train 与 holdout。

| 数据集 | 输入行数 | SmolLM3 | OLMo3 | Gemma E2B/E4B |
| --- | ---: | ---: | ---: | ---: |
| rst-cap10 | 10,778 | 6,271 / 10,778 | 6,321 / 10,778 | 5,154 / 10,742 |
| ota | 14,312 | 9,608 / 14,312 | 9,679 / 14,312 | 8,538 / 14,285 |
| nemo | 139,676 | 15,048 / 139,274 | 15,245 / 139,276 | 10,097 / 136,532 |
| tmax | 5,645 | 4,345 / 5,645 | 4,300 / 5,645 | 3,758 / 5,625 |
| seta | 1,076 | 541 / 1,076 | 531 / 1,076 | 386 / 1,076 |
| terminal-lego-deepseek | 12,138 | 7,667 / 12,138 | 7,726 / 12,138 | 6,603 / 12,076 |
| terminal-lego-opus-unscored | 8,266 | 7,237 / 8,266 | 7,249 / 8,266 | 6,819 / 8,263 |
| rst-v1 | 8,886 | 5,136 / 8,886 | 5,176 / 8,886 | 4,229 / 8,856 |
| nemo-adapters | 220,381 | 79,324 / 208,275 | 79,578 / 208,321 | 70,122 / 201,100 |
| swegym-sft-source | 491 | 57 / 458 | 58 / 458 | 35 / 425 |
| terminalevo-audit-v18 | 526 | 514 / 526 | 514 / 526 | 505 / 526 |

七组历史训练集、旧版 RST、SWE-Gym SFT 和 TerminalEvo 审计池没有模板/掩码错误。额外的 **Nemotron adapters 有 13 条不同轨迹包含保留标记**：SmolLM3 拒绝 1 条，OLMo3 拒绝 2 条，Gemma 两款各拒绝 11 条，模型间存在重叠。标记包括 `<|im_start|>`、`<|endoftext|>` 和 `<eos>`，确实出现在消息正文；拒绝行为已逐条复现。
严格导出这份额外数据前，须按逐模型清单过滤上述行，否则 `--strict` 会中止。清单位于报告 JSON 的 `input_exclusions`，以及 `outputs/model-validation-20260910/nemo-adapters-exclusions.json`，包含源 SHA-256、零起始行号、trajectory_id 和匹配标记。上表可用数已经排除了这些行。
TerminalEvo 的 526 行仅为审计池，尚非正式训练发布。另逐文件检查了 Gemma 模板会移除的原生 channel 标记，消息正文中未发现此类标记。
默认 8K 会丢掉大量长轨迹。复现此前实验宜显式评估 32K 配置；Gemma 换词表后仍有一部分轨迹超过 32K，需按导出 manifest 排除并记录，不能悄悄截断。

原始 SWE-Gym sampled rollouts（6,055 行）与 TMax（5,795 行）均带原生工具字段，当前文本入口明确拒绝，需要显式转换。TMax 已转换版本在上表。TermiGen（3,556 行）只有提示，无 assistant 监督。另列的 SWE-Gym SFT（491 行）本身就是文本消息，检查通过。

## DPO 全量结果

| 版本 | 原始 train / holdout | SmolLM3、OLMo3（32K） | Gemma E2B/E4B（32K） |
| --- | ---: | ---: | ---: |
| dpo-v1 | 1,228 / 102 | 1,228 / 102 | 1,215 / 102 |
| dpo-v2 | 2,448 / 225 | 2,448 / 225 | 2,432 / 222 |
| dpo-smoke | 12 / 4 | 12 / 4 | 12 / 4 |

Gemma 共排除 32 对超长偏好，模板/配对条件无失败。已检查分组不泄漏，完整重编码 parquet 位于 `outputs/model-validation-20260910/dpo-retokenized/`。Gemma 两款可以共享 token 数据，reference logprobs 必须分别计算。旧 Qwen token parquet 对四个目标模型均被指纹门禁拒绝。

## 训练与回归验证

- 官方完整权重：四模型各 21 个完整行严格导出及真实 dataset 加载，共 84 例；SFT 的实际 verl 输入/输出与 loss 路径使用 64-token 窗口前向、反向。三版 DPO 每模型各取一个真实偏好对，检查 reference/policy 一致、有限梯度和参数更新。没有执行全量 reference 打分或完整 epoch。
- CPU 全套：**496 passed，11 skipped**。跳过为缺少 verl 8 项、需显式开启 GPU 2 项、缺少 Harbor 1 项。真实 verl 环境另有 **26 passed、0 skipped**；最终 hook 绑定再通过 4 项。
- H100 小配置：SmolLM3 / OLMo3 / Gemma 各 FP32、BF16 共 6 例通过。单 rank 的 FSDP2 与相同精度的未分片 DPO 梯度一致；新增 GPU 回归分别验证实际 SFT wrapper 和 DPO 分组，并直接检查 OLMo3 旋转编码输入仍为 FP32。显存限额为 1–2 GiB，现有生产进程保留。
- OLMo3 修复前，BF16 分片会将 cos/sin 降精度，梯度 L2 偏差约 0.98%；修复后与未分片 DPO 对齐。此前 Gemma 的较大差异来自基准误将浮点 buffer 也转为 BF16，对齐 buffer 精度后通过原逐元素容差。
- BF16 原生 verl CPU loss 归约与 HF FP32 CE 有量化差异；与独立 BF16 计算一致，原始两组数值保存在 JSON。FP32 梯度回归另行通过。
- **未验证全尺寸 8K/32K GPU 训练、多 rank 通信、整轮训练效果或吞吐。** GPU 结果来自小配置，官方完整权重的检查在 CPU 短窗口执行。

## 复现

以下命令从仓库根目录运行。详细模型路径与固定 revision 在 inventory/报告 JSON 中；同名输出目录只允许相同代码、输入和 tokenizer 恢复。

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/16b_validate_model_datasets.py \
  --inventory outputs/model-validation-20260910/sft-inventory.json \
  --out outputs/model-validation-20260910/sft-audit --workers 20 --batch-size 128
.venv/bin/python -m pytest tests/ -q -rs
# 在包含 verl 的训练环境运行：
python tests/run_tests.py test_generic_verl test_model_precision test_multimodel_training
RST_RUN_CUDA_TESTS=1 python tests/run_tests.py test_model_precision_cuda
```

分词缓存和测试产物保留于忽略目录 `outputs/model-validation-20260910/`；报告 JSON 保存关键产物及实现文件哈希。训练 checkpoint 的 200-step 保存约定保持不变。
