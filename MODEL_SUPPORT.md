# 多模型 SFT 与 DPO

本轮扩展的是 **文本轨迹的 SFT（verl/FSDP2）与离线 DPO**。原有 Qwen3.5 路径保留；新增模型默认使用原生 Transformers forward、padding、8K 序列，关闭 Qwen 专用 fused kernels、Liger 和 Ulysses SP。这些配置不是全尺寸 GPU 性能承诺。

Llama-3.2-3B-Instruct / Phi-4-mini-instruct 在七组历史数据上的完整执行矩阵见
[`LLAMA_PHI_TRAINING_PROMPT.md`](LLAMA_PHI_TRAINING_PROMPT.md)：14 次独立 SFT +
2 次 RST DPO，按历史实验明确覆盖为 32K，并要求先完成实际 GPU smoke。

## 实验模型选择

| MODEL_KEY | 官方 checkpoint | 参数量 / 代际 | 建议用途 |
| --- | --- | --- | --- |
| `smollm3-3b` | [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3.1B，2025 | 低成本、Apache-2.0 的主力对照 |
| `llama3.2-3b` | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 3.2B，2024 | 与 Qwen 不同家族的较早对照；不是同期发布 |
| `llama3.2-1b` | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1.2B，2024 | 流程验证、极弱学生实验 |
| `llama3.1-8b` | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8.0B，2024 | 较大 Llama 对照；需要 Meta 模型访问许可 |
| `phi4-mini` | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | 3.8B，2025 | MIT 许可的小模型补充对照 |
| `olmo3-7b` | [allenai/Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) | 7.3B，2025 年末 | 更接近 Qwen3.5 时间窗口、开放训练资料的对照 |
| `gemma4-e2b` | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | 约 5.1B 总参数，2026 | 同年模型的优先候选；E2B 指有效计算规模 |
| `gemma4-e4b` | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | 约 8.0B 总参数，2026 | 更大预算的同年对照 |

资料核查于 2026-09-09。Llama 使用对应版本的社区许可证；Gemma 4、SmolLM3、OLMo 3 的上述发布为 Apache-2.0。Gemma 4 的 E2B/E4B 含大规模 per-layer embeddings，训练显存应按总参数量估计。Llama 4 的 MoE 总权重也不能按激活参数量视作“小模型”。模型的终端任务质量仍应在同一评测集上实测。

建议先用 **Qwen3.5-4B + SmolLM3-3B + Llama3.2-3B** 验证跨家族效果，再加入 Gemma4-E2B。固定任务划分、轨迹来源和训练 token 预算，分别汇报基础模型分数、训练增益、截长丢弃率和推理成本。

## SFT

先将完整的 HF checkpoint（包括 tokenizer、模板和 config）下载到本地。使用能识别目标架构的 Transformers；CPU 验证环境为 Transformers 5.15.0。**原有 Qwen 的 verl 路径仍有自己的 Transformers/FLA 版本检查**，不要为新模型覆盖正在运行的 Qwen 环境。

```bash
export BASE_FOLDER=/shared/rst
export MODEL_KEY=smollm3-3b
export MODEL_PATH=$BASE_FOLDER/SmolLM3-3B
export DATA_DIR=$BASE_FOLDER/sft-v1-cap10

python scripts/14_prepare_tokenizer.py --model "$MODEL_PATH" --verify-verl

python scripts/15_export_pretokenized.py \
  --parquet "$DATA_DIR/rst_sft_train.parquet" \
  --tokenizer "$MODEL_PATH" \
  --out "$DATA_DIR/$MODEL_KEY/pretokenized_train.parquet" \
  --max-seq-len 8192 --strict

NNODES=1 NGPUS=2 MAX_SEQ_LEN=8192 bash scripts/30_run_sft_verl.sh
```

`NGPUS=2` 仅是调用示例，实际配置须通过启动器的显存检查和短 GPU smoke。未提供 `PRETOK` 时，新模型自动使用 `DATA_DIR/MODEL_KEY/`；若文件不存在，启动器会从 messages 重新导出。Qwen 沿用原来的数据路径。序列超长会被计数并丢弃，不会悄悄截断。

### 固定 tokenizer padding

导出前须在 checkpoint 中明确保存 `pad_token`（BUG-28）。否则 verl 自动补 EOS，
会使训练指纹与普通加载时导出的数据不一致。`14_prepare_tokenizer.py` 默认只检查，
`--apply` 才备份并持久化配置；`--verify-verl` 要在训练环境运行，核验真实加载路径。
Llama-3.2 可使用现有 `<|finetune_right_pad_id|>`：

```bash
python scripts/14_prepare_tokenizer.py --model "$MODEL_PATH" \
  --pad-token '<|finetune_right_pad_id|>' --apply --verify-verl
```

脚本从实际词表解析 ID，不新增 token，并同步已有 model/generation padding 配置。
Phi-4-mini 官方 tokenizer 已有 EOS/pad `199999`，检查一致时保留。配置变化后从 messages
重新导出受影响的 SFT/DPO 数据，并重算相应 DPO reference；不能重写旧指纹绕过校验。
具体修复及版本归档步骤见 `LLAMA_PHI_TRAINING_PROMPT.md` 第 2.1 节。

## DPO

将 SFT checkpoint 导出为**完整 HF 目录**并设为 `POLICY`；DPO 的初始 policy 与 reference 必须是同一个 checkpoint。新模型不会默认下载已有 Qwen preference parquet。

```bash
export POLICY=$BASE_FOLDER/smollm3-3b-sft-hf
export PAIRS_DIR=$BASE_FOLDER/dpo-smollm3-3b
export TRAJ_ROOT=$BASE_FOLDER/rst-trajectories
NNODES=1 NGPUS=2 MAX_SEQ_LEN=8192 bash scripts/33_run_dpo.sh
```

脚本依次构建偏好对、计算 reference logprobs、执行 DPO。两侧使用目标 tokenizer 重编码；允许复用重建后的文本轨迹 cache，不能复用别的模型的 token IDs、loss masks 或 reference 分数。新 reference 带数据 SHA-256；改变数据后需另建 reference 目录重新打分。

## 数据协议与验证边界

- 新导出在 parquet 内嵌 tokenizer、chat template、模板参数和 mask 指纹。SFT dataset、DPO 打分与训练都会核验，单纯改文件夹名不能绕过检查。历史无指纹数据仅保留旧 Qwen 兼容路径。
- 新家族当前支持纯文本消息和 `step_loss_mask`。原生 `tool_calls`、独立 reasoning 字段、多模态 content 需要显式转成文本；不能静默丢失内容。实际终端输出仍然只作上下文。
- SmolLM3 使用 `enable_thinking=False`，固定模板日期为 `2026-09-09`；Llama 的模板日期也固定。推理应沿用 `rst_common.tokenization.render_kwargs(profile)`，不能恢复模板的动态日期或不同 thinking 默认值。
- Qwen 的既有 mask 保持不变。其他模型不套用 Qwen 的 25%–45% 监督比例；统计实际比例，并核验逐 token 边界。
- CPU 测试涵盖各家族的原生 SFT loss、分块 DPO logprob/梯度、Gemma soft-capping、共享 embedding 分组、DPO 单步训练/保存和失败路径。真实官方 tokenizer 另做本地检查；Llama 受访问许可限制，模板单测使用合成 tokenizer。
- **尚未验证全尺寸 GPU/多卡训练与质量收益**。新增家族的 Megatron/slime 与在线 RL 会明确拒绝；本轮不宣称支持。

本次全量 CPU 检查为 **484 passed、1 skipped**；跳过项需要未安装的 Harbor。
最后的参考缓存兼容性修改另通过相关回归。四个开放访问的官方 tokenizer 均通过整段渲染、
观察文本排除和屏蔽指定 assistant 回合的检查；检查记录位于
`reports/model_tokenizer_compatibility_20260909.json`。
该记录还包含 Phi 官方 tokenizer 对七组历史数据各 10 条的抽样检查（Nemotron
使用 holdout）；70 条均通过模板和掩码检查。这不替代全量重新分词及 GPU 验证。
BUG-28 的官方 Phi padding/上游 tokenizer helper 检查记录在
`reports/tokenizer_padding_check_20260909.json`；实际训练环境仍须运行 `--verify-verl`。

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python tests/run_tests.py test_model_tokenization test_multimodel_training
```
