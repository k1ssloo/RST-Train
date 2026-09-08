# TerminalWorld 数据接入与试处理

本次把 TerminalWorld 接入了 RST-Train 的 **Harbor 任务池**。最终产物是
604 个训练候选任务和单独的 20 个官方评测样本；还没有生成模型训练轨迹。

## RST-Train 如何处理训练数据

SFT 入口是带模型回答及终端观察的轨迹，而不是任务描述或参考解脚本：

1. `03_build_sft_data.py` 从 RST 元数据中筛选完成、无异常、reward=1 的轨迹，
   按任务组限额，并在来源模型间轮换抽样。
2. ATIF 轨迹还原成 `messages`。Terminus-2 的首条提示是 `user`；模型回答使用
   `{analysis, plan, commands[, task_complete]}`。规范化回答时同步修正下一条
   观察中过时的格式警告，保持对话一致。
3. `sft_common.py` 执行全文去重和同任务命令序列去重，校验聊天模板两种分词方式
   一致，过滤超过 32,768 tokens 的样本，默认按任务组划分 train/holdout。
4. `15_export_pretokenized.py` 输出 `input_ids` 和 `loss_mask`，只监督助手回答；
   提示词和终端输出不参与损失。verl 读取这份已经验证的 token/mask 数据。

不同来源的 `03d/03e/03f` 转换器保留各自实际使用的交互协议，再接入共同的校验流程。
任务型数据走 `10*`：输出 `{prompt, label, metadata.task_dir}`，由 Harbor 执行。
真实成功轨迹之后通过 `03h_build_rollout_sft.py` 转成相同的 SFT schema。
DPO 则需要同一任务的成功和失败轨迹，不能仅凭参考解构造。

## 使用的数据与版本

| 来源 | 固定版本 | 本次读取内容 |
|---|---|---|
| [EuniAI/TerminalWorld](https://huggingface.co/datasets/EuniAI/TerminalWorld/tree/dda7c099cc076735aef28c03bf8d3624dc0564e1) | `dda7c099…` | full 1,530 条、verified 200 条、sample 20 条元数据；下载 sample 的 20 个任务包 |
| [andylizf/TerminalWorld-Seeds-Clean](https://huggingface.co/datasets/andylizf/TerminalWorld-Seeds-Clean/tree/e033a42eaf1b6748607bb563fe9f621b5e1452f9) | `e033a42e…` | 1,353 个修复后的任务包、训练推荐列表、上游尝试次数统计 |

第二项是第三方整理版本，沿用 RST 的 `metadata/tasks.parquet` + tar 分片布局。
两者提供任务、Docker 环境、`solution/solve.sh`、`tests/test.sh` 和
`tests/test_state.py`，不含可直接监督的模型对话。原版全量任务包约 928 MB；本次
只下载约 70 KB 的原版样本包及约 19 MB 的修复版分片。

## 处理结果与发现的问题

训练候选的筛选过程为：

```text
1,353 个 Seeds-Clean 任务
  → 663 个上游 train_ready，且 reward_verdict=pass
  → 排除 58 个官方 verified 任务，剩 605 个
  → 排除 1 个旧式资源配置任务，剩 604 个
  → 完整文件检查、TOML 解析、评测脚本泄漏检查：604 个通过
```

`Clean` 的上游定义不能替代 reward：全部 1,353 条仍含 446 个 fail 和 46 个
unknown。当前模型在这些任务上的成功率尚未测量，因此输出的 `tier="unknown"`、
`empirical_pass_rate=null`。上游 CSV 的 `pass_at_5` 实际为 solved/graded 比例，
作为单独的历史统计保留；最终候选按该统计为 480 easy、111 sweet、13 unknown，
不能直接当作当前 Qwen 策略的难度分布。

审计发现以下实际差异，详见 [BUG-20](BUG.md#bug-20--terminalworld-metadata-and-resource-aliases-disagree-with-runnable-packages)：

- 修复版分片内部的 SHA-256/大小清单已过时。下载使用固定 Hub commit 的 Git/LFS
  校验值，并记录内部清单的差异。
- 1,353 个包中，123 个 Dockerfile、6 条 instruction、10 个 solution 与 parquet
  文本不一致；129 个任务内容哈希变化。官方 20 个样本也有 6 条 instruction 不一致。
  转换器始终读取实际文件生成 prompt。
- 原始 605 个候选中，451 个因 `memory`/`memory_mb` 等资源别名冲突无法被当前
  Harbor 加载。输出副本保留数值 `memory_mb`/`storage_mb`，移除重复的旧别名，
  共整理 546 个训练配置和 17 个评测配置。源文件不变，并同时记录源包和输出包哈希。
- `tw_239821` 将资源配置写在 TOML 顶层，当前 Harbor 会忽略，因此排除等待审查。

保留了 canary 标记。训练集对官方 verified 的 **ID 重叠为零**；这不证明语义独立，
尤其 RST 的部分合成任务本就来自 TerminalWorld 种子。若使用其任务轨迹训练，
应报告训练来源，不能把完整 TerminalWorld 当成独立评测集。

## 重现命令

在 `RST-Train/` 下运行。`--out` 必须是新目录或空目录，重复执行时更换输出目录；
下载缓存可以复用，省略 `--download` 即为离线转换。

```bash
.venv/bin/python scripts/10d_build_terminalworld_taskset.py \
  --source official --source-root data/terminalworld/source \
  --split sample --download --out data/terminalworld/official-sample-v1

.venv/bin/python scripts/10d_build_terminalworld_taskset.py \
  --source seeds-clean --source-root data/terminalworld/seeds-clean-source \
  --official-root data/terminalworld/source \
  --download --out data/terminalworld/train-v1
```

每个输出目录包含：

- `rl_tasks.jsonl`：现有 RL adapter 可读取的任务记录，`task_dir` 为本机绝对路径。
- `tasks/<task_id>/`：可交给 Harbor 的任务环境、指令、参考解和验证器。
- `manifest.json`：筛选计数、输入哈希、版本、修复统计及验证范围。
- `source_audit.json`：逐任务的元数据差异、缺失文件、泄漏检查和配置修改记录。

`--split train` 为默认选项；原版使用 full 减 verified，修复版额外要求 train_ready
及上游通过标记。`sample`/`verified` 用于评测，`full` 用于全量审计。
`--max-tasks 20` 可缩小试验规模；`--resource-policy preserve` 可保留旧配置供审计。

## 本地验证

- 完整 CPU suite：**376 passed，19 skipped**。18 项缺少 torch；1 项在数据处理
  `.venv` 中缺少 Harbor。实际 Harbor 加载检查使用独立的 `AgentGen` 环境。
- 无 pytest 的独立 runner：新增 11 项测试通过。
- 当前 Harbor `Task` 解析：**604/604 训练候选、20/20 官方样本通过**，记录在
  `data/terminalworld/harbor_task_parse_v1.json`。
- 使用全新缓存执行 `--download --max-tasks 1`：公开下载、哈希校验及转换通过。

```bash
.venv/bin/python -m pytest tests/ -q -rs
.venv/bin/python tests/run_tests.py test_terminalworld_taskset
```

实际执行也完成了一个官方任务的对照测试：

| 任务 | agent | reward |
|---|---|---|
| `tw_117921`，前缀表达式计算器 | `nop`，不执行任务操作 | **0.0** |
| 同一任务，全新容器 | `oracle`，执行发布的参考解 | **1.0** |

结果保存在 `data/terminalworld/smoke-official-slirp/outcomes.jsonl`，完整 Harbor
日志与 CTRF 记录在同目录的 `jobs/`。这只验证了一个任务，不代表 604 个候选均可执行。
此前的两轮、共四次尝试遇到 BuildKit 镜像导入连接中断或 Podman CNI 网络错误，
保留为 `harness_infra`、reward=null，没有计入上述有效评分。

成功运行使用本机 rootless Podman 的传统构建方式和试验专用网络覆盖文件；未修改
系统网络配置。当前环境不执行每任务 CPU 配额，因此结果不用于性能比较。

```bash
# 写入试验专用配置；已有文件时直接复用。
printf 'services:\n  main:\n    network_mode: slirp4netns\n' \
  > data/terminalworld/podman-smoke-compose.yaml

DOCKER_BUILDKIT=0 COMPOSE_BAKE=false .venv/bin/python scripts/23_probe_sandbox.py \
  --out data/terminalworld/smoke-official-slirp \
  --docker-host unix:///run/user/1004/podman/podman.sock \
  --harbor-env-kwarg cpu_enforcement_policy=ignore \
  --harbor-env-kwarg 'extra_docker_compose=["/home/lys/Desktop/RST-Train/data/terminalworld/podman-smoke-compose.yaml"]' \
  --trial-timeout 300 smoke \
  --task data/terminalworld/official-sample-v1/tasks/tw_117921 --arms nop,oracle_builtin
```

以上 socket 和覆盖文件路径对应本机；其他机器需改为自己的 rootless runtime 地址。
复跑时换一个 `--out`，以保留本次运行证据。

## 从任务池到 SFT

下一阶段先抽取少量候选，用目标模型执行并保留原始轨迹，依据真实 reward 及基础设施
失败情况筛选。采集时使用 `06_eval.py --export-trajectories`，或 RL adapter 的
`RST_EXPORT_TRAJECTORIES=1`，以保留 `raw_content` 与 `linear_history`。
拿到实际成功轨迹后运行：

```bash
.venv/bin/python scripts/03h_build_rollout_sft.py \
  --jobs-dir data/terminalworld/model-rollouts \
  --tokenizer data/Qwen3.5-27B-tokenizer --out-dir data/terminalworld/sft-v1
.venv/bin/python scripts/15_export_pretokenized.py \
  --parquet data/terminalworld/sft-v1/rollout_sft_train.parquet \
  --tokenizer data/Qwen3.5-27B-tokenizer \
  --out data/terminalworld/sft-v1/pretokenized_train.parquet --strict
```

以上是后续命令，尚未执行模型 rollout、SFT 或 GPU 训练。参考解执行用于检查环境和
验证器，不能替代模型对话中的 reasoning、动作和终端观察。
