# Terminal-Lego 数据接入与验证

现成模型轨迹的下载和 SFT 转换另见
[TERMINAL_LEGO_TRAJECTORIES.md](TERMINAL_LEGO_TRAJECTORIES.md)。下面记录的是任务池接入。

已将 Terminal-Lego 接入 RST-Train 的 Harbor 任务池：**6,797 个静态候选**，
另有 **3 个通过本地 nop/oracle 对照的任务**。本次处理的是任务环境、参考解和
验证器，没有生成模型轨迹或 SFT 样本。

## 来源与固定版本

| 来源 | 固定版本 | 用途 |
|---|---|---|
| [Lego-X/Terminal-Lego-15k](https://huggingface.co/datasets/Lego-X/Terminal-Lego-15k/tree/9c197f1c2e87b64cc316b1a5bfcef57b584929f0) | `9c197f1c…` | 官方任务文件及原始 Dockerfile |
| [PrimeIntellect/Terminal-Lego-15k](https://huggingface.co/datasets/PrimeIntellect/Terminal-Lego-15k/tree/92e6b5f577610cec9b040250a94ce66cfce24839) | `92e6b5f5…` | `prime-data-excluded-tasks.jsonl` 排除名单 |

转换器固定完整 commit 和排除文件 SHA-256。Prime 名单提供历史过滤线索；其
配置引用预构建镜像，不能证明用官方 Dockerfile 在本机重建也会成功。

## 筛选结果

```text
15,049 个官方任务
  − 1,224 个第三方排除 ID
  − 7,022 个缺失 Docker COPY 源目录的任务
  −     6 个命中保守文件名检查的任务
  = 6,797 个静态候选，6,797 个 StackOverflow 问题组
```

第三方排除原因包括 nop 通过、参考解失败、超时及构建问题。本地的六个文件名
命中均为 `name_only`：环境内有 `test.sh` 或 `solve.sh`，部分确实是题目要求
修改的文件，**不等于已证实泄漏验证器**；这里沿用仓库的保守排除规则。

运行 `task_00069` 时发现，Dockerfile 的 `COPY ./task_file /app/task_file`
引用了 Git 树中不存在的目录。全量复查发现 7,022 个同类任务，当前将它们排除
待审查；没有推测缺失附件或生成空目录来改变发布内容。最初的 `train-v1` 有
13,819 个候选，保留供审计；**当前候选池是 `train-v2`**。

转换期间校验了 109,508 个 Git blob；211 处 LFS 引用对应 189 个唯一对象，
合计 24,182 字节，输出中使用校验后的实际附件。HF `repo-info.siblings` 对该
大型仓库的列表不完整，因此任务枚举使用固定 commit 的完整 Git tree。

所有保留任务的原始文件、权限和 canary 注释均保留。旧式 `memory="1G"`、
`storage="5G"` 在当前 Harbor 下可解析，不需要 TerminalWorld 那样的别名修复。
类别包括 programming 1,706、system-administration 490、shell-scripting 279、
version-control 256、networking 131、containerization 45、web-server 165、
database 156、general 3,569。

## 重现转换

在 `RST-Train/` 下运行，`--out` 必须是新目录或空目录。复跑时换一个输出目录；
下载缓存可以复用，省略 `--download` 即为离线转换。

```bash
.venv/bin/python scripts/10e_build_terminal_lego_taskset.py \
  --download --out data/terminal-lego/train-v2

# 小规模复现：这三个 ID 已通过本机执行对照。
.venv/bin/python scripts/10e_build_terminal_lego_taskset.py \
  --task-ids task_00529 task_00705 task_05795 \
  --out data/terminal-lego/small-v1
```

`--max-tasks 20` 限制审计的候选数，不保证最终保留 20 个。下载器创建浅 Git
checkout，并禁用 LFS smudge；LFS 附件由转换器下载到校验缓存。手动准备源目录
时同样使用 `GIT_LFS_SKIP_SMUDGE=1`，不要改动固定版本的文件。

输出包括：

- `rl_tasks.jsonl`：`prompt`、命名空间化的 `label` 和 `metadata.task_dir`。
- `tasks/terminal_lego_task_XXXXX/`：完整 Harbor 任务目录。
- `source_audit.jsonl`：逐任务内容哈希、附件、过滤原因。
- `manifest.json`：来源版本、筛选计数、分类、完整性及验证范围。

`task_dir` 为本机绝对路径，迁移到其他机器时重新生成任务记录。按
`metadata.task_group_id` 分组切分，避免同一来源问题跨 train/holdout；尚未执行
跨数据集的语义去重，不能据此声明与 TerminalWorld、RST 或评测集完全独立。

## 本地执行结果

使用 Harbor 0.21.0 和 rootless Podman，五个指定任务各执行 nop 与官方 oracle：

| 任务 | nop reward | oracle reward | 结果 |
|---|---:|---:|---|
| `task_00000`，递归查找文件 | 0 | 0 | 参考解写入不存在的输出目录 |
| `task_00069`，Python subprocess | 未评分 | 未评分 | COPY 源目录缺失，已从 v2 排除 |
| `task_00529`，定位 Bash 脚本目录 | 0 | 1 | 参考解通过 11/11 项验证 |
| `task_00705`，读取 Git 分支 | 0 | 1 | 参考解通过 8/8 项验证 |
| `task_05795`，CSV 导入 SQLite | 0 | 1 | 参考解通过 13/13 项验证 |

共 10 次尝试，其中 8 次有有效评分；另外 2 次记录为 `harness_infra`、
reward=null，具体原因是缺失构建上下文。该分类不表示本机 daemon 宕机，也不把
未启动的 agent 计作任务失败。五个任务为定向 smoke 样本，不用于估算总体通过率。

`task_00000` 仍在静态候选池中，不能直接视为通过运行验证的训练任务。
`data/terminal-lego/controls-passed-v1.jsonl` 单独保存三个通过对照的任务，指向
v2 目录，并附带评分证据。它们在 v1 执行时的文件哈希与 v2 完全一致。

证据文件：

- `data/terminal-lego/runtime_controls_v1.json`：汇总、结果文件哈希及任务内容匹配。
- `data/terminal-lego/smoke-v1/outcomes.jsonl`：首个任务的失败对照。
- `data/terminal-lego/smoke-diverse-v1/outcomes.jsonl`：其余四个任务；各目录
  `jobs/` 中保留完整日志、CTRF 和参考解输出。

复跑成功任务可使用以下命令。覆盖文件
`data/terminal-lego/podman-smoke-compose.yaml` 的内容为：

```yaml
services:
  main:
    network_mode: slirp4netns
```

```bash
DOCKER_BUILDKIT=0 COMPOSE_BAKE=false .venv/bin/python scripts/23_probe_sandbox.py \
  --out data/terminal-lego/smoke-repeat-v1 \
  --docker-host unix:///run/user/1004/podman/podman.sock \
  --harbor-env-kwarg cpu_enforcement_policy=ignore \
  --harbor-env-kwarg 'extra_docker_compose=["/home/lys/Desktop/RST-Train/data/terminal-lego/podman-smoke-compose.yaml"]' \
  --trial-timeout 360 smoke \
  --task data/terminal-lego/train-v2/tasks/terminal_lego_task_00529 \
  --arms nop,oracle_builtin
```

socket 和覆盖文件路径对应本机，迁移时需替换。每次复跑使用新 `--out` 保留旧
日志。当前环境不执行每任务 CPU 配额，耗时不用于性能比较。

## 验证与后续使用

- 全部 **6,797/6,797** 任务通过 Harbor 解析、prompt 一致性及落盘文件哈希检查，
  报告为 `data/terminal-lego/harbor_task_parse_v2.json`。
- 新增 11 项回归测试，pytest 与独立 runner 均通过。
- 完整 CPU suite：**400 passed，19 skipped**；18 项缺 torch，1 项在数据处理
  `.venv` 中缺 Harbor。上述实际 Harbor 检查使用独立的 `AgentGen` 环境。

```bash
.venv/bin/python -m pytest tests/ -q -rs
.venv/bin/python tests/run_tests.py test_terminal_lego_taskset
```

候选的 `tier="unknown"`、`empirical_pass_rate=null`，三项控制成功也不代表当前
模型成功率。先用通过对照的小任务池采集真实模型轨迹，再通过
`03h_build_rollout_sft.py` 和 `15_export_pretokenized.py` 生成 SFT。扩大训练前
继续做任务运行、验证器质量和来源重叠检查；本次没有启动模型 rollout 或 GPU 训练。
