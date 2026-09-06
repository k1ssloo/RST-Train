# RSI 数据生产环境：生产就绪审查与升级提案

日期：2026-09-03　　审查对象：`RST-Train`（`6d32024` + 本轮未提交改动）、`TerminalEvo`（`827cf62`）
判据：每条结论都写明是**实测**、**读源码**还是**推断**；三者不混用。

---

## 0. 结论

**作为数据管线，可以进生产；作为训练闭环，不能——因为它还没有证明过自己。**

- 数据侧的可复现性、门禁、测试是同类项目里少见的严格：三份数据集用新代码重建后与线上发布版**逐字节一致**；333 条测试不需要 GPU/集群/数据集；每个 launcher 都是"先证明配置可跑，再启动 32 个进程"。
- 但整条链路的核心交付物——**"训练后的模型解出了更多任务"**——至今没有一个数字。11 个本地 checkpoint 里，7 个有离线 NLL 评测，**0 个有 agentic pass rate**。离线指标自己都写着 `substitutes_for_benchmark: False`。
- RSI 环路在代码层已闭合，但**一次带 `--export-trajectories` 的真实 rollout 都还没跑过**；TerminalEvo 那一侧目前是 GPT-5.5 在解题、Qwen 在学，严格说是**蒸馏**而不是自我改进。

所以生产就绪的缺口不在代码，在**测量**：一块空闲 GPU 就能补上大半。

---

## 1. 生产需求逐项审查

| 维度 | 判定 | 依据 |
|---|---|---|
| 可复现性 | ✅ | RST/OTA/TMax 三份数据集重建后 parquet 逐字节/逐列一致（实测）；每个产物带 `manifest.json`，seed 固定 |
| 正确性门禁 | ✅ | launcher 在启动前检查 fused CE、transformers 窗口、flash_attn、FLA 内核、rendezvous、静态显存；`resume_guard` 拒绝 lr 曲线被重置的续训；333 passed / 18 skipped（实测） |
| **核心交付物已验证** | ❌ | 7 个 checkpoint 的离线 NLL 都下降（−0.06 ~ −0.19，top-1 +1.2 ~ +3.8 pt），但**无一有 pass rate**；27B 在 4×8 的重跑（BUG-1 修复后）未执行；GRPO 从未执行；RSI 从未真跑 |
| 评测可用性 | ⚠️ | rootless docker 链路已通（`probe2` 正确分类）；但本机 H100 的 55.9 GB 被他人作业占用，4B 都起不来带 KV cache 的服务；tb-hard 任务集拿不到；cgroup `cpu` 未委派 → 用 `cpu_enforcement_policy=ignore`，结果**不能与论文严格可比** |
| 数据质量控制 | ⚠️ | 硬门（verifier）✅；软门（`03g`）新建，阈值手设，判官只用 0.8B 校准了 6 行；见 §2 的质量画像 |
| 评测污染防护 | ⚠️ | 本次实测：tb2 的 89 条任务与三个任务池共 11,119 条 prompt **0 条精确重合、0 对 Jaccard≥0.5**——结果干净，但**检查不在代码里**，下次换池子没人会再查 |
| 数据溯源 | ⚠️ | 计数全有；但 `03h` 只记录 `model_name` 字符串，不记录产生数据的 checkpoint 指纹（`dpo_common.checkpoint_fingerprint` 早就存在却没接）；已发布的 cap10（10,778 行）与当前代码重建（10,976 行）**不一致**，发布版落后于代码 |
| 工程基础设施 | ❌ | 无 CI（没有 `.github/`）、无 lint 配置、本地环境无依赖锁；`OPERATOR_PROMPT.md` 仍写 "252 tests"；`notes/RUN_LOG.md` 与 `DEVIATIONS.md` 是 0 字节空文件；仓库根目录三张未跟踪 PNG；`.venv-sglang/` 未进 `.gitignore` |
| 密钥与隔离 | ⚠️ | 仓库内所有密钥来自环境变量，跟踪文件里无密钥字面量（实测 grep）；不可信任务 Dockerfile 只在 rootless daemon 构建 ✅。但本机 `~/Desktop/config.toml` 里有一枚**明文** Azure OpenAI 密钥——不在仓库里，属于环境卫生问题，建议移入 keychain/env |
| 任务生成侧（TerminalEvo） | ⚠️ | 18 个通过 admission 的子任务 / 36 条 golden episodes；规模审计 **18 / 10,000 故意失败**；blind solver 是 GPT-5.5（Azure），Azure Responses 不返回 token id/logprob → episodes 全部 `online_rl_ready: false` |

### 数据质量画像（实测，`03g` 在 30,536 行上的特征统计）

| 语料 | 轮数 p50/p90 | 重复命令 ≥25% | 报错观测 ≥35% | **宣告完成后又继续** | harness 抱怨 |
|---|---|---|---|---|---|
| RST release（论文数据） | 11 / 19 | 8.0 % | 6.1 % | **27.7 %** | 8.1 % |
| OpenThoughts-Agent | 7 / 13 | 6.7 % | 4.2 % | 1.0 % | 2.1 % |
| TMax | 11 / 24 | 0.5 % | 0.6 % | 0.0 % | 0.0 % |
| Nemotron（holdout） | 7 / 12 | 11.5 % | 11.0 % | 13.7 % | 10.2 % |

两条值得单独指出：

1. **论文自己的数据是四份里"反悔率"最高的**——27.7 % 的成功轨迹至少一次宣告完成、被 harness 否定、再继续干。这些都是 reward=1 的"成功"。用它做 SFT，模型学到的第一个习惯就是"先说做完了再看"。
2. RST 有 58.1 % 的 assistant turn 经过了规范化重写（fenced JSON、前导 prose）。这个比例本身不是问题（normalizer 是确定性的），但它说明原始策略的**格式纪律很差**，而这部分噪声已经被我们清掉了——下一轮自产数据应当明显低于这个值，这是一个可以盯的健康指标。

---

## 2. 进生产之前必须关掉的缺口（按阻塞程度）

**P0 — 没有这三个数字，任何"效果"陈述都是空话**

1. 任意一个已训练 checkpoint 的 **agentic pass rate**（tb2 即可，89 题）。需要一块空卡；`06_eval.py` 已能在共享卡上起服务（`--mem-fraction-static`、`--max-running-requests`），差的只是显存。
2. **27B 在 4×8 的重跑**。BUG-1（FSDP2 未分片 fp32 梯度，103.5 GiB/GPU）的修复只在单卡上测量过，`[rst-fsdp2]` 是否在 32 个 rank 上都出现过，无人见过。
3. **一轮带 `--export-trajectories` 的真实 RSI**。`03h` 的 raw 路径只在合成 ATIF 上测过；本机唯一真实轨迹是渲染过的。

**P1 — 生产纪律**

4. CI：一个 GitHub Actions 跑 `pytest` + `ruff check --select F,E9`，PR 门禁。今天这两项全靠人手跑。
5. **评测污染门**进代码：任务池构建时对 tb2 / tb-hard 的 instruction 做精确哈希 + token Jaccard 检查，命中即拒绝入池并写进 manifest（本次实测为 0，但要变成机制）。
6. **数据溯源**：`03h` 与 `21_rsi_round.sh` 把 `checkpoint_fingerprint(CKPT)` 写进 manifest；每次训练在 run 目录写 `datasets.lock`（消费的每个 parquet 的 sha256、curation manifest hash、判官模型名）。"这个模型是拿什么训的"必须一年后还能回答。
7. **重新发布 cap10**：当前 Hub 上的 10,778 行是旧 normalizer 的产物，代码已经能多恢复 198 行。要么发 `cap10-v2` 并在 README 说明差异，要么把代码钉回去——**不能让发布版和代码各说各话**。

**P2 — 卫生**

8. `OPERATOR_PROMPT.md` 的测试计数；两个空的 notes 文件要么写要么删；根目录 PNG；`.venv-sglang/` 进 `.gitignore`。

---

## 3. 针对 RSI 的升级提案：目标是**更高质量**的数据，不是更多

下面每条都写清机制、为什么它提高质量、成本、落在哪个脚本、怎么验证。按"质量收益 ÷ 成本"排序。**全部是提案，未实现。**

### 3.1 分歧点偏好对（Divergence-point DPO）

**机制。** `03h` 已经把每轮所有 rollout（含失败）连 reward 写进 `rollouts.jsonl`。同一任务、同一 prompt、`--runs ≥ 2` 的两条轨迹，逐 turn 比较 `command_signature`：它们共享一段相同的命令前缀，在第 *t* 个 turn 分叉，一条最终 reward=1、另一条 0。把 **(公共前缀 + 成功分支的第 t 个 turn) vs (公共前缀 + 失败分支的第 t 个 turn)** 作为偏好对，loss mask 只覆盖第 *t* 个 turn。

**为什么更好。** 现有 DPO（`17_build_dpo_data.py`）比较的是两条**完整**轨迹，信号被摊在几千个 token 上，`DPO_PLAN.md` 自己担心的长度偏差就来自这里。分歧点对把 credit 收缩到**那一个决定**：上下文逐 token 相同，唯一的差别就是那一步选了什么。这是 on-policy 的、无长度偏差的、可解释的偏好信号——而且**零额外 rollout 成本**，全是已经付过钱的数据。

**成本。** CPU；只需 `21_rsi_round.sh --runs ≥ 2`（默认 1，建议改 4）。
**落点。** `17_build_dpo_data.py --rollouts rollouts.jsonl --pair-at-divergence`。
**验证。** 配对数量、分叉点分布（若 90 % 在第 1 turn 分叉，说明任务本身歧义大，是任务质量问题不是策略问题——这个副产品本身就有价值）。

### 3.2 Verifier 进度曲线：不用 LLM 的过程信号（Verifier-certified truncation）

**机制。** 对边界带里的成功轨迹，在新容器里**重放**它的命令序列，在第 25/50/75/100 % 的 turn 之后各跑一次任务测试。得到"任务是在第几步真正解出的"。若在第 *k* 步已通过、之后还有 *m* 个 turn，那 *m* 个 turn 就是 thrashing——**在第 k 步截断**，截断点由 verifier 认证，不是判官猜的。

**为什么更好。** 这是本仓库"绝不发明文本"原则下唯一合法的**编辑**：只删后缀，不改任何字；正确性由任务自己的测试保证。它把 27.7 % 的"宣告完成后又继续"从**丢弃**变成**修复**，把长而杂的成功变成短而干净的示范。同一份进度曲线还是 GRPO 的**稠密奖励**（partial credit）——这是 RL 侧最缺的东西。

**成本。** 每条边界带轨迹 4 次 verifier 运行。以 RST 边界带 3,851 条计约 15k 次容器运行；32 并发、每次 ~1 min → ~8 小时，一次性。可先只做"有反悔"的 27.7 %。
**落点。** 新脚本 `03i_replay_truncate.py`（读 `03g` 的 `curation.parquet`，只处理 `borderline`），重放用 Harbor 的 `oracle` 路径或直接 `docker exec`。
**风险。** 测试非幂等（依赖最终状态）——但截断点是"测试**在此已通过**"，所以按定义安全。**未验证：** 需要一批 tb2 任务实测重放稳定性。

### 3.3 级联判官 + 校准集（Cascade judge）

**机制。** `rst_common/judge.py` 支持一个配置；改成两个：本地小模型（sglang，零成本）先审边界带，只把**低置信或与启发式相反**的判决送 API 强模型。两位判官的一致率（Cohen's κ）写进 manifest。另外在仓库里固定一份 **200 行校准集**（强模型标注 + 人工抽查），`03g --calibrate` 先报一致率，低于阈值自动降级为 `NullJudge`。

**为什么更好。** 本次 0.8B 判官 40 次调用有 9 次吐不出合法 JSON、只校准了 6 行——这证明的是管路，不是判决。没有校准集，判官的"更高质量"是一个无法反驳也无法证实的陈述。级联把 API 成本再压 3–5×，κ 让"判官今天还可信吗"变成一个每轮都能看的数字。

**成本。** 校准集一次性 200 次强模型调用（几美元）+ 人工 1 小时。
**落点。** `judge.py` 增加 `CascadeJudge`；`tests/fixtures/curation_calibration.jsonl`；`03g --calibrate`。

### 3.4 用自己的 pass rate 重新分层（Own-pass-rate curriculum）

**机制。** `10_build_rl_taskset.py` 的 tier 来自**其他策略**的历史 pass rate，结果是 37,484 个任务里 **28,059 个（75 %）是 `unknown`——从未被采样过**。RSI 第 N 轮的 `rollouts.jsonl` 给出的是**当前策略**在每个任务上的 pass rate：p=1 → 退役（饱和）；0<p<1 → sweet（同时有 SFT 成功样本和 3.1 的偏好对）；p=0 → hard 队列。第 N+1 轮从重新分层后的池子采样。

**为什么更好。** 拒绝采样只能强化"策略有时会做对"的事，所以采样预算必须花在 0<p<1 的任务上，而这个集合**每轮都在移动**——静态 tier 一轮之后就过时。同时把 75 % 从未探索的池子逐步开垦，是数据**覆盖面**最大的杠杆。

**成本。** CPU。
**落点。** 新脚本 `10d_retier_from_rollouts.py`：`rollouts.jsonl` → 新的 `rl_tasks.jsonl`（`tier`、`empirical_pass_rate` 标注来源为 `self`）；`21_rsi_round.sh --tasks` 指向它。
**验证。** 每轮的 tier 迁移矩阵（多少任务从 unknown → sweet、sweet → 退役）写进 `rsi_round.json`。

### 3.5 让 Qwen 成为 TerminalEvo 的 blind solver（真正的 RSI）

**机制。** TerminalEvo 的核心思想是**行为条件化的任务演化**：观察 solver 的 transcript，诊断行为，据此打补丁生成子任务。今天这个 solver 是 GPT-5.5——所以演化出来的任务瞄准的是 GPT-5.5 的行为，Qwen 学的是 GPT-5.5 的示范。把 `06_eval.py` 起的 sglang 端点（OpenAI 兼容）配成 TerminalEvo 的 model provider，让**当前 Qwen checkpoint** 做 blind solver。

**为什么更好。** 这才是递归：子任务针对**我们自己策略的失败模式**生成，等于一条自动化的失败驱动课程；本地 sglang 还能返回 logprob 和 token id，`online_rl_ready` 从 0 变成全部。它同时解决 TerminalEvo 的规模瓶颈——18 个通过任务的低接受率部分来自 GPT-5.5 在这些任务上"太强"，很多子任务过不了 solver-reachability 之外的 behavior-kill 门。

**成本。** 一块能 serve 9B 的卡 + TerminalEvo 一个 provider 配置。
**落点。** TerminalEvo `runner/harbor.py` 的模型参数；RST-Train 侧无需改动。
**风险。** Qwen 4B/9B 的 pass rate 可能过低导致 admission 通不过（需要 ≥1/3 blind 成功）；先在 3.4 筛出的 sweet 任务上试。

### 3.6 任务侧 LLM 质检：instruction–verifier 一致性

**机制。** 判官在**任务**侧的正确落点：给它 `instruction.md` 和 `tests/test.sh`，问三件事——测试检查的是不是指令要求的东西；指令是否无歧义；指令是否泄漏解法。每个任务一次调用，永久缓存。不一致的任务在花任何 rollout 之前剔除。

**为什么更好。** 一个测试与指令不一致的任务，reward 是噪声——模型按指令做对了却拿 0，或做错了却拿 1。这种噪声进了 3.1 的偏好对就是毒。termigen 的 3,541 个任务和 TerminalEvo 的每个子任务今天都没有这层检查（TerminalEvo 的 admission 检查因果有效性和可达性，不检查语义一致性）。

**成本。** 每任务 ~1.5k token，一次性；termigen 全池约 5M token。
**落点。** `10e_judge_taskpool.py`，或并入 TerminalEvo `quality.py` 的 pre-sandbox 门。

### 3.7 压缩段策略：只训模型真正写过开头的对话

**机制。** `03h --on-compaction split` 会把上下文压缩后的每一段都当作独立样本，但第 2 段起的第一个 user turn 是 Terminus-2 生成的**摘要**——模型从未见过那种开头的真实任务。建议默认只保留第 0 段进 SFT，其余段单独存为 `continuation` 配置，做一次消融。

**为什么更好。** 这是纯数据质量决定：训练分布与服务分布对齐。本机那条 probe 轨迹 28 次压缩切出 24 段，说明这在长任务上不是边角情况。
**成本。** 一行默认值 + 一次消融。

### 3.8 每轮健康指标：把"坍缩"变成可报警的数字

**机制。** `rsi_round.json` 目前记 pass rate、band、保留行数。补三项：命令类型直方图对种子语料的 KL；每任务去重后的解法数（`command_signature` 粒度今天恒为 1，需要更粗的"方法"签名，如去参数后的命令名序列）；`n_rewritten_turns` 占比（§1 的 58 % 是基线，自产数据应更低）。任一项越界，脚本退出码 4，下一轮不允许自动启动。

**为什么更好。** 自训练坍缩不会以 loss 上升的形式出现，它以"越来越像自己"的形式出现。今天只有 mix-back 这一道防线，而防线需要探测器。
**成本。** CPU，几十行。

---

## 4. 建议顺序

| 周 | 做什么 | 产出 |
|---|---|---|
| 1 | P0-1、P0-3：一块空卡，跑 4B 的 tb2 pass rate；跑一轮 `21_rsi_round.sh --runs 4 --export-trajectories` | 第一个 pass rate；第一批 raw ATIF；`03h` raw 路径实测 |
| 1 | P1-4、P1-5、P1-6：CI、污染门、指纹 | 生产纪律到位 |
| 2 | 3.1（分歧点 DPO）+ 3.4（自适应分层） | 第二轮用自己的 pass rate 选任务，用自己的分叉造偏好 |
| 3 | 3.3（校准集 + 级联判官） | 判官从"管路通"变成"可信度可测" |
| 4 | 3.2（verifier 截断）在 27.7 % 反悔轨迹上试 | 把最大的一块质量问题从丢弃变修复 |
| 之后 | 3.5（Qwen 做 TerminalEvo solver）、3.6、3.7、3.8 | 真正的递归 |

---

## 5. 本文引用的实测与其来源

| 数字 | 来源 |
|---|---|
| 三份数据集重建逐字节一致 | 本机重建 + `cmp` / pandas 逐列比对（见 `WORKLOG.md` §6） |
| 333 passed / 18 skipped | `.venv/bin/python -m pytest tests/ -q` |
| 7 个 checkpoint 的 NLL delta | `data/eval/*/offline_results.json` 的 `delta_vs_base` |
| tb2 vs 任务池 0 重合 | 本次 ad hoc 脚本：sha256 精确匹配 + 小写 token 集 Jaccard，阈值 0.5 |
| 质量画像表 | `03g_curate_sft.py` 对四份语料的 `curation.parquet` |
| 28,059 `unknown` 任务 | `data/rl-sweet/manifest.json` 的 `tier_counts_all_tasks` |
| 65/70 步被渲染、24 段/28 次压缩 | `data/eval/probe2` 的 ATIF（`BUG.md` BUG-19） |
| TerminalEvo 18/36、solver=GPT-5.5、无 logprob | `TerminalEvo/GOLDEN_DATASET_REPORT.md`、`PROGRESS.md` |
