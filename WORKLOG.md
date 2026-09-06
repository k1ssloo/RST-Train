# 工作日志 · 代码结构审查与 RSI 数据范式优化

日期：2026-09-03　　基线 commit：`6d32024`　　执行环境：本机 1×H100（其中 55.9 GB 被他人作业占用）

本次三件事：**去冗余并重组代码**、**把 LLM 监督接到该接的地方**、**把自产数据的闭环补上**。
所有数字都是本机实测，不是估算；没跑过的一律标注为未验证。

---

## 一、结论摘要

| | 变更前 | 变更后 |
|---|---|---|
| 已有代码净变化 | — | 24 个 `.py`/`.sh`，+515 / −837 行（**净删 322 行**） |
| 新增共享模块 | — | 4 个（`siblings` / `sft_common` / `hf_publish` / `taskpool_common`） |
| 新增能力 | — | LLM 监督客户端、轨迹策展、rollout→SFT、RSI 单轮编排 |
| 单元测试 | 269 passed / 18 skipped | **333 passed / 18 skipped**（~1.6 s，仍然不需要 GPU、集群、容器、数据集） |
| `ruff check --select F,E9` | 10 errors | **0** |

关键验证：重构后重新构建三份数据集，**parquet 与线上发布版逐字节/逐内容一致**（见 §5）。

---

## 二、去冗余：五处真重复，一处假重复

判断标准只有一条：**两份拷贝漂移了会不会让两个数字不可比**。会，就合并；不会，就留着并写清楚为什么。

### 2.1 `spec_from_file_location` 加载器 ×5 → `scripts/siblings.py`

`scripts/` 不是 package 且文件名以数字开头，所以 `03d` / `03e` / `03f` / `17` / `06b` 各自手写了一份按路径加载兄弟脚本的代码——五份，五个不同的模块名，三种不同的缓存策略，两种关于要不要写 `sys.modules` 的答案。

这个加载器正是本仓库"只保留一份 `normalize_assistant`、一份 `qwen3_5_mask`"的实现手段，它自己有五份显然说不过去。现在统一为 `load_script(stem)`：同一 stem 全局返回**同一个模块对象**，并注册进 `sys.modules`（`ProcessPoolExecutor` 把加载来的函数发给 worker 时需要能反序列化）。`tests/_util.py` 也改为调用它，于是测试和被测脚本拿到的是同一个对象。

### 2.2 SFT 构建器的公共尾部 → `scripts/sft_common.py`

四个构建器（RST / OpenThoughts / TMax / Nemotron）各自实现了去重、模板契约门、按组切分、token 统计。**它们已经开始漂移了，而且漂在没人看的地方**：manifest 里的 `p90`/`p99`，RST 用 `numpy.quantile`（线性插值），另外三个用 `statistics.quantiles`（exclusive 方法），同一批长度算出不同的分位数。

现在四个都走 `token_stats()`，统一到 numpy 的定义——也就是 `PLAN.md` 里引用的那些 p50/p90/p99 所依据的定义。

### 2.3 HF 发布机制 ×6 → `scripts/hf_publish.py`

六个 `13*_upload_*_hf.py` 里，`check_card` 的函数体 **md5 完全相同**，create-repo / 上传文件 / 上传 README 的序列也只差字面量。真正不同的是数据卡（CARD）、卡上声明的数字（CLAIMS）和文件清单——那三样留在各自脚本里，因为那就是脚本本身。

顺手补了一个真实的加固：**所有输入文件在创建任何 Hub repo 之前统一检查**，一个笔误不会再留下一个半成品的公开仓库。

### 2.4 RL 任务池的公共约定 → `scripts/taskpool_common.py`

三个任务池（RST / termigen / SWE-Gym）各自定义了 `TIERS`。"sweet"在一个项目里必须只有一个含义，否则 GRPO 启动器按 tier 分配预算就没有意义。现在三者共享**同一个元组对象**（测试断言的是 `is`，不是 `==`：拷贝会漂移，同一个对象不会）。

verifier 泄漏检查此前有两份实现、两种输入形状（磁盘上的目录 vs 内存里的 bytes 字典）。现在**决策逻辑只有一份**，两种形状都归一成 `(路径, sha256)` 序列喂进去。同时修正了一个顺序缺陷：原实现在 name-only 命中后仍继续扫描，但 byte-identical 命中才 break——现在无论文件顺序如何，byte-identical 都胜出。

### 2.5 harbor 命令行 ×3 → `rst_common.harbor.run_argv`

`06_eval.py`、`rl/generate.py`、`verl_backend/harbor_agent_loop.py` 各自拼装 `harbor run`，`--n-attempts 1 --n-concurrent 1 --max-retries 0` 这套"一次尝试"契约写了三遍。合并后，§4 那个新开关只需要加在一个地方。

### 2.6 假重复：`command_signature` 的三份**不合并**

它在 `03`/`03e`/`03f` 各有一份，看着像该合并，其实不该：RST 版读 JSON 里的 `commands`；TMax 版读原生 `tool_calls` 的参数（TMax 根本没有 `commands` 这个键）；Nemotron 版要先切掉 `<think>` 前缀才能 parse。三者作用在**不同管线阶段的不同数据形状**上，强行合并只会造出一个到处是 if 的函数。已在代码注释里写明这个判断。

同理保留：`verl_backend` 两个模块各自的 `apply()`（两个不同的 patch，只是名字撞了）。

### 2.7 顺带修掉的死代码

`16_smoke_forward_backward.py` 加载了 tokenizer 却从不使用。删掉是一种做法，但更值的是把它变成一个真检查：**配置声明的 LM head 词表大小 vs tokenizer 实际大小**。Qwen3.5 的 head 会 padding 到比 tokenizer 大（248,320 vs 246,400 余），所以 `head >= tokenizer` 正常，`head < tokenizer` 致命——数据里的 token id 会越界，而失败现场是 loss 里的 device-side assert，完全不会提到"词表"两个字。

---

## 三、发现并修复的缺陷：BUG-19（自产数据闭环是断的）

这是本次最重要的发现，已写入 `BUG.md` BUG-19。

**现象**：仓库能 serve checkpoint、驱动 Harbor、用任务自带 verifier 打分——然后把 job 目录删掉。**没有任何代码把通过 verifier 的自家 rollout 变回训练数据**，所谓"递归自我改进"到评测表就断了。

**更糟的是**：就算有人天真地把它接上，数据也是错的。Terminus-2 2.0.0（Harbor 0.21.0）默认把 ATIF 里 agent step 的 `message` 写成**它自己的渲染**：

```
Analysis: The terminal is at /app directory. I need to ...
Plan: Check if R is available, then ...
```

命令被挪进 `tool_calls`，`task_complete` 变成一个 `mark_task_complete` 调用。模型真正输出的那段 `{analysis, plan, commands}` JSON **不在文件里**——只有 parse 失败的那些 turn 才保留原文。

**本机实测**（`data/eval/probe2/.../agent/trajectory.json`）：70 个 agent step 中 **65 个是渲染过的，5 个是原文**，整个文件里 raw JSON 出现 0 次。

**根因**（读 `harbor/agents/terminus_2/terminus_2.py` 得到，不是猜的）：形状由 `trajectory_config` 这个 agent kwarg 决定。`raw_content: true` 存 `llm_response.content`（Harbor 自己的注释就写着 "Useful for SFT data export"）；`linear_history: true` 在每次上下文压缩处切分文件，使每个文件恰好是模型真实看到过的历史。**两个默认都是关的。**

**修复**：
- `rst_common.harbor.export_agent_kwargs()` 是这个 kwarg 的唯一定义；
- `06_eval.py --export-trajectories` 转发它（并强制 `--keep-jobs`；若 harbor 构建不支持 `--agent-kwarg` 则**直接拒绝启动**，而不是几小时后在 03h 才失败）；
- RL 两条路径用 `RST_EXPORT_TRAJECTORIES=1`；
- 是否导出、导出了什么，写进 `results.json` 的 `protocol.trajectory_export`。

---

## 四、LLM 监督：接在哪里，以及**故意不接**在哪里

用户的要求是"有些地方可以接 API 做 LLM 监督，有些地方不能，因为会拖慢数据生产"。我把这条判断固化进了 `rst_common/judge.py` 的模块文档，因为它是个会被后人反复挑战的设计决定。

### 4.1 绝不接的地方（这是硬边界）

**verifier 就是 reward。** 本仓库报告的每一个数字——pass rate、DPO 隐式奖励、GRPO advantage——都来自任务自带的测试。LLM 的意见一旦渗进去，跑出来的结果就既不能和论文比、也不能和自己的上一轮比。因此判官**绝不**出现在：

- rollout 内部（reward 属于 verifier；而且每个 turn 一次 HTTP 往返会直接压在每个 sandbox 的热路径上）；
- loss mask、normalizer、tokenizer 契约（这些是确定性的，且 mask 的两份实现已经互相校验）；
- 评测打分。

### 4.2 接的地方：只在**边界带**，且离开关键路径

判官只在"verifier 必要但不充分"的地方出现，且**只看模糊的那一段**：

`scripts/03g_curate_sft.py` 分两级：

1. **确定性启发式**跑全量。重复命令、报错观测占比、harness 抱怨、完成声明反悔次数、相对同组兄弟的轮数 z 分数。每条轨迹毫秒级、零网络，直接判掉清晰的两端（`clear_keep` / `clear_drop`）。
2. **LLM 判官只看 `borderline`**，外加一小撮 `clear_keep` 随机抽样做**校准**。

**这就是速度论证的全部**——本机在 30,536 行真实数据上实测的分带：

| 分带 | 行数 | 占比 |
|---|---|---|
| clear_keep | 24,722 | 81.0 % |
| **borderline** | **5,493** | **18.0 %** |
| clear_drop | 321 | 1.1 % |

判官成本按 18 % 的边界带走，不按轮次规模走；而且**结果按内容哈希落盘缓存**，重跑一次策展是零调用。

### 4.3 客户端本身的契约（`rst_common/judge.py`，361 行，仅标准库）

- 任意 OpenAI 兼容 `/chat/completions`：OpenAI、Azure（`api-key` 头）、LiteLLM 代理，或**本机 sglang/vLLM**——在已经在 serve 模型的集群上，后者是零成本选项。
- 全部从环境变量配置，任何脚本都不需要长出 provider 参数。
- **没配 `RST_JUDGE_BASE_URL` 就返回 `NullJudge`**：所有消费方照常跑完，并在 manifest 里写明 `judge.enabled: false`——绝不假装边界带被审过。
- 预算耗尽时降级为**被计数的拒绝**（`budget_exhausted`），而不是抛异常把已经付过钱的工作丢掉。
- 响应必须是含指定 key 的 JSON 对象；不是就追加一次"只要 JSON"的重问，仍不是则记 `unparseable`。**两次往返算一个预算单位**，因为那是一个条目而不是两个判决。
- 失败**不写缓存**（否则一次坏响应会被永久固化）。

### 4.4 本机端到端实测（真的调了模型，不是 mock）

用本机剩余显存起了一个 sglang 服务（Qwen3.5-0.8B，`--mem-fraction-static 0.15`，因为整卡 66 % 被他人作业占着），对 Nemotron holdout 真实数据跑：

```
[bands] {'clear_keep': 260, 'borderline': 111, 'clear_drop': 30}
[judge] enabled=True borderline_sent=30 audit_sent=10
        calls=40 cache_hits=0 failures=9 budget_refusals=0
decisions: drop:heuristics 30 | drop:judge 2 | keep:judge 23 | keep:borderline_default 86 | keep:heuristics 260
calibration: audit_rows_judged 6 | judge_would_drop_clear_keep 0 | efficiency_mean_clear_keep 4.83/5
```

两点值得单独说：

- **校准是有信号的**：判官对 6 条被启发式判为 `clear_keep` 的样本，一条都不想丢，平均效率给 4.83/5。这正是"阈值是不是太松"这个问题该有的答法。
- **9 次失败**是 0.8B 模型吐不稳 JSON，属预期；重跑时它们**没有**被缓存，正确地重试了。

**缓存复跑验证**：第二次跑同样的输入，`cache_hits=31, calls=9`（9 次正是上次解析失败、未入缓存的那些）。

---

## 五、闭环：`03h` 与 `21_rsi_round.sh`

### 5.1 `scripts/03h_build_rollout_sft.py` —— 把自家 rollout 变回训练数据

输入是 Harbor job 目录树（`06_eval.py --out/jobs`、`RST_JOBS_ROOT`），或 TerminalEvo 的 `golden_episodes.jsonl`（每条 episode 内嵌 ATIF）。输出是**和其他所有 03* 构建器一模一样的 canonical `messages` parquet**，因为它走的是**同一个** `reconstruct_trajectory` / `normalize_assistant` / 去重 / 模板门。

三条设计决定：

- **reward 是门**：通过 `rst_common.harbor.read_reward` 读取，所以基础设施故障是"未测量"，永远不是"reward 0"。只有 `reward >= --min-reward` 的进 SFT parquet。
- **但失败也全部留下**：每条重建出的轨迹连同 reward 写进 `rollouts.jsonl`。这就是**on-policy 的成功/失败对池**——离线的发布数据集给不了这个，未来的 DPO 轮次可以直接在上面配对。
- **渲染过的轨迹默认拒绝**（BUG-19），计入 `drop_rendered_trajectory` 并打印修复指引。`--allow-rendered` 可以有损抢救，每个被重合成的 turn 都计入 `n_resynthesized_turns`。

**本机实测**：对三个 probe job 目录跑默认路径 → 正确拒绝（0 行，并打印"请用 `--export-trajectories` 重跑"）；加 `--allow-rendered --min-reward 0` → 从 1 条轨迹的 28 个压缩边界切出 **24 个线性段**，5 个 turn 因原文 parse 失败被丢弃，reward 正确读为 0.0。

**后三段串联实测**（用上面抢救出的 24 行真实 rollout 数据，跑 03h → 03g → 15）：

```
[bands]  {'borderline': 18, 'clear_keep': 6}      judge enabled=False → 启发式独立完成
[write]  rollout_sft_train_curated.parquet kept=24/24
[pretok] rows_out 24 | total_tokens 56,003 | trained_tokens 18,472 | trained_fraction 0.3298
```

最后那个数字是重点：**0.3298 落在 `30_run_sft_verl.sh` 门禁的 0.25–0.45 带内**，和发布语料的 0.3242 基本一致。也就是说自产数据经过这条链路后，训练启动器会直接接受它，不需要放宽任何阈值。

### 5.2 `scripts/21_rsi_round.sh` —— 一轮自我改进

```
1. 06_eval.py --export-trajectories    serve + rollout + 打分 + 保留 ATIF   （唯一需要沙箱的一步）
2. 03h_build_rollout_sft.py            通过 verifier 的 rollout → messages parquet
3. 03g_curate_sft.py                   反复横跳/重复/提前宣告完成 的过滤 + 边界带判官
4. 15_export_pretokenized.py           input_ids + loss_mask，直接喂 30_run_sft_verl.sh
```

只有第 1 步贵且需要沙箱；2–4 步是 CPU 且可重入，**换一组阈值或换一个判官重新策展不需要重跑任何 rollout**。

脚本里写死了两条防退化措施，因为拒绝采样自训练最典型的失败就是策略收窄：

- **`--mix-ratio`**：本轮数据是**混入**种子语料而不是替换它，默认 1:1，混合比例写进 `mix.json`。
- **`rsi_round.json`**：记录本轮 pass rate、各分带、判官统计、校准结果，并附一句必须读的话——**pass rate 没动、或保留行数塌了的一轮，就是什么都没教会**；池子饱和（pass rate 太高）或无望（接近 0）时该换 tier，而不是再跑一轮。

文档里也明确写了它**不是** GRPO：没有 advantage、没有 importance ratio、失败不产生梯度。它存在的理由是**完全不需要 trainer 侧的 rollout 管线**——一个 sglang 服务、一个 Harbor 循环、一个 parquet。在能跑容器但立不起 colocated actor 的 pod 上，这是唯一能闭环的路径。

---

## 六、重构的正确性验证：重建并逐字节比对

重构 SFT 构建器最大的风险是**悄悄改变已发布数据集**。所以三份都用重构后的代码在本机重建，与线上发布版比对：

| 数据集 | 结果 |
|---|---|
| `sft-v1-cap10`（RST，10,976 行） | **parquet 逐字节一致**（`cmp` 通过） |
| `openthoughts-agent-v1`（14,112 行） | 内容逐列一致 |
| `tmax-sft`（5,445 行） | 内容逐列一致 |

manifest 里只有两个字段移动，且是**有意为之**：`token_stats.p90` / `p99` 从 `statistics.quantiles` 的 exclusive 方法改为 numpy 的线性插值（§2.2）。例如 TMax 的 p90：13919.2 → 13908.0。RST 的 manifest 本来就用 numpy，数值不变。

另外新增 `command_signatures_shared_across_tasks` 字段（RST manifest 此前缺这个统计，另外三个有）。

---

## 七、新增测试（+64 条，全部无需 GPU/集群/数据集）

| 文件 | 钉住什么 |
|---|---|
| `test_siblings.py` | 测试助手与脚本拿到同一个模块对象；没有任何脚本还留着自己的 `spec_from_file_location` |
| `test_sft_common.py` | 共享的切分算法与**被它替换掉的内联实现逐一比对**（测试里重写了一份原始实现当 oracle）；分位数与 `numpy.quantile` 逐点吻合 |
| `test_taskpool_common.py` | 三个池共享**同一个** TIERS 对象（断言 `is`）；byte-identical 泄漏在任何文件顺序下都压过 name-only |
| `test_hf_publish.py` | 缺文件在创建任何 repo 之前就拒绝；六个上传脚本都不再自带 `check_card` / 直连 `HfApi` |
| `test_judge.py` | 先查缓存再走网络、预算降级不抛异常、一次 JSON-only 重问、两种鉴权头、失败不入缓存、429 重试而 401 不重试 |
| `test_curate_sft.py` | 两种 dialect 都能 parse；**连续两次完成声明是正常收尾而不是反悔**（Terminus-2 会要求复述）；轮数 z 分数需要足够多的兄弟样本 |
| `test_rollout_sft.py` | 渲染轨迹默认拒绝、抢救路径的重合成结果、压缩边界切分、基础设施故障不被当成 reward 0 |
| `test_harbor_invocation.py` | 三个 rollout 发起方都走 `run_argv`；harbor 不支持 `--agent-kwarg` 时导出**拒绝启动**而不是产出废数据 |
| `test_rsi_round.py` | 采样温度默认非 greedy（否则 `--runs>1` 毫无意义）；serving 模板覆盖被转发（0.8B 不覆盖会静默错服）；种子语料默认混回；空轮次以独立退出码停下而不是拿空数据去训练 |

这两条是被真实数据打脸后才加上的（先跑了一次全量策展，看结果不对再回来修）：

- TMax 的 `<tool_call>` turn 一开始全被记成 "unparseable"——策展器当时自带了一个只认 JSON 的 parser。改为复用 `06b_eval_offline.py` 的双 dialect parser。
- 每一条干净的 RST 成功轨迹都被判成"宣告完成后又继续"——因为 Terminus-2 会要求 agent 复述完成声明，**连续两次声明是正常结尾**。改为只统计"声明之后又出现非声明 turn"的反悔次数。

修正前后的对比很说明问题：`clear_keep` 从 **156 行涨到 24,722 行**，`borderline` 从 21,200 降到 5,493。改之前这个策展器等于把整个语料都推给判官——正是设计上要避免的那件事。

---

## 八、当前状态与未验证项

**已在本机实测**：全部单元测试、ruff、三份数据集重建比对、判官对真实数据的端到端评审（含缓存复跑）、`03h` 的拒绝路径与抢救路径、`06_eval.py --export-trajectories` 的参数装配与拒绝逻辑。

**未验证（需要资源）**：

| 项 | 缺什么 |
|---|---|
| 带 `--export-trajectories` 的真实 rollout | 整卡显存（现有 55.9 GB 被他人作业占用，4B 都起不来带 KV cache 的服务） |
| `21_rsi_round.sh` 端到端 | 同上，且需要 tb2 任务池 + 容器运行时 |
| `03h` 的 raw ATIF 路径 | 同上——目前只在合成 ATIF 上测过，本机唯一的真实轨迹是渲染过的 |
| 判官在强模型上的判决质量 | 本次只证明了**管路**通，0.8B 的判决质量不代表 GPT-4 class 的 |

`README.md` 的状态表沿用同样的 ✅/⚠️/⏳ 约定，我把新增行按这个标准填了。

---

## 九、遗留的两个已知问题（本次未动）

1. **`README.md` 和 `OPERATOR_PROMPT.md` 都写着 "252 tests"**，实际本次开始时是 269，现在是 326。已在 README 更新；`OPERATOR_PROMPT.md` 属于交给集群 LLM 的启动词，改动会影响正在跑的作业，留给你决定。
2. **`README.md` 的目录树缺 `03e`/`03f`/`10b`/`10c`/`13d`–`13g`/`resume_guard.py`**（上一轮工作新增但没进树）。本次一并补齐了，包括新增的 4 个共享模块和 3 个新脚本。
