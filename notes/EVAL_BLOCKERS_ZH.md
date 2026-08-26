# 评测环境：已修复的问题，以及仍需申请的权限

本机：Ubuntu 22.04.5 / kernel 6.8.0-1046-nvidia / 1×H100 80GB / 账号 `lys`（uid 1004，属于 `sudo` 组，**不属于** `docker` 组）。

结论先说：**容器问题已经全部解决，不需要任何权限**。真正需要向管理员申请的只有一项，而且它不是阻塞项，只影响 benchmark 的严格可比性。

---

## 一、已解决：容器运行时（零权限）

之前的判断是「podman 3.4.4 有 bug、`lys` 不在 docker 组、所以本机跑不了 agentic 评测」。这个判断**不完整**。实际情况：

### 1.1 本机早就有一个属于 `lys` 的 rootless dockerd 在跑

```
lys  dockerd --data-root /home/lys/.cache/terminalevo-rootless-docker.zawPhG/data \
             --host unix:///home/lys/.cache/terminalevo-rootless-docker.zawPhG/docker.sock
```

实测：docker **29.2.1**、API **1.53**、`SecurityOptions` 含 `rootless`、存储驱动 `overlayfs`。它属于 `lys`，所以：

- 不需要加入 `docker` 组
- 不需要 `sudo`
- 不需要装任何第三方二进制

之前没发现它，是因为 `scripts/00b_setup_sandbox.sh` 只在 `DOCKER_HOST` **已经被设置** 的情况下才接受非默认 socket，它不会主动去**发现**一个用户自己的 rootless daemon。已修复：新增 `discover_owned_docker()`，依次检查 `DOCKER_HOST` → `$XDG_RUNTIME_DIR/docker.sock` → 从进程表里解析 `dockerd --host unix://...`（本机 socket 在 `~/.cache/` 下的非标准路径，只能从进程表拿到），并且仍然拒绝共享的 `/var/run/docker.sock`（那是 root 的 daemon，不该用来构建不可信的任务 Dockerfile）。

### 1.2 rootless docker 比 podman 好，不是退而求其次

podman 3.4.4 在本机被实测出**三个**互相矛盾的缺陷，任选其一都跑不通：

| 操作 | podman 3.4.4 | rootless docker 29.2.1 |
|---|---|---|
| buildkit 构建并导出镜像 | ❌ `Build result will only remain in the build cache`（就是之前看到的 "sending tarball / connection reset"） | ✅ `naming to docker.io/library/... done` |
| Dockerfile heredoc（约 31% 的任务镜像用到） | ❌ `Unknown instruction: "SET"` | ✅ 构建并运行成功 |
| 通过 socket `docker run` | ❌ `unable to upgrade to tcp, received 409` | ✅ |

也就是说 podman 3.4.4 **只能二选一：要么支持 heredoc，要么能导出镜像，永远不能同时**。而且 Ubuntu 22.04 的 apt 里只有 3.4.4（`apt-cache policy podman` 的 Candidate 就是 3.4.4），harbor 0.21.0 也**没有原生 podman 后端**（全库只有 `environments/base.py` 的一句 docstring 提到 podman）。所以 podman 这条路是死的，不必再修。

### 1.3 第二个坑：cgroup 没把 `cpu` 委派给用户

换到 rootless docker 后，镜像能拉、网络能建，但容器创建失败：

```
Error response from daemon: NanoCPUs can not be set, as your kernel does not
support CPU CFS scheduler or the cgroup is not mounted
```

这条报错**很有误导性**——它说的不是内核不支持。内核支持得很好：

```
/sys/fs/cgroup/cgroup.controllers                      -> cpuset cpu io memory hugetlb pids rdma misc
/sys/fs/cgroup/user.slice/user-1004.slice/cgroup.controllers -> memory pids       ← 只有这两个
```

任务在自己的 `task.toml` 里声明了 `cpus = 1`，harbor 把它翻译成 NanoCPUs；而 systemd 默认只把 `memory pids` 委派给用户 slice，**不委派 `cpu`**，所以 rootless daemon 无权设置 CPU 限额。

**已用零权限方式绕过**：harbor 的 environment 构造参数 `cpu_enforcement_policy` 支持 `ignore`，`00b_setup_sandbox.sh` 现在会自动检测「rootless 且 `cpu` 未委派」并追加 `--environment-kwarg cpu_enforcement_policy=ignore`，同时**大声打印后果**。

### 1.4 端到端验证通过（两次，逐步加码）

**第一次：不带模型，验证容器 + verifier。** 用 harbor 的 `oracle` agent（直接跑任务自带的参考解）跑真实 tb2 任务：

```
harbor run --path /tmp/tbtasks/terminal-bench/password-recovery \
  --agent oracle --env docker --environment-kwarg cpu_enforcement_policy=ignore
→ Reward 1.0    Total runtime: 54s
```

镜像拉取、网络创建、容器启动、参考解执行、verifier 判分，**全链路通**。

**第二次：带真实模型，验证完整 agentic 链路。** sglang 服务本机 merge 出来的 4B TMax checkpoint，harbor + terminus-2 驱动它跑 tb2 任务：

```
[serve] ready
The server is fired up and ready to roll!
[tb2] run 1/1: 1 tasks, concurrency 1
```

trial.log 里能看到 terminus-2 在真的干活：模型输出被解析成 JSON action、trajectory 不断 dump、subagent 在做 summarization。也就是说 **sglang 服务 → harbor → terminus-2 → 容器 → verifier 这条完整链路是通的**。

拿到**带分数的** tb2 结果还差一步，原因在 §3.1（GPU 被占，只能给 16k context，导致 agent 疯狂 summarize 而超时），那是资源问题，不是代码或权限问题。

### 1.5 为了让 sglang 起来，另外修了三处

这三个是叠在一起的，修掉一个才能看到下一个：

1. **`ninja` 不在 PATH 上**。sglang 启动时会 shell out 调 `ninja` 编译 JIT kernel，而 `ninja` 装在 venv 自己的 `bin/` 里，没有人 activate venv 所以子进程找不到。**最坑的是它死的位置**：在打印完 `Mamba Cache is allocated` 和 `max_total_num_tokens=424903` **之后**才死，所以日志最后一行看起来像启动成功了。已修：给 serve 子进程的 PATH 前置当前解释器的 `bin/`。

2. **`--mem-fraction-static` 写死 0.85**。它是相对「sglang 可用的显存」而不是整卡算的，所以在共享卡上 0.85 反而是对的；**往下调是错的**——调到 0.30 会被拒：`Loaded weights leave no GPU memory for the KV cache ... minimum viable = 1 - available/pre = 0.3366`。已改成可配置，默认仍是 0.85。

3. **`--max-running-requests` 没设，而报错信息不提它**。每个在途请求要预留自己的 gated-delta-net state（4B 实测 49 MiB/请求），sglang 按 CUDA graph 的 max batch（256）来定这个池子的大小 = **还没分 KV cache 就先吃掉约 12.6 GB**。报错是 `Not enough GPU memory for hybrid (mamba/linear-attention) state cache`，给了四条建议，第一条正是调这个、第二条是调 mem-fraction——很容易被引到错的那个旋钮上。已加成可配置，默认 0 = 不改 sglang 自己的行为。

另外验证到：**光 merge 出来的 checkpoint 不能直接 serve**。verl 只写 text tensor，而 config 声明的是 `Qwen3_5ForConditionalGeneration` 且带 `vision_config`，sglang 会因为缺 `preprocessor_config.json` 而失败。要用 `scripts/07_restore_vision.py` 补齐（实测：426 个 tensor 来自训练、312 个 vision/mtp 来自原始 checkpoint、0 个意外回退、0 个 shape 不匹配）。

---

## 二、唯一需要申请的权限（不阻塞，只影响可比性）

### 把 cgroup v2 的 `cpu` 控制器委派给用户 slice

**为什么要**：现在用 `cpu_enforcement_policy=ignore` 绕过，代价是**任务拿到整台机器的 CPU，而不是它声明的 `cpus = 1`**。这只会让 agent 更容易做完事情，所以任何用这套环境跑出来的 pass rate 都**不能直接和论文/官方榜单对比**，报告里必须写明这一点。要做严格可比的评测，就需要真正施加 CPU 限额。

**要什么**（root，一个文件，改完重新登录一次；对系统其它部分无影响）：

```ini
# /etc/systemd/system/user@.service.d/delegate.conf
[Service]
Delegate=cpu cpuset io memory pids
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart user@1004.service    # 或者让 lys 重新登录
```

**验证**：

```bash
grep -w cpu /sys/fs/cgroup/user.slice/user-1004.slice/cgroup.controllers
```

出现 `cpu` 就成了。之后 `00b_setup_sandbox.sh` 会自动**不再**加 `cpu_enforcement_policy=ignore`，任务按声明的额度运行。

**风险**：这是 systemd 官方文档推荐的 rootless 容器配置方式（Docker / Podman 的 rootless 文档都写了这一条）。它只是允许普通用户在**自己的** cgroup 子树里设置资源限额，不提升任何其它权限。

**明确不需要的东西**（请不要批这些，都不必要）：

- ❌ 把 `lys` 加进 `docker` 组 —— 不需要了，rootless daemon 已经够用；而且 docker 组等价于 root。
- ❌ `sudo` / `--privileged` / `SYS_ADMIN` / `/dev/fuse`
- ❌ 安装第三方 podman 静态二进制
- ❌ 升级 podman

---

## 三、其它不是权限问题、但会影响评测的事

### 3.1 GPU 被另一个用户占了 53.9 GB —— 这是目前**唯一**卡住「带分数的 agentic 结果」的东西

```
$ nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
53912 MiB   ← /home/lzh/miniconda3/envs/diffsynth/bin/python
              /home/lzh/DiffSynth-Studio/scripts/eval/.../eval_context.py
```

**属于用户 `lzh`，不是我们的进程，我没有动它。** 整卡 81.5 GB 被占掉 66%，只剩约 27.6 GB。

后果是一条因果链，值得写清楚：

1. 27.6 GB 里放下 4B 权重（8.8 GB）后，只能给到 **`--context-length 16384`**，而仓库默认是 65536。
2. terminus-2 在 16k 上下文里，可用余量只有约 7,500 token（日志：`Proactively summarizing. Free tokens: approximately 7492`）。
3. 于是它**不停地做 proactive summarization**——单个任务实测跑出 **76 个** summarization 产物，每一轮都要额外调用模型。
4. 结果单个任务远远超过默认的 900 s `--task-timeout`，被判为失败，而**失败原因和模型能力无关**。

所以：**目前拿不到可信的 tb2 pass rate，不是因为链路不通（链路已验证通，见 §1.4），而是因为显存不够导致上下文被砍到 1/4。**

需要你确认的事（不是权限问题，是资源协调）：

- `lzh` 那个 job 能不能停 / 什么时候结束？只要拿回整卡，4B 就能用 65536 上下文正常跑。
- **9B / 27B 的 agentic 评测在现状下完全不可能**：27B 光 bf16 权重就 56 GB，比整张卡剩下的空间还多。这两个尺寸需要一张空卡（或多卡）。
- 本机只有 1 张 H100（`nvidia-smi` 只有 index 0），没有别的卡可以挪。

### 3.2 tb-hard（100 个任务）拿不到

`harbor download terminal-bench` 能拿到 **89 个**任务（= 仓库里 `EXPECTED_TASK_COUNTS` 的 `tb2`），六件套完整（`instruction.md` / `environment/` / `tests/` / `solution/` / `task.toml`）。但 `terminal-bench-hard` 在 harbor 默认 registry 里 **不存在**（试过 `terminal-bench-hard` / `tbh` / `terminal_bench_hard` / `terminal-bench-core` 全部 `not found`）。

所以目前只能评 **tb2（89 题）**。tb-hard 需要另外的来源（官方 registry URL 或 `--repo`），这不是权限问题。lhtb 按仓库既有结论**不可评**（verifier 上游不发布）。

### 3.3 `.venv-gpu` 里的 transformers 是 5.15.0，训练会被拒

不影响评测（评测只用 transformers 自己做推理，5.15.0 自洽），但如果要在本机跑 **verl 训练**会被 `30_run_sft_verl.sh` 的门禁拦住：5.15.0 移除了 `Qwen3_5GatedDeltaNet.__init__` 里的 `self.chunk_gated_delta_rule`，而 verl 0.9.0 无条件读它。仓库要求 `transformers>=5.11,<5.15`。本机只做评测，所以先不动。

---

## 四、附：这次一并修掉的三个代码缺口

这些不涉及权限，已经修完并提交，写在这里是因为它们会让评测「看起来成功但其实什么都没测」：

1. **`06_eval.py` 从不传 `model_info`** → harbor 0.21.0 会在 agent 发出第一条命令之前就 `ValueError: hosted_vllm models require model_info`，每个任务都失败。而这类失败被正确归类为「harness 基础设施故障」并从 pass rate 分母里排除——于是整个 run「成功」，分母是 0。现在按 `--context-length` 推导出 `max_input_tokens` 后通过 `--agent-kwarg` 传入，并把是否传成功记进 `protocol.agent_model_info`。

2. **`06b_eval_offline.py` 的 action 探针只认一种格式** → 它走 `normalize_assistant`，要求 `{analysis, plan, commands}` JSON；而 TMax 这类语料用的是原生 `<tool_call><function=bash>` XML。实测 4B TMax checkpoint：**80/80** 个 turn 被记为 `reference_unparseable`，所有 action 指标都是 `None`。修完后同一个 checkpoint、同样 40 行：探到 76 个 action，0 个无法解析，解析率 96.1%。

3. **`task_complete_agreement` 恒为真** → 代码读 `is_task_complete`，但所有数据集写的都是 `task_complete`（Nemotron holdout 里 398/398）。两边都是 `None`，`None == None` 成立，于是这个指标对每一对都报 1.0。修完后同一个 checkpoint 报 84.2%。

---

*本文件由本次会话生成，所有数字都是在本机实测得到的，不是推测。*
