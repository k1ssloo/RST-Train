# SmolLM3 / OLMo3 / Gemma 4：单节点历史数据 SFT + DPO 执行指令

你是远端训练服务器上的执行 LLM。请在 `RST-Train` 完成以下模型的训练、checkpoint
导出、评估和结果归档，持续执行，不只提交计划或启动命令。一项失败或缺少访问权限时
记录具体原因并继续其余独立任务。

## 1. 必须遵守的训练范围

本轮“三个模型”固定为 **SmolLM3-3B、OLMo-3-7B-Instruct、Gemma-4-E2B-it**。
本地报告也验证了 Gemma E4B，本轮选择 E2B 规格。

七组历史主数据集各做三次独立 SFT，共 **21 次 SFT**。每次从该模型的官方 checkpoint
重新初始化，不跨数据集连续微调，不将七组混合成一个实验。延续 **SFT + DPO**：再做
**3 次 RST DPO**，各自从同模型的 `rst-cap10` SFT final 初始化，基本矩阵共 **24 项**。

先核对服务器原训练台账：若此前还有正式训练过的额外数据版本，按原配置补入三模型
矩阵，记录来源、划分和 epochs，不能漏做。下表七组是必做下限；额外数据的处理规则
见第 3 节。没有真实成功/失败配对的数据集只做 SFT，不伪造 DPO 标签。

**每次训练始终只使用一个节点**，包括 smoke、重试和 SFT 恢复；允许使用该节点内的
多张 GPU。不得因 OOM、速度或排队原因增加节点。所有正式训练每 **200 个
optimizer/global steps** 保存 checkpoint，并额外保存 final；不是每 200 个 micro-batch。
总步数不足 200 的任务保存 final 即可，不为产生 step-200 擅自增加 epochs。

本文件的模型、单节点和保存要求优先于旧的 `LLAMA_PHI_TRAINING_PROMPT.md`、
`TRAINING_PROMPT.md` 等文档中的多节点示例。先读 `MODEL_SUPPORT.md`、
`reports/cross_family_validation_20260910.md`、对应 JSON 和实际调用脚本。

## 2. 固定代码、模型与运行环境

检查 `git status`、分支和 origin，干净时 `git pull --ff-only`，记录代码 SHA。
有未提交修改或分叉时使用独立 checkout，保留现有实验现场。确认包含 BUG-30 的
`rst_common/model_precision.py` 和 `verl_backend/model_precision.py`。

| MODEL_KEY | 官方 HF repo | 固定 revision | MODEL_DIR_NAME |
|---|---|---|---|
| `smollm3-3b` | `HuggingFaceTB/SmolLM3-3B` | `a07cc9a04f16550a088caea529712d1d335b0ac1` | `SmolLM3-3B` |
| `olmo3-7b` | `allenai/Olmo-3-7B-Instruct` | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` | `Olmo-3-7B-Instruct` |
| `gemma4-e2b` | `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | `gemma-4-E2B-it` |

`BASE_FOLDER` 使用足够大的持久目录，如 `/shared/rst-cross-family`。以
`hf download <repo> --revision <sha> --local-dir "$BASE_FOLDER/<MODEL_DIR_NAME>"`
下载完整权重、config、tokenizer 和模板。Gemma E2B 约有 5.1B 总参数，不能按 2B
估算训练显存。使用服务器已有的安全认证，不把凭据写进命令参数、文档或日志。

使用独立的 Transformers/PyTorch CUDA/verl 环境，确认识别三种架构；保留已有 Qwen
和数据生产环境及进程。记录实际 Python、Torch、Transformers、verl、CUDA、GPU/驱动
版本。本地官方权重检查使用 Transformers 5.15.0，真实 verl 检查使用 verl 0.9.0；
这些版本信息不代表服务器任意依赖组合都能工作。

在实际训练环境对每个 `MODEL_PATH` 执行：

```bash
python scripts/14_prepare_tokenizer.py --model "$MODEL_PATH" --verify-verl \
  --report "$BASE_FOLDER/tokenizer-preparation/$MODEL_KEY.json"
```

若需同步 padding，在独立模型副本中用该模型词表中已有的 pad token 执行
`--apply --verify-verl`，记录备份和前后指纹。不能套用 Llama 的 pad ID、新增 token
或改指纹绕过检查；配置变化后重导受影响的数据。不要在任务运行时修改模型目录。

## 3. 所有历史数据的矩阵与来源

文件路径相对于 HF dataset repo；行数是目标模型重分词前的发布数量。

| DATA_KEY | HF dataset repo | messages 文件 | train / holdout | epochs |
|---|---|---|---:|---:|
| `rst-cap10` | `NiuNiu0110/RST-SFT-Qwen3.5-27B` | `data/cap10/{train,holdout}.parquet` | 10,578 / 200 | 1 |
| `ota` | `NiuNiu0110/OpenThoughts-Agent-v1-SFT-terminus` | `data/messages/{train,holdout}.parquet` | 14,112 / 200 | 1 |
| `nemo` | `NiuNiu0110/Nemotron-Terminal-SFT-terminus` | 合并 `data/train_skill_based_{easy,medium,mixed}.parquet`；`data/holdout.parquet` | 139,275 / 401 | 1 |
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

下载指定 revision 的 messages 与发布 metadata，验证 SHA-256、行数和任务划分。
Nemotron 的三个 skill 文件只合并一次，holdout 独立；不要再混入 `default` 或 adapters。
私有数据不可访问时记录 blocked，不能换一个同名数据集充数。

额外版本的处理规则：

- `rst-v1`、RST cap8、`nemo-adapters` 等若在服务器历史正式矩阵中出现，作为独立
  实验补入，保留原任务划分，不和 cap10/主版 Nemotron 重复拼接。
- `nemo-adapters` 有 13 条不同源轨迹含保留标记。使用**已提交**报告 JSON 的
  `input_exclusions`，先核对源 SHA-256，再按逐模型零起始行号、trajectory_id 和标记
  排除，保存过滤 manifest。不能盲套不匹配文件的行号或修改正文逃过检查。
- SWE-Gym 的 491 条文本 SFT 源可验证；原始 sampled rollouts 和原始 TMax 含工具
  字段，须先转换。TermiGen 只有提示，TerminalEvo v18 的 526 行仍是未发布审计池，
  均不能直接当正式 SFT 数据。历史台账只写名字时须查清实际版本和发布状态。
- DPO v1 若曾用于正式训练，也补做该版本；DPO smoke 仅用于检查。不同版本各自
  重编码、分组和打分，不把重叠偏好合并扩充数量。

报告中的 `outputs/` 是忽略的本地产物，不随 GitHub checkout 出现在服务器。
从固定源重建，不能引用不存在的本地绝对路径作为远端输入。

## 4. 按目标 tokenizer 重编码和审计

保留原始数据及 metadata，统一提供：

```text
$BASE_FOLDER/data/<DATA_KEY>/rst_sft_{train,holdout}.parquet
$BASE_FOLDER/data/<DATA_KEY>/source_metadata/
$BASE_FOLDER/data/<DATA_KEY>/<MODEL_KEY>/32k-v1/pretokenized_{train,holdout}.parquet
```

每个模型用自己的原生模板重新分词，不使用 Qwen 的 token IDs、loss masks 或 reference
分数。**MAX_SEQ_LEN=32768**，不沿用 registry 的 8192；8K 只保留主版 Nemotron
约 7%–11% 的完整轨迹。超长整行排除并报告，不截断轨迹，不悄悄改训练窗口。

```bash
export DATA_DIR="$BASE_FOLDER/data/$DATA_KEY"
export PRETOK_DIR="$DATA_DIR/$MODEL_KEY/32k-v1"
for split in train holdout; do
  python scripts/15_export_pretokenized.py \
    --parquet "$DATA_DIR/rst_sft_${split}.parquet" \
    --tokenizer "$MODEL_PATH" --loss-mask-type auto \
    --out "$PRETOK_DIR/pretokenized_${split}.parquet" \
    --max-seq-len 32768 --strict
done
```

审查每个导出 manifest：允许明确计数的 `too_long`；其他失败须查明处理。任一 split
为空、零监督、模板/掩码不一致不能开训。保存保留/排除 ID、原因、token 数、长度分布、
文件哈希和 train/holdout 任务组交集，并对照本地报告解释差异。SmolLM3 使用
`enable_thinking=False` 和固定日期 `2026-09-09`；推理沿用
`rst_common.tokenization.render_kwargs(profile)`，不能恢复动态模板默认值。

保留已发布 SETA/TMax 文本中的 reasoning 和序列化工具内容，不再次转换为另一协议。
保留 Opus 的 `reward=null/unknown`、unscored 和来源信息，不冒称成功轨迹。原 metadata
和预分词 ID 映射须可追溯。新文件使用新 manifest，不将旧 Qwen release manifest
设置为 `SFT_DATA_MANIFEST`。

复查全量模板/掩码可用 `scripts/16b_validate_model_datasets.py`，先创建包含本机
`models: {key: {path: ...}}` 和 `datasets: [{key, split, path}]` 的 inventory JSON，
再指定 `--inventory`、独立 `--out`、`--workers`。该脚本记录错误行，退出码 0 不表示
全部数据可训练；须读取结果，再经过严格导出验收。

## 5. 单节点资源约束和训练准入

调度器始终请求 `--nodes=1 --ntasks-per-node=1`，由一个 launcher 在节点内启动 GPU
worker。非 Slurm 环境也须限制一个主机。独立实验可分配不同节点，但每个 run 只有
一个节点及独立输出目录。每次启动、重试、恢复都重新执行以下设置，不继承旧队列参数：

```bash
set -euo pipefail
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export NGPUS="${RST_TRAIN_GPUS:?set allocated GPUs on this single node}"
export MASTER_PORT="${RST_TRAIN_PORT:-29500}"
[[ "$NNODES" == 1 && "$NODE_RANK" == 0 && "$NGPUS" -ge 1 ]]
[[ "${SLURM_NNODES:-1}" == 1 && "${SLURM_JOB_NUM_NODES:-1}" == 1 ]]
# CUDA_VISIBLE_DEVICES 必须由实际分配确定；保留调度器提供的设备列表。
```

从最终命令及所有 rank 日志验证 `world_size=NGPUS`、唯一 hostname、`nnodes=1`。
不能仅在报告中称单节点。不得启动跨主机 rendezvous 或为一次训练添加第二个节点。
只使用已分配的 GPU，保留现有 rollout、TerminalEvo 和其他服务；同主机独立作业
使用不同 GPU 子集和 `RST_TRAIN_PORT`，不得共享输出或日志目录。

本地已完成完整权重的 CPU 64-token SFT/DPO 检查、全量数据审计及 tiny 单 rank GPU
检查；**尚未验证完整模型 32K GPU、多 rank 通信或完整 epochs**。远端先运行相关
CPU/verl 回归，单列 skipped；每个模型在独立 smoke 目录验证真实多 GPU
forward/backward、参数更新、保存和重载，再用接近 32768 tokens 的真实样本测量
显存、吞吐与梯度，通过后才排正式训练。

三模型使用原生输出头：`FUSED_KERNELS=0`，关闭 Liger/remove-padding/Ulysses，
每个 micro-batch 一条完整轨迹。不要照搬 Llama/Phi 的 fused Torch head 覆盖。
OLMo3 SFT 实际 rank 日志应出现 `[rst-fsdp2] OLMo3: preserve FP32 rotary inputs`；
DPO 通过 `shard_model` 应用同一精度修复，不移除 BUG-30 hook。

OOM 时保留第一次完整 traceback，核对分片、attention 和 logits 内存；在**同一节点**
内验证更多已分配 GPU、合法 offload 或数值等价的内存优化。需要代码修复时补回归并
重做准入，不降低正式序列长度、跳过 OOM batch 或扩为多节点。单节点仍无法满足时
只阻塞该项，继续其他模型，并报告适合的单节点资源需求。

## 6. 21 次独立 SFT 的实际命令

GBS **128**、LR **3e-6**、cosine floor **3e-7**、warmup **0.03**、seed **1228**，
epochs 按表执行。“训练一遍”指完成一次历史配置，三 epoch 数据集仍跑三 epoch。
为矩阵每项设置 `MODEL_KEY`、`MODEL_PATH`、`DATA_KEY`、`EPOCHS`，先执行第 5 节
单节点设置，再调用：

```bash
export DATA_DIR="$BASE_FOLDER/data/$DATA_KEY"
export PRETOK_DIR="$DATA_DIR/$MODEL_KEY/32k-v1"
export PRETOK="$PRETOK_DIR/pretokenized_train.parquet"
export RUN_NAME="${MODEL_KEY}-${DATA_KEY}-sft-v1"
export FSDP_SIZE=-1 ULYSSES_SP=1 MAX_SEQ_LEN=32768 SAVE_HF_MODEL=1
export FUSED_KERNELS=0 WANDB_MODE=offline
unset SFT_DATA_MANIFEST
mkdir -p "$BASE_FOLDER/$RUN_NAME/logs"

bash scripts/30_run_sft_verl.sh \
  trainer.total_epochs="$EPOCHS" trainer.save_freq=200 trainer.seed=1228 \
  trainer.resume_mode=auto data.val_files=null \
  data.train_batch_size=128 data.use_dynamic_bsz=False \
  data.micro_batch_size_per_gpu=1 data.pad_mode=no_padding \
  model.use_remove_padding=False model.use_liger=False model.use_fused_kernels=False \
  optim.lr=3e-6 optim.lr_scheduler_type=cosine optim.min_lr_ratio=0.1 \
  optim.lr_warmup_steps_ratio=0.03 \
  2>&1 | tee "$BASE_FOLDER/$RUN_NAME/logs/train.log"
```

`30` 原本默认 `NNODES=4`、`trainer.save_freq=20`；本文件的环境与尾部 Hydra overrides
必须真正传入，核验最终配置为 **200**。registry 会重设部分超参数，不能只 export
LR/NUM_EPOCH 代替 overrides。不要用会启动额外历史阶段的总入口脚本。

核验应有的 `global_step_200`、`global_step_400` 等保存点及 final。`SAVE_HF_MODEL=1`
同时请求 HF 权重和恢复状态；检查权重分片、optimizer/extra 齐全，不能把只有配置的
目录当 checkpoint。若所装 verl 未保存非整 200 的末步，补上最终保存并验证。

SFT 中断使用 `trainer.resume_mode=auto`，保持原模型、数据、GPU 数、GBS 和 schedule，
仍为单节点；保留 `resume_guard.py` 的记录。更改数据/epoch/并行度须新 run 并说明，
不删除保护文件、不把重训说成续训。记录每 epoch 的样本/token、尾批丢弃及用时。

## 7. RST DPO：重建偏好、单节点打分和训练

只对真实同任务成功/失败轨迹做偏好训练。源为
`Zhongzhi1228/Recursive-Task-Synthesis-Trajectories`，revision
`2ec62e6cbbc9e6b014d016ffc31bf266049326fa`；下载 `data/*.tar`、`metadata/*` 到
`$BASE_FOLDER/rst-trajectories`，核验 shard manifest。可复用经过来源签名验证的文本
cache，不能复用 Qwen token 数据。主配置沿用 per-side 14、seed 1228 的 RST DPO v2
构建规则；模型重分词后配对数可能变化，以 manifest 为准。

先将同模型 `rst-cap10` SFT final 导出为完整 HF 目录并验证，作为以下 `POLICY`。
不同模型/版本使用独立 pairs、reference、输出和日志目录。重新执行第 5 节单节点
设置。为保留调度器 GPU 子集映射，直接调用 `17`、`18`、`19`；`33` 的 reference
子进程会覆盖设备列表为 `0..NGPUS-1`，不要套到非零起始或非连续 GPU 分配上。

```bash
export POLICY="$BASE_FOLDER/exports/${MODEL_KEY}-rst-cap10-sft-v1/final"
export PAIRS_DIR="$BASE_FOLDER/dpo-data/$MODEL_KEY/rst-v2-32k"
export REF_DIR="$PAIRS_DIR/ref-rst-cap10-sft-final"
export OUT_DIR="$BASE_FOLDER/${MODEL_KEY}-rst-dpo-v1"
mkdir -p "$OUT_DIR/logs"
python scripts/14_prepare_tokenizer.py --model "$POLICY" --verify-verl

python scripts/17_build_dpo_data.py \
  --traj-root "$BASE_FOLDER/rst-trajectories" --tokenizer "$POLICY" \
  --loss-mask-type auto --out-dir "$PAIRS_DIR" --per-side 14 \
  --pairs-per-prompt 2 --holdout-groups 40 --seed 1228 --max-seq-len 32768 \
  --cache "$PAIRS_DIR/reconstructed.jsonl.gz"

# 明确列出本节点已分配设备；支持物理编号或 CUDA 接受的 GPU UUID。
IFS=',' read -r -a DPO_DEVICES <<< "${CUDA_VISIBLE_DEVICES:?set allocated devices}"
[[ "${#DPO_DEVICES[@]}" -eq "$NGPUS" ]]
REF_DTYPE=fp32
if (( NGPUS > 1 )); then REF_DTYPE=bf16; fi
REF_PIDS=()
for (( shard=0; shard<NGPUS; shard++ )); do
  CUDA_VISIBLE_DEVICES="${DPO_DEVICES[$shard]}" \
    python scripts/18_dpo_ref_logprobs.py \
      --pairs "$PAIRS_DIR" --model-path "$POLICY" --out "$REF_DIR" \
      --dtype "$REF_DTYPE" --max-seq-len 32768 --logit-chunk 512 \
      --shard "$shard" --num-shards "$NGPUS" \
      > "$OUT_DIR/logs/ref-shard${shard}.log" 2>&1 &
  REF_PIDS+=("$!")
done
REF_RC=0
for pid in "${REF_PIDS[@]}"; do
  wait "$pid" || REF_RC=1
done
[[ "$REF_RC" == 0 ]]

# 继承原始 CUDA_VISIBLE_DEVICES，整个 torchrun 只在当前节点执行。
torchrun --nnodes=1 --nproc_per_node="$NGPUS" --node_rank=0 \
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
  scripts/19_train_dpo.py \
  --pairs "$PAIRS_DIR" --ref-logps "$REF_DIR" --model-path "$POLICY" \
  --out "$OUT_DIR" --beta 0.1 --lr 5e-7 --grad-accum 4 --epochs 1 \
  --max-seq-len 32768 --param-dtype fp32 --logit-chunk 512 \
  --length-normalize --seed 1228 --save-every 200 \
  2>&1 | tee "$OUT_DIR/logs/train.log"
```

有效 DPO batch 为 `NGPUS × 4` 对。reference 与初始 policy 必须是同一个 SFT checkpoint，
forward dtype、chunk size 相同。验证指纹、reference 完整覆盖、step-0 loss≈log(2)、
train/holdout 任务组互斥、共同 prompt、两侧非零监督、非相同 chosen/rejected；不关闭
calibration 或放宽容差。reference 可恢复，但改变模型/数据/分片数须用新目录重算，
不能混入旧分片。读取完整打分日志及 `dpo_training_summary.json`。

`--save-every 200` 产生 `hf-step200`、`hf-step400` 等 HF 权重；最终为 `$OUT_DIR/hf`。
**当前 DPO 不保存可恢复的 optimizer/schedule**：中断后保留旧产物，在新目录从同一
SFT POLICY 重跑，不能将阶段 HF 权重声称为完整续训断点。仍只使用单节点。

## 8. 导出、评估与验收

SFT 每个 200-step 保存点及实际 final 都导出并验证；`STEP` 取真实 optimizer 步数。

```bash
bash scripts/08_prepare_eval_ckpt.sh --model-key "$MODEL_KEY" \
  --ckpt "$BASE_FOLDER/$RUN_NAME/global_step_${STEP}" \
  --out "$BASE_FOLDER/exports/$RUN_NAME/step-${STEP}"

python scripts/06b_eval_offline.py \
  --model-path "$BASE_FOLDER/exports/$RUN_NAME/step-${STEP}" \
  --base-model "$MODEL_PATH" \
  --holdout "$PRETOK_DIR/pretokenized_holdout.parquet" \
  --out "$BASE_FOLDER/eval/$RUN_NAME/step-${STEP}" \
  --max-rows 0 --max-seq-len 32768 --max-actions 0
```

`08` 通过 `$BASE_FOLDER/$MODEL_DIR_NAME` 找基础模型，使用第 2 节的目录名。验证分片
齐全、文本参数实际改变、tokenizer 指纹一致、模型能重载和生成。将通过验收的末步
导出登记为 `exports/<RUN_NAME>/final`，供 DPO 使用；保留准确 step 和源路径。

离线评估用目标模型自己的**预分词 holdout**，不用 `06b` 的旧 Qwen messages 分支。
`--max-actions 0` 关闭 Terminus 专属探针，完整评估保留的 holdout。DPO 各保存点使用
RST holdout，与自己的 SFT POLICY 比较，另报告偏好准确率、ties、margin 和裁剪比例。
NLL 仅在同模型同数据下比较。资源和协议适配允许时，再对 base/SFT/DPO 做同一终端
任务集的真实 agentic eval；未做则明确标注，不把 NLL 或偏好准确率当任务成功率。

持续维护 `$BASE_FOLDER/reports/smollm_olmo_gemma_training_matrix.{json,md}`，至少
24 行，追加历史正式版本时扩展。每行记录 `pending / running / completed / blocked /
failed`、代码/模型/数据 SHA、独立初始化来源、行数/token/排除原因、epochs/实际步数、
超参数、唯一节点 hostname、GPU 列表/world size、显存/吞吐/耗时、所有 checkpoint
路径与实际步号、重载和评估结果、失败/阻塞原因。附源 manifest 与重导 manifest 差异。

先在服务器持久目录交付模型、tokenizer、配置、脱敏日志和报告；模型上传沿用用户已有
明确授权的发布目的地，没有发布约定则保留本地产物。`completed` 必须有正式训练完成
日志、最终完整权重、重载验证和离线评估；smoke、排队或只有目录均不算完成。最终报告
逐项列出完成与缺口，包括未上传、未做 agentic eval，不能只报告一个成功示例。
