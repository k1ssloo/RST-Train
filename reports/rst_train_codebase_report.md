---
marp: true
theme: default
paginate: true
style: |
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Raleway:wght@100;200;300&display=swap');

  :root {
    --accent: #ff6b1a;
    --cyan: #38bdf8;
    --green: #22c55e;
    --yellow: #f5a623;
    --red: #ef4444;
    --dark: #000;
    --card: #080808;
    --card2: #0e0e0e;
    --border: #171717;
    --body: #999;
    --label: #666;
    --muted: #555;
    --light: #fff;
  }

  section {
    background: var(--dark);
    color: var(--light);
    font-family: 'Raleway', 'Noto Sans SC', sans-serif;
    font-weight: 200;
    padding: 50px 68px;
    line-height: 1.5;
  }

  h1 {
    font-family: 'Outfit', 'Noto Sans SC', sans-serif;
    font-weight: 800;
    font-size: 2.85em;
    color: var(--light);
    line-height: 1.05;
    margin: 0 0 8px;
  }

  h2 {
    font-family: 'Raleway', 'Noto Sans SC', sans-serif;
    font-weight: 100;
    font-size: 1.15em;
    color: #888;
    margin: 0 0 20px;
  }

  h3 {
    font-family: 'Outfit', 'Noto Sans SC', sans-serif;
    font-weight: 600;
    font-size: 0.58em;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin: 0 0 6px;
  }

  strong { color: var(--accent); font-weight: 400; }
  code { color: #ddd; background: #111; border: 1px solid #1a1a1a; border-radius: 4px; padding: 1px 5px; }
  section::after { font-family: 'Outfit'; font-size: 0.58em; color: #1b1b1b; }

  section.lead {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
  }

  section.lead h1 { font-size: 3.65em; }

  .chips { display: flex; gap: 8px; justify-content: center; margin-top: 20px; flex-wrap: wrap; }
  .chip {
    background: #ff6b1a14;
    border: 1px solid #ff6b1a33;
    border-radius: 18px;
    padding: 4px 13px;
    font-family: 'Outfit';
    font-size: 0.55em;
    color: #ffb27f;
    font-weight: 600;
    letter-spacing: 0.06em;
  }

  .metric-row { display: flex; gap: 14px; margin-top: 18px; }
  .metric {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px;
    position: relative;
    overflow: hidden;
    min-height: 116px;
  }
  .metric::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
  }
  .metric .label {
    font-family: 'Outfit';
    font-weight: 600;
    font-size: 0.48em;
    color: var(--label);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .metric .value {
    font-family: 'Outfit';
    font-weight: 800;
    font-size: 1.9em;
    line-height: 1.05;
    color: var(--light);
    margin-top: 10px;
  }
  .metric .note { font-size: 0.62em; color: var(--body); margin-top: 8px; }

  .flow { display: flex; align-items: stretch; gap: 8px; margin-top: 18px; }
  .node {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 14px;
    min-height: 118px;
  }
  .node .k {
    font-family: 'Outfit';
    font-size: 0.52em;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--accent);
    text-transform: uppercase;
  }
  .node .t { font-family: 'Outfit'; font-size: 0.95em; font-weight: 700; margin-top: 7px; }
  .node .d { font-size: 0.62em; color: var(--body); margin-top: 6px; }
  .arrow { width: 28px; display: flex; align-items: center; justify-content: center; color: #333; font-family: 'Outfit'; font-size: 1.2em; }

  .rows { margin-top: 12px; font-size: 0.68em; }
  .row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 8px;
    border-bottom: 1px solid #101010;
    border-radius: 6px;
  }
  .row:hover { background: #0c0c0c; }
  .row .name { width: 210px; color: #d0d0d0; font-weight: 300; }
  .row .desc { flex: 1; color: var(--body); }
  .row .tag { width: 86px; text-align: center; }

  .tag {
    font-family: 'Outfit';
    font-weight: 700;
    font-size: 0.55em;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 4px;
    display: inline-block;
  }
  .ok { color: var(--green); background: #22c55e12; border: 1px solid #22c55e2c; }
  .warn { color: var(--yellow); background: #f5a62312; border: 1px solid #f5a6232c; }
  .risk { color: var(--red); background: #ef444412; border: 1px solid #ef44442c; }
  .info { color: var(--cyan); background: #38bdf812; border: 1px solid #38bdf82c; }

  .terminal {
    background: #050505;
    border: 1px solid #181818;
    border-radius: 8px;
    overflow: hidden;
    font-family: 'Courier New', monospace;
    font-size: 0.62em;
    margin-top: 16px;
  }
  .terminal-bar { background: #111; padding: 8px 12px; display: flex; gap: 6px; align-items: center; }
  .dot { width: 9px; height: 9px; border-radius: 50%; }
  .terminal-body { padding: 14px 16px; color: #aaa; line-height: 1.7; }
  .prompt { color: var(--accent); }
  .success { color: var(--green); }
  .muted { color: var(--muted); }

  .split { display: flex; gap: 16px; margin-top: 18px; }
  .panel {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 19px;
  }
  .panel h4 {
    font-family: 'Outfit';
    font-size: 0.9em;
    margin: 0 0 10px;
    color: #f1f1f1;
  }
  .panel p { color: var(--body); font-size: 0.68em; margin: 0; line-height: 1.75; }

  .small { font-size: 0.66em; color: var(--body); }
  .center { text-align: center; }
header: ''
footer: ''
---

<!-- _class: lead -->
<!-- _paginate: false -->

<svg width="54" height="54" viewBox="0 0 24 24" fill="none" stroke="#ff6b1a" stroke-width="1.2">
  <path d="M12 2v20M5 7h9a3 3 0 0 1 0 6H8a3 3 0 0 0 0 6h11"/>
</svg>

# RST-Train Codebase

## Qwen3.5 终端任务训练、评测与自我改进流水线

<div class="chips">
  <span class="chip">SFT</span>
  <span class="chip">DPO DEFAULT</span>
  <span class="chip">GRPO OPT-IN</span>
  <span class="chip">RSI LOOP</span>
  <span class="chip">2026-09-03</span>
</div>

---

### Executive Summary

# 这个仓库解决什么

<div class="metric-row">
  <div class="metric">
    <div class="label">Objective</div>
    <div class="value">Agentic Training</div>
    <div class="note">把 RST 终端任务轨迹变成 Qwen3.5 的可复现 SFT / DPO / RL 路径</div>
  </div>
  <div class="metric">
    <div class="label">Primary Backend</div>
    <div class="value">verl + FSDP</div>
    <div class="note">主路径服务 4x8 A100；slime/Megatron 保留为次级路径</div>
  </div>
  <div class="metric">
    <div class="label">Core Contract</div>
    <div class="value">loss mask</div>
    <div class="note">Qwen3.5 的监督边界先离线固化，再交给 trainer</div>
  </div>
  <div class="metric">
    <div class="label">Current State</div>
    <div class="value">Ops Repo</div>
    <div class="note">重点不是库 API，而是能被审计的集群运行和报告</div>
  </div>
</div>

<div class="terminal">
  <div class="terminal-bar"><span class="dot" style="background:#ef4444"></span><span class="dot" style="background:#f5a623"></span><span class="dot" style="background:#22c55e"></span></div>
  <div class="terminal-body">
    <span class="prompt">$</span> MODEL_KEY=qwen3.5-9b bash scripts/20_run_all.sh<br>
    <span class="success">SFT -> eval -> report -> DPO</span><br>
    <span class="muted">RUN_RL=1 才进入 agentic GRPO</span>
  </div>
</div>

---

### Positioning

# 它不是严格 MAS

<div class="split">
  <div class="panel" style="border-top: 3px solid var(--accent);">
    <h4>当前系统</h4>
    <p>一个 policy agent 解终端任务；程序 verifier 给 reward；LLM judge 只在离线数据策展边界带发表意见。它更准确地叫 agentic self-improvement pipeline。</p>
  </div>
  <div class="panel" style="border-top: 3px solid var(--cyan);">
    <h4>MAS 扩展方向</h4>
    <p>把 solver、critic、planner、repair agent 变成有通信协议和共享状态的多个自治角色，再由 verifier 保持最终 reward 的可比性。</p>
  </div>
</div>

<div style="margin-top: 24px;">
  <svg width="100%" height="170" viewBox="0 0 900 170" preserveAspectRatio="none">
    <defs>
      <marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
        <path d="M0,0 L0,6 L9,3 z" fill="#333"/>
      </marker>
    </defs>
    <rect x="5" y="45" width="145" height="70" rx="8" fill="#080808" stroke="#171717"/>
    <text x="77" y="75" text-anchor="middle" fill="#fff" font-family="Outfit" font-size="18" font-weight="800">Policy</text>
    <text x="77" y="96" text-anchor="middle" fill="#777" font-family="Outfit" font-size="10">Qwen3.5</text>
    <line x1="155" y1="80" x2="265" y2="80" stroke="#333" stroke-width="2" marker-end="url(#arr)"/>
    <rect x="270" y="45" width="145" height="70" rx="8" fill="#080808" stroke="#171717"/>
    <text x="342" y="75" text-anchor="middle" fill="#fff" font-family="Outfit" font-size="18" font-weight="800">Harbor</text>
    <text x="342" y="96" text-anchor="middle" fill="#777" font-family="Outfit" font-size="10">sandbox run</text>
    <line x1="420" y1="80" x2="530" y2="80" stroke="#333" stroke-width="2" marker-end="url(#arr)"/>
    <rect x="535" y="45" width="145" height="70" rx="8" fill="#080808" stroke="#171717"/>
    <text x="607" y="75" text-anchor="middle" fill="#22c55e" font-family="Outfit" font-size="18" font-weight="800">Verifier</text>
    <text x="607" y="96" text-anchor="middle" fill="#777" font-family="Outfit" font-size="10">reward source</text>
    <line x1="685" y1="80" x2="795" y2="80" stroke="#333" stroke-width="2" marker-end="url(#arr)"/>
    <rect x="800" y="45" width="95" height="70" rx="8" fill="#080808" stroke="#171717"/>
    <text x="847" y="75" text-anchor="middle" fill="#ff6b1a" font-family="Outfit" font-size="18" font-weight="800">Data</text>
    <text x="847" y="96" text-anchor="middle" fill="#777" font-family="Outfit" font-size="10">SFT/DPO</text>
  </svg>
</div>

---

### Pipeline Map

# 主链路是编号脚本

<div class="flow">
  <div class="node"><div class="k">00-02</div><div class="t">Preflight</div><div class="d">环境、sandbox、模型和数据下载</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">03 / 15</div><div class="t">Data</div><div class="d">轨迹重建、mask 校验、预 tokenization</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">30</div><div class="t">SFT</div><div class="d">verl + FSDP2，带显存和 kernel 门禁</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">07 / 08</div><div class="t">Export</div><div class="d">合并 shard，恢复 vision / MTP</div></div>
</div>

<div class="flow" style="margin-top: 8px;">
  <div class="node"><div class="k">06 / 06b</div><div class="t">Eval</div><div class="d">agentic eval 或离线 fallback</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">14</div><div class="t">Report</div><div class="d">FAIL / WARN / OK 与 verdict.json</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">33</div><div class="t">DPO</div><div class="d">默认后训练，无容器依赖</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">12 / 21</div><div class="t">RL / RSI</div><div class="d">GRPO opt-in；RSI 生成新 SFT 数据</div></div>
</div>

---

### Data Contract

# 训练安全性来自一条硬契约

<div class="rows">
  <div class="row"><span class="tag ok">fixed</span><div class="name">统一 chat template</div><div class="desc">Qwen3.5 家族共用同一 tokenizer 和训练模板，不按模型重新切数据。</div></div>
  <div class="row"><span class="tag ok">mask</span><div class="name"><code>qwen3_5</code> loss mask</div><div class="desc">只训练 assistant 目标段，不能训练 user、system、harness 输出。</div></div>
  <div class="row"><span class="tag ok">pretok</span><div class="name"><code>input_ids + loss_mask</code></div><div class="desc">verl 直接消费离线产物，避免 MultiTurnSFTDataset 每轮注入 think block。</div></div>
  <div class="row"><span class="tag warn">gate</span><div class="name">首 token 不可监督</div><div class="desc">防止 packed batch 的 cyclic roll 把监督泄漏到样本边界。</div></div>
  <div class="row"><span class="tag info">audit</span><div class="name">两份 mask 实现互验</div><div class="desc">构建器与 validator 分开实现，同一 conversation 必须给出一致 mask。</div></div>
</div>

<div style="margin-top: 26px;">
  <svg width="100%" height="90" viewBox="0 0 900 90" preserveAspectRatio="none">
    <rect x="0" y="18" width="250" height="28" rx="4" fill="#111"/>
    <rect x="252" y="18" width="170" height="28" rx="4" fill="#1a1a1a"/>
    <rect x="424" y="18" width="306" height="28" rx="4" fill="#ff6b1a"/>
    <rect x="732" y="18" width="168" height="28" rx="4" fill="#111"/>
    <text x="125" y="37" text-anchor="middle" fill="#666" font-family="Outfit" font-size="12">prompt / user</text>
    <text x="337" y="37" text-anchor="middle" fill="#666" font-family="Outfit" font-size="12">think opener</text>
    <text x="577" y="37" text-anchor="middle" fill="#000" font-family="Outfit" font-size="13" font-weight="800">supervised assistant span</text>
    <text x="816" y="37" text-anchor="middle" fill="#666" font-family="Outfit" font-size="12">next turn</text>
    <line x1="424" y1="66" x2="730" y2="66" stroke="#ff6b1a" stroke-width="2"/>
    <text x="577" y="84" text-anchor="middle" fill="#ffb27f" font-family="Outfit" font-size="11">loss_mask = 1 only here</text>
  </svg>
</div>

---

### Model And Checkpoints

# 当前权重范围已经对齐项目模型族

<div class="metric-row">
  <div class="metric"><div class="label">Supported Family</div><div class="value">Qwen3.5</div><div class="note">0.8B / 4B / 9B / 27B / 35B-A3B 注册在 models.json</div></div>
  <div class="metric"><div class="label">Local CKPT Dirs</div><div class="value">11</div><div class="note">已覆盖 khazic 下符合项目命名的 rst-qwen3.5 权重</div></div>
  <div class="metric"><div class="label">Large Artifacts</div><div class="value">~452G</div><div class="note">data/ckpts 下当前量级，27B-sft 同时含 FSDP 和 HF 形态</div></div>
  <div class="metric"><div class="label">Excluded</div><div class="value">non-RST</div><div class="note">gemma / heat / tfm / appov3 等不在本项目 registry 内</div></div>
</div>

<div class="rows">
  <div class="row"><span class="tag info">4B</span><div class="name">sft / ota / tmax / nemo</div><div class="desc">主要是 FSDP shard 目录，约 17G 每组。</div></div>
  <div class="row"><span class="tag info">9B</span><div class="name">sft / ota / tmax / nemo</div><div class="desc">nemo 有 eval-ready HF，其他为训练 shard 形态。</div></div>
  <div class="row"><span class="tag info">27B</span><div class="name">sft / ota / cap10</div><div class="desc">ota 和 cap10 已是 out-hf-full；sft 保留 global_step_82 与 hf。</div></div>
</div>

---

### Backend Decision

# 为什么主路径是 verl + FSDP

<div class="split">
  <div class="panel" style="border-top: 3px solid var(--green);">
    <h4>verl + FSDP</h4>
    <p>当前主线。能在集群限制下推进 SFT，启动器前置检查 fused CE、FSDP2 梯度累积补丁、静态显存、rendezvous 和预 tokenized parquet。</p>
  </div>
  <div class="panel" style="border-top: 3px solid var(--yellow);">
    <h4>slime + Megatron</h4>
    <p>保留为次级路径。优势是原生 rollout 适配，但 A100 集群上的 cuDNN 和 ops 条件更难满足，当前不是默认。</p>
  </div>
</div>

<div style="display: flex; gap: 22px; margin-top: 26px; align-items: center;">
  <div style="flex: 1;">
    <svg width="100%" height="190" viewBox="0 0 420 190" preserveAspectRatio="none">
      <rect x="20" y="18" width="70" height="152" rx="6" fill="#22c55e22" stroke="#22c55e55"/>
      <rect x="110" y="48" width="70" height="122" rx="6" fill="#ff6b1a22" stroke="#ff6b1a55"/>
      <rect x="200" y="88" width="70" height="82" rx="6" fill="#f5a62322" stroke="#f5a62355"/>
      <rect x="290" y="118" width="70" height="52" rx="6" fill="#ef444422" stroke="#ef444455"/>
      <text x="55" y="182" text-anchor="middle" fill="#666" font-family="Outfit" font-size="10">mask</text>
      <text x="145" y="182" text-anchor="middle" fill="#666" font-family="Outfit" font-size="10">fused CE</text>
      <text x="235" y="182" text-anchor="middle" fill="#666" font-family="Outfit" font-size="10">FSDP2</text>
      <text x="325" y="182" text-anchor="middle" fill="#666" font-family="Outfit" font-size="10">rendezvous</text>
    </svg>
  </div>
  <div style="flex: 1;" class="small">
    <div style="font-family:'Outfit'; color:#fff; font-weight:800; font-size:1.15em; margin-bottom:8px;">设计原则</div>
    训练脚本不是“尽量启动”，而是先证明配置可运行。能在 30 秒内发现的错误，不应该等 32 个进程加载完权重后才暴露。
  </div>
</div>

---

### Checkpoint Integrity

# text-only 训练后必须恢复完整模型

<div class="flow">
  <div class="node"><div class="k">FSDP shard</div><div class="t">global_step_N</div><div class="d">要求 rank 文件完整，拒绝混合 world size</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">merge</div><div class="t">HF weights</div><div class="d">优先 verl merger，失败走本仓库兼容实现</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">restore</div><div class="t">vision / MTP</div><div class="d">从原始 checkpoint 流式补回未训练模块</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">diff</div><div class="t">trust gate</div><div class="d">确认 text stack 变了，视觉侧边文件齐全</div></div>
</div>

<div class="rows">
  <div class="row"><span class="tag ok">07</span><div class="name">restore_vision</div><div class="desc">复制 <code>model.visual.*</code> 和 <code>mtp.*</code>，默认拒绝未知 fallback。</div></div>
  <div class="row"><span class="tag ok">08</span><div class="name">prepare_eval_ckpt</div><div class="desc">能识别已合并 HF 目录，也能从 Hub 只拉取所需 shard。</div></div>
  <div class="row"><span class="tag warn">risk</span><div class="name">rank0 log</div><div class="desc">BUG.md 记录 rank0 训练日志仍需保留，方便静态显存核验。</div></div>
</div>

---

### Evaluation And Report

# verifier 是 reward，LLM judge 不是 reward

<div class="split">
  <div class="panel" style="border-top: 3px solid var(--green);">
    <h4>Agentic eval</h4>
    <p><code>06_eval.py</code> 用 SGLang + Harbor + Terminus-2 运行任务。基础设施故障不进分母；agent 预算失败留在分母且 reward 为 0。</p>
  </div>
  <div class="panel" style="border-top: 3px solid var(--cyan);">
    <h4>Offline fallback</h4>
    <p><code>06b_eval_offline.py</code> 做 teacher-forced scoring 和 action protocol probe。它不是 pass rate，但能在 sandbox 不可用时给 checkpoint 质量信号。</p>
  </div>
</div>

<div style="margin-top: 24px;">
  <svg width="100%" height="160" viewBox="0 0 900 160" preserveAspectRatio="none">
    <rect x="0" y="20" width="900" height="18" rx="5" fill="#111"/>
    <rect x="0" y="20" width="620" height="18" rx="5" fill="#22c55e"/>
    <rect x="620" y="20" width="130" height="18" fill="#f5a623"/>
    <rect x="750" y="20" width="150" height="18" rx="5" fill="#ef4444"/>
    <text x="310" y="62" text-anchor="middle" fill="#22c55e" font-family="Outfit" font-size="13" font-weight="700">OK: checkpoint + measurement</text>
    <text x="685" y="62" text-anchor="middle" fill="#f5a623" font-family="Outfit" font-size="13" font-weight="700">WARN</text>
    <text x="825" y="62" text-anchor="middle" fill="#ef4444" font-family="Outfit" font-size="13" font-weight="700">FAIL</text>
    <text x="450" y="115" text-anchor="middle" fill="#999" font-family="Outfit" font-size="17" font-weight="800">verdict.json: in_range gates GRPO, checkpoint_trustworthy gates DPO</text>
  </svg>
</div>

---

### Post-Training Paths

# 默认 DPO，可选 GRPO，新增 RSI

<div class="metric-row">
  <div class="metric">
    <div class="label">DPO</div>
    <div class="value" style="color:var(--green);">default</div>
    <div class="note">无需容器；0.8B/H100 已端到端跑通，multi-rank 仍待实测。</div>
  </div>
  <div class="metric">
    <div class="label">GRPO</div>
    <div class="value" style="color:var(--yellow);">opt-in</div>
    <div class="note">需要 sandbox；<code>RUN_RL=1</code> 才进入 agentic rollout。</div>
  </div>
  <div class="metric">
    <div class="label">RSI</div>
    <div class="value" style="color:var(--cyan);">new loop</div>
    <div class="note">把本模型成功 rollout 重新变成 SFT parquet。</div>
  </div>
</div>

<div class="flow" style="margin-top: 26px;">
  <div class="node"><div class="k">sample</div><div class="t">06_eval</div><div class="d">必须带 export-trajectories，保留 raw_content</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">harvest</div><div class="t">03h</div><div class="d">reward >= 1 的 rollout 进入 messages parquet</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">curate</div><div class="t">03g</div><div class="d">启发式全量，LLM 只看 borderline</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">train</div><div class="t">15 + 30</div><div class="d">混入 seed corpus，再预 tokenized SFT</div></div>
</div>

---

### LLM Supervision Boundary

# judge 只处理模糊数据，不参与打分

<div style="display: flex; gap: 24px; margin-top: 12px;">
  <div style="width: 320px; display: flex; align-items: center; justify-content: center;">
    <svg width="260" height="260" viewBox="0 0 260 260">
      <circle cx="130" cy="130" r="94" fill="none" stroke="#111" stroke-width="32"/>
      <circle cx="130" cy="130" r="94" fill="none" stroke="#22c55e" stroke-width="32" stroke-dasharray="478 590" stroke-dashoffset="0" transform="rotate(-90 130 130)"/>
      <circle cx="130" cy="130" r="94" fill="none" stroke="#38bdf8" stroke-width="32" stroke-dasharray="106 590" stroke-dashoffset="-478" transform="rotate(-90 130 130)"/>
      <circle cx="130" cy="130" r="94" fill="none" stroke="#ef4444" stroke-width="32" stroke-dasharray="6 590" stroke-dashoffset="-584" transform="rotate(-90 130 130)"/>
      <text x="130" y="122" text-anchor="middle" fill="#fff" font-family="Outfit" font-size="34" font-weight="800">18%</text>
      <text x="130" y="146" text-anchor="middle" fill="#666" font-family="Outfit" font-size="11" font-weight="700">TO JUDGE</text>
    </svg>
  </div>
  <div style="flex: 1;">
    <div class="rows" style="margin-top: 18px;">
      <div class="row"><span class="tag ok">81.0%</span><div class="name">clear_keep</div><div class="desc">确定性启发式直接保留。</div></div>
      <div class="row"><span class="tag info">18.0%</span><div class="name">borderline</div><div class="desc">只有这部分会调用 LLM judge，且按内容哈希缓存。</div></div>
      <div class="row"><span class="tag risk">1.1%</span><div class="name">clear_drop</div><div class="desc">重复、报错、提前完成等坏示范直接剔除。</div></div>
      <div class="row"><span class="tag warn">never</span><div class="name">reward / eval / mask</div><div class="desc">LLM 意见不进入 reward，保持 benchmark 可比。</div></div>
    </div>
  </div>
</div>

---

### Shared Modules

# 重构目标：一个定义对应一个可比较数字

<div class="rows">
  <div class="row"><span class="tag info">new</span><div class="name"><code>scripts/siblings.py</code></div><div class="desc">统一加载数字开头脚本，避免五份 spec_from_file_location 漂移。</div></div>
  <div class="row"><span class="tag info">new</span><div class="name"><code>scripts/sft_common.py</code></div><div class="desc">统一 dedup、长度门、group split、token_stats。</div></div>
  <div class="row"><span class="tag info">new</span><div class="name"><code>scripts/taskpool_common.py</code></div><div class="desc">统一 RL task tier、Docker FROM、verifier leak 规则。</div></div>
  <div class="row"><span class="tag info">new</span><div class="name"><code>scripts/hf_publish.py</code></div><div class="desc">统一 Hugging Face 上传和数据卡 claim 校验。</div></div>
  <div class="row"><span class="tag info">new</span><div class="name"><code>rst_common/judge.py</code></div><div class="desc">统一 OpenAI-compatible judge、缓存、预算和 JSON contract。</div></div>
</div>

<div class="metric-row">
  <div class="metric"><div class="label">Worklog</div><div class="value">+515 / -837</div><div class="note">已有代码净删约 322 行，新增共享模块与测试覆盖。</div></div>
  <div class="metric"><div class="label">Published Data</div><div class="value">stable</div><div class="note">RST / OpenThoughts / TMax 重建后与已发布数据对齐。</div></div>
</div>

---

### Test And Gate Status

# 本地能证明的是 CPU 约束

<div class="metric-row">
  <div class="metric"><div class="label">Local Runner</div><div class="value" style="color:var(--green);">0 fail</div><div class="note"><code>python tests/run_tests.py</code></div></div>
  <div class="metric"><div class="label">Observed</div><div class="value">311 ok</div><div class="note">当前轻量环境统计，部分模块按依赖跳过。</div></div>
  <div class="metric"><div class="label">Skipped</div><div class="value" style="color:var(--yellow);">29</div><div class="note">torch / transformers / numpy 缺失时跳过，不代表 GPU 路径已验证。</div></div>
  <div class="metric"><div class="label">Design</div><div class="value">source gates</div><div class="note">大量测试钉住脚本内容和启动门禁，而非只跑 mock。</div></div>
</div>

<div class="terminal">
  <div class="terminal-bar"><span class="dot" style="background:#ef4444"></span><span class="dot" style="background:#f5a623"></span><span class="dot" style="background:#22c55e"></span></div>
  <div class="terminal-body">
    <span class="prompt">$</span> python tests/run_tests.py<br>
    <span class="success">OK: 0 failure(s)</span><br>
    <span class="muted">GPU、transformers、numpy 相关路径需要对应环境复测</span>
  </div>
</div>

---

### Open Risks

# 需要集群或真实 rollout 才能关闭

<div class="rows">
  <div class="row"><span class="tag warn">OPEN-1</span><div class="name">4-node FSDP2 throughput</div><div class="desc">TCP 拓扑可能吞吐受限，需要测量后决定是否用 HSDP。</div></div>
  <div class="row"><span class="tag warn">OPEN-2</span><div class="name">Ulysses SP correctness</div><div class="desc">Gated-delta-net + SP 路径需要与 sp=1 对齐 loss 曲线。</div></div>
  <div class="row"><span class="tag warn">OPEN-3</span><div class="name">vision / MTP 未训练</div><div class="desc">已恢复但未训练；是否冻结和显存收益仍需实测。</div></div>
  <div class="row"><span class="tag risk">OPEN-4</span><div class="name">rank0 log</div><div class="desc">需要保留 rank0 stdout，才能审计 FSDP 静态 footprint。</div></div>
  <div class="row"><span class="tag risk">OPEN-5</span><div class="name">cluster code freshness</div><div class="desc">集群实际拉取版本要明确，不能只看本地 cached ref。</div></div>
</div>

---

### Next Run Plan

# 下一步应该验证什么

<div class="flow">
  <div class="node"><div class="k">1</div><div class="t">Raw rollout</div><div class="d">跑一次带 <code>--export-trajectories</code> 的真实 Harbor eval，关闭 BUG-19 raw path 风险</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">2</div><div class="t">SP check</div><div class="d">同 seed、同数据，sp=1 vs sp=8 小步 loss 曲线对照</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">3</div><div class="t">27B SFT</div><div class="d">确认 fused CE、FSDP2 patch、token budget gate 均在日志出现</div></div>
  <div class="arrow">→</div>
  <div class="node"><div class="k">4</div><div class="t">DPO multi-rank</div><div class="d">验证修复后的 sharding layout 在 torchrun 下可过 step-0 gate</div></div>
</div>

<div class="terminal">
  <div class="terminal-bar"><span class="dot" style="background:#ef4444"></span><span class="dot" style="background:#f5a623"></span><span class="dot" style="background:#22c55e"></span></div>
  <div class="terminal-body">
    <span class="prompt">$</span> MODEL_KEY=qwen3.5-27b bash scripts/20_run_all.sh<br>
    <span class="prompt">$</span> MODEL_KEY=qwen3.5-27b RUN_RL=1 bash scripts/20_run_all.sh<br>
    <span class="muted">先测量，再把 README 状态从 UNVERIFIED 提升为 run</span>
  </div>
</div>

---

<!-- _class: lead -->
<!-- _paginate: false -->

<svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="1.4">
  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
  <polyline points="22 4 12 14.01 9 11.01"/>
</svg>

# Bottom Line

## 当前 repo 已经具备可审计的训练闭环；下一步的价值在集群实测，而不是继续堆脚本。

<div class="chips">
  <span class="chip">mask correctness</span>
  <span class="chip">checkpoint integrity</span>
  <span class="chip">verifier reward</span>
  <span class="chip">RSI data loop</span>
</div>
