---
marp: true
theme: default
paginate: true
style: |
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Raleway:wght@300;400;500;600&display=swap');

  :root {
    --accent: #ea580c;
    --accent-soft: #fff1e8;
    --cyan: #0284c7;
    --green: #15803d;
    --yellow: #b45309;
    --red: #dc2626;
    --bg: #ffffff;
    --card: #f8f8fa;
    --border: #e6e7eb;
    --body: #4b5563;
    --muted: #8b8f98;
    --label: #6b7280;
    --text: #111827;
  }

  section {
    background: var(--bg);
    color: var(--text);
    font-family: 'Raleway', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif;
    font-weight: 400;
    padding: 46px 66px;
    line-height: 1.42;
  }
  h1 { font-family: 'Outfit', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif; font-weight: 800; font-size: 2.42em; color: var(--text); line-height: 1.08; margin: 0 0 9px; }
  h2 { font-family: 'Raleway', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif; font-weight: 400; font-size: 1.08em; color: #6b7280; margin: 0 0 18px; }
  h3 { font-family: 'Outfit', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif; font-weight: 700; font-size: 0.58em; color: var(--muted); text-transform: uppercase; margin: 0 0 6px; }
  strong { color: var(--accent); font-weight: 600; }
  code { color: #111827; background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 4px; padding: 1px 5px; }
  section::after { font-family: 'Outfit'; font-size: 0.58em; color: #9ca3af; }

  section.lead { display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
  section.lead h1 { font-size: 3.12em; max-width: 980px; }
  section.lead h2 { max-width: 900px; color: #4b5563; }

  .chips { display: flex; gap: 8px; justify-content: center; margin-top: 20px; flex-wrap: wrap; }
  .chip { background: var(--accent-soft); border: 1px solid #fdba74; border-radius: 18px; padding: 4px 12px; font-family: 'Outfit', sans-serif; font-size: 0.55em; color: #c2410c; font-weight: 700; }

  .metric-row { display: flex; gap: 14px; margin-top: 18px; }
  .metric { flex: 1; background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; min-height: 116px; position: relative; overflow: hidden; }
  .metric::before { content: ''; position: absolute; inset: 0 auto auto 0; height: 2px; width: 100%; background: linear-gradient(90deg, var(--accent), transparent); }
  .metric .label { font-family: 'Outfit'; font-size: 0.54em; font-weight: 700; color: var(--label); text-transform: uppercase; }
  .metric .value { font-family: 'Outfit'; font-size: 1.63em; font-weight: 800; line-height: 1.04; margin-top: 9px; color: var(--text); }
  .metric .note { font-size: 0.62em; color: var(--body); margin-top: 8px; line-height: 1.55; }

  .split { display: flex; gap: 15px; margin-top: 18px; }
  .panel { flex: 1; background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 18px; min-height: 172px; }
  .panel h4 { font-family: 'Outfit'; font-size: 0.92em; font-weight: 800; margin: 0 0 9px; color: var(--text); }
  .panel p { color: var(--body); font-size: 0.66em; line-height: 1.72; margin: 0; }

  .rows { margin-top: 13px; }
  .row { display: flex; align-items: center; gap: 10px; padding: 8px 7px; border-bottom: 1px solid #ececef; border-radius: 6px; font-size: 0.66em; }
  .row .name { width: 218px; color: #1f2937; font-weight: 600; }
  .row .desc { flex: 1; color: var(--body); line-height: 1.55; }

  .tag { font-family: 'Outfit'; font-weight: 800; font-size: 0.54em; text-transform: uppercase; padding: 3px 8px; border-radius: 4px; min-width: 58px; text-align: center; display: inline-block; }
  .ok { color: var(--green); background: #15803d14; border: 1px solid #15803d33; }
  .warn { color: var(--yellow); background: #b4530914; border: 1px solid #b4530933; }
  .risk { color: var(--red); background: #dc262614; border: 1px solid #dc262633; }
  .info { color: var(--cyan); background: #0284c714; border: 1px solid #0284c733; }

  .quote { background: var(--card); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0; padding: 14px 20px; color: #374151; font-size: 0.76em; line-height: 1.7; margin-top: 16px; }
  .formula { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 20px; margin-top: 14px; font-family: 'Outfit', 'Noto Sans CJK SC', sans-serif; font-size: 0.78em; color: var(--text); }
  .codebox { background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 14px; color: #1f2937; font-family: 'Courier New', monospace; font-size: 0.54em; line-height: 1.5; white-space: pre-wrap; margin: 0; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start; margin-top: 12px; }
  .prov { font-size: 0.52em; color: var(--muted); margin-top: 10px; line-height: 1.5; }
header: ''
footer: ''
---

<!-- _class: lead -->
<!-- _paginate: false -->

<svg width="62" height="62" viewBox="0 0 24 24" fill="none" stroke="#ea580c" stroke-width="1.25">
  <circle cx="7" cy="7" r="3"/>
  <circle cx="17" cy="7" r="3"/>
  <circle cx="12" cy="17" r="3"/>
  <path d="M9.6 8.7l1.9 5.3M14.4 8.7l-1.9 5.3M10 17H8.7a5.8 5.8 0 0 1-4.5-2.2M14 17h1.3a5.8 5.8 0 0 0 4.5-2.2"/>
</svg>

# Recursive Self-Improving Data Production

## 对抗递归进化：由求解、验证、改环境三个 agent 组成的多智能体系统，持续生产高难度、不可作弊的终端任务与轨迹

<div class="chips">
  <span class="chip">ADVERSARIAL RECURSIVE EVOLUTION</span>
  <span class="chip">THREE-AGENT MAS</span>
  <span class="chip">PARAMETRIC TASK INSTANCES</span>
  <span class="chip">VERIFIER IS THE ONLY REWARD</span>
</div>

---

### Motivation

# 能力的瓶颈，已经从模型转移到题目

<div class="split">
  <div class="panel" style="border-top: 3px solid var(--accent);">
    <h4>能力从哪里来</h4>
    <p>终端 agent 要学的不是知识，是<strong>行为</strong>：在信息不全的环境里试探、判断、纠错、收尾。这类行为只出现在"真的把一件事做完"的过程里，也只能从这样的过程中学到。而模型越强，还能提供梯度的题越少——能力的边界越来越取决于我们造得出多难的题。</p>
  </div>
  <div class="panel" style="border-top: 3px solid var(--red);">
    <h4>难在自洽，不在数量</h4>
    <p>一道题不是一段文字，而是一个自洽的小世界：目标说得清、环境立得住、确实存在一条走得通的路、而且有人能客观判定走通了没有。人来写，一道题的成本高到无法规模化；交给模型批量生成，这四者很容易各说各话。最常见的失效不是"太难"，而是<strong>"看起来合格"</strong>。</p>
  </div>
  <div class="panel" style="border-top: 3px solid var(--cyan);">
    <h4>"合格"本身有漏洞</h4>
    <p>今天判定一道题合格的方式是：存在一个正确解，且验收脚本认它。这只证明了<strong>有路可走</strong>，没证明<strong>必须走这条路</strong>。若照抄答案、伪造产物也能通过，这道题给出的分数就不再代表能力；用它训练，模型会朝"让验收通过"进化，而不是朝"把事做成"进化。</p>
  </div>
</div>

<div class="quote">
所以要提高的不是题目的长度或数量，而是<strong>"合格"这个标准本身的强度</strong>：不仅要存在正确解，还要在给定预算内找不到廉价解。难度也随之换了定义——不是标准解有多长，而是<strong>最短的通过解有多长</strong>。
</div>

---

### The System

# 三个 agent，一个 verifier

<div class="split">
  <div class="panel" style="border-top: 3px solid var(--accent);">
    <h4>Solving Agent</h4>
    <p>在沙盒里做任务，产生轨迹。它有两种模式：<strong>正常模式</strong>尽力解题，成功轨迹成为训练数据和后续的"证人"；<strong>攻击模式</strong>只给 1 到 8 条命令的预算、只看指令与环境，用最短最懒的方式试图通过 verifier。攻击模式的每一次成功，就是一个反例。</p>
  </div>
  <div class="panel" style="border-top: 3px solid var(--green);">
    <h4>Verify Agent</h4>
    <p>唯一的裁决者。在多个环境实例上执行标准解并运行 verifier，给出可解性证书；回放攻击模式的轨迹，判定它是反例还是合法快解；检查修复后的 verifier 是否仍让所有证人通过。它输出逐项 check 结果，不输出意见。</p>
  </div>
  <div class="panel" style="border-top: 3px solid var(--cyan);">
    <h4>Environment Agent</h4>
    <p>唯一能改任务的角色。提出更难的任务；收到反例后写出一条新 check，使反例失败而标准解与证人通过；把写死的 fixture 改成按 seed 生成的参数化实例，让常数不再是常数。</p>
  </div>
</div>

<div class="quote">
分工的原则：Solving Agent 只产生行为，Verify Agent 只裁决，Environment Agent 只改环境。LLM 的意见永远不进入 reward、loss mask 或评测分，reward 只来自 verifier。
</div>

---

### The Loop

# 对抗递归进化怎么做

<div class="rows" style="margin-top: 6px;">
  <div class="row"><span class="tag info">1 提出</span><div class="name">Environment Agent</div><div class="desc">在 seed 任务上加一个新的终端子目标，产出指令、参数化环境、标准解、verifier。</div></div>
  <div class="row"><span class="tag ok">2 证书</span><div class="name">Verify Agent</div><div class="desc">在 n 个环境实例上运行标准解并跑 verifier。全部通过才算"有通解"，只在一个实例上通过不算。</div></div>
  <div class="row"><span class="tag risk">3 攻击</span><div class="name">Solving Agent · 攻击模式</div><div class="desc">在一个新实例上，用预算 B 条命令、只看指令与环境，尝试通过 verifier。预算从 1 逐级放大到 8，记录首次得手的预算。</div></div>
  <div class="row"><span class="tag warn">4 修复</span><div class="name">Environment Agent</div><div class="desc">对每个反例写一条分离 check：反例失败，标准解和所有证人通过。写得出来，说明 verifier 有缺口；写不出来，说明反例是合法解，任务只是容易，记低难度而不修。</div></div>
  <div class="row"><span class="tag ok">5 接受</span><div class="name">Verify Agent</div><div class="desc">预算内再找不到反例即接受，预算耗尽即拒绝。接受的任务带一个新标签：首次得手预算，它就是任务的难度。</div></div>
  <div class="row"><span class="tag info">6 训练</span><div class="name">Solving Agent · 正常模式</div><div class="desc">在存活任务上 rollout、筛选、训练；成功轨迹回流为证人，失败画像告诉 Environment Agent 先加固哪一类 check。</div></div>
</div>

<div class="formula">
accept(τ) ⇔ ∀θ ∈ 实例样本: 标准解通过 V<sub>θ</sub> ｜ ∧ ∀ξ ∈ 攻击类 A<sub>B</sub>: ξ 在新实例上不通过 ｜ ∧ ∀w ∈ 证人集: w 在新实例上通过<br>
难度(τ) = 最短通过解的命令数，而不是标准解的长度。
</div>

---

### A Real Task · Round 0

# 起点：一道已经"验收合格"的任务

<div class="two-col">
  <div>
    <h4 style="font-family:Outfit; font-size:0.9em; margin:0 0 6px;">任务指令（节选）</h4>
    <pre class="codebox">用 ~/keys 里的私钥 SSH 到 10.188.74.102，运行
migrationConfigGenerator.py --cluster cl1 --host c7401...，
输出重定向到 /app/result.txt。
在 /etc 下找到 required_services.txt，把生成结果里出现
而清单里缺少的服务追加进去。
对清单里每个服务，确认它出现在 result.txt 中，
把 "&lt;服务名&gt; PASS" 逐行写进 /app/validation.txt。</pre>
    <h4 style="font-family:Outfit; font-size:0.9em; margin:12px 0 6px;">环境 v0：写死的 fixture</h4>
    <pre class="codebox">printf 'ZOOKEEPER\nAMBARI_INFRA_SOLR\nRANGER\nATLAS\n' \
  &gt; /etc/service-lists/required_services.txt
# 远端生成器的输出固定多出一个 LOGSEARCH
# 标准解里也写死了: grep -qF "LOGSEARCH" result.txt</pre>
  </div>
  <div>
    <h4 style="font-family:Outfit; font-size:0.9em; margin:0 0 6px;">verifier v0 的四项 check</h4>
    <div class="rows" style="margin-top:0;">
      <div class="row" style="padding:6px 7px;"><span class="tag ok">01</span><div class="desc">清单文件含 <code>LOGSEARCH</code>，且 mtime 晚于 result.txt</div></div>
      <div class="row" style="padding:6px 7px;"><span class="tag ok">02</span><div class="desc">result.txt 非空，含三个固定字符串：开始标记、主机名、结束标记</div></div>
      <div class="row" style="padding:6px 7px;"><span class="tag ok">03</span><div class="desc">清单里每个服务在 validation.txt 里有一行 <code>X PASS</code>，且无 FAIL</div></div>
      <div class="row" style="padding:6px 7px;"><span class="tag risk">04</span><div class="desc">"反捷径"：validation.txt 不是清单的逐字复制；mtime 晚于 result.txt；再查一次三个标记</div></div>
    </div>
    <div class="quote" style="margin-top:14px;">
    标准解跑通，四项全绿，这道题进了任务池。但 v0 检查的全是<strong>常数和时间戳</strong>：三个固定字符串、一个写死的服务名、两个 mtime 先后。标准解和 verifier 都只对这一个实例成立。下一页看它如何被两个反例推着进化。
    </div>
  </div>
</div>

---

### Evolution

# 两个反例，两次进化，然后存活

<style scoped>section h1 { font-size: 2.0em; margin-bottom: 2px; }</style>

<style scoped>
section .stage { display:grid; grid-template-columns: 0.9fr 1.2fr 1.2fr; gap:10px; margin-top:6px; }
section .box { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:8px 10px; font-size:0.54em; line-height:1.5; color:var(--body); }
section .box b { display:block; font-family:Outfit; font-size:1.05em; color:var(--text); margin-bottom:4px; }
section .box pre { background:#fff; border:1px solid #e5e7eb; border-radius:6px; padding:4px 7px; font-size:0.86em; line-height:1.35; margin:4px 0 0; white-space:pre-wrap; font-family:'Courier New',monospace; color:#1f2937; }
section .lab { font-family:Outfit; font-size:0.48em; font-weight:800; text-transform:uppercase; color:var(--muted); margin-top:7px; }
</style>

<div class="lab">Round 0 → 1 · 驱动进化的反例：换一个实例</div>
<div class="stage">
  <div class="box"><b>环境 v0</b>缺失服务固定为 LOGSEARCH；verifier 找固定字符串；标准解写死 LOGSEARCH。</div>
  <div class="box" style="border-left:3px solid var(--red);"><b>反例 A · Verify Agent 生成新实例 θ₂</b>远端生成器这次多出的是 KAFKA。标准解仍追加 LOGSEARCH，check 01 仍要求 LOGSEARCH：<strong>标准解与 verifier 同时失效</strong>。这是"只对一个实例可解"的证据。</div>
  <div class="box" style="border-left:3px solid var(--cyan);"><b>Environment Agent 的修复 → v1</b>缺失服务集合与主机名按 seed 生成；标准解改为集合差 <code>结果 − 清单</code>；verifier 验证时自己拉取本实例生成器的真实输出，逐行比对 result.txt，不再找固定字符串。</div>
</div>

<div class="lab">Round 1 → 2 · 驱动进化的反例：跳过校验</div>
<div class="stage">
  <div class="box"><b>环境 v1</b>result.txt 须等于真实输出；清单 ⊇ 结果中的服务；validation 对清单里每个服务有 PASS。</div>
  <div class="box" style="border-left:3px solid var(--red);"><b>反例 B · Solving Agent 攻击模式，预算 3</b><pre>ssh ... &gt; /app/result.txt
grep -oE '^[A-Z_]+' /app/result.txt &gt;&gt; 清单
sed 's/$/ PASS/' 清单 &gt; /app/validation.txt</pre>第三条命令给<strong>所有</strong>服务写 PASS，从未检查它是否真在结果里。v1 分不清"校验过"与"假定过"。</div>
  <div class="box" style="border-left:3px solid var(--cyan);"><b>Environment Agent 的修复 → v2</b>清单初值里按 seed 放一个<strong>诱饵服务</strong>（结果里不存在）；指令改为逐行写 PASS 或 FAIL；verifier 从两个文件重算 PASS 集 = 清单 ∩ 结果、FAIL 集 = 清单 − 结果。反例 B 对诱饵写了 PASS，失败。证人检查：通解版标准解与两条真正逐项校验过的学生轨迹仍通过。</div>
</div>

<div class="lab">Round 2 · 存活</div>
<div class="stage">
  <div class="box"><b>环境 v2</b>参数化实例、真实输出比对、诱饵、交叉重算。</div>
  <div class="box" style="border-left:3px solid var(--green);"><b>攻击预算放大到 8，无反例</b>伪造 result.txt 过不了真实输出比对；不读结果写不出诱饵的 FAIL；复制清单过不了交叉重算。最短通过解必须含 SSH、集合差与逐项校验。</div>
  <div class="box" style="border-left:3px solid var(--green);"><b>接受，难度标签"预算 8 存活"</b>Round 0 的最短通过解是 4 条命令、零次 SSH；Round 2 已等于任务本身。任务没有变长，变的是<strong>最短捷径的长度</strong>。</div>
</div>

<div class="prov">Round 0 来自任务池中的真实任务及其 verifier 源码；反例 A、B 与修复 v1、v2 是对该 verifier 的推演，尚未在沙盒中执行。</div>

---

### Evidence

# 为什么要这么做

## 两个缺口相加：一道"合格"的题可以不迫使求解者读实例，而模型已经学会提前收工

<style scoped>
section h1 { font-size: 1.84em; margin-bottom: 0; }
section h2 { font-size: 0.84em; margin: 0 0 2px; }
section .metric { min-height: 74px; padding: 9px 11px; }
section .metric .value { font-size: 1.24em; margin-top: 5px; }
section .metric .note { font-size: 0.49em; margin-top: 4px; line-height: 1.5; }
section .metric-row { margin-top: 6px; gap: 10px; }
section .lab { font-family: Outfit; font-size: 0.45em; font-weight: 800; text-transform: uppercase; color: var(--muted); margin-top: 9px; }
section .prov { margin-top: 8px; font-size: 0.43em; }
</style>

<div class="lab">任务侧 · 验收标准留下的缺口：不重算、不换实例，答案就是常数</div>
<div class="metric-row">
  <div class="metric"><div class="label">No recompute needed</div><div class="value">23.9%</div><div class="note">不必从实例重算就能满足的检查：只查存在性 9.2% + 只比常数 2.4% + 读产物再比常数 12.3%。它们对"复制标准解的产物"没有抵抗力。对照：67.2% 确实在重算或重跑。</div></div>
  <div class="metric"><div class="label">Single-instance tasks</div><div class="value">85.7%</div><div class="note">只有 14.3% 的任务在构建时引入随机性。其余答案是常数，写死在原理上不可检出——参数化实例最直接的理由。</div></div>
  <div class="metric"><div class="label">Verifier reads the oracle</div><div class="value">599</div><div class="note">去读标准解脚本找硬编码值的 verifier 数；其中 249 个断言该脚本存在，而 <code>/solution</code> 只在跑 oracle 时挂载，这些任务对任何 agent 都不可解。</div></div>
  <div class="metric"><div class="label">Mutation-ready</div><div class="value">64.1%</div><div class="note">24,028 个任务有"Dockerfile 烤进去、被读、标准解不写"的 fixture，可自动改一个数据 token。换实例实验的可行上界。</div></div>
</div>

<div class="lab">模型侧 · 失败不是做不完，是做了六成就宣布做完</div>
<div class="metric-row">
  <div class="metric"><div class="label">Failures that claim done</div><div class="value">91.9%</div><div class="note">失败轨迹里最后一步宣称完成的比例；撞到轮数上限的是 0%。失败是提前收工，不是跑不完。</div></div>
  <div class="metric"><div class="label">Partial credit at failure</div><div class="value">0.60</div><div class="note">失败时已通过的 check 占比中位数。模型做了六成就宣布完成。</div></div>
  <div class="metric"><div class="label">Did it, tripped process</div><div class="value">35%</div><div class="note">带编号 check 的 712 条失败中，最终语义已过、却挂在过程或反捷径 check 上的比例。是抓到捷径还是过度特化，只有证人能分辨。</div></div>
  <div class="metric"><div class="label">Success length vs oracle</div><div class="value">9 → 14</div><div class="note">标准解从 50 行长到 400 行，成功轨迹的中位轮数只从 9 到 14。任务在变长，不在变难。</div></div>
</div>

<div class="prov">上排：全部 37,484 个任务、309,712 个测试函数，规则版本 2026-09-04（<code>scripts/22_probe_verifiers.py</code>，两套口径均存于 manifest）。下排前三格：<code>trajectories-00000.tar</code> 的 4,429 条干净轨迹，全部 Qwen3.5-27B；第四格为全量 60,932 条成功轨迹。轨迹侧尚未全量复核。</div>

---

### Produced Data

# 生产出来的一条优质 agent 数据

<style scoped>
section .turns { font-size:0.52em; line-height:1.45; color:var(--body); }
section .turn { display:grid; grid-template-columns: 34px 1fr; gap:6px; padding:4px 0; border-bottom:1px solid #ececef; }
section .turn .n { font-family:Outfit; font-weight:800; color:var(--accent); }
section .turn code { font-size:0.95em; }
section .turn .a { color:#374151; }
</style>

<div class="two-col" style="grid-template-columns: 1.25fr 0.75fr;">
  <div>
    <div class="turns">
      <div class="turn"><div class="n">T1</div><div><span class="a">先看密钥是否存在，再建立连接并运行生成器。</span><br><code>ls -la ~/keys/</code> · <code>chmod 600 ~/keys/insecure_private_key</code> · <code>ssh -i ... root@10.188.74.102 'migrationConfigGenerator.py ...' &gt; result.txt</code></div></div>
      <div class="turn"><div class="n">T2</div><div><span class="a">SSH 成功。查看输出内容，并定位清单文件。</span><br><code>cat /app/result.txt</code> · <code>find /etc -name 'required_services.txt'</code></div></div>
      <div class="turn"><div class="n">T3</div><div><span class="a">结果里检测到 5 个服务，其中 LOGSEARCH 不在清单里。</span><br><code>cat /etc/service-lists/required_services.txt</code></div></div>
      <div class="turn"><div class="n">T4</div><div><span class="a">追加缺失服务，逐项校验并写报告。</span><br><code>echo LOGSEARCH &gt;&gt; required_services.txt</code> · <code>while read s; do grep -q "Service detected: $s" result.txt &amp;&amp; echo "$s PASS"; done &gt; validation.txt</code></div></div>
      <div class="turn"><div class="n">T5</div><div><span class="a">validation.txt 是空的，grep 模式可能没匹配上。手动测试模式，逐个服务检查。</span><br><code>grep 'Service detected:' result.txt</code> · <code>... grep -q "$s" result.txt &amp;&amp; echo Found</code></div></div>
      <div class="turn"><div class="n">T6</div><div><span class="a">模式修正后 5 个服务全部 Found。重写报告，并核对两个文件都非空。</span><br><code>while read s; do grep -q "$s" result.txt &amp;&amp; echo "$s PASS" &gt;&gt; validation.txt; done</code> · <code>ls -la result.txt validation.txt</code></div></div>
      <div class="turn"><div class="n">T7</div><div><span class="a">逐条对照指令确认五项要求均已满足，宣称完成。</span> <code>task_complete: true</code></div></div>
    </div>
  </div>
  <div>
    <div class="metric-row" style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:0;">
      <div class="metric" style="min-height:74px; padding:10px 12px;"><div class="label">Verifier</div><div class="value" style="font-size:1.3em;">4 / 4</div><div class="note" style="font-size:0.56em;">四项 check 全部通过</div></div>
      <div class="metric" style="min-height:74px; padding:10px 12px;"><div class="label">Commands</div><div class="value" style="font-size:1.3em;">17</div><div class="note" style="font-size:0.56em;">7 个回合，15 条不重复</div></div>
    </div>
    <div class="rows" style="margin-top:8px;">
      <div class="row" style="padding:5px 6px; font-size:0.58em;"><span class="tag ok">探索</span><div class="desc">先 ls、cat、find 再动手，缺失服务是从输出里读出来的，不是猜的。</div></div>
      <div class="row" style="padding:5px 6px; font-size:0.58em;"><span class="tag ok">恢复</span><div class="desc">T4 写出空文件，T5 定位到 grep 模式问题，T6 修正。一次真实的错误诊断与恢复。</div></div>
      <div class="row" style="padding:5px 6px; font-size:0.58em;"><span class="tag ok">自检</span><div class="desc">宣称完成前用 ls -la 核对产物，再逐条对照指令。</div></div>
      <div class="row" style="padding:5px 6px; font-size:0.58em;"><span class="tag ok">干净</span><div class="desc">从未触碰 tests/ 或 verifier；保留原始 completion 与逐项 check 结果，可直接重构为训练样本。</div></div>
    </div>
    <div class="prov">来源：任务池同一任务上 27B 求解模型的一条成功轨迹，8 个回合，1,696 个输出 token，reward 1.0。文字为模型 analysis 字段的节译。</div>
  </div>
</div>

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Bottom Line

## 对抗递归进化：每一轮，求解者攻击环境，环境在验证者的裁决下进化，存活的任务才进入训练。标准解通过只是入场券；还要在预算内找不到捷径、此前的合法解仍然通过、换一个实例后答案不再是常数。三个 agent 各管一段，verifier 始终是唯一的 reward。

<div class="chips">
  <span class="chip">NOT CHEAPLY SOLVABLE</span>
  <span class="chip">SOLVE · VERIFY · REPAIR</span>
  <span class="chip">PARAMETRIC INSTANCES</span>
  <span class="chip">VERIFIER-ONLY REWARD</span>
</div>
