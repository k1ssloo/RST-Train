#!/usr/bin/env python3
"""Build DPO preference pairs from the RST trajectory release. No container needed.

    python scripts/17_build_dpo_data.py \
        --traj-root  $BASE_FOLDER/rst-trajectories \
        --tokenizer  $BASE_FOLDER/Qwen3.5-27B \
        --out-dir    $BASE_FOLDER/dpo-v1

WHY THIS EXISTS
    GRPO needs a sandbox: every rollout builds a task image and drives tmux inside
    it. On a pod whose AppArmor profile denies mount(2) that is impossible, and if
    no off-machine backend is available either (see BACKENDS.md) then on-policy RL
    cannot run at all.

    Preference learning does not need one. The trajectory release already contains
    both successes and failures on the same tasks, scored by the tasks' own
    verifiers. That is a preference dataset that was paid for once and can be
    replayed forever with zero container privilege -- the honest fallback when the
    rollout loop is blocked, and a reasonable warm-up even when it is not.

    MEASURED on the release (231,092 clean trajectories, reward is binary 0/1):
        61,575 successes, 169,517 failures
        2,246 task groups have any clean data
        1,290 groups contain BOTH a success and a failure   <- the usable pool
        of those: 59,406 successes / 111,991 failures

WHAT A PAIR MUST SATISFY, AND WHY THE OBVIOUS SHORTCUTS ARE WRONG
    1. THE SAME PROMPT, UP TO THE CONTAINER ID. The DPO objective compares
       log pi(y_w|x) against log pi(y_l|x) for the SAME x. The trajectory metadata
       has no `task_id` -- only `task_group_id`, which spans several task VARIANTS
       with DIFFERENT instructions. Pairing on `task_group_id` would silently
       compare answers to two different questions. So we pair on the actual first
       user message, read out of the trajectory itself.

       Hashing it verbatim, however, pairs NOTHING. Measured: 2,064 trajectories on
       one shard produced 2,064 distinct prompts and exactly 0 pairs, because the
       prompt ends with a live terminal screen:

           Current Terminal Screen:
           root@fa40b1a3-91e9-4ef4-bb3f-bde3c60fe5fc:~/exercises#

       -- the container's UUID hostname, fresh per run. So the hash is taken over a
       CANONICALIZED copy (`canonical_prompt()`) that masks UUIDs and docker's
       12-hex hostnames. That is a narrow substitution, and it is worth being sure
       it does not over-merge: on the same shard it collapses 2,064 prompts to 360
       and yields 241 pairs, and every group that still splits after
       canonicalization splits on genuinely rewritten instructions ("Run the tool
       with the -u flag and write its full standard output to /app/result.txt" vs
       "Write the tool's output to /app/result.txt") -- i.e. real variants, which
       must stay unpaired. The instruction text itself is never touched.

       The pair is then verified at the token level: the two sides must share a real
       common prefix, and the residual divergence inside the prompt is RECORDED
       (`prompt_divergence_tokens`), never hidden. It is a handful of hostname
       tokens in masked context, which is why this is acceptable -- but it is
       measured, and the manifest reports the distribution. The trained text is
       always the exact bytes the model saw; canonicalization is used for grouping
       only, never written into the data.
    2. A TOKEN-LEVEL LOSS MASK. These are multi-turn agentic episodes: the
       "response" interleaves assistant actions with terminal output the model did
       not produce. Scoring whole suffixes would (a) credit the policy for
       predicting environment output and (b) push DOWN the observations of a failed
       episode, which are facts, not choices. So each side carries the same
       verified `loss_mask` the SFT path uses, and the trainer sums logprobs over
       mask==1 only.
    3. SAME MODEL ON BOTH SIDES, by default. If chosen came from Qwen3.5-27B and
       rejected from gpt-oss-120b, part of what DPO learns is the difference
       between two decoding styles rather than between solving and failing.
       `--cross-model` allows the mixed pairs and the manifest counts them.
    4. FORMAT CONTROL. Both sides go through the same `normalize_assistant()` used
       to build the SFT data, and a side whose action JSON does not parse is
       dropped. This deliberately removes the "emitted unparseable garbage" failure
       mode: keeping it would make the pair differ in FORMATTING, and DPO would
       spend its capacity on something SFT already taught. The drop is counted per
       side (`dropped_unparseable_rejected`) because that count IS the bias.

LENGTH BIAS -- CHECK THE MANIFEST, DO NOT ASSUME
    Standard DPO compares SUMS of token logprobs, so the objective is biased toward
    shorter sequences. The worry is that failures are systematically longer (an agent
    that cannot solve a task keeps flailing until the step limit), which would make
    "prefer the shorter trajectory" correlate with "prefer the winner" and let a run
    look like competence while it learns brevity.

    MEASURED on the built pairs, that confound is small here: 46% of pairs have the
    longer side rejected and the median rejected/chosen supervised-token ratio is
    0.98 -- i.e. the two sides are about the same length. Failed episodes do run
    longer in the raw pool, but the pairing pipeline removes most of that (an
    over-long side is dropped, not truncated, and failures lose more rows to the
    format gate). The mechanism is still real, so this script always reports both
    length distributions and sets `length_bias_warning` when the numbers move;
    `19_train_dpo.py --length-normalize` is the mitigation if they do.

OUTPUT
    <out-dir>/dpo_train.parquet     chosen/rejected input_ids + loss_mask
    <out-dir>/dpo_holdout.parquet   split by task_group_id, so no group leaks
    <out-dir>/manifest.json         every counter, both length distributions
    <out-dir>/reconstructed.jsonl.gz   cache; reading 23 GB of tars takes ~20 min,
                                       so pairing knobs can be retuned in seconds
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
_SIBLINGS: dict[str, Any] = {}


def _load_sibling(stem: str) -> Any:
    """Import a sibling script whose module name starts with a digit.

    Same shim as `06b_eval_offline.py`, and for the same reason: the trajectory
    reconstruction here MUST be the one that built the SFT data, and the mask MUST
    be the one that pretokenized it. A second copy of either would drift.
    """
    if stem in _SIBLINGS:
        return _SIBLINGS[stem]
    path = HERE / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.lstrip("0123456789_") or stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _SIBLINGS[stem] = module
    return module


def _order_key(seed: int, trajectory_id: str) -> str:
    return hashlib.sha256(f"{seed}:{trajectory_id}".encode("utf-8")).hexdigest()


# Volatile identifiers that make two runs of the SAME task look like two different
# prompts. Both patterns are anchored so they cannot eat task content: the UUID
# form is the full 8-4-4-4-12 hex layout Harbor gives its containers, and the short
# form only matches a 12-hex docker hostname immediately after an "@".
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                      r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_DOCKER_HOST_RE = re.compile(r"(?<=@)[0-9a-f]{12}(?=[:\s])")


def canonical_prompt(text: str) -> str:
    """Mask per-run container identity so one task hashes to one prompt.

    Used ONLY as a grouping key. The text written to the dataset is always the
    verbatim prompt the model actually conditioned on.
    """
    return _DOCKER_HOST_RE.sub("<HOST>", _UUID_RE.sub("<UUID>", text))


def cache_signature(traj_root: Path, *, per_side: int, seed: int, cross_model: bool) -> str:
    """Identity of a reconstruction cache: which trajectories it should contain."""
    return hashlib.sha256(
        json.dumps({"traj_root": str(traj_root.resolve()), "per_side": per_side,
                    "seed": seed, "cross_model": cross_model}, sort_keys=True).encode()
    ).hexdigest()


def read_cache(path: Path, signature: str) -> tuple[dict[str, dict], Counter] | None:
    """Return the cached reconstruction, or None if absent/stale/corrupt.

    The reconstruction counters ride along in the header so a cache hit reports the
    same manifest numbers as a cold build -- a cache that quietly zeroed
    `drop_unparseable` would make the format bias disappear from the record.
    """
    if not path.is_file():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            return None
        try:
            meta = json.loads(header)
        except json.JSONDecodeError:
            return None
        if meta.get("signature") != signature:
            return None
        built: dict[str, dict] = {}
        for line in handle:
            record = json.loads(line)
            built[record["trajectory_id"]] = record
    return built, Counter(meta.get("counters", {}))


def write_cache(path: Path, signature: str, built: dict[str, dict],
                counters: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"signature": signature,
                                 "n": len(built),
                                 "counters": {k: int(v) for k, v in sorted(counters.items())}},
                                sort_keys=True) + "\n")
        for tid in sorted(built):
            handle.write(json.dumps(built[tid], sort_keys=True) + "\n")
    tmp.replace(path)


def common_prefix_len(left: list[int], right: list[int]) -> int:
    n = min(len(left), len(right))
    i = 0
    while i < n and left[i] == right[i]:
        i += 1
    return i


# ------------------------------------------------------------------- selection

def select_candidates(frame, *, per_side: int, seed: int, cross_model: bool):
    """Pick success/failure candidates per group, keeping same-model pairs likely.

    The selection is per (group, model) FIRST, because a pair whose two sides come
    from one model isolates the success signal. Only if `cross_model` is set do we
    top up from the group at large.
    """
    frame = frame.copy()
    frame["_key"] = [_order_key(seed, tid) for tid in frame["trajectory_id"]]
    frame["_win"] = frame.reward >= 1.0
    picked: list[Any] = []

    for _group, rows in frame.groupby("task_group_id", sort=True):
        rows = rows.sort_values("_key")
        taken_s: list[Any] = []
        taken_f: list[Any] = []
        for model in sorted(rows.model_name.unique()):
            sub = rows[rows.model_name == model]
            wins = list(sub[sub._win].index)
            losses = list(sub[~sub._win].index)
            # Only worth reading if this model has both sides in this group.
            if not wins or not losses:
                continue
            budget = min(per_side, len(wins), len(losses))
            taken_s.extend(wins[:budget])
            taken_f.extend(losses[:budget])
        if cross_model:
            have = set(taken_s) | set(taken_f)
            wins = [i for i in rows[rows["_win"]].index if i not in have]
            losses = [i for i in rows[~rows["_win"]].index if i not in have]
            room = max(0, per_side - min(len(taken_s), len(taken_f)))
            taken_s.extend(wins[:room])
            taken_f.extend(losses[:room])
        picked.extend(taken_s)
        picked.extend(taken_f)

    return frame.loc[picked].drop(columns=["_key", "_win"])


# ------------------------------------------------------------------- pairing

def form_pairs(members: list[dict], *, cap: int, cross_model: bool) -> list[tuple[dict, dict]]:
    """Pair successes with failures inside ONE prompt, same model first."""
    wins = [m for m in members if m["win"]]
    losses = [m for m in members if not m["win"]]
    pairs: list[tuple[dict, dict]] = []

    by_model_w: dict[str, list[dict]] = defaultdict(list)
    by_model_l: dict[str, list[dict]] = defaultdict(list)
    for m in wins:
        by_model_w[m["model_name"]].append(m)
    for m in losses:
        by_model_l[m["model_name"]].append(m)
    for model in sorted(set(by_model_w) & set(by_model_l)):
        w, lo = by_model_w[model], by_model_l[model]
        for i in range(min(len(w), len(lo))):
            if len(pairs) >= cap:
                return pairs
            pairs.append((w[i], lo[i]))

    if cross_model and len(pairs) < cap:
        used = {id(x) for pair in pairs for x in pair}
        w = [m for m in wins if id(m) not in used]
        lo = [m for m in losses if id(m) not in used]
        for i in range(min(len(w), len(lo))):
            if len(pairs) >= cap:
                break
            pairs.append((w[i], lo[i]))
    return pairs


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-root", type=Path, required=True,
                    help="dir containing data/*.tar and metadata/trajectories.parquet")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-side", type=int, default=4,
                    help="candidates per (group, model) per side before pairing")
    ap.add_argument("--pairs-per-prompt", type=int, default=2,
                    help="cap on pairs sharing one prompt; keeps one long task from "
                         "dominating the gradient")
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = no cap")
    ap.add_argument("--max-seq-len", type=int, default=32768,
                    help="a side longer than this is DROPPED, never truncated: a "
                         "truncated trajectory is a different trajectory")
    ap.add_argument("--max-len-ratio", type=float, default=0.0,
                    help="drop pairs whose rejected/chosen supervised-token ratio "
                         "exceeds this (0 = keep all, and report the confound)")
    ap.add_argument("--max-prompt-divergence", type=int, default=128,
                    help="drop a pair whose two prompts differ by more than this many "
                         "tokens. Measured on the built set: median 40, max 56 -- the "
                         "container id, tokenized")
    ap.add_argument("--cross-model", action="store_true",
                    help="allow pairs whose two sides come from different models")
    ap.add_argument("--holdout-groups", type=int, default=40,
                    help="task groups reserved for the holdout split; clamped to at "
                         "most --max-holdout-fraction of the groups actually paired")
    ap.add_argument("--max-holdout-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1228)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache", type=Path, default=None,
                    help="reconstruction cache path (default <out-dir>/reconstructed.jsonl.gz)")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="ignore any cache and re-read the trajectory tars")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the selection and stop before reading any tar")
    args = ap.parse_args()

    import pandas as pd

    builder = _load_sibling("03_build_sft_data")
    exporter = _load_sibling("15_export_pretokenized")

    meta = pd.read_parquet(args.traj_root / "metadata" / "trajectories.parquet")
    clean = meta[
        (meta.status == "completed")
        & meta.has_trajectory
        & (~meta.has_exception)
        & meta.reward.notna()
    ]
    # Deliberately NOT filtering on task_present_in_task_dataset: DPO never
    # materializes a task dir, so a trajectory whose task is absent from the task
    # release is still perfectly usable here. (It costs 11 groups to filter it.)
    stats = clean.groupby("task_group_id").agg(
        n=("reward", "size"), s=("reward", lambda x: int((x >= 1.0).sum())),
    )
    stats["f"] = stats.n - stats.s
    paired_groups = stats[(stats.s > 0) & (stats.f > 0)]
    pool = clean[clean.task_group_id.isin(paired_groups.index)]
    print(f"[gate] clean={len(clean):,} groups={len(stats):,} "
          f"groups_with_both={len(paired_groups):,} pool={len(pool):,}", flush=True)

    selected = select_candidates(pool, per_side=args.per_side, seed=args.seed,
                                cross_model=args.cross_model)
    n_sel_win = int((selected.reward >= 1.0).sum())
    print(f"[select] candidates={len(selected):,} "
          f"(success {n_sel_win:,} / fail {len(selected) - n_sel_win:,}) "
          f"groups={selected.task_group_id.nunique():,}", flush=True)
    if args.dry_run:
        print("[dry-run] stopping before tar extraction")
        return 0

    # ---- reconstruct conversations -----------------------------------------
    # Reading every shard costs ~20 minutes of tar I/O, and the pairing knobs below
    # are exactly the ones worth retuning. Cache the reconstruction, keyed by the
    # selection it belongs to, so a retune is seconds instead of a coffee break.
    signature = cache_signature(args.traj_root, per_side=args.per_side, seed=args.seed,
                                cross_model=args.cross_model)
    cache_path = args.cache or (args.out_dir / "reconstructed.jsonl.gz")
    stats_counter: Counter = Counter()
    built: dict[str, dict] | None = None
    if not args.refresh_cache:
        hit = read_cache(cache_path, signature)
        if hit is not None:
            built, stats_counter = hit
            print(f"[build] cache hit {cache_path} ({len(built):,} trajectories); "
                  f"--refresh-cache to re-read the tars", flush=True)

    if built is None:
        jobs: dict[str, dict[str, str]] = defaultdict(dict)
        for row in selected.itertuples():
            jobs[row.shard][row.member_prefix] = row.trajectory_id
        built = {}
        payload = [(shard, str(args.traj_root), wanted) for shard, wanted in sorted(jobs.items())]
        with ProcessPoolExecutor(max_workers=args.workers) as pool_exec:
            for records, shard_stats in pool_exec.map(builder.build_from_shard, payload):
                for record in records:
                    built[record["trajectory_id"]] = record
                stats_counter.update(shard_stats)
        print(f"[build] reconstructed={len(built):,} of {len(selected):,} "
              f"stats={dict(stats_counter)}", flush=True)
        write_cache(cache_path, signature, built, stats_counter)

    reward_of = dict(zip(selected.trajectory_id, selected.reward))
    model_of = dict(zip(selected.trajectory_id, selected.model_name))
    group_of = dict(zip(selected.trajectory_id, selected.task_group_id))

    # How the reconstruction loss splits across sides. A much higher loss on the
    # rejected side is the format bias described in the docstring, made visible.
    lost = {"chosen": 0, "rejected": 0}
    for row in selected.itertuples():
        if row.trajectory_id not in built:
            lost["chosen" if row.reward >= 1.0 else "rejected"] += 1
    print(f"[build] unreconstructed by side: {lost}", flush=True)

    # ---- group by the ACTUAL prompt, canonicalized -------------------------
    # Verbatim hashing yields one bucket per trajectory (the prompt embeds the
    # container's UUID hostname) and therefore zero pairs. See rule 1 in the module
    # docstring for the measurement and for why this substitution is safe.
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    raw_prompts: set[str] = set()
    for tid, record in built.items():
        prompt = record["messages"][0]["content"]
        raw_prompts.add(prompt)
        digest = hashlib.sha256(canonical_prompt(prompt).encode("utf-8")).hexdigest()
        by_prompt[digest].append({
            "trajectory_id": tid,
            "messages": record["messages"],
            "win": reward_of[tid] >= 1.0,
            "model_name": model_of[tid],
            "task_group_id": group_of[tid],
            "prompt_sha256": digest,
            "prompt_verbatim_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        })
    prompts_per_group: Counter = Counter()
    for members in by_prompt.values():
        prompts_per_group[members[0]["task_group_id"]] += 1
    multi = sum(1 for v in prompts_per_group.values() if v > 1)
    print(f"[prompts] verbatim={len(raw_prompts):,} -> canonical={len(by_prompt):,} "
          f"across {len(prompts_per_group):,} groups; {multi:,} groups still hold >1 "
          f"canonical prompt (real task variants -- which is exactly why pairing on "
          f"task_group_id would be wrong)", flush=True)

    candidate_pairs: list[tuple[dict, dict]] = []
    # Why the yield is what it is, in one number. Selection (above) happens per
    # (group, model) because that is all the metadata knows; pairing happens per
    # VARIANT, which is only visible after reconstruction. So a group's chosen
    # candidates can land on variant A while its rejected ones land on variant B,
    # and that bucket pairs nothing. `--per-side` is the lever: measured yield went
    # 8,024 candidates -> ~2 pairs at --per-side 2 and 18,572 -> 1,330 at
    # --per-side 5. Raising --pairs-per-prompt instead barely helps (1,330 -> 1,533
    # from 2 to 8) because the shortage is buckets with both sides, not pairs per
    # bucket. Reconstruction is cached, so a bigger --per-side costs one pass.
    one_sided = 0
    for digest in sorted(by_prompt):
        members = by_prompt[digest]
        if all(m["win"] for m in members) or not any(m["win"] for m in members):
            one_sided += 1
        candidate_pairs.extend(form_pairs(members, cap=args.pairs_per_prompt,
                                          cross_model=args.cross_model))
    print(f"[pair] candidate pairs={len(candidate_pairs):,}; {one_sided:,} of "
          f"{len(by_prompt):,} canonical prompts hold only one outcome and pair "
          f"nothing (raise --per-side to cover more variants)", flush=True)

    # ---- tokenize both sides with the verified mask -------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    if not tokenizer.is_fast:
        sys.exit("a fast tokenizer is required (the mask needs offset mapping)")

    dropped = Counter()
    rows: list[dict] = []
    for chosen, rejected in candidate_pairs:
        try:
            c_ids, c_mask = exporter.qwen3_5_mask(tokenizer, [dict(m) for m in chosen["messages"]])
            r_ids, r_mask = exporter.qwen3_5_mask(tokenizer, [dict(m) for m in rejected["messages"]])
        except ValueError:
            dropped["contract_mismatch"] += 1
            continue
        if len(c_ids) > args.max_seq_len:
            dropped["too_long_chosen"] += 1
            continue
        if len(r_ids) > args.max_seq_len:
            dropped["too_long_rejected"] += 1
            continue
        c_trained, r_trained = sum(c_mask), sum(r_mask)
        if c_trained == 0 or r_trained == 0:
            dropped["no_trained_tokens"] += 1
            continue
        # The two sides must really answer the same question. If the shared token
        # prefix is trivial, the prompt hash matched something degenerate and the
        # DPO comparison would be meaningless.
        shared = common_prefix_len(c_ids, r_ids)
        if shared < 16:
            dropped["no_common_prefix"] += 1
            continue
        # How far into the prompt the two sides actually agree. `c_first` is where
        # the chosen side's first supervised (assistant) token starts, so everything
        # before it is prompt; a divergence of 0 means the prompts are token-
        # identical and the DPO derivation holds exactly. Anything larger is the
        # canonicalized container id, and is capped rather than assumed small.
        c_first = next((i for i, m in enumerate(c_mask) if m), len(c_mask))
        divergence = max(0, c_first - shared)
        if divergence > args.max_prompt_divergence:
            dropped["prompt_divergence"] += 1
            continue
        if args.max_len_ratio and r_trained > args.max_len_ratio * c_trained:
            dropped["length_ratio"] += 1
            continue
        rows.append({
            "pair_id": hashlib.sha256(
                f"{chosen['trajectory_id']}|{rejected['trajectory_id']}".encode("utf-8")
            ).hexdigest(),
            "prompt_sha256": chosen["prompt_sha256"],
            "task_group_id": chosen["task_group_id"],
            "chosen_trajectory_id": chosen["trajectory_id"],
            "rejected_trajectory_id": rejected["trajectory_id"],
            "chosen_model": chosen["model_name"],
            "rejected_model": rejected["model_name"],
            "same_model": chosen["model_name"] == rejected["model_name"],
            "chosen_input_ids": c_ids,
            "chosen_loss_mask": c_mask,
            "rejected_input_ids": r_ids,
            "rejected_loss_mask": r_mask,
            "chosen_n_tokens": len(c_ids),
            "rejected_n_tokens": len(r_ids),
            "chosen_n_trained": c_trained,
            "rejected_n_trained": r_trained,
            "common_prefix_tokens": shared,
            "prompt_tokens": c_first,
            "prompt_divergence_tokens": divergence,
        })
        if args.max_pairs and len(rows) >= args.max_pairs:
            break
    print(f"[tokenize] pairs={len(rows):,} dropped={dict(dropped)}", flush=True)
    if not rows:
        sys.exit("no usable pairs were produced -- nothing to write")

    # ---- split by GROUP so no task group appears on both sides -------------
    rows.sort(key=lambda r: r["pair_id"])
    groups = sorted({r["task_group_id"] for r in rows},
                    key=lambda g: hashlib.sha256(f"{args.seed}:{g}".encode()).hexdigest())
    # The requested holdout size is a ceiling, not a promise: how many groups survive
    # pairing is only known here. Clamp loudly rather than either failing or silently
    # handing back a holdout that swallowed the training set.
    n_holdout = min(args.holdout_groups, int(len(groups) * args.max_holdout_fraction))
    if n_holdout < args.holdout_groups:
        print(f"[split] --holdout-groups {args.holdout_groups} clamped to {n_holdout} "
              f"({args.max_holdout_fraction:.0%} of the {len(groups):,} paired groups)",
              flush=True)
    if n_holdout == 0:
        sys.exit(f"only {len(groups)} groups paired; too few to hold any out. Raise "
                 f"--per-side, or lower --max-holdout-fraction if you accept no holdout.")
    holdout_groups = set(groups[:n_holdout])
    train = [r for r in rows if r["task_group_id"] not in holdout_groups]
    holdout = [r for r in rows if r["task_group_id"] in holdout_groups]
    if not train:
        sys.exit(f"--holdout-groups {args.holdout_groups} consumed every group; lower it")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write(subset: list[dict], name: str) -> Path:
        path = args.out_dir / name
        pd.DataFrame(subset).to_parquet(path, index=False)
        return path

    train_path = write(train, "dpo_train.parquet")
    holdout_path = write(holdout, "dpo_holdout.parquet")

    # ---- length-bias diagnostic -------------------------------------------
    import numpy as np

    c_len = np.array([r["chosen_n_trained"] for r in rows], dtype=float)
    r_len = np.array([r["rejected_n_trained"] for r in rows], dtype=float)
    rejected_longer = float((r_len > c_len).mean())
    ratio = float(np.median(r_len / np.maximum(c_len, 1.0)))
    warning = None
    if rejected_longer > 0.65 or ratio > 1.5:
        warning = (
            f"{rejected_longer:.0%} of pairs have a LONGER rejected side (median "
            f"rejected/chosen supervised-token ratio {ratio:.2f}). Summed-logprob DPO "
            f"is biased toward shorter sequences, so a run on this data can lower the "
            f"loss by learning brevity rather than competence. Train with "
            f"19_train_dpo.py --length-normalize, or rebuild with --max-len-ratio, and "
            f"report which you did."
        )

    manifest = {
        "source_dataset": "Zhongzhi1228/Recursive-Task-Synthesis-Trajectories",
        "tokenizer": str(args.tokenizer),
        "clean_trajectories": int(len(clean)),
        "groups_with_any_clean_data": int(len(stats)),
        "groups_with_both_outcomes": int(len(paired_groups)),
        "candidates_selected": int(len(selected)),
        "candidates_reconstructed": int(len(built)),
        "unreconstructed_by_side": lost,
        "distinct_prompts_verbatim": int(len(raw_prompts)),
        "distinct_prompts_canonical": int(len(by_prompt)),
        "groups_with_multiple_prompts": int(multi),
        "prompt_canonicalization": "UUIDs and 12-hex docker hostnames masked for "
                                   "GROUPING ONLY; the trained text is verbatim",
        "candidate_pairs": int(len(candidate_pairs)),
        "one_sided_prompts": int(one_sided),
        "pairs_final": int(len(rows)),
        "pairs_train": int(len(train)),
        "pairs_holdout": int(len(holdout)),
        "holdout_groups": sorted(holdout_groups),
        "same_model_pairs": int(sum(r["same_model"] for r in rows)),
        "cross_model_allowed": bool(args.cross_model),
        "dropped": {k: int(v) for k, v in sorted(dropped.items())},
        "reconstruction_counters": {k: int(v) for k, v in sorted(stats_counter.items())},
        "length": {
            "chosen_trained_p50": float(np.quantile(c_len, 0.5)),
            "chosen_trained_p90": float(np.quantile(c_len, 0.9)),
            "rejected_trained_p50": float(np.quantile(r_len, 0.5)),
            "rejected_trained_p90": float(np.quantile(r_len, 0.9)),
            "fraction_rejected_longer": round(rejected_longer, 4),
            "median_rejected_over_chosen": round(ratio, 4),
            "max_total_tokens": int(max(max(r["chosen_n_tokens"], r["rejected_n_tokens"])
                                        for r in rows)),
        },
        "prompt_agreement": {
            "pairs_token_identical_prompt": int(sum(r["prompt_divergence_tokens"] == 0
                                                    for r in rows)),
            "divergence_tokens_p50": float(np.quantile(
                [r["prompt_divergence_tokens"] for r in rows], 0.5)),
            "divergence_tokens_max": int(max(r["prompt_divergence_tokens"] for r in rows)),
            "note": "tokens between the first chosen/rejected mismatch and the start of "
                    "the response. Nonzero means the two runs had different container "
                    "hostnames; that text is masked context on both sides, never a target.",
        },
        "length_bias_warning": warning,
        "model_mix_chosen": dict(Counter(r["chosen_model"] for r in rows)),
        "model_mix_rejected": dict(Counter(r["rejected_model"] for r in rows)),
        "mask_source": "slime/utils/mask_utils.py::gen_multi_turn_loss_mask_qwen3_5 "
                       "(via scripts/15_export_pretokenized.py)",
        "schema": {
            "chosen_input_ids/rejected_input_ids": "list[int], whole-conversation render",
            "chosen_loss_mask/rejected_loss_mask":
                "list[int] aligned 1:1 with input_ids, no offset. 1 = a token the "
                "POLICY produced, so it enters the DPO logprob sum. 0 = prompt, "
                "harness text or terminal output, which must not.",
            "common_prefix_tokens": "shared leading tokens; proof both sides answer "
                                    "the same prompt",
        },
        "seed": args.seed,
        "train_parquet": str(train_path),
        "holdout_parquet": str(holdout_path),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if warning:
        print(f"\nWARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
