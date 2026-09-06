# TODO

Open items from the 2026-09-03 review. **Engineering** items make the existing loop more
measurable or more correct; they are not the research contribution. **Algorithmic** items are
candidates until one has a measured result behind it. Every number quoted here comes from ONE
trajectory shard (`trajectories-00000.tar`, 5,000 trials, four solver models mixed) unless
stated otherwise -- re-measure on all 66 shards before any of it goes into `README.md` or `BUG.md`.

## Engineering -- measurement (CPU only, data already on disk)

- [ ] Failure taxonomy of the release, all 66 shards, split per solver model: premature
      completion (final step claims `task_complete`, reward 0), zero-progress (0 checks passed),
      horizon (episode cap), exception. Shard 0, clean failures: 91.9% / 5.0% / 0.0%.
      Extend per synthesis round once round provenance is recovered (`tasks.parquet` carries no
      round column; `rl-sweet` has 4,860 `rewrite_target.json`).
- [ ] Per-check failure profile from `verifier/ctrf.json`: which contract check fails most
      (01 required_evidence / 02 intermediate_artifact / 03 final_semantics / 04 no_shortcut) and
      whether pass/fail vectors are prefix-shaped. Shard 0: top two are `no_shortcut` and
      `final_semantics`; median partial credit at failure 0.60.
- [ ] Template-overfitting index: SFT gain on held-out RST-style tasks (`Quality-1K`) divided by
      the gain on TB2 / TB-Hard.
- [ ] Behaviour-level diversity of our own rollouts: `command_signature` entropy, share of
      trajectories that write unprompted report/validation files, PASS-line echo rate.
- [ ] Lineage-aware dedup of the SFT corpus (paper: R15 children keep ~82% of the parent
      solution tokens): SFT gain per training token vs the flat mixture.

## Engineering -- borrow from `alexhuang13/viewer` (read-only audit UIs, NOT the synthesis code)

- [ ] `03g_curate_sft.py`: add `verifier_fresh` (last mutation precedes last verification),
      recovery-chain status, and an anti-tampering hard drop (a command touches `tests/` or the
      verifier). Borrow the event extraction, not the rubric scores -- those were never validated
      against human labels (their own docs say so).
- [ ] `03h_build_rollout_sft.py`: follow `continued_trajectory_ref` instead of glob-sorting
      `trajectory*.json`, and report an incomplete chain. Harbor's terminus_2 writes
      `trajectory.cont-N.json` through that ref. Local rollouts so far are single-segment, so this
      is unexercised, not verified. `BUG.md` OPEN candidate.
- [ ] `rst_common/harbor.py`: a second command line for oracle re-validation (`agents: oracle`,
      `n_attempts: 1`) so an audited or repaired environment is proven solvable before it
      re-enters the pool.
- [ ] `17_build_dpo_data.py`: record the first-divergence turn (per-turn command Jaccard against
      the paired trajectory) as pair metadata.
- [ ] Loop B on-disk contract: emit `rewrite_records.jsonl` and
      `validation_round_N/validation_results.jsonl` in the viewer's shape so its Task Viewer
      works on our runs unchanged.

## Algorithmic -- research candidates (none measured yet)

Context. The paper's RSI is two uncoupled processes: the task pool evolves by fixed operators +
oracle acceptance + pass-rate-band reseeding; the policy is updated by success-only rejection
sampling (SFT) and PPO whose reward is *already* shaped by verifier-check count (paper section 5.5,
"customized reward shaping"). So "partial credit instead of binary reward" is NOT a contribution.
The candidates below couple the two processes; each is an objective or an update rule, not tooling.

- [ ] Teacher policy over (solver failure profile, lineage) -> (operator, contract stage to harden)
      as a contextual bandit with a learning-progress / ZPD objective, replacing family caps +
      inverse-frequency selection. Evidence (rl-sweet, 2,963 provenance-tagged tasks, 4 solver
      models mixed, selection-biased to the sweet band): per-operator mean pass rate spans
      0.15..0.54 (3.6x), operator explains R^2=0.10 of pass-rate variance, family explains 0.002 --
      the paper balances at the family level, which is blind to solver difficulty.
- [ ] Lineage hindsight verification: score a failed child rollout with the verifiers of its
      ancestors; deepest ancestor passed = progress. Real, oracle-validated verifiers rather than the
      designer's within-task check split. Precondition to measure first: share of (child, parent)
      pairs where the child's oracle solution passes the parent verifier (path/workdir operators
      will break it). Do NOT relabel as SFT positives (prompt differs); use as RL progress + ZPD test.
- [ ] Calibrated closure: `task_complete` is an action. r = r_verifier - lambda * 1[claim] *
      (1 - r_verifier) plus a budget cost; before claiming, the policy predicts the check vector,
      scored against ctrf with a proper scoring rule (327k labelled trials exist). Test-time stopping
      rule where no verifier exists (TB2). Evidence, shard 0: 91.9% of clean failures claim
      completion at median partial credit 0.60; 35% of scheme-named failures pass final_semantics
      but fail process / no_shortcut checks.
- [ ] Progressive disclosure curriculum: anneal how much of the acceptance contract is visible in
      `instruction.md` (visible -> partial -> original hidden form) by solver success; edits one file,
      no regeneration, final stage is the unmodified task so nothing leaks.
- [ ] Solver-relative difficulty k*: execute the first k fraction of the task's oracle `solve.sh`
      (cut at contract-stage / command boundaries), hand the transcript to the policy, let it finish;
      k* = smallest k at which the policy succeeds, found by bisection over k with a few rollouts.
      Uses: (a) reverse curriculum -- train at k*-delta and anneal to 0 (Florensa 2017 / Salimans &
      Chen 2018 / Backplay, with RST's oracle as the free demonstration); (b) acceptance + designer
      objective on k* (target band; reward dk*/dlength, i.e. difficulty per added command, not
      length); (c) the jump in the success-vs-k curve localises the decisive step to an oracle line
      range = the subgoal to harden or to disclose. Prediction to test: late-round RST tasks have k*
      concentrated near the end of the oracle (consistent with 60% partial credit at failure).
      First experiment: ~40 sweet-pool tasks x 4 k-values x 2 rollouts with the base model, rootless
      docker; ~13 h at concurrency 8.
- [ ] Dropped: task-blind reward debiasing (bias-model baseline / min-max acceptance). Shard 0,
      Qwen3.5-27B only, 4,429 clean trials: a task-blind logistic model predicts reward with AUC
      0.71, but the signal is length (shorter => success; log_cmd_chars alone 0.70) and the style
      features (report/PASS//app/ ratios) are weaker (0.65) and NEGATIVELY associated with success.
      That is difficulty, not an exploitable proxy; a length baseline is what GRPO's per-group
      baseline already provides.
- [ ] Dropped: strict prefix-ordered stage reward. Only 25.6% of shard-0 failures carry the numbered
      check scheme and only 64.7% of those pass-vectors are prefix-shaped (101 alone is 18.7%).
- [ ] Prior art to read before claiming any of the above: SETA, CLI-Universe, R-Zero, PAIRED / ACCEL /
      POET, LHTB dense rewards, Fu et al. 2025 (self-verification prevents collapse), BenchEvolver,
      TRACE, HER.

See `notes/RSI_ALGORITHM_CANDIDATES_ZH.md` for the formalisation and the three algorithm-level candidates.
