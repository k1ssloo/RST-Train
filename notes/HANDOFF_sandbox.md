# Handoff: the pod cannot mount(2), and RL does not have to wait for that

Written 2026-08-18, in reply to a field report from the training cluster. Forward
this whole file; it is meant to be read by whoever is running the job there.

## Your diagnosis is right, and the repo was wrong

You found: podman 5.8.4 / crun 1.28 / conmon / netavark all installed, `podman info`
clean, blobs pulling through the proxy, and then

    ApplyLayer ... remount /, flags: 0x44000: permission denied

with `unshare -U` / `unshare -Ur` succeeding, every `mount` returning EACCES even
inside your own user namespace, `/proc/self/attr/current = docker-default (enforce)`,
and `Seccomp: 0`.

That is correct and complete. `0x44000` is `MS_REC|MS_PRIVATE`: containers/storage
isolating mount propagation, not tar extraction failing. EACCES rather than EPERM is
the LSM signature — a missing capability gives EPERM.

You were also right that the repo's advice ("ask for the podman + uidmap packages")
was useless. That has been fixed: `OPERATOR_PROMPT.md` and `BACKENDS.md` now carry
the AppArmor case, and `scripts/00b_setup_sandbox.sh --diagnose` runs the probes
you ran by hand (`00c_probe_sandbox.py` calls `unshare(2)`/`mount(2)` directly and
reads the AppArmor profile) so the next operator gets the verdict in one command
instead of 12 minutes of bisection.

## One thing to add: there is no in-pod workaround, at all

Worth knowing so you don't spend time on it. AppArmor profiles are inherited across
`unshare()` **and** `execve()`, `docker-default` contains a literal `deny mount,`
(it allows only `umount`), and it grants no `change_profile` rule — so nothing you
run can transition out of the profile. fuse-overlayfs, vfs, bubblewrap, a newer
podman, a different storage driver: all of them mount, so all of them are dead. Stop
at the diagnosis; it is final.

## Trim the ops ask to one flag

Ask for **only** this:

    --security-opt apparmor=unconfined
    # k8s >= 1.30:  securityContext.appArmorProfile.type: Unconfined
    # k8s <  1.30:  container.apparmor.security.beta.kubernetes.io/<name>: unconfined

Drop `--cap-add SYS_ADMIN` and `/dev/fuse` from the request:

- SYS_ADMIN buys nothing. Inside the user namespace you already hold full
  capabilities — that is why `unshare -Ur` works — and AppArmor is overriding them.
  It would override SYS_ADMIN too.
- `/dev/fuse` is only for fuse-overlayfs. The vfs storage driver needs no FUSE, and
  vfs already initialized fine for you.
- seccomp needs no change; you measured `Seccomp: 0`.

This matters practically: a one-flag request that names the exact syscall and the
exact profile gets approved far more often than a four-item request that reads like
"give me privileged". Mentioning that `apparmor=unconfined` still leaves user
namespaces, seccomp and the capability bounding set in place usually helps too.

## Answering "AppArmor request now, or SFT first?": both, in this order

1. **File the AppArmor request now.** It has queue latency you cannot compress, and
   it is the cheapest long-term fix. Send it and stop waiting on it.
2. **Start SFT immediately.** It needs no container runtime whatsoever. Nothing
   about this blocker touches it.
3. **Unblock eval and RL without ops, in parallel** — see below. Do not treat the
   AppArmor request as the critical path for RL.

## RL does not actually need this pod to run containers

Harbor 0.21.0 ships ~25 environment backends. Several build the task's own
`environment/Dockerfile` **on someone else's infrastructure**, so a pod with zero
container privilege is enough:

| `RST_HARBOR_ENV` | needs | builds the RST Dockerfile |
|---|---|---|
| `daytona` | outbound HTTPS + `DAYTONA_API_KEY` | yes — declarative build → content-hash snapshot. **This is what the RST paper used.** |
| `e2b` | outbound HTTPS + `E2B_API_KEY` | yes — `Template().from_dockerfile` → `AsyncTemplate.build` |
| `modal` | outbound HTTPS + a Modal token | yes — `Image.from_dockerfile` |
| remote Docker daemon | a reachable `DOCKER_HOST`, plus the task tree at the **same absolute path** on the daemon host (Harbor bind-mounts it) | yes |
| `gke`/`ack`/`openshift` | a namespace + RBAC for sibling pods | yes |

Not usable for RST tasks: `hf-sandbox` and `singularity` both want a prebuilt image
(or `.sif`) and refuse a Dockerfile.

**Why this works without any network exposure:** terminus-2 is a *host-side* agent.
It drives the container through `environment.exec(...)`, so the agent loop and every
model call stay in the training pod, hitting the local vLLM/SGLang shim. No reverse
tunnel, nothing inbound to the pod, and `--network none` inside the task container is
preserved. The only new egress is HTTPS to the provider API — which your proxy
already carries, since you pulled image blobs through it.

To use it:

    export RST_HARBOR_ENV=daytona DAYTONA_API_KEY=...
    # or let 00b pick it up from the credentials it finds:
    source scripts/00b_setup_sandbox.sh

`06_eval.py`, `rl/generate.py` and `verl_backend/harbor_agent_loop.py` all honour
`RST_HARBOR_ENV` (plus `RST_HARBOR_ENV_KWARGS`, space-separated `K=V` passed through
as `--environment-kwarg`). Two things change with an off-machine backend, both
already handled:

- **Proxy handling inverts.** `--env docker` needs `HTTP(S)_PROXY` stripped; an
  off-machine backend needs it kept, with the local shim added to `NO_PROXY`. Note
  that Ray's `runtime_env` *replaces* the worker environment rather than extending
  it, so `12_run_grpo.sh` names the proxy vars explicitly in `RUNTIME_ENV_JSON`.
- **Prebuilding is skipped, not failed.** `11_prebuild_images.py` cannot warm a cache
  it does not own, so it writes `prebuild_report.json` with `"skipped": true` plus a
  reason and exits 0. The first rollout per distinct image pays the provider-side
  build once; after that the snapshot is reused. Also: `RST_MAX_SANDBOXES` is now the
  provider's **concurrency quota**, not a function of local cores/RAM. Set it from
  your plan's actual limit — over it you get 429s, which the rollout code classifies
  as infrastructure failures and drops.

## On "we must report that evaluation could not be performed"

Right, and thank you for saying it plainly. That rule is now enforced mechanically
rather than left to judgement:

- `scripts/06b_eval_offline.py` is new: container-free evaluation. Held-out loss,
  perplexity and next-token top-1 over the supervised spans only, a base-vs-tuned
  delta (the only number that answers "did SFT do anything"), and greedy
  action-protocol agreement — parse rate, first-keystroke match, command-list match,
  `is_task_complete` agreement. It reuses `normalize_assistant()` from
  `03_build_sft_data.py` and `qwen3_5_mask()` from `15_export_pretokenized.py`
  by path import, so it cannot drift from the training contract.

      python scripts/06b_eval_offline.py \
          --model-path $BASE_FOLDER/out-hf-full \
          --base-model $BASE_FOLDER/$MODEL_DIR_NAME \
          --holdout    $DATA_DIR/rst_sft_holdout.parquet \
          --tokenizer  $BASE_FOLDER/$MODEL_DIR_NAME \
          --out        $BASE_FOLDER/eval/offline

  `--dry-run` validates the parquet and prints the plan without loading a model.
  Give it `pretokenized_holdout.parquet` instead and you get the loss half only,
  with the action probe marked unavailable-and-why rather than silently absent.
  Two details worth knowing: it never materializes `[1, T, V]` logits (32k × ~152k ×
  2 B ≈ 10 GB for one row) — it runs the decoder and applies `lm_head` in slices; and
  it counts truncated generations separately, because a generation that hit the token
  cap is evidence about `--gen-tokens`, not evidence that the model lost the protocol.

- `scripts/14_make_report.py --offline-eval <offline_results.json>` renders those
  numbers under a heading that states they are not a benchmark, **and still records a
  FAIL** on benchmark coverage. That FAIL sets `in_range: false` in `verdict.json`,
  which is what `20_run_all.sh` gates RL on. Previously a run with no eval results
  produced only a WARN and a green verdict — a run that never measured anything could
  read as a checkpoint that was fine. That hole is closed.

- `20_run_all.sh` runs the offline eval automatically whenever `00b_setup_sandbox.sh`
  reports no sandbox, and prints the not-run banner.

Be aware of the limit in the other direction too: teacher-forced agreement with a
recorded expert cannot tell you whether the agent recovers from a wrong command,
which is most of the difficulty in terminal work. Report these as a sanity check
that training did what the data asked, never as a pass rate.

## Two unrelated things from your report

- **Trained-token fraction 32.4217% vs 32.42% expected** — that is the expected
  value rounded; nothing is wrong.
- **The manifest/handoff contradiction you caught was real, and you resolved it
  correctly.** HF CausalLM and Liger's fused CE both shift internally, so the right
  rule is `labels[i] = input_ids[i] where loss_mask[i]==1 else -100` with **no**
  shift. The manifest sentence telling trainers to "apply the usual shift-by-one
  themselves" was wrong on the verl/HF path and would have double-shifted. Fixed in
  `15_export_pretokenized.py`, in both local manifests, and on the Hub (commit
  `448bce17549bb9c57844a1f515aac8b105b4e677`). Decoding actual tokens to settle it
  was the right call; keep doing that.

## One operational warning

If ops fixes this by giving you a **new** pod, everything not on a PVC is gone: the
8–9 GB conda env, the downloaded model, the data. Confirm what is on persistent
storage before you accept a pod recreation, or ask them to apply the flag to the
existing pod spec for the next restart while you keep working in this one.
