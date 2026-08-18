#!/usr/bin/env bash
# Find a place to run task containers for eval + RL rollouts, WITHOUT needing
# permission to use the host Docker daemon.
#
#   source scripts/00b_setup_sandbox.sh            # exports RST_HARBOR_ENV etc.
#   bash   scripts/00b_setup_sandbox.sh --check    # select, then live smoke test
#   bash   scripts/00b_setup_sandbox.sh --diagnose # capability report only, no selection
#
# WHY THIS EXISTS
#   SFT needs no sandbox at all. Evaluation and RL rollouts do: Harbor builds each
#   task's Dockerfile and drives a tmux session inside the container.
#
# THERE ARE THREE PLACES A SANDBOX CAN LIVE, and this script picks the first that
# works. Only the first needs any container privilege on this machine:
#
#   local          rootless podman (or a non-default docker socket) right here
#   remote daemon  a Docker-API endpoint on another host you can reach
#   managed        Daytona / E2B / Modal — Harbor builds the task's Dockerfile on
#                  the provider's infrastructure and runs the sandbox there
#
#   The managed row matters more than it looks. Harbor 0.21.0 has first-class
#   backends for it (harbor/environments/{daytona,e2b,modal}.py) and all three
#   build `environment/Dockerfile` server-side -- Daytona into a content-hashed
#   snapshot, E2B into a template, Modal via Image.from_dockerfile. So a pod with
#   NO container capability whatsoever is still enough to run eval and RL. All it
#   needs is outbound HTTPS and an API key. Daytona is what the RST paper used.
#
# WHEN A LOCAL RUNTIME LOOKS INSTALLED AND STILL DOES NOT WORK
#   Read the mount probe below before asking for packages. If the pod is confined
#   by an LSM (AppArmor's stock `docker-default` contains a literal `deny mount,`)
#   then podman, uidmap, subuid entries and podman version are all irrelevant:
#   `podman info` succeeds, blobs download, and unpacking a layer dies with
#   `ApplyLayer ... remount /, flags: 0x44000: permission denied`. There is no
#   in-container workaround -- the profile is inherited across unshare() and
#   execve() and permits no transition. Either the pod gets relaunched, or the
#   sandbox moves off this machine. `--diagnose` tells you which case you are in
#   and prints the exact, minimal thing to ask for.
#
# VERIFIED for the local path on a box with no docker.sock permission
# (podman 3.4.4, cgroup v2, rootless, subuid configured):
#   * `podman build` of a real RST task Dockerfile: 26.6 s, including
#     `apt-get install build-essential git curl`, a `git clone`, and `pip install`.
#     Rootless builds run as uid 0 inside a user namespace, so apt works.
#   * docker CLI -> podman socket: client 29.2.1 / server 3.4.4 / API 1.40
#   * `docker run -d --network none`, then `docker exec`: OK
#   * tmux 3.2a inside: new-session / send-keys / capture-pane all OK
#   * no network inside the container: confirmed
#
# THREE THINGS THAT ONLY SHOW UP WHEN YOU ACTUALLY BUILD THE TASK IMAGES, all
# found by doing exactly that on the real pool (local path only):
#   1. 16% of task Dockerfiles use `SHELL`, which podman's default OCI image format
#      ignores and then errors on -> always build with `--format docker`.
#   2. 31% use heredoc (`RUN <<EOF`), which needs podman >= 4.4 -> see the version
#      gate below. podman 3.4.4 reports "Unknown instruction: IF", which reads like
#      a broken task rather than an old toolchain.
#   3. A `git clone` inside a build can fail with "hardlink different from source"
#      under rootless kernel overlay -> retry in a vfs store under a separate
#      --root. Verified to fix a real failing task. scripts/11_prebuild_images.py
#      does this retry automatically.
#
# SECURITY NOTE, and it is an improvement rather than a compromise: task
# Dockerfiles are untrusted third-party build scripts. Rootless podman is a
# *better* place to build them than a root Docker daemon -- there is no privileged
# daemon to talk to, and a container escape lands in an unprivileged user
# namespace. A managed provider is better still: the build never touches cluster
# hardware at all. The earlier guidance ("use a dedicated daemon, never the
# host's") is satisfied more strongly by both, not waived.
set -uo pipefail

# A mode is read from the command line ONLY when this file was executed as a
# program. 20_run_all.sh *sources* it (it has to: RST_HARBOR_ENV has to survive into
# the caller), and a sourced script sees the PARENT's positional parameters. So
# `bash scripts/20_run_all.sh --check` used to make this script exit 0 in the middle
# of the orchestration, and `--diagnose` used to `exec python` over the whole run.
MODE=select
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-}" in
    --check)    MODE=check ;;
    --diagnose) MODE=diagnose ;;
  esac
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
: "${XDG_RUNTIME_DIR:=/run/user/$(id -u)}"
export XDG_RUNTIME_DIR
PODMAN_SOCK="$XDG_RUNTIME_DIR/podman/podman.sock"
PROBE_JSON="${TMPDIR:-/tmp}/rst_sandbox_probe.json"

log() { echo "[sandbox] $*"; }

# ---------------------------------------------------------------- diagnose mode
if [[ "$MODE" == diagnose ]]; then
  exec python "$HERE/00c_probe_sandbox.py" --out "$PROBE_JSON"
fi

# Can this machine mount at all? Everything local depends on it, and the answer
# changes what you should ask for. Cached, because it forks and unshares.
mount_ok() {
  [[ -f "$PROBE_JSON" ]] || python "$HERE/00c_probe_sandbox.py" --json --out "$PROBE_JSON" >/dev/null 2>&1
  python - "$PROBE_JSON" <<'EOF_PY' 2>/dev/null
import json, sys
try:
    f = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)          # probe unavailable: do not block, just try podman
m = f.get("mount", {})
bad = [k for k in ("make_rprivate", "tmpfs")
       if isinstance(m.get(k), dict) and not m[k].get("ok")]
if not bad:
    sys.exit(0)
first = m[bad[0]]
print(f"{bad[0]} -> {first.get('errno_name')} ({first.get('strerror','')})")
prof = f.get("confinement", {}).get("apparmor_profile", "?")
if first.get("errno_name") == "EACCES":
    print(f"LSM denial; apparmor profile = {prof}")
sys.exit(1)
EOF_PY
}

RST_CONTAINER_RUNTIME=""
RST_DOCKER_HOST=""
RST_BUILD_CMD=""
RST_HARBOR_ENV="${RST_HARBOR_ENV:-}"
RST_HARBOR_ENV_KWARGS="${RST_HARBOR_ENV_KWARGS:-}"
RST_SANDBOX_LOCATION=""

# --------------------------------------------------- 0. explicit operator choice
# Set RST_HARBOR_ENV yourself to force a backend -- including ones this script
# will not auto-select because they need a decision it cannot make for you
# (gke/ack/openshift need a namespace and RBAC; singularity needs prebuilt
# images). Anything other than "docker" is treated as off-machine.
if [[ -n "$RST_HARBOR_ENV" && "$RST_HARBOR_ENV" != docker ]]; then
  RST_SANDBOX_LOCATION=preset
  log "using operator-selected harbor env: --env $RST_HARBOR_ENV ${RST_HARBOR_ENV_KWARGS:+(kwargs: $RST_HARBOR_ENV_KWARGS)}"
  case "$RST_HARBOR_ENV" in
    daytona) [[ -n "${DAYTONA_API_KEY:-}${DAYTONA_JWT_TOKEN:-}" ]] || log "WARNING: DAYTONA_API_KEY is not set" ;;
    e2b)     [[ -n "${E2B_API_KEY:-}"    ]] || log "WARNING: E2B_API_KEY is not set" ;;
    modal)   [[ -n "${MODAL_TOKEN_ID:-}" && -n "${MODAL_TOKEN_SECRET:-}" ]] || log "WARNING: MODAL_TOKEN_ID/MODAL_TOKEN_SECRET are not both set" ;;
    hf-sandbox)
      [[ -n "${HF_TOKEN:-}${HUGGINGFACE_HUB_TOKEN:-}" ]] || log "WARNING: HF_TOKEN is not set"
      log "NOTE: hf-sandbox requires a PREBUILT image per task ([environment].docker_image)."
      log "      RST tasks ship a Dockerfile and no docker_image, so you must build and"
      log "      push each image to a registry first. Prefer daytona/e2b/modal, which"
      log "      build the Dockerfile for you." ;;
    singularity)
      log "NOTE: harbor's singularity backend converts a docker image to .sif; it cannot"
      log "      read a Dockerfile. Set [environment].docker_image per task, or pass"
      log "      RST_HARBOR_ENV_KWARGS='singularity_image_cache_dir=/path/to/sif-cache'"
      log "      with pre-converted .sif files." ;;
  esac
fi

# ---------------------------------------------------------------- 1. real docker
if [[ -z "$RST_SANDBOX_LOCATION" ]] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  # Only accept a NON-default socket: building untrusted Dockerfiles on the shared
  # host daemon is what we are avoiding. If DOCKER_HOST is unset or points at the
  # host socket, fall through.
  if [[ -n "${DOCKER_HOST:-}" && "$DOCKER_HOST" != "unix:///var/run/docker.sock" ]]; then
    RST_CONTAINER_RUNTIME=docker
    RST_DOCKER_HOST="$DOCKER_HOST"
    RST_BUILD_CMD=docker
    RST_HARBOR_ENV=docker
    RST_SANDBOX_LOCATION=local
    log "using docker at $DOCKER_HOST"
  else
    log "docker reachable but only via the shared host socket; looking for something else"
  fi
fi

# ------------------------------------------------- 2. a Docker daemon on another host
# The client side of the Docker API does no mounting, so this works from a fully
# mount-less pod. Set RST_REMOTE_DOCKER_HOST=tcp://host:2375 or ssh://user@host.
if [[ -z "$RST_SANDBOX_LOCATION" && -n "${RST_REMOTE_DOCKER_HOST:-}" ]]; then
  if command -v docker >/dev/null 2>&1 && DOCKER_HOST="$RST_REMOTE_DOCKER_HOST" docker version >/dev/null 2>&1; then
    RST_CONTAINER_RUNTIME=docker
    RST_DOCKER_HOST="$RST_REMOTE_DOCKER_HOST"
    RST_BUILD_CMD=docker
    RST_HARBOR_ENV=docker
    RST_SANDBOX_LOCATION=remote-daemon
    log "using a remote docker daemon: $RST_DOCKER_HOST"
    log "CAVEAT: bind mounts resolve on the DAEMON host, not here. Harbor mounts the"
    log "        task dir into the container, so the task tree must exist at the SAME"
    log "        absolute path over there. Sync it before running:"
    log "          rsync -a --delete \"\$BASE_FOLDER/rst-tasks/\" <host>:\"\$BASE_FOLDER/rst-tasks/\""
    log "        Build contexts are streamed over the socket and need no sync."
  else
    log "RST_REMOTE_DOCKER_HOST=$RST_REMOTE_DOCKER_HOST is set but unreachable (needs a docker CLI here)"
  fi
fi

# ------------------------------------------------------- 3. rootless podman (local)
if [[ -z "$RST_SANDBOX_LOCATION" ]] && command -v podman >/dev/null 2>&1; then
  if ! mount_reason="$(mount_ok)"; then
    # Fail fast with the real cause. Without this you get an ApplyLayer error
    # 40 minutes into a prebuild run and conclude the task pool is broken.
    log "podman is installed but this machine cannot mount(2): $mount_reason"
    log "  Installing packages will NOT fix that. Run:  bash scripts/00b_setup_sandbox.sh --diagnose"
  else
    missing=()
    [[ $(grep -c "^$(id -un):" /etc/subuid 2>/dev/null || echo 0) -ge 1 ]] || missing+=("no /etc/subuid entry for $(id -un)")
    command -v newuidmap >/dev/null 2>&1 || missing+=("newuidmap absent (install uidmap)")
    [[ $(cat /proc/sys/user/max_user_namespaces 2>/dev/null || echo 0) -gt 0 ]] || missing+=("user namespaces disabled")
    if (( ${#missing[@]} )); then
      log "podman present but rootless prerequisites missing: ${missing[*]}"
    elif ! podman info >/dev/null 2>&1; then
      log "podman present but 'podman info' failed; see: podman info"
    else
      mkdir -p "$(dirname "$PODMAN_SOCK")"
      if [[ ! -S "$PODMAN_SOCK" ]]; then
        log "starting podman's Docker-compatible API service"
        nohup podman system service --time=0 "unix://$PODMAN_SOCK" \
          >"${TMPDIR:-/tmp}/podman_service.log" 2>&1 &
        for _ in $(seq 1 20); do [[ -S "$PODMAN_SOCK" ]] && break; sleep 1; done
      fi
      if [[ -S "$PODMAN_SOCK" ]]; then
        RST_CONTAINER_RUNTIME=podman
        RST_DOCKER_HOST="unix://$PODMAN_SOCK"
        RST_BUILD_CMD=podman
        RST_HARBOR_ENV=docker      # podman serves the Docker API; harbor needs no patch
        RST_SANDBOX_LOCATION=local
        log "using rootless podman: $RST_DOCKER_HOST ($(podman --version))"
        # VERSION GATE. 31% of the RST task Dockerfiles use heredoc syntax
        # (RUN <<EOF), which needs buildah >= 1.29 / podman >= 4.4. Older podman
        # fails them with "Unknown instruction: IF", which looks like a broken task
        # rather than a stale toolchain -- so it would silently cost you a third of
        # the task pool. Ubuntu 22.04 ships 3.4.4, which is too old.
        PV=$(podman --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
        PV_MAJOR=${PV%%.*}; PV_MINOR=${PV##*.}
        if (( PV_MAJOR < 4 || (PV_MAJOR == 4 && PV_MINOR < 4) )); then
          log "WARNING: podman $PV is too old for Dockerfile heredocs (need >= 4.4)."
          log "         31% of RST task Dockerfiles use them and will fail to build."
          log "         Install a static rootless podman -- NO root, NO package manager:"
          log "           curl -sSL -o /tmp/podman.tgz \\"
          log "             https://github.com/mgoltzsche/podman-static/releases/latest/download/podman-linux-amd64.tar.gz"
          log "           mkdir -p \$HOME/.local && tar xzf /tmp/podman.tgz -C \$HOME/.local --strip-components=1"
          log "           export PATH=\$HOME/.local/bin:\$PATH   # then re-run this script"
          export RST_PODMAN_TOO_OLD=1
        fi
        # NOTE: do NOT create a CNI network named "default" here. podman 3.4.x
        # writes a conflist whose version its own firewall plugin rejects, and every
        # later podman call then emits a validation warning. Agent sandboxes want
        # `--network none` anyway, which needs no network definition at all.
      else
        log "podman service socket never appeared; see ${TMPDIR:-/tmp}/podman_service.log"
      fi
    fi
  fi
fi

# ------------------------------------- 4. managed provider (no local privilege at all)
# Ordered by how well they fit RST tasks: all three build environment/Dockerfile
# server-side, so nothing needs a container runtime here.
if [[ -z "$RST_SANDBOX_LOCATION" ]]; then
  if [[ -n "${DAYTONA_API_KEY:-}${DAYTONA_JWT_TOKEN:-}" ]]; then
    RST_HARBOR_ENV=daytona; RST_SANDBOX_LOCATION=managed
    log "using Daytona (declarative Dockerfile build -> content-hashed snapshot, cached across rollouts)"
  elif [[ -n "${E2B_API_KEY:-}" ]]; then
    RST_HARBOR_ENV=e2b; RST_SANDBOX_LOCATION=managed
    log "using E2B (Dockerfile -> template build)"
  elif [[ -n "${MODAL_TOKEN_ID:-}" && -n "${MODAL_TOKEN_SECRET:-}" ]]; then
    RST_HARBOR_ENV=modal; RST_SANDBOX_LOCATION=managed
    log "using Modal (Image.from_dockerfile)"
  fi
fi

# ------------------------------------------------------------------- 5. verdict
if [[ -z "$RST_SANDBOX_LOCATION" ]]; then
  cat >&2 <<'EOF'
[sandbox] NO SANDBOX AVAILABLE YET.

  Consequences, stated precisely:
    * SFT is UNAFFECTED. Training needs no sandbox; run it.
    * Agentic evaluation is BLOCKED. Every benchmark task builds a Dockerfile and
      runs a tmux session inside it.
    * Agentic RL is BLOCKED for the same reason.
  You can still measure the checkpoint without any container:
      python scripts/06b_eval_offline.py --model-path <ckpt> ...
  That gives held-out loss, next-action agreement and tool-call parse rate. It is
  a weaker signal than a benchmark score and must be reported as such -- but
  "unmeasured" is not an acceptable deliverable, and this is not unmeasured.

  Do not guess at the fix. Run this first, it tells you which case you are in:
      bash scripts/00b_setup_sandbox.sh --diagnose

  The four cases, and what each one actually needs:
    A. packages missing        -> ask for podman + uidmap, or install podman-static
    B. no /etc/subuid entry    -> ask for one line in /etc/subuid + /etc/subgid
    C. userns disabled         -> ask for user.max_user_namespaces > 0
    D. mount(2) denied by an   -> NOTHING INSTALLABLE HELPS. Either relaunch the
       LSM (EACCES, e.g. the      pod with `--security-opt apparmor=unconfined`
       docker-default AppArmor    (k8s: securityContext.appArmorProfile.type:
       profile)                   Unconfined), or move the sandbox off this host:

  Moving the sandbox off this host -- any ONE of these unblocks eval and RL:
    * a Docker API endpoint on another machine you can reach:
        export RST_REMOTE_DOCKER_HOST=tcp://host:2375   # or ssh://user@host
    * a managed provider, which needs only outbound HTTPS and a key. Harbor 0.21.0
      builds the task's own Dockerfile on their side:
        export DAYTONA_API_KEY=...   # what the RST paper used; snapshots are cached
        export E2B_API_KEY=...
        export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
    * Kubernetes sibling pods, if your service account may create pods:
        kubectl auth can-i create pods
        export RST_HARBOR_ENV=gke     # or ack / openshift
        export RST_HARBOR_ENV_KWARGS="namespace=<your-namespace>"
EOF
  [[ "$MODE" == check ]] && exit 1
  return 1 2>/dev/null || exit 1
fi

export RST_CONTAINER_RUNTIME RST_DOCKER_HOST RST_BUILD_CMD
export RST_HARBOR_ENV RST_HARBOR_ENV_KWARGS RST_SANDBOX_LOCATION
[[ -n "$RST_DOCKER_HOST" ]] && export DOCKER_HOST="$RST_DOCKER_HOST"

if [[ "$MODE" == check ]]; then
  log "RST_HARBOR_ENV=$RST_HARBOR_ENV  RST_SANDBOX_LOCATION=$RST_SANDBOX_LOCATION"
  if [[ "$RST_HARBOR_ENV" != docker ]]; then
    # There is nothing to exec into locally; the real smoke test is one harbor
    # trial, which costs provider time. Say so instead of pretending to check.
    log "off-machine backend selected; no local smoke test applies."
    log "smoke-test it for real with ONE task before launching a sweep:"
    log "  harbor run --path <one-task-dir> --env $RST_HARBOR_ENV --n-attempts 1"
    exit 0
  fi
  log "RST_DOCKER_HOST=$RST_DOCKER_HOST"
  log "smoke test: run + exec"
  img="docker.io/library/alpine:3.18"
  "$RST_BUILD_CMD" pull -q "$img" >/dev/null 2>&1 || true
  cid=$("$RST_BUILD_CMD" run -d --network none "$img" sleep 60 2>/dev/null | tail -1)
  if [[ -n "$cid" ]]; then
    "$RST_BUILD_CMD" exec "$cid" sh -c 'echo "  exec OK"' 2>&1 | tail -1
    "$RST_BUILD_CMD" rm -f "$cid" >/dev/null 2>&1
  else
    log "WARNING: could not start a test container"; exit 1
  fi
  log "OK"
  exit 0
fi

log "RST_HARBOR_ENV=$RST_HARBOR_ENV  location=$RST_SANDBOX_LOCATION${RST_DOCKER_HOST:+  RST_DOCKER_HOST=$RST_DOCKER_HOST}"
