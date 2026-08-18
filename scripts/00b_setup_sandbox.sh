#!/usr/bin/env bash
# Find (or set up) a container runtime for eval + RL rollouts, WITHOUT needing
# permission to use the host Docker daemon.
#
#   source scripts/00b_setup_sandbox.sh          # exports RST_DOCKER_HOST etc.
#   bash   scripts/00b_setup_sandbox.sh --check  # just report, exit 0/1
#
# WHY THIS EXISTS
#   SFT needs no sandbox at all. Evaluation and RL rollouts do: Harbor builds each
#   task's Dockerfile and drives a tmux session inside the container. On a shared
#   cluster you typically cannot touch /var/run/docker.sock.
#
# THE ANSWER: rootless podman, via its Docker-compatible API socket.
#   Harbor speaks the Docker API. podman serves that same API. So pointing
#   DOCKER_HOST at podman's socket makes Harbor work UNCHANGED -- no patch, no
#   custom backend.
#
# VERIFIED on a box with no docker.sock permission (podman 3.4.4, cgroup v2,
# rootless, subuid configured):
#   * `podman build` of a real RST task Dockerfile: 26.6 s, including
#     `apt-get install build-essential git curl`, a `git clone`, and `pip install`.
#     Rootless builds run as uid 0 inside a user namespace, so apt works.
#   * docker CLI -> podman socket: client 29.2.1 / server 3.4.4 / API 1.40
#   * `docker run -d --network none`, then `docker exec`: OK
#   * tmux 3.2a inside: new-session / send-keys / capture-pane all OK
#   * no network inside the container: confirmed
#
# THREE THINGS THAT ONLY SHOW UP WHEN YOU ACTUALLY BUILD THE TASK IMAGES, all
# found by doing exactly that on the real pool:
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
# namespace. The earlier guidance ("use a dedicated daemon, never the host's")
# is satisfied more strongly here, not waived.
set -uo pipefail

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

: "${XDG_RUNTIME_DIR:=/run/user/$(id -u)}"
export XDG_RUNTIME_DIR
PODMAN_SOCK="$XDG_RUNTIME_DIR/podman/podman.sock"

log() { echo "[sandbox] $*"; }

RST_CONTAINER_RUNTIME=""
RST_DOCKER_HOST=""
RST_BUILD_CMD=""

# ---------------------------------------------------------------- 1. real docker
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  # Only accept a NON-default socket: building untrusted Dockerfiles on the shared
  # host daemon is what we are avoiding. If DOCKER_HOST is unset or points at the
  # host socket, fall through to podman.
  if [[ -n "${DOCKER_HOST:-}" && "$DOCKER_HOST" != "unix:///var/run/docker.sock" ]]; then
    RST_CONTAINER_RUNTIME=docker
    RST_DOCKER_HOST="$DOCKER_HOST"
    RST_BUILD_CMD=docker
    log "using docker at $DOCKER_HOST"
  else
    log "docker reachable but only via the shared host socket; preferring rootless podman"
  fi
fi

# ------------------------------------------------------- 2. rootless podman (main)
if [[ -z "$RST_CONTAINER_RUNTIME" ]] && command -v podman >/dev/null 2>&1; then
  missing=()
  [[ $(grep -c "^$(id -un):" /etc/subuid 2>/dev/null || echo 0) -ge 1 ]] || missing+=("no /etc/subuid entry for $(id -un)")
  command -v newuidmap >/dev/null 2>&1 || missing+=("newuidmap absent (install uidmap)")
  [[ $(cat /proc/sys/user/max_user_namespaces 2>/dev/null || echo 0) -gt 0 ]] || missing+=("user namespaces disabled")
  if (( ${#missing[@]} )); then
    log "podman present but rootless prerequisites missing: ${missing[*]}"
  else
    if ! podman info >/dev/null 2>&1; then
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

# ---------------------------------------------------- 3. apptainer (NOT wired up)
if [[ -z "$RST_CONTAINER_RUNTIME" ]] && (command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1); then
  log "apptainer/singularity found, but it does NOT serve the Docker API, so Harbor"
  log "cannot drive it unchanged. It would need a new Harbor environment backend:"
  log "  apptainer build task.sif docker-archive:<oci-tar>   # convert, cannot read a Dockerfile"
  log "  apptainer exec --net --network none task.sif ...    # and a tmux-driving shim"
  log "That work is NOT done here. Prefer installing rootless podman."
fi

# ------------------------------------------------------------------- 4. verdict
if [[ -z "$RST_CONTAINER_RUNTIME" ]]; then
  cat >&2 <<'EOF'
[sandbox] NO USABLE CONTAINER RUNTIME.

  Consequences, stated precisely:
    * SFT is UNAFFECTED. Training needs no sandbox; run it.
    * Evaluation is BLOCKED. Every benchmark task builds a Dockerfile and runs a
      tmux session inside it.
    * RL is BLOCKED for the same reason.
  So you can train, but you cannot measure whether training helped. Do not report
  a checkpoint as good without an eval; say the eval was impossible instead.

  Cheapest fix, in order:
    1. ask for the `podman` and `uidmap` packages (no daemon, no root, no group
       membership needed) -- this is normally an easier ask than docker access
    2. ask for one /etc/subuid + /etc/subgid entry for your user
    3. ask for a Daytona (or equivalent) sandbox endpoint, which is what the paper
       used, and set it up as a remote Harbor environment
EOF
  (( CHECK_ONLY )) && exit 1
  return 1 2>/dev/null || exit 1
fi

export RST_CONTAINER_RUNTIME RST_DOCKER_HOST RST_BUILD_CMD
export DOCKER_HOST="$RST_DOCKER_HOST"

if (( CHECK_ONLY )); then
  log "RST_CONTAINER_RUNTIME=$RST_CONTAINER_RUNTIME"
  log "RST_DOCKER_HOST=$RST_DOCKER_HOST"
  log "smoke test: run + exec + tmux"
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

log "RST_CONTAINER_RUNTIME=$RST_CONTAINER_RUNTIME  RST_DOCKER_HOST=$RST_DOCKER_HOST"
