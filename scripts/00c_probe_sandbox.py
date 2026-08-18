#!/usr/bin/env python3
"""Decide *why* container sandboxing is unavailable, and what would actually fix it.

    python scripts/00c_probe_sandbox.py            # human report + a verdict
    python scripts/00c_probe_sandbox.py --json     # machine-readable facts

WHY THIS EXISTS
    "podman does not work here" has at least four different causes with four
    different fixes, and guessing wrong costs a day:

      missing packages          -> ask for podman + uidmap
      no /etc/subuid entry      -> ask for one line in /etc/subuid
      user namespaces disabled  -> ask for user.max_user_namespaces > 0
      LSM policy denies mount   -> NONE of the above helps; the pod itself has
                                   to be relaunched, or the sandbox has to move
                                   off this machine entirely

    The fourth case is the one that wastes the most time, because everything
    looks installed and correct: `podman info` succeeds, image blobs download,
    and then unpacking a layer fails with `ApplyLayer ... remount /, flags:
    0x44000: permission denied`. 0x44000 is MS_REC|MS_PRIVATE -- that is
    containers/storage isolating mount propagation, not tar failing.

HOW WE TELL THEM APART
    We call mount(2) directly, inside a throwaway user+mount namespace, and
    report errno:

      success  -> mounting works; any podman failure is a podman problem
      EPERM    -> the kernel refused: no privilege in this namespace
      EACCES   -> an LSM (AppArmor/SELinux) refused. Capabilities are irrelevant;
                  the process holds CAP_SYS_ADMIN in its own namespace and is
                  still denied. Docker's stock `docker-default` AppArmor profile
                  contains a literal `deny mount,` and allows only `umount`.

    An AppArmor profile is inherited across unshare() and execve(), and
    `docker-default` grants no `change_profile` rule, so there is no escape from
    inside the container. Nothing installable fixes it.

The probe never mounts anything in the caller's namespace: it forks, unshares,
and reports over a pipe. The parent's mount table is untouched either way.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import shutil
import stat
import subprocess
import tempfile

CLONE_NEWNS = 0x00020000
CLONE_NEWUSER = 0x10000000
MS_RDONLY = 0x1
MS_REC = 0x4000
MS_PRIVATE = 0x40000

# What the operator should ask for, keyed by what we actually measured. Kept
# separate from the probing so the advice is auditable against the evidence.
OPS_ASK_APPARMOR = """\
Relaunch this pod with AppArmor confinement off for the container.
This is a ONE-FLAG change. It does NOT need --privileged and it does NOT need
extra capabilities: rootless podman already works with no added caps once mount
is permitted, and seccomp is not the blocker here.

  docker:      --security-opt apparmor=unconfined
  podman:      --security-opt apparmor=unconfined
  k8s >= 1.30: spec.containers[].securityContext.appArmorProfile.type: Unconfined
  k8s <  1.30: annotation
               container.apparmor.security.beta.kubernetes.io/<container>: unconfined

Optional, only if you want fuse-overlayfs instead of the vfs storage driver
(vfs works without it; it just uses more disk):  --device /dev/fuse

BEFORE the pod is recreated: anything not on a persistent volume is lost,
including the conda env and the downloaded model/data. Check what is on a PVC
first, or budget an hour to rebuild."""

ASK_PACKAGES = """\
Ask for the `podman` and `uidmap` packages, or install a static rootless podman
without root:
  curl -sSL -o /tmp/podman.tgz \\
    https://github.com/mgoltzsche/podman-static/releases/latest/download/podman-linux-amd64.tar.gz
  mkdir -p $HOME/.local && tar xzf /tmp/podman.tgz -C $HOME/.local --strip-components=1
  export PATH=$HOME/.local/bin:$PATH"""


def _probe_mount_in_userns() -> dict[str, object]:
    """Fork, unshare user+mount namespaces, and try two mounts. Report errno.

    The two operations are chosen to match what a container runtime actually
    does first:
      * `MS_REC|MS_PRIVATE` on "/"   -- mount-propagation isolation, the exact
                                        call in the reported ApplyLayer failure
      * a fresh tmpfs                 -- the simplest possible new mount
    """
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(read_fd)
        out: dict[str, object] = {}
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            uid, gid = os.getuid(), os.getgid()
            if libc.unshare(CLONE_NEWUSER | CLONE_NEWNS) != 0:
                e = ctypes.get_errno()
                out["unshare"] = {"ok": False, "errno": e, "errno_name": errno.errorcode.get(e, "?")}
            else:
                out["unshare"] = {"ok": True}
                # Map ourselves to uid 0 inside the new namespace. Not required
                # for mounting, but it is what podman does, so a failure here is
                # itself a finding.
                for path, data in (
                    ("/proc/self/setgroups", "deny"),
                    ("/proc/self/uid_map", f"0 {uid} 1"),
                    ("/proc/self/gid_map", f"0 {gid} 1"),
                ):
                    try:
                        with open(path, "w") as fh:
                            fh.write(data)
                    except OSError as exc:
                        out.setdefault("map_errors", []).append(f"{path}: {exc.strerror}")  # type: ignore[union-attr]

                def attempt(name: str, *args: object) -> None:
                    ctypes.set_errno(0)
                    rc = libc.mount(*args)
                    if rc == 0:
                        out[name] = {"ok": True}
                    else:
                        e = ctypes.get_errno()
                        out[name] = {
                            "ok": False,
                            "errno": e,
                            "errno_name": errno.errorcode.get(e, "?"),
                            "strerror": os.strerror(e),
                        }

                attempt("make_rprivate", b"", b"/", None, MS_REC | MS_PRIVATE, None)
                target = tempfile.mkdtemp(prefix="rst-mountprobe-")
                attempt("tmpfs", b"none", target.encode(), b"tmpfs", 0, None)
                # Negative control: a filesystem type that cannot exist. If this
                # comes back EACCES too, the denial is blanket policy rather than
                # anything specific to tmpfs.
                attempt("bogus_fstype", b"none", target.encode(), b"nosuchfs", MS_RDONLY, None)
        except Exception as exc:  # pragma: no cover - defensive
            out["error"] = f"{type(exc).__name__}: {exc}"
        os.write(write_fd, json.dumps(out).encode())
        os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    chunks = []
    while True:
        block = os.read(read_fd, 65536)
        if not block:
            break
        chunks.append(block)
    os.close(read_fd)
    os.waitpid(pid, 0)
    try:
        return json.loads(b"".join(chunks) or b"{}")
    except json.JSONDecodeError:
        return {"error": "probe child produced no parsable output"}


def _read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _version(*argv: str) -> str:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else ""


def collect() -> dict[str, object]:
    facts: dict[str, object] = {}

    # ---------------------------------------------------------------- confinement
    profile = _read("/proc/self/attr/current") or "unavailable"
    status = _read("/proc/self/status")
    seccomp = next((ln.split()[1] for ln in status.splitlines() if ln.startswith("Seccomp:")), "?")
    cap_eff = next((ln.split()[1] for ln in status.splitlines() if ln.startswith("CapEff:")), "?")
    facts["confinement"] = {
        "apparmor_profile": profile,
        # "unconfined" is the only value that means "no AppArmor mediation".
        # Anything else -- docker-default, cri-containerd.apparmor.d, a custom
        # name -- mediates, and the (enforce) suffix says it denies rather than
        # logs.
        "apparmor_mediating": profile not in ("unconfined", "unavailable", ""),
        "apparmor_enforcing": "(enforce)" in profile,
        "selinux": _read("/sys/fs/selinux/enforce") or "absent",
        "seccomp_mode": seccomp,
        "cap_eff": cap_eff,
    }

    # -------------------------------------------------------------- namespaces
    facts["userns"] = {
        "max_user_namespaces": _read("/proc/sys/user/max_user_namespaces") or "?",
        "unprivileged_userns_clone": _read("/proc/sys/kernel/unprivileged_userns_clone") or "n/a",
        "subuid_entry": any(
            ln.split(":")[0] == _whoami() for ln in _read("/etc/subuid").splitlines()
        ),
        "newuidmap": bool(shutil.which("newuidmap")),
        "dev_fuse": os.path.exists("/dev/fuse"),
    }
    facts["mount"] = _probe_mount_in_userns()

    # ---------------------------------------------------------------- runtimes
    sockets = {}
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    for label, path in (
        ("host_docker", "/var/run/docker.sock"),
        ("rootless_docker", f"{runtime_dir}/docker.sock"),
        ("podman", f"{runtime_dir}/podman/podman.sock"),
    ):
        try:
            sockets[label] = stat.S_ISSOCK(os.stat(path).st_mode)
        except OSError:
            sockets[label] = False
    facts["runtimes"] = {
        "docker_cli": _version("docker", "--version"),
        "podman": _version("podman", "--version"),
        "apptainer": _version("apptainer", "--version") or _version("singularity", "--version"),
        "sockets": sockets,
        "DOCKER_HOST": os.environ.get("DOCKER_HOST", ""),
        "CONTAINER_HOST": os.environ.get("CONTAINER_HOST", ""),
    }

    # ------------------------------------------- remote sandbox providers
    # Presence only. Never print or return a credential value.
    facts["providers"] = {
        "daytona": bool(os.environ.get("DAYTONA_API_KEY") or os.environ.get("DAYTONA_JWT_TOKEN")),
        "e2b": bool(os.environ.get("E2B_API_KEY")),
        "modal": bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")),
        "hf_sandbox": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
    }

    # --------------------------------------------------------------- kubernetes
    k8s: dict[str, object] = {
        "kubectl": bool(shutil.which("kubectl")),
        "oc": bool(shutil.which("oc")),
        "in_cluster_sa": os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"),
        "kubeconfig": os.path.exists(os.environ.get("KUBECONFIG", os.path.expanduser("~/.kube/config"))),
    }
    if k8s["kubectl"]:
        probe = subprocess.run(
            ["kubectl", "auth", "can-i", "create", "pods"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        k8s["can_create_pods"] = probe.stdout.strip().startswith("yes")
    facts["kubernetes"] = k8s

    facts["verdict"] = _verdict(facts)
    return facts


def _whoami() -> str:
    import pwd

    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return str(os.getuid())


def _verdict(f: dict[str, object]) -> dict[str, object]:
    mount = f["mount"]  # type: ignore[index]
    conf = f["confinement"]  # type: ignore[index]
    providers = f["providers"]  # type: ignore[index]
    k8s = f["kubernetes"]  # type: ignore[index]

    def denied(op: str) -> str | None:
        entry = mount.get(op) if isinstance(mount, dict) else None
        if isinstance(entry, dict) and not entry.get("ok"):
            return str(entry.get("errno_name"))
        return None

    rprivate, tmpfs = denied("make_rprivate"), denied("tmpfs")
    local_ok = rprivate is None and tmpfs is None

    options: list[str] = []
    if not local_ok:
        lsm_denied = "EACCES" in (rprivate or "", tmpfs or "")
        if lsm_denied and conf.get("apparmor_mediating"):  # type: ignore[union-attr]
            cause = (
                f"AppArmor profile {conf['apparmor_profile']!r} denies mount(2). "  # type: ignore[index]
                "Installing packages cannot fix this."
            )
            options.append("ops: relaunch the pod with apparmor=unconfined (see ops_ask)")
        elif lsm_denied:
            cause = "an LSM denies mount(2) (EACCES) though no AppArmor profile is reported"
            options.append("ops: identify the mediating LSM; SELinux -> a permissive type, AppArmor -> unconfined")
        else:
            cause = f"mount(2) refused by the kernel ({rprivate or tmpfs})"
            options.append("ops: allow unprivileged user namespaces, or grant CAP_SYS_ADMIN")
    else:
        cause = "mount(2) works in a user namespace; a local rootless runtime should be viable"

    # Remote paths, ordered by how little they need. These three build the task's
    # own Dockerfile server-side, so a fully mount-less pod is still enough.
    for name, flag, note in (
        ("daytona", "daytona", "builds environment/Dockerfile into a content-hashed snapshot; the paper's provider"),
        ("e2b", "e2b", "builds environment/Dockerfile into a template"),
        ("modal", "modal", "Image.from_dockerfile"),
    ):
        if providers.get(flag):  # type: ignore[union-attr]
            options.append(f"ready now: --env {name} ({note}); credential already in the environment")
        else:
            options.append(f"needs a key: --env {name} ({note})")
    if k8s.get("can_create_pods"):
        options.append("ready now: --env gke|ack|openshift — run each task as a sibling pod, no local runtime needed")
    options.append("no sandbox at all: scripts/06b_eval_offline.py still measures the checkpoint (weaker, but not nothing)")

    return {
        "local_container_runtime_possible": local_ok,
        "cause": cause,
        "ops_ask": OPS_ASK_APPARMOR if (not local_ok and conf.get("apparmor_mediating")) else (  # type: ignore[union-attr]
            ASK_PACKAGES if local_ok and not f["runtimes"]["podman"] else ""  # type: ignore[index]
        ),
        "options": options,
    }


def _print_human(f: dict[str, object]) -> None:
    conf, mount = f["confinement"], f["mount"]  # type: ignore[index]
    v = f["verdict"]  # type: ignore[index]
    print("=== confinement ===")
    print(f"  apparmor profile   : {conf['apparmor_profile']}")  # type: ignore[index]
    print(f"  seccomp mode       : {conf['seccomp_mode']}   (0 = no filter)")  # type: ignore[index]
    print(f"  effective caps     : {conf['cap_eff']}")  # type: ignore[index]
    print(f"  selinux enforce    : {conf['selinux']}")  # type: ignore[index]
    print("=== mount(2) inside a fresh user+mount namespace ===")
    for op in ("unshare", "make_rprivate", "tmpfs", "bogus_fstype"):
        entry = mount.get(op) if isinstance(mount, dict) else None  # type: ignore[union-attr]
        if not isinstance(entry, dict):
            continue
        if entry.get("ok"):
            print(f"  {op:<14}: OK")
        else:
            print(f"  {op:<14}: {entry.get('errno_name')}  ({entry.get('strerror', '')})")
    print("     ^ EACCES = policy (LSM) denial; EPERM = missing privilege;")
    print("       ENODEV on bogus_fstype only = mounting is allowed and working")
    print("=== runtimes ===")
    for k, val in f["runtimes"].items():  # type: ignore[union-attr]
        print(f"  {k:<16}: {val}")
    print("=== remote sandbox credentials present ===")
    for k, val in f["providers"].items():  # type: ignore[union-attr]
        print(f"  {k:<16}: {'yes' if val else 'no'}")
    print("=== kubernetes ===")
    for k, val in f["kubernetes"].items():  # type: ignore[union-attr]
        print(f"  {k:<16}: {val}")
    print("=== verdict ===")
    print(f"  local runtime possible: {v['local_container_runtime_possible']}")  # type: ignore[index]
    print(f"  cause: {v['cause']}")  # type: ignore[index]
    print("  options, cheapest first:")
    for opt in v["options"]:  # type: ignore[index]
        print(f"    - {opt}")
    if v["ops_ask"]:  # type: ignore[index]
        print("\n=== paste this to whoever owns the pod spec ===")
        print(v["ops_ask"])  # type: ignore[index]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit facts as JSON")
    ap.add_argument("--out", help="also write the JSON here")
    args = ap.parse_args()
    facts = collect()
    if args.json:
        print(json.dumps(facts, indent=2, sort_keys=True))
    else:
        _print_human(facts)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(facts, fh, indent=2, sort_keys=True)
            fh.write("\n")
    # Exit 0 always: this is a report, and a nonzero exit would make callers
    # treat "diagnosed successfully" as "the probe broke".
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
