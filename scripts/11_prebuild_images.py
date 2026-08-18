#!/usr/bin/env python3
"""Pre-build and cache the Docker images for the GRPO task pool.

This is the single largest rollout prerequisite and the one most likely to be
underestimated. Measured on the release: **99% of the 37,484 task Dockerfiles run
`apt-get install` / `pip install` / `npm install` at BUILD time**. If you let
Harbor build them lazily during training, then every rollout pays a network-bound
build, and any registry hiccup surfaces as an infrastructure failure that stalls
the trainer.

So: build once, up front, with bounded concurrency, and record exactly which
tasks are ready. Tasks that fail to build are excluded from the pool rather than
discovered mid-run.

SECURITY: these Dockerfiles are untrusted third-party build scripts fetched from
the hub. Build them on a DEDICATED (ideally rootless) daemon, never the host's
default one. This script refuses to run without an explicit non-default
DOCKER_HOST.

    export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
    python scripts/11_prebuild_images.py --taskset data/rl-sweet --workers 8

    # size probe first (strongly recommended before committing disk):
    python scripts/11_prebuild_images.py --taskset data/rl-sweet --sample 40

NOT APPLICABLE TO OFF-MACHINE SANDBOXES. With `RST_HARBOR_ENV` set to daytona /
e2b / modal, Harbor hands the task's Dockerfile to the provider and the image is
built and cached THERE -- Daytona keyed by a content hash of the build spec, E2B
as a template, Modal via Image.from_dockerfile. Building the same image locally
would not populate that cache, and on a pod that cannot mount(2) it cannot even
be attempted. This script detects that case and exits 0 having built nothing,
because a warm-cache step that is structurally impossible is not a failure. The
first rollout of each distinct base image then pays the provider-side build once;
after that the snapshot is reused. Budget for that in the first sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IMAGE_PREFIX = "rst-task"


# Which CLI to shell out to. Rootless podman is the primary path on clusters where
# you cannot use the host Docker daemon; it builds task Dockerfiles fine because a
# rootless build runs as uid 0 inside a user namespace, so `apt-get install` works.
# Verified: a real RST task image built in 26.6s this way.
BUILD_CMD = os.environ.get("RST_BUILD_CMD") or (
    "podman" if shutil.which("podman") else "docker"
)


def docker(*argv: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BUILD_CMD, *argv], capture_output=True, text=True, timeout=timeout, check=False
    )


VFS_ROOT = os.environ.get("RST_PODMAN_VFS_ROOT", "")


def _is_hardlink_failure(proc: subprocess.CompletedProcess) -> bool:
    text = ((proc.stderr or "") + (proc.stdout or "")).lower()
    return "hardlink different from source" in text


def _build_with_vfs(argv: list[str], timeout: int) -> subprocess.CompletedProcess:
    root = VFS_ROOT or os.path.join(os.environ.get("TMPDIR", "/tmp"), "rst-podman-vfs")
    os.makedirs(root, exist_ok=True)
    return subprocess.run(
        [BUILD_CMD, "--root", root, "--storage-driver", "vfs", *argv],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def offmachine_reason() -> str | None:
    """Is the sandbox somewhere this script cannot pre-warm? Then say so, don't fail.

    Only two of the sandbox locations from 00b_setup_sandbox.sh can be warmed from
    here: `local` (obviously) and `remote-daemon` (the docker CLI streams the build
    context over the socket, so the images land on the daemon Harbor will use).
    Everything else builds on infrastructure we do not address.
    """
    location = os.environ.get("RST_SANDBOX_LOCATION", "")
    harbor_env = os.environ.get("RST_HARBOR_ENV", "")
    if location in ("local", "remote-daemon"):
        return None
    if location in ("managed", "preset"):
        return (f"sandbox location is {location!r} (harbor --env {harbor_env or '?'}); "
                f"images are built by the provider, not here")
    if harbor_env and harbor_env != "docker":
        return f"RST_HARBOR_ENV={harbor_env} is an off-machine backend; it builds its own images"
    return None


def image_tag(task_id: str) -> str:
    return f"{IMAGE_PREFIX}:{task_id}"


def compose_task(task_dir: Path) -> bool:
    """Multi-container tasks are driven by docker-compose, not a single image."""
    env = task_dir / "environment"
    return any((env / n).is_file() for n in ("docker-compose.yaml", "docker-compose.yml"))


def build_one(row: dict, timeout: int, pull_base: bool) -> dict:
    task_id = row["task_id"]
    task_dir = Path(row["task_dir"])
    result = {"task_id": task_id, "base_image": row.get("base_image"), "compose": False}
    started = time.time()

    if compose_task(task_dir):
        # `docker compose build` needs the compose file's own context; Harbor
        # handles the lifecycle, we just warm the layers.
        result["compose"] = True
        # podman implements `podman compose` (delegating to podman-compose/docker-compose)
        proc = subprocess.run(
            [BUILD_CMD, "compose", "-f", str(task_dir / "environment" /
              ("docker-compose.yaml" if (task_dir / "environment/docker-compose.yaml").is_file()
               else "docker-compose.yml")), "build"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    else:
        env_dir = task_dir / "environment"
        if not (env_dir / "Dockerfile").is_file():
            result.update(ok=False, reason="no Dockerfile", seconds=0.0)
            return result
        argv = ["build", "-t", image_tag(task_id), "-f", str(env_dir / "Dockerfile")]
        if BUILD_CMD == "podman":
            # Discovered by running this rootless on a real task set:
            #  --format docker : 16% of these Dockerfiles use SHELL, which podman's
            #                    default OCI image format silently ignores and then
            #                    errors on. Docker format supports it.
            #  --network host  : builds legitimately fetch packages; the AGENT
            #                    sandbox is what must have no network, not the build.
            # Both go AFTER the subcommand: `--format` is an option of `podman build`,
            # not a global. podman's globals are --root/--runroot/--storage-driver/
            # --storage-opt/--remote (which is why _build_with_vfs prefixes exactly
            # those). Putting --format in front makes podman exit on an unknown flag
            # before it ever reads the Dockerfile, so every build fails and the reason
            # recorded in prebuild_report.json is a CLI parse error rather than
            # anything about the task.
            argv[1:1] = ["--format", "docker"]
            argv.append("--network=host")
        if pull_base:
            argv.append("--pull")
        argv.append(str(env_dir))
        proc = docker(*argv, timeout=timeout)
        if proc.returncode != 0 and _is_hardlink_failure(proc):
            # Rootless kernel-overlay cannot reproduce hardlinks that a `git clone`
            # inside the build creates ("hardlink different from source"). A vfs
            # store in a separate root does; verified fix on a real failing task.
            result["retried_vfs"] = True
            proc = _build_with_vfs(argv, timeout)

    result["seconds"] = round(time.time() - started, 1)
    result["ok"] = proc.returncode == 0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
        result["reason"] = " | ".join(tail)[:400]
    return result


def image_size_bytes(task_id: str) -> int | None:
    proc = docker("image", "inspect", image_tag(task_id), "--format", "{{.Size}}")
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taskset", type=Path, required=True, help="output dir of 10_build_rl_taskset.py")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1800, help="per-image build timeout (s)")
    parser.add_argument("--sample", type=int, default=0, help="build only N tasks (size probe)")
    parser.add_argument("--no-pull", action="store_true", help="skip --pull on base images")
    # Skipping images that already exist is the default because this script is
    # re-run to fill in failures. `--rebuild-existing` is the way to turn it off:
    # a `--skip-existing` store_true that already defaults to True can only ever be
    # set to the value it already has, which reads like a switch and is not one.
    parser.add_argument("--rebuild-existing", action="store_true",
                        help="rebuild task images that are already present locally")
    parser.add_argument("--force-local", action="store_true",
                        help="build here even if the sandbox backend is off-machine")
    args = parser.parse_args()

    if not args.force_local and (skip := offmachine_reason()):
        report = {
            "skipped": True,
            "reason": skip,
            "harbor_env": os.environ.get("RST_HARBOR_ENV", ""),
            "sandbox_location": os.environ.get("RST_SANDBOX_LOCATION", ""),
        }
        print(f"[skip] {skip}")
        print("[skip] The provider builds each task's Dockerfile on its own side and caches")
        print("[skip] the result, so there is nothing useful to warm locally. The first")
        print("[skip] rollout per distinct image pays that build once; budget for it.")
        if args.taskset.is_dir():
            out = args.taskset / "prebuild_report.json"
            out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"[skip] wrote {out}")
        print("[skip] Pass --force-local if you really do want local images too.")
        return 0

    if not shutil.which(BUILD_CMD):
        sys.exit(f"{BUILD_CMD} not found on PATH (set RST_BUILD_CMD)")
    docker_host = os.environ.get("DOCKER_HOST", "") or os.environ.get("RST_DOCKER_HOST", "")
    if BUILD_CMD == "docker" and not docker_host:
        sys.exit(
            "refusing to run: DOCKER_HOST is unset and the runtime is docker.\n"
            "Task Dockerfiles are untrusted third-party build scripts and must not be built\n"
            "on a shared root daemon. Run `source scripts/00b_setup_sandbox.sh`, which\n"
            "prefers rootless podman and exports the right socket."
        )
    # `--version` rather than `info --format`: podman's info schema is not Docker's
    # (.Version.Version vs .ServerVersion), so a Docker-shaped template errors out.
    ver = docker("--version")
    if ver.returncode != 0:
        sys.exit(f"cannot reach {BUILD_CMD}: {(ver.stderr or ver.stdout).strip()[:300]}")
    reachable = docker("info", timeout=120)
    if reachable.returncode != 0:
        sys.exit(f"{BUILD_CMD} present but not usable: {(reachable.stderr or '').strip()[:300]}")
    print(f"[runtime] {ver.stdout.strip()} host={docker_host or '(local cli)'}")

    rows = [json.loads(line)["metadata"] for line in
            (args.taskset / "rl_tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.sample:
        # stride-sample so the probe spans many base images, not just the first ones
        stride = max(1, len(rows) // args.sample)
        rows = rows[::stride][: args.sample]
    print(f"[plan] {len(rows)} tasks, {len({r['base_image'] for r in rows})} distinct base images")

    # Dockerfile heredocs (RUN <<EOF) need buildah >= 1.29 / podman >= 4.4. On older
    # podman they fail with `Unknown instruction: IF` or similar, which reads like a
    # broken task rather than a stale toolchain. Measured on the sweet pool: 31% of
    # Dockerfiles use heredocs, so an old podman silently loses a third of the pool.
    if BUILD_CMD == "podman":
        version = re.search(r"(\d+)\.(\d+)", ver.stdout or "")
        major, minor = (int(version.group(1)), int(version.group(2))) if version else (0, 0)
        heredoc = re.compile(r"^\s*(?:RUN|COPY)\s.*<<-?\s*[\'\"]?[A-Za-z_]", re.M)
        affected = 0
        for r in rows:
            df = Path(r["task_dir"]) / "environment" / "Dockerfile"
            if df.is_file() and heredoc.search(df.read_text(errors="replace")):
                affected += 1
        if affected and (major, minor) < (4, 4):
            print(f"[precheck] podman {major}.{minor} cannot parse Dockerfile heredocs, and "
                  f"{affected}/{len(rows)} ({affected/len(rows):.0%}) of these tasks use them.")
            print("[precheck] Install a newer rootless podman WITHOUT root -- a static build is")
            print("[precheck] ~32 MB and needs no package manager:")
            print("[precheck]   https://github.com/mgoltzsche/podman-static/releases (v5.8.4+)")
            print("[precheck]   tar xzf podman-linux-amd64.tar.gz -C $HOME/.local --strip-components=1")
            print("[precheck] Aborting rather than silently building only two thirds of the pool.")
            sys.exit(2)
        elif affected:
            print(f"[precheck] podman {major}.{minor} OK for the {affected} heredoc Dockerfiles")

    if not args.rebuild_existing:
        existing = set()
        listing = docker("images", "--format", "{{.Repository}}:{{.Tag}}", timeout=120)
        if listing.returncode == 0:
            existing = {line.strip() for line in listing.stdout.splitlines()}
        before = len(rows)
        rows = [r for r in rows if image_tag(r["task_id"]) not in existing]
        if before != len(rows):
            print(f"[skip] {before - len(rows)} images already present")

    results: list[dict] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_one, r, args.timeout, not args.no_pull): r for r in rows}
        for done, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append({"task_id": row["task_id"], "ok": False,
                                "reason": f"{type(exc).__name__}: {exc}"[:300]})
            if done % 25 == 0 or done == len(rows):
                ok = sum(1 for r in results if r.get("ok"))
                rate = done / max(1e-9, time.time() - started)
                remaining = (len(rows) - done) / max(1e-9, rate)
                print(f"  {done}/{len(rows)} ok={ok} "
                      f"{rate*60:.1f}/min eta={remaining/60:.0f}min", flush=True)

    ok_rows = [r for r in results if r.get("ok")]
    sizes = [s for s in (image_size_bytes(r["task_id"]) for r in ok_rows if not r["compose"]) if s]
    total = sum(sizes)
    report = {
        "runtime": BUILD_CMD,
        "docker_host": docker_host,
        "attempted": len(results),
        "built": len(ok_rows),
        "failed": len(results) - len(ok_rows),
        "compose_tasks": sum(1 for r in results if r.get("compose")),
        "wall_clock_sec": round(time.time() - started, 1),
        "measured_images": len(sizes),
        "measured_total_gib": round(total / 2**30, 2),
        "mean_image_gib": round(total / max(1, len(sizes)) / 2**30, 3),
        "median_build_sec": (sorted(r["seconds"] for r in ok_rows)[len(ok_rows)//2] if ok_rows else None),
        "failures": [{k: r.get(k) for k in ("task_id", "base_image", "reason")}
                     for r in results if not r.get("ok")][:100],
        "ready_task_ids": sorted(r["task_id"] for r in ok_rows),
    }
    if args.sample and sizes:
        # No extrapolated total on purpose: with shared base layers the honest
        # number cannot be computed from a sample, so we say what to measure
        # instead of publishing mean_image_gib * task_count as if it were one.
        report["extrapolation_note"] = (
            f"sampled {len(sizes)} of {args.sample} requested images. Layers are shared "
            "across tasks with the same base image, so the naive per-image sum "
            "double-counts and no total is extrapolated here. Compare `docker system df` "
            "before and after this probe for the true incremental cost."
        )
    out = args.taskset / "prebuild_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "ready_task_ids"}, indent=2)[:2500])
    print(f"\nwrote {out}")
    print("True disk cost (shared layers accounted): run `docker system df` and compare.")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
