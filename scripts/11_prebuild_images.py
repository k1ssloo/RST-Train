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
            argv[:0] = ["--format", "docker"]
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
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

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

    if args.skip_existing:
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
        full = len(rows) if not args.sample else None
        report["extrapolation_note"] = (
            "NOTE: layers are shared across tasks with the same base image, so the "
            "naive per-image sum double-counts. Compare `docker system df` before and "
            "after this probe for the true incremental cost."
        )
    out = args.taskset / "prebuild_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "ready_task_ids"}, indent=2)[:2500])
    print(f"\nwrote {out}")
    print("True disk cost (shared layers accounted): run `docker system df` and compare.")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
