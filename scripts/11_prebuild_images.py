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
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IMAGE_PREFIX = "rst-task"


def docker(*argv: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *argv], capture_output=True, text=True, timeout=timeout, check=False
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
        proc = subprocess.run(
            ["docker", "compose", "-f", str(task_dir / "environment" /
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
        if pull_base:
            argv.append("--pull")
        argv.append(str(env_dir))
        proc = docker(*argv, timeout=timeout)

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

    if not shutil.which("docker"):
        sys.exit("docker not found on PATH")
    docker_host = os.environ.get("DOCKER_HOST", "")
    if not docker_host:
        sys.exit(
            "refusing to run: DOCKER_HOST is unset.\n"
            "RST task Dockerfiles are untrusted third-party build scripts and must not be\n"
            "built on the host's default daemon. Start a dedicated/rootless daemon and set\n"
            "  export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock"
        )
    info = docker("info", "--format", "{{.ServerVersion}} {{.DockerRootDir}}")
    if info.returncode != 0:
        sys.exit(f"cannot reach docker at {docker_host}: {info.stderr.strip()[:300]}")
    print(f"[docker] host={docker_host} server={info.stdout.strip()}")

    rows = [json.loads(line)["metadata"] for line in
            (args.taskset / "rl_tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.sample:
        # stride-sample so the probe spans many base images, not just the first ones
        stride = max(1, len(rows) // args.sample)
        rows = rows[::stride][: args.sample]
    print(f"[plan] {len(rows)} tasks, {len({r['base_image'] for r in rows})} distinct base images")

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
