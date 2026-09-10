#!/usr/bin/env python3
"""Audit every messages row against local model tokenizers, without truncation.

Input JSON: {"models": {key: {"path": ...}}, "datasets": [{"key": ...,
"split": ..., "path": ...}]}. Original artifacts are read-only. Equal tokenizer
fingerprints share computation; model-weight compatibility requires a separate
training probe. Chunk receipts allow interruption/resume with identical inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rst_common.tokenization import (
    load_training_tokenizer, mask_type_for_model, template_loss_mask, tokenization_identity,
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


_TOKENIZERS = {}


def init_worker(models: dict) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.nice(10)
    for key, model in models.items():
        tok = load_training_tokenizer(model["path"])
        config = json.loads((Path(model["path"]) / "config.json").read_text())
        vocab = config.get("text_config", config)["vocab_size"]
        _TOKENIZERS[key] = (tok, mask_type_for_model(model["path"]), vocab)


def check_chunk(rows: list[dict], start: int) -> dict:
    """Run the production masking implementation, preserving every row outcome."""
    output = {}
    for key, (tok, profile, vocab) in _TOKENIZERS.items():
        checks = []
        for offset, row in enumerate(rows):
            result = {"row": start + offset}
            try:
                ids, mask = template_loss_mask(tok, row["messages"], profile)
                if not ids or len(ids) != len(mask):
                    raise ValueError("empty or misaligned input_ids/loss_mask")
                if min(ids) < 0 or max(ids) >= vocab:
                    raise ValueError("token ID outside model vocabulary")
                if mask[0] != 0 or any(m not in (0, 1) for m in mask):
                    raise ValueError("invalid assistant loss mask")
                trained = sum(mask)
                result.update(status="ok" if trained else "no_trained_tokens",
                              tokens=len(ids), trained=trained)
            except Exception as exc:
                result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            checks.append(result)
        output[key] = checks
    return {"start": start, "rows": len(rows), "models": output}


def summarize(root: Path, plan: dict) -> dict:
    summaries = []
    for dataset in plan["datasets"]:
        name = dataset["key"] + "--" + dataset["split"]
        stats = {key: {"rows_checked": 0, "status": Counter(), "errors": Counter(),
                       "usable_8192": 0, "usable_32768": 0, "total_tokens": 0,
                       "trained_tokens": 0, "max_tokens": 0, "first_usable_row": None,
                       "first_error_row": None} for key in plan["unique_models"]}
        covered = 0
        for path in sorted((root / "chunks" / name).glob("*.json")):
            chunk = json.loads(path.read_text())
            if chunk["start"] != covered:
                raise ValueError(f"missing/overlapping chunk in {name} at row {covered}")
            covered += chunk["rows"]
            for key, rows in chunk["models"].items():
                s = stats[key]
                for row in rows:
                    s["rows_checked"] += 1
                    s["status"][row["status"]] += 1
                    if row["status"] == "error":
                        s["errors"][row["error"]] += 1
                        if s["first_error_row"] is None:
                            s["first_error_row"] = row["row"]
                    else:
                        s["max_tokens"] = max(s["max_tokens"], row["tokens"])
                        s["total_tokens"] += row["tokens"]
                        s["trained_tokens"] += row["trained"]
                        if row["status"] == "ok":
                            for limit in (8192, 32768):
                                s[f"usable_{limit}"] += int(row["tokens"] <= limit)
                            if row["tokens"] <= 32768 and s["first_usable_row"] is None:
                                s["first_usable_row"] = row["row"]
        if covered != dataset["rows"]:
            raise ValueError(f"incomplete {name}: {covered}/{dataset['rows']}")
        summaries.append({**dataset, "models": {key: stats[rep] for key, rep in plan["aliases"].items()}})
    return {"scope": "all rows; official tokenizers; no truncation or model weights",
            "plan_sha256": sha256(root / "plan.json"), "datasets": summaries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("workers and batch-size must be positive")
    import pyarrow.parquet as pq

    plan = json.loads(args.inventory.read_text())
    plan["implementation"] = {str(p): sha256(p) for p in (
        Path(__file__), Path(__file__).resolve().parent.parent / "rst_common/tokenization.py")}
    plan["batch_size"] = args.batch_size
    aliases, unique, fingerprints = {}, {}, {}
    for key, model in plan["models"].items():
        tok = load_training_tokenizer(model["path"])
        profile = mask_type_for_model(model["path"])
        identity = tokenization_identity(tok, profile)
        model["tokenization"] = identity
        config = json.loads((Path(model["path"]) / "config.json").read_text())
        vocab = config.get("text_config", config)["vocab_size"]
        group = (identity["fingerprint"], vocab)
        if group not in fingerprints:
            unique[key] = model
            fingerprints[group] = key
        aliases[key] = fingerprints[group]
    plan.update(unique_models=unique, aliases=aliases)
    for dataset in plan["datasets"]:
        path = Path(dataset["path"])
        pf = pq.ParquetFile(path)
        if "messages" not in pf.schema_arrow.names:
            raise ValueError(f"{path}: messages column required; never reuse foreign token IDs")
        dataset.update(sha256=sha256(path), rows=pf.metadata.num_rows)
    if len({(d["key"], d["split"]) for d in plan["datasets"]}) != len(plan["datasets"]):
        raise ValueError("duplicate dataset key/split")
    target = args.out / "plan.json"
    if target.exists() and json.loads(target.read_text()) != plan:
        raise ValueError("resume inputs, code, or tokenizers changed; choose a new output directory")
    write_json(target, plan)
    total = sum(d["rows"] for d in plan["datasets"])
    completed, futures = 0, {}
    start_time = time.monotonic()

    def collect() -> None:
        nonlocal completed
        done, _ = wait(futures, return_when=FIRST_COMPLETED)
        for future in done:
            path = futures.pop(future)
            result = future.result()
            write_json(path, result)
            completed += result["rows"]
        write_json(args.out / "progress.json", {"rows_completed": completed, "rows_total": total,
                                                "seconds": time.monotonic() - start_time})

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(unique,)) as pool:
        for dataset in plan["datasets"]:
            name = dataset["key"] + "--" + dataset["split"]
            offset = 0
            for batch in pq.ParquetFile(dataset["path"]).iter_batches(
                    batch_size=args.batch_size, columns=["messages"]):
                path = args.out / "chunks" / name / f"{offset:09d}.json"
                if path.exists():
                    previous = json.loads(path.read_text())
                    if previous["start"] != offset or previous["rows"] != batch.num_rows:
                        raise ValueError(f"invalid receipt: {path}")
                    completed += batch.num_rows
                else:
                    futures[pool.submit(check_chunk, batch.to_pylist(), offset)] = path
                    if len(futures) >= args.workers * 2:
                        collect()
                offset += batch.num_rows
            print(f"submitted {name}: {offset} rows; completed={completed}/{total}", flush=True)
        while futures:
            collect()
    result = summarize(args.out, plan)
    result["elapsed_seconds"] = time.monotonic() - start_time
    write_json(args.out / "result.json", result)
    print(f"completed all {total} rows across {len(plan['models'])} models", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
