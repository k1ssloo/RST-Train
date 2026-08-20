#!/usr/bin/env python3
"""Measure a checkpoint with NO container, NO sandbox and NO network.

    python scripts/06b_eval_offline.py \
        --model-path $BASE_FOLDER/out-hf-full \
        --holdout    $BASE_FOLDER/sft-v1-cap10/pretokenized_holdout.parquet \
        --out        $BASE_FOLDER/eval/offline

WHY THIS EXISTS
    Agentic evaluation needs to build a task Dockerfile and drive a tmux session
    inside it. On a pod whose AppArmor profile denies mount(2) that is impossible,
    and no package fixes it (see scripts/00c_probe_sandbox.py). SFT still trains
    fine. The failure mode to avoid is then reporting a checkpoint as good because
    the loss curve looked nice.

    This script exists so "we could not measure it" is never the answer. It is a
    WEAKER signal than a benchmark and the output says so in the artifact itself.

WHAT IT CAN AND CANNOT TELL YOU
    CAN:    did the fine-tune move the model at all, in the direction the data
            asked for -- lower held-out NLL and higher next-token accuracy on
            exactly the supervised spans, versus the base model.
    CAN:    can the model still emit the RST action protocol? Terminus-2 parses a
            JSON object with `analysis`, `plan` and a `commands` list; anything
            else is a dead turn. A checkpoint that has lost the protocol is
            broken, and that is visible here with no container at all.
    CAN:    given the exact context the expert saw, does the model choose the same
            first keystrokes?
    CANNOT: whether the agent would actually solve a task. Terminal work is a
            closed loop -- recover from a wrong command, notice a failing test,
            try a second approach. Agreement with a recorded expert on turn N says
            nothing about what happens at turn N+1 after a mistake. Do not convert
            any number here into a pass-rate claim.

THE TWO MEASUREMENT FAMILIES
    A. teacher-forced scoring        needs input_ids + loss_mask
    B. action-protocol generation    needs `messages` (so: the parquet from
                                     03_build_sft_data.py, not the pretokenized one)
    Give it the pretokenized parquet and you get A only, with B marked
    unavailable-and-why rather than silently missing.

MEMORY NOTE (the reason this is not four lines of transformers)
    A 32k sequence times a ~150k vocab is ~10 GB of bf16 logits for ONE row, so
    `model(input_ids, labels=...)` OOMs before it computes anything. We run the
    decoder, then apply the LM head in slices over the sequence, accumulating NLL
    and top-1 hits. Peak logit memory becomes chunk*vocab instead of T*vocab.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


_SIBLINGS: dict[str, Any] = {}


def _load_sibling(stem: str) -> Any:
    """Import a sibling script whose module name starts with a digit.

    Done by path, deliberately: `normalize_assistant` here MUST be the same
    function that built the training data, and `qwen3_5_mask` the same one that
    pretokenized it. A second copy of either would drift.
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


# ------------------------------------------------------------------ B: actions

def strip_think(text: str) -> tuple[str, str]:
    """Drop a reasoning block; return the remainder and the block's state.

    State is "none", "closed" (stripped, parse the remainder) or "open".

    The chat template opens `<think>\\n` as part of the generation prompt and the
    training target is the bare action JSON, so a trained checkpoint emits the
    object immediately. A base or under-trained model reasons in prose first.
    Closed block -> parse what follows. Never closed -> there is no action to
    parse, and scoring that as "lost the protocol" would be wrong; it ran out of
    budget mid-thought. Measured on this holdout: 0 of 578 expert turns contain a
    think block, so any reasoning prose here comes from the model, not the data.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1], "closed"
    if "<think>" in text:
        return text, "open"
    return text, "none"


def action_of(text: str) -> dict | None:
    """Parse one assistant output into the RST action dict, or None."""
    builder = _load_sibling("03_build_sft_data")
    canonical, _rewritten, _reason = builder.normalize_assistant(text)
    if canonical is None:
        return None
    try:
        obj = json.loads(canonical)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def keystrokes_of(action: dict | None) -> list[str]:
    if not action:
        return []
    out: list[str] = []
    for command in action.get("commands") or []:
        if isinstance(command, dict):
            out.append(str(command.get("keystrokes", "")))
    return out


def compare_actions(reference: dict, predicted: dict | None) -> dict[str, bool]:
    """Score one predicted action against the expert's.

    `analysis` and `plan` are free prose and are deliberately NOT compared: two
    different words for the same plan are not a disagreement. What is comparable
    is the executable part.
    """
    ref_keys, pred_keys = keystrokes_of(reference), keystrokes_of(predicted)
    return {
        "parsed": predicted is not None,
        "commands_exact": bool(pred_keys) and pred_keys == ref_keys,
        "first_keystrokes_exact": bool(pred_keys) and bool(ref_keys)
        and pred_keys[0] == ref_keys[0],
        "command_count_match": len(pred_keys) == len(ref_keys),
        "task_complete_agreement": bool(predicted is not None)
        and bool(reference.get("is_task_complete")) == bool(predicted.get("is_task_complete")),
    }


def pick_turn_indices(messages: list[dict], per_row: int) -> list[int]:
    """Indices of supervised assistant turns to probe, spread across the episode.

    Spread rather than "the first N": early turns are almost always `ls`/`cat`
    orientation moves, so first-turn-only agreement is both easy and uninformative.
    """
    supervised = [
        i for i, m in enumerate(messages)
        if m.get("role") == "assistant" and m.get("step_loss_mask", 1) == 1
    ]
    if not supervised or per_row <= 0:
        return []
    if len(supervised) <= per_row:
        return supervised
    step = (len(supervised) - 1) / (per_row - 1) if per_row > 1 else 0
    return [supervised[round(k * step)] for k in range(per_row)]


# --------------------------------------------------------------- model loading

AUTO_CLASSES = ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel")


def load_model(model_path: str, dtype_name: str) -> tuple[Any, str]:
    """Load whatever kind of model this checkpoint is, and say which class worked."""
    import torch
    import transformers

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_name]
    errors: list[str] = []
    for name in AUTO_CLASSES:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(
                model_path, dtype=dtype, device_map="auto", trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001 - we are probing which class fits
            errors.append(f"{name}: {type(exc).__name__}: {exc}"[:300])
            continue
        model.eval()
        return model, name
    raise SystemExit("could not load the checkpoint with any auto class:\n  " + "\n  ".join(errors))


# ------------------------------------------------------- A: teacher-forced loss

def score_rows(model, rows: list[tuple[list[int], list[int]]], *, chunk: int,
               progress_every: int = 20) -> dict[str, Any]:
    """Sum NLL and top-1 hits over supervised positions, without materializing logits.

    `loss_mask[i] == 1` means token i is a supervised TARGET (see the dataset
    manifest note: the mask is aligned 1:1 with input_ids, no offset). Token i is
    predicted from hidden state i-1, so position i contributes
    `-log p(ids[i] | ids[:i])` using hidden[i-1].
    """
    import torch

    decoder = model.get_decoder() if hasattr(model, "get_decoder") else None
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if decoder is None or head is None:
        raise SystemExit(
            "this model exposes no get_decoder()/get_output_embeddings(), so the "
            "chunked scoring path cannot run. Scoring it would need full logits "
            "(~10GB per 32k row) -- reduce --max-seq-len drastically, or add an "
            "explicit path for this architecture."
        )

    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    per_row: list[dict[str, float]] = []
    started = time.time()

    with torch.inference_mode():
        for index, (ids, mask) in enumerate(rows, 1):
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            hidden = decoder(input_ids=input_ids).last_hidden_state[0]  # [T, H]
            targets = torch.tensor(ids, dtype=torch.long, device=hidden.device)
            supervised = torch.tensor(mask, dtype=torch.bool, device=hidden.device)
            # Position 0 has no predecessor, so it can never be a target.
            supervised[0] = False
            positions = torch.nonzero(supervised, as_tuple=False).flatten()
            row_nll = 0.0
            row_correct = 0
            for start in range(0, positions.numel(), chunk):
                block = positions[start : start + chunk]
                logits = head(hidden[block - 1]).float()
                # `head` and `decoder` can sit on different devices when the model
                # is sharded across GPUs, so the targets have to follow the logits
                # rather than the hidden states.
                gold = targets[block].to(logits.device)
                row_nll += torch.nn.functional.cross_entropy(
                    logits, gold, reduction="sum").item()
                row_correct += int((logits.argmax(-1) == gold).sum().item())
                del logits
            n = int(positions.numel())
            total_nll += row_nll
            total_tokens += n
            total_correct += row_correct
            per_row.append({
                "n_supervised": n,
                "loss": round(row_nll / n, 6) if n else None,
                "top1": round(row_correct / n, 6) if n else None,
            })
            del hidden, input_ids
            if index % progress_every == 0 or index == len(rows):
                rate = index / max(1e-9, time.time() - started)
                print(f"  scored {index}/{len(rows)} rows  "
                      f"loss={total_nll / max(1, total_tokens):.4f}  "
                      f"{rate * 60:.1f} rows/min", flush=True)

    mean_loss = total_nll / total_tokens if total_tokens else float("nan")
    return {
        "rows": len(rows),
        "supervised_tokens": total_tokens,
        "loss": round(mean_loss, 6),
        "perplexity": round(math.exp(mean_loss), 4) if math.isfinite(mean_loss) else None,
        "top1_accuracy": round(total_correct / total_tokens, 6) if total_tokens else None,
        "seconds": round(time.time() - started, 1),
        "per_row": per_row,
    }


# ---------------------------------------------------------- B: greedy generation

def probe_actions(model, tokenizer, frame, *, max_actions: int, per_row: int,
                  gen_tokens: int, max_prompt_tokens: int) -> dict[str, Any]:
    """Greedy-decode one action per probed turn and compare it to the expert's."""
    import torch

    tallies = {k: 0 for k in ("parsed", "commands_exact", "first_keystrokes_exact",
                              "command_count_match", "task_complete_agreement")}
    attempted = 0
    skipped = {"prompt_too_long": 0, "reference_unparseable": 0}
    # A generation that hit the cap is NOT evidence that the model lost the
    # protocol -- it is evidence that --gen-tokens is too small for this model.
    # Measured on the holdout: expert turns are p50=225, p90=406, p99=890 tokens,
    # so the default leaves headroom; a model that reasons in prose first can still
    # blow past it. Counted separately so the parse rate stays interpretable.
    shape = {"truncated": 0, "unterminated_think": 0, "had_think_block": 0}
    samples: list[dict[str, Any]] = []
    started = time.time()

    with torch.inference_mode():
        for row in frame.itertuples():
            if attempted >= max_actions:
                break
            messages = [dict(m) for m in row.messages]
            for turn in pick_turn_indices(messages, per_row):
                if attempted >= max_actions:
                    break
                reference = action_of(messages[turn].get("content", ""))
                if reference is None:
                    # The expert turn itself does not parse. That is a data
                    # problem, not a model problem -- do not score it either way.
                    skipped["reference_unparseable"] += 1
                    continue
                prompt_ids = tokenizer.apply_chat_template(
                    messages[:turn], tokenize=True, add_generation_prompt=True,
                    return_dict=False,
                )
                if len(prompt_ids) > max_prompt_tokens:
                    skipped["prompt_too_long"] += 1
                    continue
                device = next(model.parameters()).device
                out = model.generate(
                    input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=device),
                    max_new_tokens=gen_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                new_ids = out[0][len(prompt_ids):]
                truncated = len(new_ids) >= gen_tokens
                raw = tokenizer.decode(new_ids, skip_special_tokens=True)
                text, think = strip_think(raw)
                shape["truncated"] += int(truncated)
                shape["unterminated_think"] += int(think == "open")
                shape["had_think_block"] += int(think != "none")
                scores = compare_actions(reference, action_of(text))
                for key, hit in scores.items():
                    tallies[key] += int(hit)
                attempted += 1
                if len(samples) < 5:
                    samples.append({
                        "trajectory_id": getattr(row, "trajectory_id", None),
                        "turn": turn,
                        "reference_keystrokes": keystrokes_of(reference)[:4],
                        "predicted_raw_head": raw[:400],
                        "think_block": think,
                        "truncated": truncated,
                        "scores": scores,
                    })
                if attempted % 10 == 0:
                    print(f"  probed {attempted}/{max_actions} actions  "
                          f"parse={tallies['parsed'] / attempted:.1%}", flush=True)

    rates = {f"{k}_rate": (round(v / attempted, 4) if attempted else None)
             for k, v in tallies.items()}
    unparsed = attempted - tallies["parsed"]
    if not attempted:
        note = "no turns were probed"
    elif unparsed == 0:
        note = "every probed turn produced a valid RST action object"
    elif shape["truncated"] >= unparsed:
        note = (f"{unparsed}/{attempted} turns did not parse, and {shape['truncated']} hit "
                f"the {gen_tokens}-token cap -- raise --gen-tokens before concluding "
                f"anything about the protocol")
    else:
        note = (f"{unparsed}/{attempted} turns did not parse and only {shape['truncated']} "
                f"were truncated, so at least "
                f"{unparsed - shape['truncated']} are genuine protocol failures")
    return {
        "actions_attempted": attempted,
        "counts": tallies,
        **rates,
        "skipped": skipped,
        "generation_shape": shape,
        "truncated_rate": (round(shape["truncated"] / attempted, 4) if attempted else None),
        "gen_tokens": gen_tokens,
        "note": note,
        "seconds": round(time.time() - started, 1),
        "examples": samples,
    }


# ------------------------------------------------------------------- data load

def read_holdout(path: Path, tokenizer_path: str | None, max_seq_len: int,
                 max_rows: int) -> tuple[Any, list[tuple[list[int], list[int]]], dict[str, Any]]:
    """Return (frame, scored_rows, info). Accepts either holdout parquet shape."""
    import pandas as pd

    frame = pd.read_parquet(path)
    columns = list(frame.columns)
    info: dict[str, Any] = {"path": str(path), "rows_in_file": int(len(frame)),
                            "columns": columns}
    if max_rows and len(frame) > max_rows:
        frame = frame.head(max_rows)
    info["rows_used"] = int(len(frame))

    # A row longer than --max-seq-len is DROPPED, not shortened. Truncating changes
    # what is being scored: cut the tail off a trajectory and the held-out loss is
    # measured over a different, easier target than the one the row describes (see
    # 17_build_dpo_data.py --max-seq-len: "a truncated trajectory is a different
    # trajectory"). The drop is deterministic given max_seq_len, so two checkpoints
    # scored with the same value are still comparable -- which is why the count and
    # the value both go into the results json.
    too_long = 0

    if "input_ids" in columns and "loss_mask" in columns:
        info["source"] = "pretokenized"
        rows = []
        for r in frame.itertuples():
            ids, mask = list(r.input_ids), list(r.loss_mask)
            if len(ids) > max_seq_len:
                too_long += 1
                continue
            rows.append((ids, mask))
    elif "messages" in columns:
        if not tokenizer_path:
            raise SystemExit(
                f"{path} holds `messages`, so scoring needs a tokenizer: pass "
                f"--tokenizer. (Or point --holdout at pretokenized_holdout.parquet.)"
            )
        from transformers import AutoTokenizer

        info["source"] = "messages+tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        masker = _load_sibling("15_export_pretokenized")
        rows, dropped = [], 0
        for r in frame.itertuples():
            try:
                ids, mask = masker.qwen3_5_mask(tokenizer, [dict(m) for m in r.messages])
            except ValueError:
                dropped += 1
                continue
            if len(ids) > max_seq_len:
                too_long += 1
                continue
            rows.append((ids, mask))
        info["dropped_contract_failures"] = dropped
    else:
        raise SystemExit(
            f"{path} has neither (input_ids, loss_mask) nor messages; columns={columns}"
        )
    info["max_seq_len"] = int(max_seq_len)
    info["dropped_too_long"] = too_long
    before_unsupervised = len(rows)
    rows = [(i, m) for i, m in rows if sum(m) > 0]
    info["dropped_no_trained_tokens"] = before_unsupervised - len(rows)
    info["rows_scored"] = len(rows)
    if too_long:
        print(f"[holdout] dropped {too_long} row(s) longer than {max_seq_len} tokens "
              f"(not truncated -- a truncated trajectory is a different trajectory). "
              f"Raise --max-seq-len to score them.")
    if not rows:
        raise SystemExit(
            f"no scorable rows left from {path}: {info['rows_used']} used, "
            f"{too_long} over --max-seq-len={max_seq_len}, "
            f"{info['dropped_no_trained_tokens']} with no trained tokens."
        )
    return frame, rows, info


# ------------------------------------------------------------------------ main

INTERPRETATION = {
    "substitutes_for_benchmark": False,
    "what_it_measures": (
        "agreement with recorded expert trajectories under teacher forcing, plus "
        "whether the model can still emit the RST JSON action protocol"
    ),
    "what_it_does_not_measure": (
        "task success. The agent loop is closed: recovering from a wrong command "
        "is most of the difficulty, and no teacher-forced metric sees it. Never "
        "restate a number here as a pass rate."
    ),
    "how_to_report": (
        "If agentic eval was blocked, say so explicitly and cite the cause "
        "(scripts/00b_setup_sandbox.sh --diagnose). Report these numbers as a "
        "sanity check that training did what the data asked, not as a result."
    ),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="HF checkpoint to score")
    p.add_argument("--base-model", default=None,
                   help="also score this (the un-finetuned model) for a delta -- "
                        "the only comparison that answers 'did SFT do anything'")
    p.add_argument("--holdout", type=Path, required=True,
                   help="pretokenized_holdout.parquet, or rst_sft_holdout.parquet "
                        "with --tokenizer")
    p.add_argument("--tokenizer", default=None,
                   help="needed only for a `messages` parquet, or for the action probe")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--max-rows", type=int, default=0, help="0 = all")
    p.add_argument("--max-seq-len", type=int, default=32768)
    p.add_argument("--chunk", type=int, default=2048,
                   help="supervised positions per LM-head slice; lower it if OOM")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--max-actions", type=int, default=120,
                   help="0 disables the action probe (family B)")
    p.add_argument("--turns-per-row", type=int, default=2)
    # 1024, not 384: expert turns on this holdout run p50=225 / p90=406 / p99=890
    # tokens, so a smaller cap truncates the long ones and quietly depresses every
    # agreement rate. Truncation is counted and reported either way.
    p.add_argument("--gen-tokens", type=int, default=1024)
    p.add_argument("--max-prompt-tokens", type=int, default=16384)
    p.add_argument("--dry-run", action="store_true",
                   help="validate inputs and print the plan; load no model")
    args = p.parse_args()

    frame, rows, data_info = read_holdout(
        args.holdout, args.tokenizer, args.max_seq_len, args.max_rows)
    can_probe_actions = "messages" in data_info["columns"] and args.max_actions > 0
    action_unavailable = None
    if args.max_actions > 0 and not can_probe_actions:
        action_unavailable = (
            f"{args.holdout} holds token ids only, and the action probe needs the "
            f"conversation text. Point --holdout at rst_sft_holdout.parquet (with "
            f"--tokenizer) to get it."
        )

    print(f"[data] {data_info['rows_scored']} scorable rows from {args.holdout} "
          f"({data_info['source']}), "
          f"{sum(sum(m) for _, m in rows):,} supervised tokens")
    print("[plan] teacher-forced scoring: yes")
    print(f"[plan] action probe: "
          f"{'yes' if can_probe_actions else 'no -- ' + str(action_unavailable)}")
    if args.dry_run:
        print("[dry-run] stopping before any model load")
        return 0

    tokenizer = None
    if can_probe_actions:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_path)

    results: dict[str, Any] = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "data": data_info,
        "settings": {
            "max_seq_len": args.max_seq_len, "chunk": args.chunk, "dtype": args.dtype,
            "max_actions": args.max_actions, "turns_per_row": args.turns_per_row,
            "gen_tokens": args.gen_tokens,
        },
        "interpretation": INTERPRETATION,
    }

    model, auto_class = load_model(args.model_path, args.dtype)
    print(f"[model] loaded {args.model_path} via {auto_class}")
    results["auto_class"] = auto_class
    results["scoring"] = score_rows(model, rows, chunk=args.chunk)
    print(f"[scoring] loss={results['scoring']['loss']} "
          f"ppl={results['scoring']['perplexity']} "
          f"top1={results['scoring']['top1_accuracy']}")

    if can_probe_actions:
        results["actions"] = probe_actions(
            model, tokenizer, frame, max_actions=args.max_actions,
            per_row=args.turns_per_row, gen_tokens=args.gen_tokens,
            max_prompt_tokens=args.max_prompt_tokens)
        a = results["actions"]
        print(f"[actions] parse={a['parsed_rate']} "
              f"commands_exact={a['commands_exact_rate']} "
              f"first_keystrokes={a['first_keystrokes_exact_rate']} "
              f"over {a['actions_attempted']} turns "
              f"(truncated={a['truncated_rate']})")
        print(f"[actions] {a['note']}")
    elif action_unavailable:
        results["actions"] = {"available": False, "reason": action_unavailable}

    if args.base_model:
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cache clearing is best effort
            pass
        base, base_class = load_model(args.base_model, args.dtype)
        print(f"[model] loaded base {args.base_model} via {base_class}")
        results["base_scoring"] = score_rows(base, rows, chunk=args.chunk)
        delta = results["scoring"]["loss"] - results["base_scoring"]["loss"]
        results["delta_vs_base"] = {
            "loss": round(delta, 6),
            "top1_accuracy": round(
                (results["scoring"]["top1_accuracy"] or 0)
                - (results["base_scoring"]["top1_accuracy"] or 0), 6),
            "reading": ("SFT lowered held-out NLL on the supervised spans"
                        if delta < 0 else
                        "SFT did NOT lower held-out NLL -- suspect the loss mask, "
                        "the LR, or a train/serve template mismatch"),
        }
        print(f"[delta] loss {results['base_scoring']['loss']} -> "
              f"{results['scoring']['loss']} ({delta:+.4f})")

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "offline_results.json"
    target.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {target}")
    print("REMINDER: this is not a benchmark score. If agentic eval was blocked, the")
    print("report must say the benchmark did not run -- see the interpretation block.")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
