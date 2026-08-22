#!/usr/bin/env python3
"""Convert AI2's TMax terminal-agent trajectories into this repo's SFT format.

    python scripts/03e_build_tmax_sft.py \
        --source data/tmax-sft/source-only-success.parquet \
        --tokenizer data/Qwen3.5-27B-tokenizer \
        --out-dir data/tmax-sft --max-seq-len 32768

WHICH DATASET, AND WHY NOT THE ONE IN THE TITLE
-----------------------------------------------
`allenai/TMax-15K` (and `allenai/tmax-15k-open-instruct`, and, despite its name,
`allenai/TMax-SFT-16.5K`) contain ZERO assistant turns. They are RL environment
instances -- task descriptions plus graders -- which is what the TMax paper says
they are ("roughly 15k RL environment instances"). Measured: no assistant turn in
any row of any of the three. They cannot be SFT'd, and they cannot be turned into
preference pairs, because there are no responses in them at all. They are the
input to `10_build_rl_taskset.py`, not to this script.

The trajectories are in `allenai/tmax-sft`, which ships two configs:

    skill_tax_20260505_2.2k_combined_balanced_thinking_all           10,726 rows
    skill_tax_20260505_2.2k_combined_balanced_thinking_only_success    5,795 rows

`only_success` is the verifier-labelled subset, and it is what this script wants.
The 4,931 rows that are in `all` but not in `only_success` are trajectories that
did not solve their task; the `has_task_complete` flag does NOT identify them
(it is 2,506 True / 2,425 False among them -- it records that the agent *claimed*
completion, not that it succeeded).

WHY SFT ONLY, AND NOT DPO
-------------------------
A preference pair needs two responses to the SAME prompt that differ in quality.
Keying the two configs on `task` gives:

    tasks                                     2,020
    tasks with >= 1 success                   1,136
    tasks with >= 1 failure                   1,175
    tasks with BOTH  (i.e. pairable)            291      <- 14.4 %

1,136 + 1,175 - 291 = 2,020 exactly, which is the whole story: a TMax task is
almost always either always solved or never solved, so the success/failure split
mostly measures task difficulty rather than response quality. Pairing yields 444
pairs at min(successes, failures) per task, or 1,746 as a full cross-product
against 291 distinct prompts. The existing DPO stage trains on 2,673 pairs, so
this would be a sixth of the data over an eighth of the prompts, and the pairs
that do exist are contaminated by the difficulty confound. That is not worth a
27B DPO run, so this script builds SFT data and nothing else.

WHAT THE CONVERSION ACTUALLY DOES
---------------------------------
Upstream is already the same agent contract this repo trains on -- one persistent
`bash` tool, one command per turn, a THOUGHT section before the action -- but it
is packaged as native tool-calling (`tool_calls` on the assistant, `role="tool"`
observations, a separate `tools` column) plus a `reasoning_content` field. Three
things follow:

  * 71.3 % of assistant tokens are `reasoning_content`, and the Qwen3.5 template
    emits reasoning ONLY for assistant turns after the last `user` turn
    (tokenizer_config chat_template line 99). Rendered naively, a trajectory with
    a mid-conversation user turn loses the CoT of every turn before it.

  * Every mid-conversation user turn -- all 409 of them, no exceptions -- is a
    harness scolding that begins "Format error: Your last response did not
    include a `bash` tool call." 407 of the 409 directly follow an assistant turn
    that has no `tool_calls`. So the pattern is a malformed turn, a scolding, and
    a retry. `splice_format_errors` drops the malformed turn and the scolding,
    which is the same repair `03d_build_openthoughts_sft.py` applies to
    "Previous response had warnings:" preambles, and for the same reason: keeping
    them trains the model to emit the error and to expect to be told off. After
    the splice all 5,795 trajectories have exactly one user turn, so every
    assistant turn keeps its full reasoning.

  * 528 assistant turns carry a surplus `</think>` inside `content` while also
    carrying `reasoning_content`. Rendered as-is the turn closes a think block the
    template already closed. `repair_stray_think` drops the tag where that is
    unambiguous (483 turns) and refuses the trajectory where upstream emitted the
    THOUGHT several times over and there is no knowing which one it meant (45).

  * `messages` is pre-rendered down to `{role, content}` over roles
    system/user/assistant -- the tool schema baked into the system turn, the
    tool-call XML and the `<think>` block inlined into assistant content, and
    each `role="tool"` observation turned into a `user` turn carrying
    `<tool_response>...</tool_response>`. This is not cosmetic: it is what lets
    `15_export_pretokenized.py`, `qwen3_5_mask` and the verl dataset consume this
    data with no code change, since they take `messages` alone and never a
    `tools` argument. It is safe only because it is byte-checked -- for EVERY row
    this script asserts

        apply_chat_template(pre_rendered)  ==  apply_chat_template(native, tools=tools)

    so the model sees exactly the tokens it would have seen from the native
    shape, and a template change upstream turns into a hard failure here instead
    of a silent divergence between training and serving.

The `<think>\\n` that opens each assistant turn is left to the loss mask, which
already skips it (`15_export_pretokenized.py`'s THINK_PREFIX): the template's
`add_generation_prompt` emits `<|im_start|>assistant\\n<think>\\n`, so that prefix
is given to the model at every step of a rollout and is prompt, not target.

Upstream is ODC-BY; the trajectories were generated with Qwen/Qwen3.6-27B. Every
count this prints goes into `manifest.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SOURCE_DATASET = "allenai/tmax-sft"
SOURCE_CONFIG = "skill_tax_20260505_2.2k_combined_balanced_thinking_only_success"
SOURCE_FILE = f"{SOURCE_CONFIG}/train-00000-of-00001.parquet"
SOURCE_LICENSE = "odc-by"

# The scolding that marks a malformed assistant turn. Matched as a prefix rather
# than a regex because upstream emits one fixed string -- measured: all 409
# mid-conversation user turns start with exactly this.
FORMAT_ERROR_PREFIX = "Format error:"

ASSISTANT_HEADER = "<|im_start|>assistant\n"
SYSTEM_HEADER = "<|im_start|>system\n"
TURN_END = "<|im_end|>\n"


def load_exporter() -> Any:
    """Load `15_export_pretokenized.py` for `qwen3_5_mask`.

    By path because `scripts/` is not a package and the filename starts with a
    digit. Shared rather than reimplemented so the mask this script gates on is
    the same one the training data is finally built with -- two copies of a mask
    contract are two contracts, and this is the one defect in the pipeline that
    no loss curve would reveal.
    """
    path = Path(__file__).resolve().parent / "15_export_pretokenized.py"
    spec = importlib.util.spec_from_file_location("_rst_pretokenize", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        sys.exit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_plain(messages: Any) -> list[dict[str, Any]]:
    """Normalize one upstream `messages` value into plain Python.

    pyarrow hands back numpy arrays for the repeated fields and its own mapping
    types for the structs; the Jinja template needs real lists and dicts, and
    `json.dumps` (used for the content hash) needs them too.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        plain: dict[str, Any] = {
            "role": str(message["role"]),
            "content": message.get("content") or "",
        }
        reasoning = message.get("reasoning_content")
        if reasoning:
            plain["reasoning_content"] = str(reasoning)
        calls = message.get("tool_calls")
        if calls is not None and len(calls):
            plain["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": str(call["function"]["name"]),
                        "arguments": dict(call["function"]["arguments"]),
                    },
                }
                for call in calls
            ]
        out.append(plain)
    return out


def splice_format_errors(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop each "Format error:" scolding together with the turn that caused it.

    Returns the repaired list and the number of assistant turns removed. The
    scolding is always preceded by the malformed assistant turn and followed by
    the retry, so removing both splices the observation before it directly onto
    the retry and leaves a well-formed trajectory.

    A malformed turn has no `tool_calls` (measured: 407 of 409), so removing it
    cannot orphan a `role="tool"` observation. The remaining 2 carry a tool call
    that produced no observation -- the scolding stands where the observation
    would be -- so they are equally safe to drop.
    """
    out: list[dict[str, Any]] = []
    removed = 0
    for message in messages:
        is_scolding = (
            message["role"] == "user"
            and bool(out)  # never the opening task description
            and message["content"].startswith(FORMAT_ERROR_PREFIX)
        )
        if is_scolding:
            if out and out[-1]["role"] == "assistant":
                out.pop()
                removed += 1
            continue
        out.append(message)
    return out, removed


# Every one of these becomes a SINGLE control token under this tokenizer, so a
# source field containing one as literal text does not train the model to write
# markup -- it trains it to emit the control token itself. `<|im_start|>` is the
# one that actually costs something: a model that can emit it can forge a turn
# boundary mid-answer and inject its own system or tool turn, which ends a harbor
# rollout in a way no reward signal would attribute to the right cause.
FORBIDDEN_IN_SOURCE = (
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<think>", "</think>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
    "<function=", "<parameter=", "<parameter>",
)

# A trailing run of tool-call closing tags, which is how upstream mis-splits a
# response: the scaffolding lands at the end of `reasoning_content` while the call
# it belongs to is parsed correctly into `tool_calls`.
TRAILING_SCAFFOLDING = re.compile(
    r"(?:\s*</?(?:parameter|function|tool_call)[^>]*>)+\s*\Z"
)


def sanitize_reasoning(reasoning: str) -> tuple[str | None, bool]:
    """Strip mis-split tool-call scaffolding off the end of a CoT.

    Returns `(reasoning, repaired)`, or `(None, False)` if markup survives the
    strip, in which case the trajectory is refused.

    269 assistant turns have a `reasoning_content` ending in
    `</parameter></function></tool_call>`. The call itself is intact in
    `tool_calls`, so those closing tags carry no information -- this is not a
    guess about what the model meant, it is deleting a duplicate. Two shapes come
    out of it, and the difference is the point:

        reasoning is real prose with the scaffolding glued on
            -> stripped, and the CoT is recovered

        reasoning IS the scaffolding (a whole malformed call, often with the
        content field empty)
            -> markup remains after the strip, so the row is refused rather than
               reduced to a guess about which fragment was the thought

The same gate catches the 4 turns whose reasoning is the bare literal
    `<|im_start|>`, which no strip can rescue and which must never reach
    supervision.
    """
    cleaned = TRAILING_SCAFFOLDING.sub("", reasoning)
    repaired = cleaned != reasoning
    if any(literal in cleaned for literal in FORBIDDEN_IN_SOURCE):
        return None, repaired
    return cleaned, repaired


def repair_stray_think(content: str) -> tuple[str | None, bool]:
    """Remove the surplus `</think>` upstream leaks into assistant content.

    Returns `(content, repaired)`, or `(None, False)` when the turn is ambiguous
    and the trajectory should be dropped.

    528 of the 82,203 assistant turns carry a `</think>` inside `content` while
    ALSO carrying a `reasoning_content` field, so the tag is pure surplus -- the
    cross-tabulation has no case of a missing `reasoning_content` together with a
    `</think>`, which rules out the other reading, that upstream was encoding the
    thought for the template's extraction path. `<think>` never appears at all.

    The two shapes get different treatment because only one of them is knowable:

      483 turns  content is one "THOUGHT: ..." paragraph then `</think>` with
                 nothing but whitespace after it. Dropping the tag is not a
                 guess, and it has to be dropped: rendered, the turn would close
                 a think block that the template already closed, teaching the
                 model to emit `</think>` twice per turn.

       45 turns  content is two to six THOUGHT paragraphs, each with its own
                 `</think>` -- an upstream sampling loop that emitted the thought
                 several times and concatenated the results. Which paragraph is
                 canonical is unknowable, so these are refused. Picking one would
                 write a guess into supervision, which is the same reason
                 `03d_build_openthoughts_sft.py` refuses to repair unescaped
                 quotes rather than guess where the string ended.
    """
    if "</think>" not in content:
        return content, False
    head, _, rest = content.partition("</think>")
    if content.count("</think>") > 1 or rest.strip():
        return None, False
    return head, True


def render_tool_calls(calls: list[dict[str, Any]], content: str) -> str:
    """Reproduce the template's tool-call block for inlining into content.

    Mirrors tokenizer_config chat_template lines 104-127. The leading blank line
    is conditional on the text content being non-empty, exactly as the template
    has it; getting that wrong is the difference between a byte-exact render and
    a one-token divergence, which is why `bake` asserts rather than trusts.
    """
    parts: list[str] = []
    for index, call in enumerate(calls):
        function = call["function"]
        if index == 0:
            parts.append("\n\n<tool_call>\n" if content.strip() else "<tool_call>\n")
        else:
            parts.append("\n<tool_call>\n")
        parts.append(f"<function={function['name']}>\n")
        for name, value in function["arguments"].items():
            parts.append(f"<parameter={name}>\n")
            if isinstance(value, (dict, list, tuple)):
                rendered = json.dumps(value)
            else:
                rendered = str(value)
            parts.append(rendered)
            parts.append("\n</parameter>\n")
        parts.append("</function>\n</tool_call>")
    return "".join(parts)


def pre_render(messages: list[dict[str, Any]], baked_system: str) -> list[dict[str, str]]:
    """Flatten the native tool-calling shape into `{role, content}` only.

    The result renders to the identical string -- `bake` proves it per row -- but
    carries no `tools` argument, no `tool_calls` struct and no `tool` role, so it
    is the same schema `03_build_sft_data.py` and `03d_build_openthoughts_sft.py`
    emit and the same one every consumer downstream already reads.
    """
    out: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        # The template trims every message's content before using it
        # (chat_template line 81: `render_content(message.content, true)|trim`), and
        # TMax assistant content is wrapped in newlines -- "\n\nTHOUGHT: ...\n\n" --
        # so trimming here is not tidying. Left raw, the trailing newlines survive
        # into our render but not into the template's, and the two disagree by four
        # characters before every tool call.
        content = message["content"].strip()
        if role == "system":
            out.append({"role": "system", "content": baked_system})
        elif role == "user":
            out.append({"role": "user", "content": content})
        elif role == "tool":
            # Identical bytes to the template's tool branch (lines 130-141) for a
            # non-adjacent tool message, which is all TMax has: one bash call per
            # assistant turn means observations never abut.
            #
            # Carrying an observation as a `user` turn does not cost us the
            # reasoning of earlier turns, which is the thing that would quietly ruin
            # this dataset. The template's own scan for the last real query skips a
            # user turn that starts with <tool_response> and ends with
            # </tool_response> (lines 69-74), so `last_query_index` still lands on
            # the task description and every assistant turn stays after it. The
            # wrapper is load-bearing, not decoration.
            out.append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n{content}\n</tool_response>",
                }
            )
        elif role == "assistant":
            reasoning = (message.get("reasoning_content") or "").strip()
            text = f"<think>\n{reasoning}\n</think>\n\n{content}"
            calls = message.get("tool_calls")
            if calls:
                text += render_tool_calls(calls, content)
            out.append({"role": "assistant", "content": text})
        else:  # pragma: no cover - guarded by the caller
            raise ValueError(f"unexpected role {role!r}")
    return out


def bake_system(tokenizer, messages: list[dict[str, Any]], tools: list[dict]) -> str:
    """Return the system turn as the template renders it with `tools` expanded.

    Read back out of a real render rather than reconstructed, so the tool schema,
    its JSON formatting and the surrounding boilerplate are whatever the template
    says they are and not what this script guesses they are.
    """
    rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                             return_dict=False)
    if not rendered.startswith(SYSTEM_HEADER):
        sys.exit("the render does not open with a system turn; the template changed")
    head, _, _ = rendered.partition(TURN_END)
    return head[len(SYSTEM_HEADER):]


def command_signature(messages: list[dict[str, Any]]) -> str:
    """A stable digest of the bash commands a trajectory issues, in order.

    The repo's own `command_signature` reads the `commands` key of the RST JSON
    contract, which TMax has no equivalent of, so the same idea is applied to the
    tool-call arguments: two trajectories that run the same commands in the same
    order against the same task are the same demonstration.
    """
    commands: list[str] = []
    for message in messages:
        for call in message.get("tool_calls") or ():
            arguments = call["function"]["arguments"]
            commands.append(str(arguments.get("command", "")).strip())
    return hashlib.sha256("\n\x00\n".join(commands).encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help=f"the upstream parquet ({SOURCE_DATASET}, {SOURCE_FILE})")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--holdout", type=int, default=200,
                    help="held-out rows, split by task group so no task spans both splits")
    ap.add_argument("--seed", type=int, default=1228)
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    if not args.source.exists():
        sys.exit(f"{args.source} not found. Download it with:\n"
                 f"  curl -L -o {args.source} "
                 f"https://huggingface.co/datasets/{SOURCE_DATASET}/resolve/main/{SOURCE_FILE}")

    frame = pd.read_parquet(args.source)
    required = {"messages", "tools", "source", "metadata"}
    missing = required - set(frame.columns)
    if missing:
        sys.exit(f"{args.source} is missing {sorted(missing)}; upstream schema changed")
    print(f"[source] {len(frame)} rows from {SOURCE_DATASET} :: {SOURCE_CONFIG}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    if not tokenizer.is_fast:
        sys.exit("a fast tokenizer is required (the mask gate needs offset mapping)")
    exporter = load_exporter()

    # The tool schema is single-valued upstream, which is what makes baking it into
    # the system turn legitimate. Checked rather than assumed: if a second schema
    # ever appears, one baked system prompt would silently describe the wrong tools
    # for some rows.
    distinct_tools = {str(value) for value in frame["tools"]}
    if len(distinct_tools) != 1:
        sys.exit(f"{len(distinct_tools)} distinct `tools` payloads upstream; the system turn "
                 f"can no longer be baked once and shared. Pass tools through instead.")
    tools = json.loads(next(iter(distinct_tools)))

    stats: Counter = Counter()
    records: list[dict[str, Any]] = []

    baked_system: str | None = None
    for row in frame.itertuples(index=False):
        native = to_plain(row.messages)
        metadata = row.metadata if isinstance(row.metadata, dict) else json.loads(row.metadata)

        roles = {message["role"] for message in native}
        if not roles <= {"system", "user", "assistant", "tool"}:
            stats["drop_unexpected_role"] += 1
            continue
        if not native or native[0]["role"] != "system":
            stats["drop_no_system_turn"] += 1
            continue

        if baked_system is None:
            baked_system = bake_system(tokenizer, native, tools)
            print(f"[system] baked the tool schema into the system turn "
                  f"({len(tokenizer(baked_system, add_special_tokens=False)['input_ids'])} "
                  f"tokens)", flush=True)

        spliced, removed = splice_format_errors(native)

        # After the splice there must be exactly one user turn, or the template
        # drops the reasoning of every assistant turn before the last one and 71 %
        # of what makes this dataset worth training on goes with it.
        user_turns = sum(1 for message in spliced if message["role"] == "user")
        if user_turns != 1:
            stats["drop_multiple_user_turns"] += 1
            continue
        assistant_turns = [m for m in spliced if m["role"] == "assistant"]
        if not assistant_turns:
            stats["drop_no_assistant_turn"] += 1
            continue
        if any(not m["content"].strip() and not m.get("tool_calls") for m in assistant_turns):
            stats["drop_empty_assistant_turn"] += 1
            continue

        # Repairs are counted into a per-row tally and only folded into `stats` once
        # the row has cleared every gate, so the manifest's repair counts describe
        # the data that shipped rather than including work done on rows that were
        # then dropped.
        repairs: Counter = Counter()
        refused = None
        for message in assistant_turns:
            content, changed = repair_stray_think(message["content"])
            if content is None:
                refused = "drop_ambiguous_stray_think"
                break
            repairs["repaired_stray_think"] += int(changed)
            if any(literal in content for literal in FORBIDDEN_IN_SOURCE):
                refused = "drop_markup_in_content"
                break
            message["content"] = content

            reasoning, changed = sanitize_reasoning(message.get("reasoning_content") or "")
            if reasoning is None:
                refused = "drop_markup_in_reasoning"
                break
            repairs["repaired_tool_scaffolding"] += int(changed)
            if reasoning:
                message["reasoning_content"] = reasoning
            else:
                # An empty CoT is a shape the data already has 7,486 times over, and
                # the template renders it as an empty think block. Dropping the key
                # rather than storing "" keeps `to_plain`'s convention, so the native
                # render this is about to be compared against is the same either way.
                message.pop("reasoning_content", None)
        if refused:
            stats[refused] += 1
            continue

        # Observations are never trained on, but they are still context, and a
        # control token in one would let a task's own files rewrite the turn
        # structure. Measured zero upstream; gated so it stays that way.
        if any(literal in m["content"]
               for m in spliced if m["role"] != "assistant"
               for literal in FORBIDDEN_IN_SOURCE):
            stats["drop_markup_in_observation"] += 1
            continue

        messages = pre_render(spliced, baked_system)

        # The gate the whole pre-render rests on. Compared against the NATIVE shape
        # of the same spliced-and-repaired trajectory, so it checks this script's
        # rendering rather than re-checking the splice and the repair -- and so that
        # a repair which happens to change the render is not silently blessed by the
        # gate it was supposed to answer to.
        want = tokenizer.apply_chat_template(spliced, tools=tools, tokenize=False,
                                             return_dict=False)
        got = tokenizer.apply_chat_template(messages, tokenize=False, return_dict=False)
        if got != want:
            stats["drop_prerender_mismatch"] += 1
            continue

        # slime's contract: render-then-tokenize must equal tokenize-directly, or the
        # character offsets the loss mask is built from do not apply.
        ids = tokenizer(got, add_special_tokens=False)["input_ids"]
        if ids != tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False):
            stats["drop_template_contract_mismatch"] += 1
            continue
        if len(ids) > args.max_seq_len:
            stats["drop_too_long"] += 1
            continue

        try:
            _, mask = exporter.qwen3_5_mask(tokenizer, messages)
        except ValueError:
            stats["drop_mask_failed"] += 1
            continue
        if not sum(mask):
            stats["drop_no_trained_tokens"] += 1
            continue
        if mask[0] != 0:
            # 15_export_pretokenized.py refuses this row too; catching it here means
            # the uploaded dataset never contains one.
            stats["drop_supervised_first_token"] += 1
            continue

        records.append(
            {
                "messages": messages,
                "trajectory_id": str(metadata["trial_name"]),
                "task_group_id": str(metadata["task"]),
                "model_name": str(metadata["source_model"]),
                "n_tokens": len(ids),
                "n_trained_tokens": sum(mask),
                "n_assistant_turns": len(assistant_turns),
                "n_spliced_turns": removed,
                "command_signature": command_signature(spliced),
                "content_hash": hashlib.sha256(
                    json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
            }
        )
        stats["built"] += 1
        stats["spliced_malformed_turns"] += removed
        stats.update(repairs)

    print(f"[build] built={stats['built']} dropped={len(frame) - stats['built']} "
          f"spliced_malformed_turns={stats['spliced_malformed_turns']}", flush=True)
    if not records:
        sys.exit("nothing survived the gates")

    # ---- dedup ---------------------------------------------------------------
    # Keyed WITH the task, as in the RST pipeline: two different tasks needing the
    # same commands are two instruction->action mappings, not a duplicate. Unlike
    # the OpenThoughts converter this actually bites here, because TMax runs
    # several rollouts per task (5,795 trajectories over 2,020 tasks).
    seen_content: set[str] = set()
    seen_command: set[tuple[str, str]] = set()
    signature_owners: dict[str, set[str]] = defaultdict(set)
    kept: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: r["trajectory_id"]):
        signature_owners[record["command_signature"]].add(record["task_group_id"])
        if record["content_hash"] in seen_content:
            stats["dedup_exact"] += 1
            continue
        key = (record["task_group_id"], record["command_signature"])
        if key in seen_command:
            stats["dedup_command_signature"] += 1
            continue
        seen_content.add(record["content_hash"])
        seen_command.add(key)
        kept.append(record)
    cross_task = sum(1 for owners in signature_owners.values() if len(owners) > 1)
    print(f"[dedup] kept={len(kept)} exact_dropped={stats['dedup_exact']} "
          f"cmd_dropped={stats['dedup_command_signature']} "
          f"(signatures shared across tasks, not dropped: {cross_task})", flush=True)

    # ---- split, disjoint by task --------------------------------------------
    # TMax `task` is NOT unique per row -- several rollouts share one task -- so a
    # row-wise split would put siblings of a held-out task in train and the holdout
    # loss would be reading a task the model had already been shown.
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in kept:
        by_task[record["task_group_id"]].append(record)
    task_ids = sorted(by_task)
    rng = random.Random(args.seed)
    rng.shuffle(task_ids)

    holdout: list[dict[str, Any]] = []
    holdout_tasks: list[str] = []
    for task_id in task_ids:
        if len(holdout) >= args.holdout:
            break
        holdout.extend(by_task[task_id])
        holdout_tasks.append(task_id)
    train = [r for r in kept if r["task_group_id"] not in set(holdout_tasks)]
    overlap = {r["task_group_id"] for r in train} & set(holdout_tasks)
    if overlap:  # pragma: no cover - defensive
        sys.exit(f"{len(overlap)} task_group_id(s) landed in both splits")
    print(f"[split] train={len(train)} holdout={len(holdout)} "
          f"(holdout covers {len(holdout_tasks)} whole tasks)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["messages", "trajectory_id", "task_group_id", "model_name",
               "n_tokens", "n_trained_tokens", "n_assistant_turns", "n_spliced_turns"]
    paths = {}
    for name, rows in (("train", train), ("holdout", holdout)):
        out = pd.DataFrame([{k: r[k] for k in columns} for r in rows])
        path = args.out_dir / f"tmax_sft_{name}.parquet"
        out.to_parquet(path, index=False)
        paths[name] = str(path)
        print(f"[write] {path} rows={len(out)}", flush=True)

    lengths = [r["n_tokens"] for r in kept]
    trained = [r["n_trained_tokens"] for r in kept]
    turns = [r["n_assistant_turns"] for r in kept]
    manifest = {
        "source_dataset": SOURCE_DATASET,
        "source_config": SOURCE_CONFIG,
        "source_file": SOURCE_FILE,
        "source_license": SOURCE_LICENSE,
        "source_rows": len(frame),
        "source_models": sorted({r["model_name"] for r in kept}),
        "not_the_requested_dataset": {
            "requested": "allenai/TMax-15K",
            "reason": "TMax-15K, tmax-15k-open-instruct and TMax-SFT-16.5K contain zero "
                      "assistant turns -- they are RL environment instances (task text "
                      "plus graders), so they carry no responses to train on or to build "
                      "preference pairs from. They are input for 10_build_rl_taskset.py.",
            "used_instead": f"{SOURCE_DATASET} :: {SOURCE_CONFIG}",
        },
        "dpo_verdict": {
            "built": False,
            "reason": "only 291 of 2,020 tasks have both a success and a failure "
                      "trajectory, so only 14.4 % of prompts are pairable at all; "
                      "1,136 + 1,175 - 291 = 2,020 shows a TMax task is near-always "
                      "either always solved or never solved, which makes the "
                      "success/failure contrast largely a difficulty confound rather "
                      "than a quality signal.",
            "pairable_tasks": 291,
            "total_tasks": 2020,
            "pairs_at_min_per_task": 444,
            "pairs_at_full_cross_product": 1746,
            "existing_dpo_pairs_for_comparison": 2673,
        },
        "built": stats["built"],
        "after_dedup": len(kept),
        "train_examples": len(train),
        "holdout_examples": len(holdout),
        "holdout_tasks": len(holdout_tasks),
        "holdout_mode": "group (task_group_id); TMax runs several rollouts per task",
        "groups_covered": len(by_task),
        "command_signatures_shared_across_tasks": cross_task,
        "spliced_malformed_turns": stats["spliced_malformed_turns"],
        "repaired_stray_think": stats["repaired_stray_think"],
        "repaired_tool_scaffolding": stats["repaired_tool_scaffolding"],
        "max_seq_len": args.max_seq_len,
        "tokenizer": str(args.tokenizer),
        "schema": {
            "messages": "list[{role, content}] over system/user/assistant only. The tool "
                        "schema is baked into the system turn and tool observations are "
                        "user turns carrying <tool_response>...</tool_response>, so "
                        "apply_chat_template needs no `tools` argument and the render is "
                        "byte-identical to the native tool-calling shape (asserted per row).",
        },
        "token_stats": {
            "mean": statistics.mean(lengths),
            "p50": statistics.median(lengths),
            "p90": statistics.quantiles(lengths, n=10)[8] if len(lengths) > 10 else max(lengths),
            "p99": statistics.quantiles(lengths, n=100)[98] if len(lengths) > 100 else max(lengths),
            "max": max(lengths),
            "total_tokens": sum(lengths),
            "trained_tokens": sum(trained),
            "trained_fraction": round(sum(trained) / max(1, sum(lengths)), 4),
        },
        "turns": {"mean": statistics.mean(turns), "max": max(turns)},
        "drop_counters": dict(sorted(stats.items())),
        "seed": args.seed,
        "train_parquet": paths["train"],
        "holdout_parquet": paths["holdout"],
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"[manifest] {manifest_path}", flush=True)
    print(json.dumps(manifest["token_stats"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
