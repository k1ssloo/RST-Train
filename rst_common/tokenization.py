"""Whole-conversation, assistant-only tokenization and dataset identity (BUG-27).

The new profiles cover text messages. Native tool-call/multimodal records must be
converted explicitly; silently dropping those fields would change the trajectory.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

MASK_TYPES = ("qwen3_5", "llama3", "phi3", "smollm3", "olmo3", "gemma4")
MODEL_MASKS = {
    "qwen3_5": "qwen3_5", "qwen3_5_text": "qwen3_5",
    "qwen3_5_moe": "qwen3_5", "qwen3_5_moe_text": "qwen3_5",
    "llama": "llama3", "phi3": "phi3", "smollm3": "smollm3", "olmo3": "olmo3",
    "gemma4": "gemma4", "gemma4_text": "gemma4",
}
METADATA_KEY = b"rst_tokenization"
# SmolLM's upstream template otherwise changes with the wall clock. The date is
# part of the prompt and of the fingerprint, and is reported in every export.
TEMPLATE_DATE = "2026-09-09"


def mask_type_for_model(model_path: str | Path) -> str:
    path = Path(model_path) / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if model_type not in MODEL_MASKS:
        raise ValueError(f"unsupported model_type {model_type!r}; add a tested tokenization profile")
    return MODEL_MASKS[model_type]


def resolve_mask_type(model_path: str | Path, requested: str = "auto") -> str:
    inferred = mask_type_for_model(model_path)
    if requested != "auto" and requested != inferred:
        raise ValueError(f"loss-mask type {requested!r} does not match model profile {inferred!r}")
    return inferred


def template_options(mask_type: str) -> dict:
    if mask_type == "smollm3":
        return {"enable_thinking": False, "template_date": TEMPLATE_DATE}
    if mask_type == "llama3":
        return {"date_string": datetime.fromisoformat(TEMPLATE_DATE).strftime("%d %b %Y")}
    if mask_type == "gemma4":
        return {"enable_thinking": False}
    return {}


def render_kwargs(mask_type: str) -> dict:
    options = dict(template_options(mask_type))
    date = options.pop("template_date", None)
    if date:
        # Jinja context overrides its strftime_now global without rewriting the
        # official template. Pin the date across all renders of this dataset.
        options["strftime_now"] = datetime.fromisoformat(date).strftime
    return options


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def tokenization_identity(tokenizer, mask_type: str) -> dict:
    if mask_type not in MASK_TYPES:
        raise ValueError(f"unsupported loss-mask type {mask_type!r}")
    template = tokenizer.get_chat_template()
    if not template:
        raise ValueError("an instruction/chat tokenizer with a chat template is required")
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # Transient batch-padding state is not part of the single-row encoding.
    backend.pop("padding", None)
    backend.pop("truncation", None)
    payload = {
        "schema": "rst-tokenization-v1",
        "loss_mask_type": mask_type,
        "tokenizer_sha256": _sha(backend),
        "special_tokens": {k: str(v) for k, v in tokenizer.special_tokens_map.items()},
        "chat_template_sha256": _sha(template),
        "chat_template_options": template_options(mask_type),
    }
    return {**payload, "fingerprint": _sha(payload)}


def write_tokenized_parquet(frame, path: str | Path, identity: dict) -> None:
    """Keep the identity inside the parquet, so copying it preserves provenance."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[METADATA_KEY] = json.dumps(identity, sort_keys=True).encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)


def validate_tokenized_parquet(path: str | Path, tokenizer, mask_type: str) -> dict | None:
    import pyarrow.parquet as pq

    raw = (pq.read_schema(path).metadata or {}).get(METADATA_KEY)
    if raw is None:
        if mask_type == "qwen3_5":
            # Compatibility with the existing published, audited Qwen datasets.
            return None
        raise ValueError(f"{path}: missing tokenizer provenance; re-export messages for {mask_type}")
    actual = json.loads(raw)
    expected = tokenization_identity(tokenizer, mask_type)
    if actual != expected:
        raise ValueError(f"{path}: tokenizer/template/mask fingerprint mismatch; "
                         "re-export from messages using the training checkpoint's tokenizer")
    return actual


def validate_training_data(paths: list[Path], model_path: str | Path) -> str:
    """Used by both DPO entry points, before loading weights or resuming scores."""
    from transformers import AutoTokenizer

    mask_type = mask_type_for_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    for path in paths:
        validate_tokenized_parquet(path, tokenizer, mask_type)
    return mask_type


# All structural markers in a profile are forbidden inside content. Otherwise a
# literal assistant header in terminal output could be mistaken for supervision.
PROFILES = {
    "llama3": (r"<\|start_header_id\|>(\w+)<\|end_header_id\|>\n\n",
               r"<\|eot_id\|>", ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")),
    "phi3": (r"<\|(system|user|assistant|tool)\|>", r"<\|end\|>",
             ("<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|end|>")),
    "smollm3": (r"<\|im_start\|>(\w+)\n", r"<\|im_end\|>",
                ("<|im_start|>", "<|im_end|>")),
    "olmo3": (r"<\|im_start\|>(\w+)\n", r"<\|im_end\|>|<\|endoftext\|>",
              ("<|im_start|>", "<|im_end|>", "<|endoftext|>")),
    "gemma4": (r"<\|turn>(\w+)\n", r"<turn\|>", ("<|turn>", "<turn|>")),
}


def template_loss_mask(tokenizer, messages: list[dict], mask_type: str) -> tuple[list[int], list[int]]:
    """Mask explicit assistant spans in ONE native-template render, including EOS.

    Validate every role boundary, not just assistant headers. Tokens straddling a
    context/target boundary are rejected instead of training part of the prompt.
    """
    if mask_type not in PROFILES:
        raise ValueError(f"no text template profile for {mask_type!r}")
    header_re, end_re, reserved = PROFILES[mask_type]
    if not messages:
        raise ValueError("empty conversation")
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported role {message.get('role')!r}")
        content = message.get("content")
        if not isinstance(content, str):
            # Rejected dataset records use ValueError throughout the exporters.
            raise ValueError(  # noqa: TRY004
                "this profile requires text content; serialize multimodal records explicitly"
            )
        if any(message.get(key) for key in ("tool_calls", "function_call", "function_calls",
                                              "reasoning_content", "reasoning", "tools", "functions")):
            raise ValueError("native tools/reasoning fields require an explicit conversion to text")
        if any(marker in content for marker in (*reserved, tokenizer.eos_token or "\0")):
            raise ValueError("reserved chat marker inside message content")
        if message.get("step_loss_mask") not in (None, 0, 1):
            raise ValueError("step_loss_mask must be 0 or 1")

    options = render_kwargs(mask_type)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, return_dict=False,
                                             add_generation_prompt=False, **options)
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    direct = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                           add_generation_prompt=False, **options)
    if ids != direct:
        raise ValueError("chat-template contract mismatch (render vs direct tokenize)")
    headers = list(re.finditer(header_re, rendered))
    expected = list(messages)
    if headers and headers[0].group(1) == "system" and messages[0]["role"] != "system":
        expected = [{"role": "system", "content": ""}, *expected]
    role_map = {"assistant": "model"} if mask_type == "gemma4" else {}
    if mask_type == "smollm3":
        role_map["tool"] = "user"
    if mask_type == "olmo3":
        role_map["tool"] = "environment"
    if [h.group(1) for h in headers] != [role_map.get(m["role"], m["role"]) for m in expected]:
        raise ValueError("template role boundaries do not match messages (merged/dropped/unsupported turns)")

    char_mask = bytearray(len(rendered))
    for index, (header, message) in enumerate(zip(headers, expected)):
        stop = headers[index + 1].start() if index + 1 < len(headers) else len(rendered)
        endings = list(re.finditer(end_re, rendered[header.end():stop]))
        if mask_type == "smollm3" and message["role"] == "system" and not endings:
            # The upstream no-tools template omits the system terminator. Keep
            # its native render; none of this context is a supervised target.
            continue
        if len(endings) != 1:
            raise ValueError("template turn must have exactly one end marker")
        start = header.end()
        end = start + endings[0].end()
        if rendered[end:stop].strip() not in ("", tokenizer.eos_token):
            raise ValueError("unexpected text between template turns")
        if message["role"] != "assistant" or message.get("step_loss_mask") == 0:
            continue
        if mask_type == "smollm3":
            prefix = "<think>\n\n</think>\n"
            if not rendered.startswith(prefix, start):
                raise ValueError("SmolLM3 non-thinking generation prefix changed")
            start += len(prefix)
        # The suffix consists only of turn-ending markers / whitespace. It is
        # generated along with this response, including Phi's final EOS.
        char_mask[start:stop] = b"\1" * (stop - start)

    prefix_sum = [0]
    for flag in char_mask:
        prefix_sum.append(prefix_sum[-1] + flag)
    mask = []
    for start, end in offsets:
        trained = prefix_sum[end] - prefix_sum[start]
        if trained and trained != end - start:
            raise ValueError("token straddles context/assistant boundary")
        mask.append(int(trained > 0))
    if mask and mask[0]:
        raise ValueError("first token cannot be supervised")
    return ids, mask
