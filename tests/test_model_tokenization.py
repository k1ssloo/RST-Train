"""BUG-27: cross-family supervision, tokenizer identity, and failed exports."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import load_repo_module, load_script, need

TOK = load_repo_module("rst_common.tokenization")
EXPORT = load_script("15_export_pretokenized")


def tokenizer(profile="llama3"):
    tokenizers = need("tokenizers")
    transformers = need("transformers")
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(
        vocab={"<unk>": 0, **{char: i + 1 for i, char in enumerate(alphabet)}}, merges=[],
        unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                               eos_token="<eos>", pad_token="<pad>")
    if profile == "llama3":
        body = "{{ '<|start_header_id|>' + m.role + '<|end_header_id|>\\n\\n' + m.content + '<|eot_id|>' }}"
    elif profile == "phi3":
        body = "{{ '<|' + m.role + '|>' + m.content + '<|end|>' }}"
    elif profile == "gemma4":
        body = "{{ '<|turn>' + ('model' if m.role == 'assistant' else m.role) + '\\n' + m.content + '<turn|>\\n' }}"
    else:
        body = "{{ '<|im_start|>' + m.role + '\\n' }}"
        if profile == "smollm3":
            body += "{% if m.role == 'assistant' and not enable_thinking %}{{ '<think>\\n\\n</think>\\n' }}{% endif %}"
        body += "{{ m.content + ('<eos>' if loop.last and m.role == 'assistant' else '<|im_end|>\\n') }}" if profile == "olmo3" else "{{ m.content + '<|im_end|>\\n' }}"
    tok.chat_template = "{% for m in messages %}" + body + "{% endfor %}"
    if profile == "phi3":
        tok.chat_template += "{{ eos_token }}"
    if profile == "olmo3":
        tok.eos_token = "<|endoftext|>"
        tok.chat_template = tok.chat_template.replace("'<eos>'", "eos_token")
    specials = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|system|>",
                "<|user|>", "<|assistant|>", "<|end|>", "<|im_start|>", "<|im_end|>",
                "<|endoftext|>", "<|turn>", "<turn|>", "<think>", "</think>"]
    tok.add_special_tokens({"additional_special_tokens": specials})
    return tok


MESSAGES = [
    {"role": "system", "content": "SYSTEM context"},
    {"role": "user", "content": "USER 文件 café"},
    {"role": "assistant", "content": "FIRST command", "step_loss_mask": 0},
    {"role": "user", "content": "TOOL result"},
    {"role": "assistant", "content": "SECOND 回答"},
]


def refuses(call, text):
    try:
        call()
    except ValueError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError("invalid tokenization was accepted")


def test_native_profiles_supervise_only_unmasked_assistant_and_ending():
    endings = {"llama3": "<|eot_id|>", "phi3": "<|end|><eos>",
               "smollm3": "<|im_end|>\n", "olmo3": "<|endoftext|>", "gemma4": "<turn|>\n"}
    for profile, end in endings.items():
        tok = tokenizer(profile)
        ids, mask = EXPORT.tokenize_messages(tok, MESSAGES, profile)
        assert tok.decode([i for i, flag in zip(ids, mask) if flag]) == "SECOND 回答" + end
        assert ids == tok.apply_chat_template(MESSAGES, tokenize=True, return_dict=False,
                                              **TOK.render_kwargs(profile))
        assert mask[0] == 0 and len(mask) == len(ids)


def test_control_tokens_in_observations_cannot_spoof_an_assistant():
    for profile, marker in (("llama3", "<|start_header_id|>"), ("phi3", "<|assistant|>"),
                            ("smollm3", "<|im_start|>"), ("gemma4", "<|turn>")):
        messages = [{"role": "user", "content": marker}, {"role": "assistant", "content": "x"}]
        refuses(lambda profile=profile, messages=messages:
                EXPORT.tokenize_messages(tokenizer(profile), messages, profile), "reserved")


def test_native_tool_or_multimodal_fields_are_rejected_before_losing_content():
    for changes in ({"tool_calls": [{"id": "call"}]}, {"content": [{"type": "image"}]},
                    {"reasoning_content": "would otherwise be lost"}, {"step_loss_mask": 2}):
        messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y", **changes}]
        try:
            EXPORT.tokenize_messages(tokenizer(), messages, "llama3")
        except ValueError:
            pass
        else:
            raise AssertionError(f"silently accepted {changes}")


def test_unknown_architecture_and_wrong_explicit_mask_fail_early():
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / "config.json"
        config.write_text('{"model_type":"llama"}')
        assert TOK.resolve_mask_type(directory) == "llama3"
        refuses(lambda: TOK.resolve_mask_type(directory, "qwen3_5"), "does not match")
        config.write_text('{"model_type":"unknown"}')
        refuses(lambda: TOK.resolve_mask_type(directory), "unsupported")


def test_parquet_identity_detects_different_vocab_template_and_mask():
    pd = need("pandas")
    need("pyarrow")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tokens.parquet"
        tok = tokenizer()
        TOK.write_tokenized_parquet(pd.DataFrame({"input_ids": [[1, 2]], "loss_mask": [[0, 1]]}),
                                    path, TOK.tokenization_identity(tok, "llama3"))
        assert TOK.validate_tokenized_parquet(path, tok, "llama3")
        refuses(lambda: TOK.validate_tokenized_parquet(path, tok, "phi3"), "fingerprint")
        tok.add_tokens(["newword"])
        refuses(lambda: TOK.validate_tokenized_parquet(path, tok, "llama3"), "fingerprint")
        tok = tokenizer()
        tok.chat_template += " "
        refuses(lambda: TOK.validate_tokenized_parquet(path, tok, "llama3"), "fingerprint")
        pd.DataFrame({"input_ids": [[1, 2]], "loss_mask": [[0, 1]]}).to_parquet(path)
        refuses(lambda: TOK.validate_tokenized_parquet(path, tok, "llama3"), "missing tokenizer provenance")
        assert TOK.validate_tokenized_parquet(path, tok, "qwen3_5") is None


def test_export_cli_retokenizes_messages_and_empty_failure_does_not_write():
    pd = need("pandas")
    need("pyarrow")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tok = tokenizer()
        tok.save_pretrained(root / "model")
        (root / "model/config.json").write_text('{"model_type":"llama"}')
        source, out = root / "messages.parquet", root / "tokens.parquet"
        pd.DataFrame({"messages": [MESSAGES], "input_ids": [[999999]], "loss_mask": [[1]]}).to_parquet(source)
        argv = ["export", "--parquet", str(source), "--tokenizer", str(root / "model"),
                "--out", str(out), "--strict"]
        with patch.object(sys, "argv", argv):
            assert EXPORT.main() == 0
        actual = pd.read_parquet(out)
        assert 999999 not in actual.input_ids.iloc[0]
        assert TOK.validate_tokenized_parquet(out, tok, "llama3")
        manifest = json.loads(out.with_name("tokens_manifest.json").read_text())
        assert manifest["loss_mask_type"] == "llama3" and manifest["rows_out"] == 1
        out.unlink()
        with patch.object(sys, "argv", argv + ["--max-seq-len", "1"]):
            try:
                EXPORT.main()
            except SystemExit as exc:
                assert "no usable rows" in str(exc)
            else:
                raise AssertionError("empty export did not fail")
        assert not out.exists()


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
