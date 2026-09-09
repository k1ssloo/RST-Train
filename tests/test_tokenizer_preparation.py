"""BUG-28: persisted padding must survive plain/verl loading and strict export."""

from __future__ import annotations

import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import load_repo_module, load_script, need
from test_model_tokenization import EXPORT, MESSAGES, TOK, refuses, tokenizer

PREPARE = load_script("14_prepare_tokenizer")


def checkpoint(path, profile="llama3", missing=True, dedicated=True):
    tok = tokenizer(profile)
    if dedicated:
        tok.add_tokens([PREPARE.LLAMA_PAD], special_tokens=True)
    if profile == "llama3":
        tok.eos_token = "<|eot_id|>"
    tok.pad_token = None if missing else tok.eos_token
    tok.save_pretrained(path)
    (path / "config.json").write_text(json.dumps({
        "model_type": "llama" if profile == "llama3" else "phi3",
        "vocab_size": len(tok), "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }))
    (path / "generation_config.json").write_text(json.dumps({
        "pad_token_id": tok.pad_token_id, "eos_token_id": [tok.eos_token_id],
        "do_sample": True,
    }))
    return tok


def verl_loader(path, **kwargs):
    # Reproduce the reported verl fallback. The deployment check calls the real
    # installed function; unit tests deliberately do not depend on a GPU stack.
    tok = need("transformers").AutoTokenizer.from_pretrained(str(path), **kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@contextmanager
def use_verl_loader(loader):
    # Restore only our stub. Restoring a snapshot of all sys.modules would
    # unload lazy pandas/Arrow imports whose native registrations remain live.
    previous = sys.modules.get("verl.utils")
    sys.modules["verl.utils"] = SimpleNamespace(hf_tokenizer=loader)
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("verl.utils", None)
        else:
            sys.modules["verl.utils"] = previous


def files(path):
    return {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}


def test_missing_padding_reproduces_mismatch_and_export_fails_before_reading_data():
    pd = need("pandas")
    need("pyarrow")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tok = checkpoint(root / "model")
        path = root / "old.parquet"
        TOK.write_tokenized_parquet(pd.DataFrame({"input_ids": [[1, 2]], "loss_mask": [[0, 1]]}),
                                    path, TOK.tokenization_identity(tok, "llama3"))
        refuses(lambda: TOK.validate_tokenized_parquet(path, verl_loader(root / "model"), "llama3"),
                "fingerprint mismatch")
        refuses(lambda: TOK.load_training_tokenizer(root / "model"), "14_prepare_tokenizer.py")
        with patch.object(sys, "argv", ["export", "--tokenizer", str(root / "model"),
                                       "--parquet", str(root / "absent.parquet"),
                                       "--out", str(root / "out.parquet")]):
            refuses(EXPORT.main, "no persisted pad_token")
        assert not (root / "out.parquet").exists()


def test_llama_preparation_roundtrip_and_export_match_verl_with_old_data_rejected():
    pd = need("pandas")
    need("pyarrow")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = root / "model"
        tok = checkpoint(model)
        original = files(model)
        before_ids, before_mask = EXPORT.tokenize_messages(tok, MESSAGES, "llama3")
        old = root / "old.parquet"
        TOK.write_tokenized_parquet(pd.DataFrame({"input_ids": [before_ids], "loss_mask": [before_mask]}),
                                    old, TOK.tokenization_identity(tok, "llama3"))
        with use_verl_loader(verl_loader):
            plan = PREPARE.prepare_tokenizer(model, verify_verl=True)
            assert plan["status"] == "needs_update" and files(model) == original
            report = PREPARE.prepare_tokenizer(model, apply=True, verify_verl=True)
            assert report["status"] == "applied"
            for entry in report["files"]:
                assert (model / entry["backup"]).read_bytes() == original[entry["file"]]
            saved = files(model)
            assert PREPARE.prepare_tokenizer(model, apply=True, verify_verl=True)["status"] == "ready"
            assert saved == files(model)
            actual = TOK.load_verl_training_tokenizer(model)
            assert actual.pad_token == PREPARE.LLAMA_PAD
            assert actual.pad_token_id == tok.get_vocab()[PREPARE.LLAMA_PAD]
            assert actual.eos_token_id == tok.eos_token_id
            assert EXPORT.tokenize_messages(actual, MESSAGES, "llama3") == (before_ids, before_mask)
            refuses(lambda: TOK.validate_tokenized_parquet(old, actual, "llama3"), "fingerprint mismatch")
            source, out = root / "messages.parquet", root / "new.parquet"
            pd.DataFrame({"messages": [MESSAGES]}).to_parquet(source)
            with patch.object(sys, "argv", ["export", "--tokenizer", str(model),
                                           "--parquet", str(source), "--out", str(out), "--strict"]):
                assert EXPORT.main() == 0
            assert TOK.validate_tokenized_parquet(out, actual, "llama3") == report["after_identity"]
            assert TOK.validate_training_data([out], model) == "llama3"
        generation = json.loads((model / "generation_config.json").read_text())
        assert generation["eos_token_id"] == [tok.eos_token_id] and generation["do_sample"]
        assert generation["pad_token_id"] == actual.pad_token_id


def test_phi_preserves_official_eos_padding_and_supervises_real_eos():
    dataset = load_repo_module("verl_backend.rst_sft_dataset")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tok = checkpoint(root, "phi3", missing=False)
        original = files(root)
        report = PREPARE.prepare_tokenizer(root, apply=True)
        assert report["status"] == "ready" and files(root) == original
        assert report["before_identity"] == report["after_identity"]
        ids, mask = EXPORT.tokenize_messages(tok, MESSAGES, "phi3")
        assert ids[-1] == tok.eos_token_id and mask[-1] == 1
        padded = dataset.build_row(ids, mask, max_length=len(ids) + 3,
                                   pad_mode="right", pad_token_id=tok.pad_token_id)
        assert padded["input_ids"][-4:] == [tok.eos_token_id] * 4
        assert padded["loss_mask"][-4:] == [1, 0, 0, 0]
        assert padded["attention_mask"][-4:] == [1, 0, 0, 0]


def test_unknown_or_unrepresentable_padding_fails_without_modifying_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint(root, dedicated=False)
        original = files(root)
        refuses(lambda: PREPARE.prepare_tokenizer(root, apply=True), "absent")
        refuses(lambda: PREPARE.prepare_tokenizer(root, apply=True, pad_token="<invented>"), "absent")
        assert files(root) == original
        checkpoint(root)
        config = json.loads((root / "config.json").read_text())
        config["vocab_size"] = 2
        (root / "config.json").write_text(json.dumps(config))
        original = files(root)
        refuses(lambda: PREPARE.prepare_tokenizer(root, apply=True), "outside model")
        assert files(root) == original


def test_existing_padding_requires_explicit_override_and_updates_legacy_map():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tok = checkpoint(root, missing=False)
        assert PREPARE.prepare_tokenizer(root, apply=True)["status"] == "ready"
        legacy = {"eos_token": tok.eos_token, "pad_token": tok.eos_token}
        (root / "special_tokens_map.json").write_text(json.dumps(legacy))
        report = PREPARE.prepare_tokenizer(root, apply=True, pad_token=PREPARE.LLAMA_PAD)
        assert report["status"] == "applied"
        actual = TOK.load_training_tokenizer(root)
        assert actual.pad_token == PREPARE.LLAMA_PAD and actual.eos_token == tok.eos_token
        assert json.loads((root / "special_tokens_map.json").read_text())["pad_token"] == PREPARE.LLAMA_PAD


def test_staged_verl_mismatch_blocks_writes_and_loader_rejects_template_changes():
    def bad_loader(path, **kwargs):
        tok = verl_loader(path, **kwargs)
        tok.chat_template += " changed"
        return tok

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint(root)
        original = files(root)
        with use_verl_loader(bad_loader):
            refuses(lambda: PREPARE.prepare_tokenizer(root, apply=True, verify_verl=True),
                    "plain/verl tokenizer fingerprint mismatch")
        assert files(root) == original
        PREPARE.prepare_tokenizer(root, apply=True)
        with use_verl_loader(bad_loader):
            refuses(lambda: TOK.load_verl_training_tokenizer(root), "plain/verl")


def test_preparation_replaces_symlink_without_writing_to_shared_cache():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = root / "model"
        checkpoint(model)
        path = model / "tokenizer_config.json"
        cache = root / "cached-tokenizer.json"
        path.rename(cache)
        path.symlink_to(cache)
        original = cache.read_bytes()
        PREPARE.prepare_tokenizer(model, apply=True)
        assert cache.read_bytes() == original and not path.is_symlink()
        assert TOK.load_training_tokenizer(model).pad_token == PREPARE.LLAMA_PAD


def test_check_cli_returns_nonzero_and_writes_actionable_report():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint(root / "model")
        argv = ["prepare", "--model", str(root / "model"), "--report", str(root / "report.json")]
        with patch.object(sys, "argv", argv):
            assert PREPARE.main() == 1
        assert json.loads((root / "report.json").read_text())["status"] == "needs_update"
        with patch.object(sys, "argv", [*argv, "--apply"]):
            assert PREPARE.main() == 0


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
