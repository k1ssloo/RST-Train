#!/usr/bin/env python3
"""Persist existing padding tokens before SFT/DPO export; never add vocabulary.

Without --apply this is a read-only check (exit 1 if changes are needed).
Use --verify-verl in the training environment to check its real loading path.
Run once before exports/jobs start, on a local checkpoint copy, not an HF cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rst_common.tokenization import (  # noqa: E402
    load_training_tokenizer, load_verl_training_tokenizer, mask_type_for_model,
    tokenization_identity,
)

LLAMA_PAD = "<|finetune_right_pad_id|>"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_tokenizer(model: Path, *, apply: bool = False, pad_token: str | None = None,
                      verify_verl: bool = False) -> dict:
    from transformers import AutoTokenizer

    model = model.resolve()
    profile = mask_type_for_model(model)
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer is required for fingerprint and mask validation")
    before = tokenization_identity(tokenizer, profile)
    old_pad = {"token": tokenizer.pad_token, "id": tokenizer.pad_token_id}
    selected = pad_token if pad_token is not None else tokenizer.pad_token
    if selected is None and profile == "llama3":
        selected = LLAMA_PAD
    vocab = tokenizer.get_vocab()
    if selected is None or selected not in vocab:
        raise ValueError(
            f"padding token {selected!r} is absent from this checkpoint's vocabulary; "
            "specify --pad-token with an existing token. No tokens will be added."
        )
    token_id = vocab[selected]
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config") or config
    head_size = text_config.get("vocab_size")
    if token_id < 0 or (head_size is not None and token_id >= head_size):
        raise ValueError(f"pad token ID {token_id} is outside model vocab_size={head_size}")
    if tokenizer.encode(selected, add_special_tokens=False) != [token_id]:
        raise ValueError("padding token must encode as exactly its existing vocabulary ID")
    tokenizer.pad_token = selected
    after = tokenization_identity(tokenizer, profile)
    # Merely assigning an existing token must not alter the encoder or template.
    for field in ("tokenizer_sha256", "chat_template_sha256", "chat_template_options"):
        if before[field] != after[field]:
            raise ValueError(f"padding assignment unexpectedly changed {field}")

    originals, updates = {}, {}
    for name in ("tokenizer_config.json", "special_tokens_map.json",
                 "config.json", "generation_config.json"):
        path = model / name
        if not path.is_file():
            continue
        original = path.read_bytes()
        content = json.loads(original)
        if name in ("tokenizer_config.json", "special_tokens_map.json"):
            current = content.get("pad_token")
            if isinstance(current, dict):
                current = current.get("content")
            if current == selected:
                continue
            content["pad_token"] = selected
        else:
            targets = [content]
            if name == "config.json" and isinstance(content.get("text_config"), dict):
                targets.append(content["text_config"])
            if all(target.get("pad_token_id") == token_id for target in targets):
                continue
            for target in targets:
                target["pad_token_id"] = token_id
        originals[name] = original
        updates[name] = (json.dumps(content, ensure_ascii=False, indent=2) + "\n").encode()

    loader = load_verl_training_tokenizer if verify_verl else load_training_tokenizer

    def verify(path: Path) -> None:
        reloaded = loader(path)
        if (reloaded.pad_token_id != token_id or reloaded.get_vocab() != vocab
                or tokenization_identity(reloaded, profile) != after):
            raise ValueError("saved tokenizer does not reproduce the planned vocabulary/fingerprint")

    if updates:
        # Validate serialization before touching the checkpoint. Symlinks make
        # even multi-GB checkpoints cheap to stage; loaders only read metadata.
        with tempfile.TemporaryDirectory(prefix="rst-pad-") as directory:
            staged = Path(directory)
            for path in model.iterdir():
                if path.name in updates:
                    (staged / path.name).write_bytes(updates[path.name])
                else:
                    (staged / path.name).symlink_to(path, target_is_directory=path.is_dir())
            verify(staged)
    else:
        verify(model)

    files = [{"file": name, "before_sha256": _sha(originals[name]),
              "after_sha256": _sha(data),
              "backup": f"{name}.rst-pad-backup-{_sha(originals[name])[:16]}"}
             for name, data in updates.items()]
    if apply and updates:
        # Refuse to overwrite a configuration changed since inspection. Back up
        # every original before replacing any file; never write through a cache symlink.
        for entry in files:
            name = entry["file"]
            if (model / name).read_bytes() != originals[name]:
                raise ValueError(f"{name} changed during preparation; retry before starting jobs")
            backup = model / entry["backup"]
            try:
                with backup.open("xb") as handle:
                    handle.write(originals[name])
            except FileExistsError:
                if backup.read_bytes() != originals[name]:
                    raise ValueError(f"backup conflict: {backup}") from None
        for name, data in updates.items():
            with tempfile.NamedTemporaryFile(dir=model, prefix=".rst-pad-", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
            try:
                os.chmod(temporary, (model / name).stat().st_mode & 0o777)
                os.replace(temporary, model / name)
            finally:
                temporary.unlink(missing_ok=True)
        verify(model)

    status = "ready"
    if updates:
        status = "applied" if apply else "needs_update"
    return {
        "schema": "rst-tokenizer-padding-v1", "model": str(model), "profile": profile,
        "status": status,
        "before_pad": old_pad, "after_pad": {"token": selected, "id": token_id},
        "before_identity": before, "after_identity": after, "files": files,
        "verl_checked": verify_verl,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="persist the checked changes with backups")
    parser.add_argument("--pad-token", help="explicit existing token; default preserves configured padding")
    parser.add_argument("--verify-verl", action="store_true", help="also load with installed verl")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = prepare_tokenizer(args.model, apply=args.apply, pad_token=args.pad_token,
                                   verify_verl=args.verify_verl)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"tokenizer preparation failed: {exc}\n")
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return int(report["status"] == "needs_update")


if __name__ == "__main__":
    raise SystemExit(main())
