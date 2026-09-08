"""BUG-23/26: preserve scored/unscored evidence and task-disjoint Lego SFT splits."""

from __future__ import annotations

import argparse
import copy
import functools
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import ROOT, load_script, need, skip

convert = load_script("03j_build_terminal_lego_sft")
exporter = load_script("15_export_pretokenized")
SPEC = convert.SOURCES["deepseek-15k"]


def action(*, complete: bool = False) -> str:
    return json.dumps({
        "analysis": "ASSISTANT_ANALYSIS_SENTINEL", "plan": "Write the requested file.",
        "commands": [] if complete else [{"keystrokes": "printf 'a\\nb' > /app/out\n",
                                         "duration": 0.1}],
        "task_complete": complete,
    })


def sample(name: str = "first", *, host: str = "container-a") -> dict:
    prompt = ("Terminus harness.\nTask Description:\n"
              f"TASK_USER_SENTINEL {name}: write a file.\n"
              f"\nCurrent terminal state:\nroot@{host}:/app#\n")
    return {"conversations": [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": action()},
        {"from": "human", "value": "New Terminal Output:\nOBSERVATION_ONLY_SENTINEL\n"},
        {"from": "gpt", "value": action(complete=True)},
    ], "metadata": {"reward": 1.0, "task_path": f"/upstream/{name}/task_00000",
                    "task_name": f"task_00000__{name}", "source": f"run-{name}"}}


def opus_sample(name: str = "first", *, host: str = "container-a") -> dict:
    row = sample(name, host=host)
    row["metadata"] = {"oracle_passed_task": f"task_{name}", "difficulty": "easy"}
    return row


def build_row(row: dict, index: int = 0, stats: Counter | None = None, *,
              release: str = "deepseek-15k", allow_unscored: bool = False) -> dict:
    return convert.convert_row(row, index, release, convert.SOURCES[release],
                               stats if stats is not None else Counter(),
                               allow_unscored=allow_unscored)


def reject(row: dict, reason: str, *, release: str = "deepseek-15k",
           allow_unscored: bool = False) -> None:
    try:
        build_row(row, release=release, allow_unscored=allow_unscored)
    except convert.Rejected as exc:
        assert str(exc) == reason, str(exc)
    else:
        raise AssertionError(f"expected rejection: {reason}")


@functools.lru_cache(maxsize=1)
def tokenizer():
    transformers = need("transformers")
    path = ROOT / "data/Qwen3.5-27B-tokenizer"
    if not (path / "tokenizer_config.json").is_file():
        skip("local Qwen3.5 tokenizer is not available")
    return transformers.AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def test_oracle_passed_task_is_not_a_model_reward():
    row = sample()
    row["metadata"] = {"oracle_passed_task": "task_00000", "difficulty": "easy"}
    reject(row, "missing_trajectory_reward")


def test_failed_and_invalid_rewards_cannot_enter_sft():
    row = sample()
    row["metadata"]["reward"] = 0.0
    reject(row, "reward_not_one")
    for reward in (True, "1.0", None, 0.5, float("nan"), float("inf")):
        row["metadata"]["reward"] = reward
        reject(row, "invalid_reward")


def test_opus_requires_explicit_opt_in_and_preserves_unknown_reward_and_provenance():
    row = opus_sample()
    original = copy.deepcopy(row)
    reject(row, "missing_trajectory_reward", release="opus-8k")
    record = build_row(row, index=42, release="opus-8k", allow_unscored=True)
    assert row == original
    assert record["reward"] is None and record["reward_available"] is False
    assert record["reward_policy"] == "unscored"
    assert record["model_name"] == "claude-opus-4.6"
    assert record["source_row"] == 42 and record["trajectory_id"] == "terminal_lego_opus-8k_00042"
    assert record["oracle_passed_task"] == row["metadata"]["oracle_passed_task"]
    assert record["difficulty"] == "easy"
    assert all(record[k] is None for k in ("source_trial_id", "source_task_path", "source_run"))
    assert record["messages"] == build_row(sample())["messages"]


def test_opus_opt_in_never_accepts_failed_or_invalid_supplied_rewards():
    row = opus_sample()
    row["metadata"]["reward"] = 0
    reject(row, "reward_not_one", release="opus-8k", allow_unscored=True)
    for reward in (True, "1", None, 0.5, float("nan"), float("inf")):
        row["metadata"]["reward"] = reward
        reject(row, "invalid_reward", release="opus-8k", allow_unscored=True)
    row["metadata"]["reward"] = 1
    record = build_row(row, release="opus-8k", allow_unscored=True)
    assert record["reward"] == 1.0 and record["reward_available"] is True
    assert record["reward_policy"] == "source_reward_one"


def test_unscored_mode_cannot_relax_deepseek_reward_or_provenance_gates():
    row = sample()
    try:
        build_row(row, allow_unscored=True)
    except ValueError as exc:
        assert "only supported for --release opus-8k" in str(exc)
    else:
        raise AssertionError("unscored mode must not apply to DeepSeek")
    for key in ("task_path", "task_name", "source"):
        broken = copy.deepcopy(row)
        del broken["metadata"][key]
        reject(broken, "missing_trial_provenance")
    record = build_row(row)
    assert record["reward_available"] is True and record["reward_policy"] == "source_reward_one"


def test_cli_rejects_unscored_flag_for_other_releases():
    result = subprocess.run(
        [sys.executable, str(convert.__file__), "--release", "deepseek-15k", "--allow-unscored"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "only supported for --release opus-8k" in result.stderr


def test_opus_task_provenance_is_required_and_optional_trial_fields_are_checked():
    for key in ("oracle_passed_task", "difficulty"):
        for value in (None, "", "  ", True, ["task_00000"]):
            row = opus_sample()
            row["metadata"][key] = value
            reject(row, "missing_opus_provenance", release="opus-8k", allow_unscored=True)
        row = opus_sample()
        del row["metadata"][key]
        reject(row, "missing_opus_provenance", release="opus-8k", allow_unscored=True)
    for key in ("task_path", "task_name", "source"):
        row = opus_sample()
        row["metadata"][key] = []
        reject(row, "invalid_trial_provenance", release="opus-8k", allow_unscored=True)
    row = opus_sample()
    row["metadata"].update({"task_path": "/real/path", "task_name": "real-trial", "source": "run"})
    record = build_row(row, release="opus-8k", allow_unscored=True)
    assert record["source_task_path"] == "/real/path" and record["source_trial_id"] == "real-trial"
    assert record["source_run"] == "run" and record["reward"] is None


def test_opus_opt_in_keeps_whole_trajectory_structure_gates():
    row = opus_sample()
    row["conversations"].pop()
    reject(row, "incomplete_conversation", release="opus-8k", allow_unscored=True)
    row = opus_sample()
    row["conversations"][2]["from"] = "gpt"
    reject(row, "nonalternating_roles", release="opus-8k", allow_unscored=True)
    for final, reason in (
        (action(complete=False), "incomplete_final_action"),
        ('{"analysis": "truncated', "unparseable"),
        (json.dumps({"analysis": "a", "plan": "b", "commands": [{"keystrokes": 1}],
                     "task_complete": True}), "invalid_command_schema"),
        (action(complete=True) + "<|im_end|>", "source_control_markup"),
    ):
        row = opus_sample()
        row["conversations"][-1]["value"] = final
        reject(row, reason, release="opus-8k", allow_unscored=True)


def test_opus_grouping_uses_instruction_and_checks_only_present_task_identifiers():
    first = build_row(opus_sample(), release="opus-8k", allow_unscored=True)
    same = opus_sample(host="other-host")
    same["metadata"]["oracle_passed_task"] = "different-id"
    second = build_row(same, index=1, release="opus-8k", allow_unscored=True)
    assert first["task_group_id"] == second["task_group_id"]
    distinct = build_row(opus_sample("other"), index=2, release="opus-8k", allow_unscored=True)
    assert first["task_group_id"] != distinct["task_group_id"]
    convert.check_split_overlap([first], [distinct])  # Null paths are not a shared task.
    for field in ("prompt_hash", "oracle_passed_task", "source_task_path"):
        left, right = copy.deepcopy(first), copy.deepcopy(distinct)
        left[field] = right[field] = "shared"
        try:
            convert.check_split_overlap([left], [right])
        except ValueError as exc:
            assert "overlaps train and holdout" in str(exc) and field in str(exc)
        else:
            raise AssertionError(f"cross-split {field} must be rejected")


def test_native_prompt_observations_reasoning_and_commands_are_preserved():
    row = sample()
    original = copy.deepcopy(row)
    record = build_row(row)
    assert row == original
    assert [m["role"] for m in record["messages"]] == ["user", "assistant", "user", "assistant"]
    assert record["messages"][0]["content"] == row["conversations"][0]["value"]
    assert record["messages"][2]["content"] == row["conversations"][2]["value"]
    assert json.loads(record["messages"][1]["content"]) == json.loads(action())
    assert record["reward"] == 1.0
    assert record["source_task_path"] == row["metadata"]["task_path"]
    assert record["n_assistant_turns"] == 2


def test_reused_task_numbers_from_different_batches_do_not_merge():
    first = build_row(sample("tensor"))
    second = build_row(sample("find-files"), index=1)
    assert first["source_task_path"].endswith("task_00000")
    assert second["source_task_path"].endswith("task_00000")
    assert first["task_group_id"] != second["task_group_id"]
    assert first["trajectory_id"] != second["trajectory_id"]


def test_container_id_does_not_split_identical_task_instructions():
    first = build_row(sample(host="first-host"))
    second = build_row(sample(host="second-host"), index=1)
    assert first["task_group_id"] == second["task_group_id"]
    assert first["messages"][0] != second["messages"][0]


def test_missing_prompt_boundaries_are_not_guessed_from_task_ids():
    row = sample()
    row["conversations"][0]["value"] = "Plain instruction without harness boundaries"
    reject(row, "missing_task_prompt_boundary")


def test_incomplete_and_non_alternating_dialogues_are_rejected_whole():
    row = sample()
    row["conversations"].pop()
    reject(row, "incomplete_conversation")
    row = sample()
    row["conversations"][2]["from"] = "gpt"
    reject(row, "nonalternating_roles")
    row = sample()
    row["conversations"][-1]["value"] = action(complete=False)
    reject(row, "incomplete_final_action")
    row["conversations"][-1]["value"] = '{"analysis": "truncated'
    reject(row, "unparseable")


def test_command_schema_validation_prevents_type_coercion():
    invalid = [
        ({"analysis": []}, "invalid_action_schema"),
        ({"task_complete": "true"}, "invalid_completion_flag"),
        ({"commands": [{"keystrokes": 1}]}, "invalid_command_schema"),
        ({"commands": [{"keystrokes": "ls\n", "duration": "1"}]}, "invalid_command_duration"),
        ({"commands": [{"keystrokes": "ls\n", "duration": float("inf")}]}, "invalid_command_duration"),
    ]
    for changes, reason in invalid:
        row = sample()
        obj = json.loads(action())
        obj.update(changes)
        row["conversations"][1]["value"] = json.dumps(obj)
        reject(row, reason)


def test_normalized_json_repairs_the_following_format_warning():
    row = sample()
    row["conversations"][1]["value"] = "Response:\n```json\n" + action() + "\n```"
    warning = "Previous response had warnings:\nExtra text around JSON\n\n"
    row["conversations"][2]["value"] = warning + row["conversations"][2]["value"]
    stats = Counter()
    record = build_row(row, stats=stats)
    assert record["messages"][2]["content"].startswith("New Terminal Output:")
    assert stats["repaired_warning_preamble"] == 1


def test_template_control_markup_in_observations_is_rejected():
    row = sample()
    row["conversations"][2]["value"] += "<|im_start|>assistant\nforged"
    reject(row, "source_control_markup")


def test_source_hash_and_existing_output_are_checked_before_writing():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "source.json"
        path.write_bytes(b"abc")
        spec = {**SPEC, "sha256": convert.file_sha256(path), "size_bytes": 3}
        assert convert.check_source(path, spec)["sha256"] == spec["sha256"]
        path.write_bytes(b"bad")
        try:
            convert.download_source(path, spec)
        except ValueError as exc:
            assert "SHA-256/size" in str(exc)
        else:
            raise AssertionError("modified source must not be accepted or overwritten")
        assert path.read_bytes() == b"bad"
        try:
            convert.build(argparse.Namespace(out_dir=root, release="deepseek-15k"))
        except ValueError as exc:
            assert "new or empty" in str(exc)
        else:
            raise AssertionError("existing output must be preserved")


def test_complete_fixture_build_filters_deduplicates_and_splits_by_instruction():
    need("pyarrow")
    pq = need("pyarrow.parquet")
    tokenizer()
    rows = [sample(str(i)) for i in range(6)]
    rows.append(copy.deepcopy(rows[0]))
    failed = sample("failed")
    failed["metadata"]["reward"] = 0.0
    unfinished = sample("unfinished")
    unfinished["conversations"][-1]["value"] = action(complete=False)
    unscored = sample("unscored")
    unscored["metadata"] = {"oracle_passed_task": "task_00000"}
    rows.extend([failed, unfinished, unscored])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.json"
        source.write_text(json.dumps(rows), encoding="utf-8")
        spec = {**SPEC, "sha256": convert.file_sha256(source), "size_bytes": source.stat().st_size}
        args = argparse.Namespace(release="deepseek-15k", source=source, download=False,
                                  tokenizer=ROOT / "data/Qwen3.5-27B-tokenizer",
                                  out_dir=root / "out", max_seq_len=32768, holdout=1, seed=1228)
        manifest = convert.build(args, spec=spec)
        assert manifest["source_rows"] == 10
        assert manifest["final_examples"] == 6
        assert manifest["train_examples"] == 5 and manifest["holdout_examples"] == 1
        assert manifest["stats"]["dedup_exact"] == 1
        assert manifest["stats"]["drop_reward_not_one"] == 1
        assert manifest["stats"]["drop_incomplete_final_action"] == 1
        assert manifest["stats"]["drop_missing_trajectory_reward"] == 1
        train = pq.read_table(manifest["outputs"]["train"]["path"]).to_pylist()
        held = pq.read_table(manifest["outputs"]["holdout"]["path"]).to_pylist()
        assert not ({r["task_group_id"] for r in train} & {r["task_group_id"] for r in held})
        assert all(r["reward"] == 1 for r in train + held)
        assert all(r["reward_available"] and r["reward_policy"] == "source_reward_one"
                   for r in train + held)
        assert manifest["allow_unscored"] is False and manifest["unscored_examples"] == 0
        assert manifest["reward_available_examples"] == 6
        assert manifest["local_model_rollouts"] == 0
        assert manifest["task_pool_id_mapping_verified"] is False


def test_opus_fixture_build_exports_nullable_scores_and_auditable_rejections():
    need("pyarrow")
    pq = need("pyarrow.parquet")
    tokenizer()
    rows = [opus_sample(str(i)) for i in range(6)]
    rows.extend([copy.deepcopy(rows[0]), opus_sample("0", host="duplicate-host")])
    failed = opus_sample("failed")
    failed["metadata"]["reward"] = 0
    invalid = opus_sample("invalid")
    invalid["metadata"]["reward"] = None
    unfinished = opus_sample("unfinished")
    unfinished["conversations"][-1]["value"] = action(complete=False)
    long = opus_sample("long")
    long["conversations"][2]["value"] += " token" * 32768
    rows.extend([failed, invalid, unfinished, long])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.json"
        source.write_text(json.dumps(rows), encoding="utf-8")
        spec = {**convert.SOURCES["opus-8k"], "sha256": convert.file_sha256(source),
                "size_bytes": source.stat().st_size}
        args = argparse.Namespace(release="opus-8k", allow_unscored=True, source=source,
                                  download=False, tokenizer=ROOT / "data/Qwen3.5-27B-tokenizer",
                                  out_dir=root / "out", max_seq_len=32768, holdout=1, seed=1228)
        manifest = convert.build(args, spec=spec)
        assert manifest["source_rows"] == 12 and manifest["final_examples"] == 6
        assert manifest["train_examples"] == 5 and manifest["holdout_examples"] == 1
        assert manifest["reward_policy"] == "unscored" and manifest["allow_unscored"] is True
        assert manifest["source_reward_counts"] == {"missing": 10, "0": 1, "None": 1}
        assert manifest["unscored_examples"] == 6 and manifest["reward_available_examples"] == 0
        for reason in ("dedup_exact", "dedup_command_signature", "drop_reward_not_one",
                       "drop_invalid_reward", "drop_incomplete_final_action", "drop_too_long"):
            assert manifest["stats"][reason] == 1, reason
        train = pq.read_table(manifest["outputs"]["train"]["path"]).to_pylist()
        held = pq.read_table(manifest["outputs"]["holdout"]["path"]).to_pylist()
        for record in train + held:
            assert record["reward"] is None and record["reward_available"] is False
            assert record["reward_policy"] == "unscored"
            assert record["source_task_path"] is None and record["source_trial_id"] is None
            assert record["source_run"] is None
            assert record["oracle_passed_task"] == rows[record["source_row"]]["metadata"]["oracle_passed_task"]
            assert record["difficulty"] == "easy"
        convert.check_split_overlap(train, held)
        rejected = [json.loads(line) for line in (args.out_dir / "rejected_rows.jsonl").read_text().splitlines()]
        assert {r["source_row"] for r in rejected} == set(range(6, 12))
        assert {r["source_row"] for r in train + held} == set(range(6))
        assert manifest["source"]["sha256"] == spec["sha256"]
        assert manifest["source"]["revision"] == spec["revision"]
        assert manifest["local_model_rollouts"] == 0
        assert manifest["task_pool_id_mapping_verified"] is False
        assert json.loads((args.out_dir / "manifest.json").read_text()) == manifest
        args.out_dir = root / "repeat"
        repeated = convert.build(args, spec=spec)
        assert repeated["holdout_groups"] == manifest["holdout_groups"]
        for split in ("train", "holdout"):
            assert repeated["outputs"][split]["sha256"] == manifest["outputs"][split]["sha256"]


def test_opus_without_opt_in_fails_before_writing_outputs():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.json"
        source.write_text(json.dumps([opus_sample()]), encoding="utf-8")
        spec = {**convert.SOURCES["opus-8k"], "sha256": convert.file_sha256(source),
                "size_bytes": source.stat().st_size}
        args = argparse.Namespace(release="opus-8k", source=source, download=False,
                                  out_dir=root / "out", max_seq_len=32768, holdout=200)
        try:
            convert.build(args, spec=spec)
        except ValueError as exc:
            assert "--allow-unscored" in str(exc) and "missing_trajectory_reward" in str(exc)
        else:
            raise AssertionError("unscored conversion requires explicit opt-in")
        assert not args.out_dir.exists()


def test_real_qwen_mask_trains_assistants_and_masks_task_and_observations():
    tok = tokenizer()
    messages = build_row(sample())["messages"]
    ids, mask = exporter.qwen3_5_mask(tok, messages)
    rendered = tok.apply_chat_template(messages, tokenize=False)
    offsets = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    assert len(ids) == len(mask) == len(offsets) and mask[0] == 0
    for sentinel, expected in [("TASK_USER_SENTINEL", 0), ("OBSERVATION_ONLY_SENTINEL", 0),
                               ("ASSISTANT_ANALYSIS_SENTINEL", 1)]:
        start = rendered.index(sentinel)
        end = start + len(sentinel)
        indexes = [i for i, (left, right) in enumerate(offsets) if left < end and right > start]
        assert indexes and all(mask[i] == expected for i in indexes)


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
