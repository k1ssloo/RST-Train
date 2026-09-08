"""`scripts/03h_build_rollout_sft.py` -- closing the loop from Harbor job dirs.

BUG.md BUG-19 is the centre of this file: a Terminus-2 trajectory written without
`raw_content` holds "Analysis:/Plan:" renderings and tool calls, not the model's
completion. The builder refuses those by default, can salvage them on request with the
loss made visible, and shares its reconstruction with the release builder so a raw
rollout and a release trajectory canonicalize identically.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import load_script  # noqa: E402

rollout = load_script("03h_build_rollout_sft")
builder = load_script("03_build_sft_data")

RAW_TURN = json.dumps({"analysis": "look", "plan": "ls", "commands": [{"keystrokes": "ls\n", "duration": 0.1}]})


def user(msg, step_id=1):
    return {"step_id": step_id, "source": "user", "message": msg}


def raw_agent(msg, obs, step_id):
    return {"step_id": step_id, "source": "agent", "message": msg,
            "observation": {"results": [{"content": obs}]}}


def rendered_agent(analysis, plan, cmds, obs, step_id, complete=False):
    calls = [{"tool_call_id": f"c{i}", "function_name": "bash_command",
              "arguments": {"keystrokes": c, "duration": 1.0}} for i, c in enumerate(cmds)]
    if complete:
        calls.append({"tool_call_id": "done", "function_name": "mark_task_complete", "arguments": {}})
    return {"step_id": step_id, "source": "agent", "message": f"Analysis: {analysis}\nPlan: {plan}",
            "tool_calls": calls, "observation": {"results": [{"content": obs}]}}


def compaction(step_id):
    return {"step_id": step_id, "source": "system", "message": "Performed context summarization",
            "observation": {"results": []}, "extra": {"context_management": {"type": "compaction"}}}


def test_raw_trajectories_reconstruct_exactly_like_the_release_builder():
    record = {"steps": [user("do it"), raw_agent(RAW_TURN, "New Terminal Output:\nfile\n", 2),
                        raw_agent(RAW_TURN, "done", 3)]}
    assert not rollout.is_rendered(record)
    stats: Counter = Counter()
    segments = rollout.build_segments(record, on_compaction="split", allow_rendered=False,
                                      builder=builder, stats=stats)
    assert len(segments) == 1
    expected = builder.reconstruct_trajectory(record, Counter())
    assert segments[0]["messages"] == expected["messages"]
    assert segments[0]["n_resynthesized_turns"] == 0 and segments[0]["segment_index"] == 0


def test_rendered_trajectories_are_refused_by_default_and_counted():
    record = {"steps": [user("do it"), rendered_agent("a", "p", ["ls\n"], "file", 2)]}
    assert rollout.is_rendered(record)
    stats: Counter = Counter()
    assert rollout.build_segments(record, on_compaction="split", allow_rendered=False,
                                  builder=builder, stats=stats) == []
    assert stats["drop_rendered_trajectory"] == 1


def test_salvage_resynthesizes_the_canonical_json_and_marks_every_turn():
    record = {"steps": [user("do it"),
                        rendered_agent("a1", "p1", ["ls\n"], "file", 2),
                        rendered_agent("a2", "p2", [], "ok", 3, complete=True)]}
    stats: Counter = Counter()
    segments = rollout.build_segments(record, on_compaction="split", allow_rendered=True,
                                      builder=builder, stats=stats)
    assert len(segments) == 1
    messages = segments[0]["messages"]
    first = json.loads(messages[1]["content"])
    assert first == {"analysis": "a1", "plan": "p1",
                     "commands": [{"keystrokes": "ls\n", "duration": 1.0}]}
    last = json.loads(messages[-1]["content"])
    assert last["task_complete"] is True and last["commands"] == []
    assert segments[0]["n_resynthesized_turns"] == 2
    assert messages[2]["content"] == "file", "the observation still becomes the next user turn"


def test_a_raw_turn_inside_a_rendered_trajectory_is_left_alone():
    # Terminus-2 falls back to the raw completion when its parser fails; that turn
    # has no tool_calls and must reach normalize_assistant untouched.
    text, changed = rollout.resynthesize_step({"source": "agent", "message": "```json\n" + RAW_TURN + "\n```"})
    assert changed is False and text.startswith("```json")


def test_compaction_boundaries_split_into_linear_segments():
    steps = [user("start"), raw_agent(RAW_TURN, "out1", 2), compaction(3),
             user("handoff: continue", 4), raw_agent(RAW_TURN, "out2", 5), raw_agent(RAW_TURN, "out3", 6)]
    pieces = rollout.split_on_compaction(steps)
    assert [len(p) for p in pieces] == [2, 3]
    assert pieces[1][0]["source"] == "user"
    stats: Counter = Counter()
    segments = rollout.build_segments({"steps": steps}, on_compaction="split",
                                      allow_rendered=False, builder=builder, stats=stats)
    assert [s["segment_index"] for s in segments] == [0, 1]
    assert segments[1]["messages"][0]["content"] == "handoff: continue"
    assert stats["trajectories_with_compaction"] == 1
    stitched = rollout.build_segments({"steps": steps}, on_compaction="stitch",
                                      allow_rendered=False, builder=builder, stats=Counter())
    assert len(stitched) == 1 and stitched[0]["n_assistant_turns"] == 3
    dropped_stats: Counter = Counter()
    assert rollout.build_segments({"steps": steps}, on_compaction="drop", allow_rendered=False,
                                  builder=builder, stats=dropped_stats) == []
    assert dropped_stats["drop_compaction"] == 1


def test_trajectory_files_are_ordered_main_then_continuations():
    with tempfile.TemporaryDirectory() as tmp:
        agent = Path(tmp)
        for name in ("trajectory.cont-2.json", "trajectory.json", "trajectory.cont-1.json",
                     "trajectory.summarization-3-questions.json"):
            (agent / name).write_text("{}")
        assert [p.name for p in rollout.trajectory_files(agent)] == [
            "trajectory.json", "trajectory.cont-1.json", "trajectory.cont-2.json"]


def _trial(root, name, reward, steps, exception=None, task="task-a"):
    trial = root / f"{name}-job" / name
    (trial / "agent").mkdir(parents=True)
    result = {"task_name": task, "trial_name": name, "exception_info": exception,
              "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
              "agent_info": {"model_info": {"name": "rst-eval"}}}
    (trial / "result.json").write_text(json.dumps(result))
    (trial / "agent" / "trajectory.json").write_text(json.dumps({"steps": steps}))


def test_the_reward_is_read_through_the_shared_classifier_and_infra_is_unmeasured():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        steps = [user("go"), raw_agent(RAW_TURN, "ok", 2)]
        _trial(root, "win", 1.0, steps)
        _trial(root, "lose", 0.0, steps)
        _trial(root, "broken", None, steps, exception={"error": "failed to pull image x"})
        stats: Counter = Counter()
        records = rollout.from_jobs_dir(root, model_name=None, stats=stats, builder=builder,
                                        on_compaction="split", allow_rendered=False)
        by_id = {r["trajectory_id"]: r for r in records}
        assert set(by_id) == {"win", "lose"}, "the infra failure is unmeasured, not a reward-0 row"
        assert by_id["win"]["reward"] == 1.0 and by_id["lose"]["reward"] == 0.0
        assert by_id["win"]["model_name"] == "rst-eval" and by_id["win"]["task_group_id"] == "task-a"
        assert stats["trials_infra_unmeasured"] == 1 and stats["trials_read"] == 2


def test_golden_episodes_are_read_from_the_embedded_trajectory():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "golden_episodes.jsonl"
        episode = {"episode_id": "e1", "task": {"task_id": "child-7"},
                   "rollout": {"trial_name": "child-7__abc", "reward": 1.0,
                               "agent_info": {"model_info": {"name": "policy-x"}},
                               "trajectory": {"steps": [user("go"), raw_agent(RAW_TURN, "ok", 2)]}}}
        path.write_text(json.dumps(episode) + "\n" + json.dumps({"rollout": {}}) + "\n")
        stats: Counter = Counter()
        records = rollout.from_golden_episodes(path, model_name=None, stats=stats, builder=builder,
                                               on_compaction="split", allow_rendered=False)
        assert len(records) == 1
        assert records[0]["task_group_id"] == "child-7" and records[0]["model_name"] == "policy-x"
        assert stats["drop_episode_without_trajectory"] == 1


def _v3_episode():
    from rst_common.rollout_provenance import content_digest

    segments = [{"session_id": "session-a", "agent": {"model_name": "policy-x"},
                 "continued_trajectory_ref": "trajectory.cont-1.json",
                 "steps": [user("start"), raw_agent(RAW_TURN, "out", 2)]},
                {"session_id": "session-a", "agent": {"model_name": "policy-x"},
                 "steps": [user("continue"), raw_agent(RAW_TURN, "done", 2)]}]
    return {"schema_version": 3, "episode_id": "episode-a", "dataset_purpose": "training",
            "policy_id": "checkpoint-a", "inference_digest": "profile", "admission_sha256": "admit",
            "task": {"task_id": "child", "parent_task_id": "parent", "root_lineage_id": "parent",
                     "ancestry": ["parent", "child"], "generation": 1, "bundle_digest": "bundle",
                     "semantic_cluster_id": "cluster", "overlap_group_id": "overlap", "split": "train"},
            "rollout": {"role": "student", "model_name": "policy-x", "reward": 1.0,
                        "trajectory": segments[0], "trajectory_segments": segments,
                        "trajectory_segment_names": ["trajectory.json", "trajectory.cont-1.json"],
                        "trajectory_content_sha256": [content_digest(s) for s in segments]}}


def _read_episode(episode, **kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "episodes.jsonl"
        path.write_text(json.dumps(episode) + "\n")
        stats = Counter()
        rows = rollout.from_golden_episodes(path, stats=stats, builder=builder,
                                            on_compaction="split", allow_rendered=False,
                                            model_name=kwargs.pop("model_name", None), **kwargs)
        return rows, stats


def test_v3_retains_both_segments_and_groups_descendants_by_root():
    # BUG.md BUG-24: a continued episode is one policy execution, with all segments.
    rows, stats = _read_episode(_v3_episode())
    assert len(rows) == 2 and stats["trajectory_files_read"] == 2
    assert {r["task_group_id"] for r in rows} == {"parent"}
    assert {r["task_id"] for r in rows} == {"child"}
    assert rows[1]["messages"][0]["content"] == "continue"
    assert rows[0]["trajectory_id"] != rows[1]["trajectory_id"]


def test_v3_rejects_missing_continuations_model_relabeling_and_changed_payload():
    for corruption in ("missing", "model", "content", "lineage", "cycle", "session", "escape"):
        episode = _v3_episode()
        kwargs = {}
        if corruption == "missing":
            episode["rollout"]["trajectory_segments"].pop()
        elif corruption == "model":
            kwargs["model_name"] = "a-different-model"
        elif corruption == "content":
            episode["rollout"]["trajectory_segments"][1]["steps"][0]["message"] = "tampered"
        elif corruption == "lineage":
            episode["task"]["root_lineage_id"] = "another-root"
        elif corruption == "cycle":
            episode["rollout"]["trajectory_segments"][1]["continued_trajectory_ref"] = "trajectory.json"
        elif corruption == "session":
            episode["rollout"]["trajectory_segments"][1]["session_id"] = "another-session"
        else:
            episode["rollout"]["trajectory"]["continued_trajectory_ref"] = "../private.json"
        rows, stats = _read_episode(episode, **kwargs)
        assert not rows and stats["drop_invalid_provenance"] == 1, corruption


def test_smoke_and_unmeasured_data_cannot_silently_enter_training():
    episode = _v3_episode()
    episode["dataset_purpose"] = "smoke"
    assert not _read_episode(episode)[0]
    assert all(r["dataset_purpose"] == "smoke" for r in _read_episode(episode, allow_smoke=True)[0])
    episode["dataset_purpose"] = "training"
    for reward in (None, True, "1", float("nan")):
        episode["rollout"]["reward"] = reward
        assert not _read_episode(episode)[0]
    episode["rollout"]["reward"] = 1
    episode["rollout"]["infrastructure_failure"] = True
    assert not _read_episode(episode)[0]


def test_disk_continuation_orphans_are_rejected_even_when_first_segment_wins():
    # BUG.md BUG-24: glob ordering is not proof of a complete execution chain.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _trial(root, "win", 1, [user("go"), raw_agent(RAW_TURN, "ok", 2)])
        (root / "win-job/win/agent/trajectory.cont-1.json").write_text("{}")
        stats = Counter()
        rows = rollout.from_jobs_dir(root, model_name=None, stats=stats, builder=builder,
                                     on_compaction="split", allow_rendered=False)
        assert not rows and stats["drop_invalid_provenance"] == 1


def test_v3_bundle_hash_and_frozen_membership_are_checked_independently():
    # BUG.md BUG-24: a train label inside an episode cannot override the frozen manifest.
    import hashlib
    from rst_common.rollout_provenance import load_export_bundle, validate_export_membership

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        episode = _v3_episode()
        profile = {"policy_id": "checkpoint-a", "model_name": "policy-x",
                   "agent_profile_sha256": "profile"}
        (root / "student-policy.json").write_text(json.dumps(profile))
        episode["student_profile_sha256"] = hashlib.sha256(
            (root / "student-policy.json").read_bytes()).hexdigest()
        path = root / "episodes.jsonl"
        path.write_text(json.dumps(episode) + "\n")
        splits = {"seeds": [{"root_lineage_id": "parent", "split": "train",
                              "overlap_group_id": "overlap"}]}
        (root / "splits.json").write_text(json.dumps(splits))
        manifest = {"schema_version": 3, "purpose": "training", "student_policy_id": "checkpoint-a",
                    "student_model": "policy-x", "files": {name: hashlib.sha256(
                        (root / name).read_bytes()).hexdigest()
                        for name in ("episodes.jsonl", "splits.json", "student-policy.json")}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        bundle = load_export_bundle(path)
        validate_export_membership(episode, bundle)
        bundle["seeds"]["parent"]["split"] = "holdout"
        try:
            validate_export_membership(episode, bundle)
        except ValueError as error:
            assert "frozen split" in str(error)
        else:
            raise AssertionError("episode overrode frozen holdout")
        path.write_text(path.read_text() + "\n")
        try:
            load_export_bundle(path)
        except ValueError as error:
            assert "hash" in str(error)
        else:
            raise AssertionError("tampered bundle was accepted")


def test_foundation_quota_preserves_raw_rows_and_cannot_count_smoke_as_mixed():
    # BUG.md BUG-24: curriculum metadata must survive without rewriting evidence.
    from rst_common.rollout_provenance import cap_foundation_records

    records = [
        {"trajectory_id": str(i), "task_id": "task", "dataset_purpose": "training",
         "training_subset": "foundation" if i < 8 else "mixed"}
        for i in range(10)
    ]
    records += [{"trajectory_id": "smoke", "dataset_purpose": "smoke"}]
    kept, dropped = cap_foundation_records(records, fraction=0.5, seed=7)
    assert len(records) == 11 and len(kept) == 5 and dropped == 6
    assert sum(row.get("training_subset") == "foundation" for row in kept) == 2
    reordered, _ = cap_foundation_records(list(reversed(records)), fraction=0.5, seed=7)
    assert {row["trajectory_id"] for row in kept} == {row["trajectory_id"] for row in reordered}
    assert all(any(row is original for original in records) for row in kept)
    missing = [{"dataset_purpose": "training", "trajectory_id": "unmeasured"}]
    try:
        cap_foundation_records(missing, fraction=0.5, seed=7)
    except ValueError as error:
        assert "calibration" in str(error)
    else:
        raise AssertionError("unknown calibration silently satisfied a curriculum quota")


def test_v3_calibration_metadata_survives_all_continuation_segments():
    # BUG.md BUG-24: later curriculum selection uses measured calibration, not file order.
    episode = _v3_episode()
    episode["task"]["training_subset"] = "mixed"
    episode["task"]["student_calibration"] = {"attempts": 4, "successes": 2}
    rows, _ = _read_episode(episode)
    assert rows and all(row["training_subset"] == "mixed" for row in rows)
    assert all(row["student_calibration"] == {"attempts": 4, "successes": 2} for row in rows)


def test_v3_family_exclusions_override_train_label_and_remain_bound_to_split():
    # BUG.md BUG-24: paraphrased source tasks can share a held-out task family.
    import hashlib
    from rst_common.rollout_provenance import load_export_bundle, validate_export_membership

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        episode = _v3_episode()
        (root / "episodes.jsonl").write_text(json.dumps(episode) + "\n")
        (root / "student-policy.json").write_text(json.dumps({
            "policy_id": "checkpoint-a", "model_name": "policy-x", "agent_profile_sha256": "profile",
        }))
        (root / "splits.json").write_text(json.dumps({"seeds": [
            {"root_lineage_id": "parent", "split": "train", "overlap_group_id": "train",
             "bundle_digest": "parent-hash"},
            {"root_lineage_id": "eval", "split": "holdout", "overlap_group_id": "eval",
             "bundle_digest": "eval-hash"},
        ]}))
        audit = {"seed_manifest_sha256": hashlib.sha256((root / "splits.json").read_bytes()).hexdigest(),
                 "excluded_roots": [{"root_lineage_id": "parent", "bundle_digest": "parent-hash",
                                     "related_holdout_root": "eval",
                                     "related_holdout_bundle_digest": "eval-hash",
                                     "reason": "Same underlying task with different wording."}]}
        (root / "seed-exclusions.json").write_text(json.dumps(audit))
        manifest = {"schema_version": 3, "purpose": "training", "student_policy_id": "checkpoint-a",
                    "student_model": "policy-x", "files": {name: hashlib.sha256(
                        (root / name).read_bytes()).hexdigest() for name in (
                            "episodes.jsonl", "splits.json", "student-policy.json", "seed-exclusions.json")}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        bundle = load_export_bundle(root / "episodes.jsonl")
        episode["task"]["seed_exclusions_sha256"] = manifest["files"]["seed-exclusions.json"]
        try:
            validate_export_membership(episode, bundle)
        except ValueError as error:
            assert "family exclusions" in str(error)
        else:
            raise AssertionError("excluded family was accepted under its original train label")
        audit["seed_manifest_sha256"] = "another-split"
        (root / "seed-exclusions.json").write_text(json.dumps(audit))
        manifest["files"]["seed-exclusions.json"] = hashlib.sha256(
            (root / "seed-exclusions.json").read_bytes()).hexdigest()
        (root / "manifest.json").write_text(json.dumps(manifest))
        try:
            load_export_bundle(root / "episodes.jsonl")
        except ValueError as error:
            assert "another frozen split" in str(error)
        else:
            raise AssertionError("unrelated exclusion audit was trusted")


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
