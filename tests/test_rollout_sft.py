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


if __name__ == "__main__":
    from run_tests import run_module

    raise SystemExit(run_module(sys.modules[__name__]))
