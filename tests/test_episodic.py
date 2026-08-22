"""
Tests for the episodic consolidation module.
All tests use injected functions — no live database required.
"""
import pytest
from agent_sleep.episodic import (
    distill_episode,
    distill_procedures,
    consolidate_episodes,
    compress_stale_episodes,
    _summarize_goal,
    MIN_GOAL_CHARS,
)


# ---------------------------------------------------------------------------
# _summarize_goal
# ---------------------------------------------------------------------------

def test_summarize_goal_bare_string():
    assert _summarize_goal("Fix the payment bug") == "Fix the payment bug"


def test_summarize_goal_json_content():
    import json
    g = json.dumps({"content": "Refactor the database layer"})
    assert _summarize_goal(g) == "Refactor the database layer"


def test_summarize_goal_dict():
    assert _summarize_goal({"goal": "Add unit tests"}) == "Add unit tests"


def test_summarize_goal_collapses_whitespace():
    assert _summarize_goal("  hello   world  ") == "hello world"


# ---------------------------------------------------------------------------
# distill_episode
# ---------------------------------------------------------------------------

def _ep(**kwargs):
    defaults = {
        "goal": "Refactor the authentication module to use JWT",
        "action": "edit_file('auth.py', ...)",
        "outcome": "success",
        "failure_reason": "",
        "episode_kind": "action",
        "emotion_label": "",
    }
    defaults.update(kwargs)
    return defaults


def test_distill_success_returns_procedural():
    result = distill_episode(_ep(outcome="success"))
    assert result is not None
    mtype, fact, value, importance, confidence, ver_status = result
    assert mtype == "procedural"
    assert "done before" in fact.lower()
    assert importance == pytest.approx(0.7)
    assert ver_status == "observed"


def test_distill_failure_returns_lesson():
    result = distill_episode(_ep(outcome="failure", failure_reason="ImportError: no module named X"))
    assert result is not None
    mtype, fact, value, importance, confidence, ver_status = result
    assert mtype == "lesson"
    assert "caution" in fact.lower()
    assert "ImportError" in value
    assert ver_status == "observed"


def test_distill_verdict_success_higher_confidence():
    result = distill_episode(_ep(outcome="success", episode_kind="task_verdict"))
    assert result is not None
    _, _, _, importance, confidence, ver_status = result
    assert importance == pytest.approx(0.85)
    assert confidence == pytest.approx(0.95)
    assert ver_status == "verified"


def test_distill_verdict_failure_higher_confidence():
    result = distill_episode(_ep(outcome="failure", episode_kind="task_verdict", failure_reason="wrong"))
    assert result is not None
    _, fact, _, importance, confidence, ver_status = result
    assert "verified" in fact.lower()
    assert importance == pytest.approx(0.85)
    assert confidence == pytest.approx(0.95)
    assert ver_status == "verified"


def test_distill_short_goal_returns_none():
    result = distill_episode(_ep(goal="fix it"))  # too short
    assert result is None


def test_distill_unrecognized_outcome_returns_none():
    result = distill_episode(_ep(outcome="pending"))
    assert result is None


def test_distill_emotion_appended():
    result = distill_episode(_ep(outcome="success", emotion_label="satisfaction"))
    assert result is not None
    _, _, value, _, _, _ = result
    assert "satisfaction" in value.lower()


# ---------------------------------------------------------------------------
# distill_procedures
# ---------------------------------------------------------------------------

def _success_ep(goal, action):
    return _ep(goal=goal, action=action, outcome="success")


def test_procedure_from_multiple_steps():
    goal = "Implement JWT authentication for the REST API"
    episodes = [
        _success_ep(goal, "read_file('auth.py')"),
        _success_ep(goal, "edit_file('auth.py', ...)"),
        _success_ep(goal, "run_tests('tests/test_auth.py')"),
    ]
    procedures = distill_procedures(episodes)
    assert len(procedures) == 1
    mtype, fact, value, importance, confidence, ver_status = procedures[0]
    assert mtype == "procedural"
    assert "procedure that worked" in fact.lower()
    assert "1." in value and "2." in value
    assert ver_status == "inferred"


def test_single_step_goal_no_procedure():
    goal = "Fix typo in README documentation file"
    episodes = [_success_ep(goal, "edit_file('README.md')")]
    procedures = distill_procedures(episodes)
    assert len(procedures) == 0


def test_procedure_deduplicates_same_action():
    goal = "Write unit tests for the payment module"
    episodes = [
        _success_ep(goal, "read_file('payments.py')"),
        _success_ep(goal, "read_file('payments.py')"),  # duplicate
        _success_ep(goal, "edit_file('test_payments.py')"),
    ]
    procedures = distill_procedures(episodes)
    assert len(procedures) == 1
    _, _, value, _, _, _ = procedures[0]
    # Should not double-list the duplicate action
    assert value.count("read_file") == 1


def test_verdict_rows_excluded_from_procedure_steps():
    goal = "Refactor the database connection module to async"
    episodes = [
        _success_ep(goal, "read_file('db.py')"),
        _success_ep(goal, "edit_file('db.py')"),
        _ep(goal=goal, action="verified by automated grading", outcome="success",
            episode_kind="task_verdict"),
    ]
    procedures = distill_procedures(episodes)
    assert len(procedures) == 1
    _, _, value, _, _, _ = procedures[0]
    assert "verified by automated grading" not in value


# ---------------------------------------------------------------------------
# consolidate_episodes (full integration with injected save_fn)
# ---------------------------------------------------------------------------

def test_consolidate_writes_to_save_fn():
    saved = []
    def fake_save(fact, value, *, memory_type, importance, confidence=0.8, verification_status="observed", scope="global", source="", **kwargs):
        saved.append({"fact": fact, "value": value, "memory_type": memory_type, "confidence": confidence})

    episodes = [
        _ep(outcome="success"),
        _ep(outcome="failure", goal="Implement JWT auth for the REST API endpoints",
            failure_reason="SyntaxError"),
    ]
    report = consolidate_episodes(episodes, save_fn=fake_save)
    assert report["written"] == 2
    assert report["procedural"] == 1
    assert report["lessons"] == 1
    assert len(saved) == 2


def test_consolidate_skips_short_goals():
    saved = []
    episodes = [_ep(goal="fix", outcome="success")]  # too short
    report = consolidate_episodes(episodes, save_fn=lambda *a, **k: saved.append(1))
    assert report["written"] == 0
    assert report["skipped"] == 1


# ---------------------------------------------------------------------------
# compress_stale_episodes
# ---------------------------------------------------------------------------

def test_compress_no_candidates():
    report = compress_stale_episodes(
        get_stale_fn=lambda **k: [],
        delete_fn=lambda ids: None,
        save_fn=lambda *a, **k: None,
    )
    assert report["episodes_scanned"] == 0
    assert report["clusters_compressed"] == 0


def test_compress_small_cluster_not_compressed():
    """A cluster below MIN_COMPRESSION_CLUSTER_SIZE should not be compressed."""
    import numpy as np
    episodes = [
        {"id": i, "goal": "Fix the login bug in auth module", "outcome": "failure",
         "failure_reason": "error"}
        for i in range(3)  # < MIN_COMPRESSION_CLUSTER_SIZE (5)
    ]
    saved = []
    deleted = []
    report = compress_stale_episodes(
        get_stale_fn=lambda **k: episodes,
        delete_fn=lambda ids: deleted.extend(ids),
        save_fn=lambda *a, **k: saved.append(1),
        embed_fn=lambda x: np.ones(384, dtype=np.float32),
    )
    assert report["clusters_compressed"] == 0
    assert not deleted


def test_compress_large_cluster_compressed():
    """A cluster of 6 similar episodes should be compressed into one summary."""
    import numpy as np
    goal = "Refactor the payment processing module to use async database calls"
    episodes = [
        {"id": i, "goal": goal, "outcome": "failure" if i < 4 else "success",
         "failure_reason": "Timeout error" if i < 4 else ""}
        for i in range(6)  # >= MIN_COMPRESSION_CLUSTER_SIZE (5)
    ]
    saved = []
    deleted = []
    report = compress_stale_episodes(
        get_stale_fn=lambda **k: episodes,
        delete_fn=lambda ids: deleted.extend(ids),
        save_fn=lambda fact, val, **k: saved.append((fact, val)),
        embed_fn=lambda x: np.ones(384, dtype=np.float32),
        similarity_threshold=0.0,  # force all into one cluster for test
        min_cluster_size=5,
    )
    assert report["clusters_compressed"] == 1
    assert report["episodes_compressed"] == 6
    assert len(saved) == 1
    _, summary_value = saved[0][0], saved[0][1]
    assert "6 similar episodes" in summary_value
    assert "Timeout error" in summary_value
    assert len(deleted) == 6

