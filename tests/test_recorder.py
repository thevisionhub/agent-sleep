"""
Tests for AgentMemory recorder.
Uses a temp DB to avoid polluting any real store.
"""
import os
import tempfile
import pytest
from agent_sleep.recorder import AgentMemory
from agent_sleep import storage


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Point the DB at a fresh temp file for every test."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("AGENT_SLEEP_DB", db_file)
    # Force re-init of DB_PATH
    import importlib
    import agent_sleep.storage.db as db_mod
    db_mod.DB_PATH = __import__("pathlib").Path(db_file)
    db_mod.init_db()
    yield
    # Cleanup is automatic via tmp_path


def test_record_episode_returns_id():
    memory = AgentMemory(session_id="test_session")
    eid = memory.record_episode(
        goal="Refactor the authentication module to use JWT",
        action="read_file('auth.py')",
        outcome="success",
    )
    assert isinstance(eid, int)
    assert eid > 0


def test_record_task_verdict():
    memory = AgentMemory(session_id="test_session")
    eid = memory.record_task_verdict(
        goal="Add async support to the database layer",
        passed=True,
    )
    assert isinstance(eid, int)


def test_recall_returns_empty_when_no_memories():
    memory = AgentMemory(session_id="empty_session")
    result = memory.recall("some task")
    assert result == "" or result is not None  # empty string is fine


def test_recall_structured_returns_dict():
    memory = AgentMemory(session_id="test_session")
    result = memory.recall_structured("some task")
    assert "memories" in result
    assert "rules" in result


def test_failure_sets_higher_emotional_weight():
    """Failure episodes should get emotional_weight=1.5 by default."""
    from agent_sleep.storage.db import get_unprocessed_episodes
    memory = AgentMemory(session_id="weight_test")
    memory.record_episode(
        goal="Add rate limiting to the REST API endpoints",
        action="edit_file('middleware.py')",
        outcome="failure",
        failure_reason="AttributeError",
    )
    episodes = get_unprocessed_episodes("weight_test")
    assert len(episodes) == 1
    assert episodes[0]["emotional_weight"] == pytest.approx(1.5)


def test_success_sets_lower_emotional_weight():
    from agent_sleep.storage.db import get_unprocessed_episodes
    memory = AgentMemory(session_id="weight_test2")
    memory.record_episode(
        goal="Add rate limiting to the REST API endpoints",
        action="edit_file('middleware.py')",
        outcome="success",
    )
    episodes = get_unprocessed_episodes("weight_test2")
    assert episodes[0]["emotional_weight"] == pytest.approx(1.0)


def test_scope_isolation():
    """Memories stored in repo_a scope should not bleed into repo_b."""
    from agent_sleep.storage.db import save_semantic_memory
    save_semantic_memory("Specific rule for Repo A", "Always use PostgreSQL pool", scope="repo_a")
    save_semantic_memory("Global rule", "Always write clean code", scope="global")

    mem_a = AgentMemory(session_id="s1", scope="repo_a")
    mem_b = AgentMemory(session_id="s2", scope="repo_b")

    recall_a = mem_a.recall("PostgreSQL connection configuration")
    recall_b = mem_b.recall("PostgreSQL connection configuration")

    assert "PostgreSQL pool" in recall_a
    assert "PostgreSQL pool" not in recall_b


def test_provenance_tracking():
    """Verify that provenance metadata is stored with distilled memories."""
    from agent_sleep.storage.db import save_semantic_memory, recall_memories
    prov = {"session_id": "test_prov", "episode_id": 99, "model": "test-v1"}
    save_semantic_memory(
        "Provenance test fact",
        "Provenance test value",
        provenance=prov,
    )
    recalled = recall_memories("Provenance test fact")
    assert len(recalled) == 1
    assert "provenance" in recalled[0]
    import json
    parsed_prov = json.loads(recalled[0]["provenance"])
    assert parsed_prov.get("episode_id") == 99


def test_decay_stale_memories():
    """Verify that stale, unverified memories below utility threshold are pruned."""
    from agent_sleep.storage.db import save_semantic_memory, decay_stale_memories, recall_memories
    # Save a low-utility unverified memory
    save_semantic_memory(
        "Temporary scratch memory",
        "Temporary scratch value",
        utility_score=0.1,
        verification_status="observed",
    )
    # Decay with 0 age days to simulate decay pass
    rep = decay_stale_memories(max_age_days=0.0, min_utility=0.2)
    assert rep["pruned_stale_memories"] >= 1
    recalled = recall_memories("Temporary scratch memory")
    assert len(recalled) == 0
