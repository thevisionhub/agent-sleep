"""
Unit tests for memory utility feedback loop (retrieval -> application -> outcome attribution).
"""
import pytest
from pathlib import Path

from agent_sleep import AgentMemory
from agent_sleep.storage.db import (
    ensure_db_initialized,
    save_semantic_memory,
    recall_memories,
    record_memory_utility_feedback,
    _cursor,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "utility_test.db"
    ensure_db_initialized(db_file)
    return db_file


def test_utility_feedback_success_boost(temp_db):
    memory = AgentMemory(session_id="s_util", scope="global", db_path=temp_db)
    
    save_semantic_memory(
        fact="Set busy_timeout to 5000ms",
        value="Prevents immediate locks under concurrency",
        memory_type="procedural",
        utility_score=0.50,
        db_path=temp_db,
    )

    recalled = recall_memories("busy_timeout concurrency", scopes=["global"], db_path=temp_db)
    assert len(recalled) >= 1
    mem_id = recalled[0]["id"]

    # Feedback: agent applied memory and succeeded
    res = memory.record_memory_feedback([mem_id], outcome="success", was_applied=True)
    assert res["updated"] == 1

    with _cursor(db_path=temp_db) as cur:
        cur.execute("SELECT times_applied, success_count, failure_count, utility_score FROM semantic_memories WHERE id=?", (mem_id,))
        row = cur.fetchone()
        assert row["times_applied"] == 1
        assert row["success_count"] == 1
        assert row["failure_count"] == 0
        assert row["utility_score"] == 0.60  # 0.50 + 0.10


def test_utility_feedback_failure_penalty(temp_db):
    memory = AgentMemory(session_id="s_util2", scope="global", db_path=temp_db)

    save_semantic_memory(
        fact="Use raw thread locks",
        value="Manual lock handling",
        memory_type="procedural",
        utility_score=0.50,
        db_path=temp_db,
    )

    recalled = recall_memories("raw thread locks", scopes=["global"], db_path=temp_db)
    mem_id = recalled[0]["id"]

    # Feedback: agent applied memory and failed
    res = memory.record_memory_feedback([mem_id], outcome="failure", was_applied=True)
    assert res["updated"] == 1

    with _cursor(db_path=temp_db) as cur:
        cur.execute("SELECT times_applied, success_count, failure_count, utility_score FROM semantic_memories WHERE id=?", (mem_id,))
        row = cur.fetchone()
        assert row["times_applied"] == 1
        assert row["success_count"] == 0
        assert row["failure_count"] == 1
        assert row["utility_score"] == 0.35  # 0.50 - 0.15


def test_utility_feedback_causal_attribution(temp_db):
    memory = AgentMemory(session_id="s_util3", scope="global", db_path=temp_db)

    # 1. Memory A: merely retrieved (not applied)
    save_semantic_memory(
        fact="Memory A: General tip",
        value="Clean code practices",
        memory_type="procedural",
        utility_score=0.50,
        db_path=temp_db,
    )
    # 2. Memory B: applied and verified as direct causal contributor
    save_semantic_memory(
        fact="Memory B: Explicit lock fix",
        value="Pass timeout=5.0 to acquire()",
        memory_type="procedural",
        utility_score=0.50,
        db_path=temp_db,
    )

    recalled_a = recall_memories("General tip", scopes=["global"], db_path=temp_db)
    recalled_b = recall_memories("Explicit lock fix", scopes=["global"], db_path=temp_db)
    id_a = recalled_a[0]["id"]
    id_b = recalled_b[0]["id"]

    # Memory A was present but not applied
    memory.record_memory_feedback([id_a], outcome="success", was_applied=False)
    # Memory B was actively applied and was the causal contributor
    memory.record_memory_feedback([id_b], outcome="success", was_applied=True, is_causal_contributor=True)

    with _cursor(db_path=temp_db) as cur:
        cur.execute("SELECT utility_score, times_applied FROM semantic_memories WHERE id=?", (id_a,))
        row_a = cur.fetchone()
        cur.execute("SELECT utility_score, times_applied FROM semantic_memories WHERE id=?", (id_b,))
        row_b = cur.fetchone()

        # Unapplied memory receives nominal co-occurrence boost (+0.02)
        assert row_a["utility_score"] == 0.52
        assert row_a["times_applied"] == 0

        # Causal contributor receives full attribution boost (+0.15)
        assert row_b["utility_score"] == 0.65
        assert row_b["times_applied"] == 1

