"""
Unit tests for epistemic memory lifecycle (RAW -> OBSERVED -> REPEATED -> VERIFIED -> QUARANTINED).
"""
import pytest
from pathlib import Path

from agent_sleep import AgentMemory, SleepConsolidator
from agent_sleep.storage.db import (
    ensure_db_initialized,
    save_semantic_memory,
    recall_memories,
    record_memory_utility_feedback,
    _cursor,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "lifecycle_test.db"
    ensure_db_initialized(db_file)
    return db_file


def test_memory_lifecycle_observation_to_repetition(temp_db):
    # 1. Single observation -> 'observed'
    save_semantic_memory(
        fact="Use SQLite WAL mode",
        value="Enables concurrent readers and single writer without lock",
        memory_type="lesson",
        verification_status="observed",
        db_path=temp_db,
    )

    recalled = recall_memories("SQLite WAL mode", scopes=["global"], db_path=temp_db)
    assert len(recalled) == 1
    assert recalled[0]["verification_status"] == "observed"
    assert recalled[0]["evidence_count"] == 1

    # 2. Second observation of identical lesson -> automatically transitions to 'repeated'
    save_semantic_memory(
        fact="Use SQLite WAL mode",
        value="Enables concurrent readers and single writer without lock",
        memory_type="lesson",
        verification_status="observed",
        db_path=temp_db,
    )

    recalled2 = recall_memories("SQLite WAL mode", scopes=["global"], db_path=temp_db)
    assert recalled2[0]["verification_status"] == "repeated"
    assert recalled2[0]["evidence_count"] == 2
    assert recalled2[0]["confidence"] > 0.80


def test_quarantined_memory_excluded_from_recall(temp_db):
    save_semantic_memory(
        fact="Disable WAL mode always",
        value="Deprecated advice that causes failures",
        memory_type="lesson",
        verification_status="observed",
        db_path=temp_db,
    )

    # Initial recall finds it
    recalled = recall_memories("Disable WAL mode", scopes=["global"], db_path=temp_db)
    assert len(recalled) == 1
    mem_id = recalled[0]["id"]

    # Repeated failures following its application -> triggers auto-quarantine
    record_memory_utility_feedback([mem_id], outcome="failure", was_applied=True, db_path=temp_db)
    record_memory_utility_feedback([mem_id], outcome="failure", was_applied=True, db_path=temp_db)
    record_memory_utility_feedback([mem_id], outcome="failure", was_applied=True, db_path=temp_db)

    # Verify status in database is quarantined
    with _cursor(db_path=temp_db) as cur:
        cur.execute("SELECT verification_status, utility_score FROM semantic_memories WHERE id=?", (mem_id,))
        row = cur.fetchone()
        assert row["verification_status"] == "quarantined"
        assert row["utility_score"] <= 0.25

    # Subsequent recall must NOT return quarantined memory
    recalled_after = recall_memories("Disable WAL mode", scopes=["global"], db_path=temp_db)
    assert len(recalled_after) == 0
