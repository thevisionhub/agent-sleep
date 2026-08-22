"""
Unit tests for mathematically active epistemic ranking in memory retrieval.
"""
import pytest
from pathlib import Path

from agent_sleep.storage.db import (
    ensure_db_initialized,
    save_semantic_memory,
    recall_memories,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "ranking_test.db"
    ensure_db_initialized(db_file)
    return db_file


def test_verified_outranks_unverified_memory(temp_db):
    # Observed memory (trust = 0.50)
    save_semantic_memory(
        fact="Database Concurrency Lock Advice",
        value="Observed advice: Increase thread sleep time",
        memory_type="lesson",
        importance=0.8,
        confidence=0.8,
        utility_score=0.5,
        verification_status="observed",
        scope="billing",
        db_path=temp_db,
    )

    # Verified memory (trust = 1.0)
    save_semantic_memory(
        fact="Database Concurrency Lock Resolution",
        value="Verified fact: Enable WAL mode and set busy_timeout=5000",
        memory_type="lesson",
        importance=0.8,
        confidence=0.8,
        utility_score=0.5,
        verification_status="verified",
        scope="billing",
        db_path=temp_db,
    )

    recalled = recall_memories("Database Concurrency Lock", scopes=["billing"], min_score=0.01, db_path=temp_db)
    assert len(recalled) == 2
    # Verified memory must rank FIRST and have higher score than observed memory
    assert recalled[0]["verification_status"] == "verified"
    assert "WAL mode" in recalled[0]["value"]
    assert recalled[0]["relevance_score"] > recalled[1]["relevance_score"]
    assert recalled[0]["epistemic_trust"] == 1.00
    assert recalled[1]["epistemic_trust"] == 0.50
