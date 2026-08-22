"""
Unit tests for crash consistency and consolidation idempotency.
"""
import pytest
from pathlib import Path

from agent_sleep import AgentMemory, SleepConsolidator
from agent_sleep.storage.db import (
    ensure_db_initialized,
    _cursor,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "idempotency_test.db"
    ensure_db_initialized(db_file)
    return db_file


def test_repeated_consolidation_is_idempotent(temp_db):
    memory = AgentMemory(session_id="s_idem", scope="payment", db_path=temp_db)
    consolidator = SleepConsolidator(scope="payment", db_path=temp_db)

    # 1. Record an episode
    ep_id = memory.record_episode(
        goal="Process payment transactions concurrently",
        action="execute_worker_threads()",
        outcome="failure",
        failure_reason="OperationalError: database is locked",
        scope="payment",
    )

    # 2. Run consolidation first time
    rep1 = consolidator.run(session_id="s_idem")
    assert rep1["episodes_processed"] == 1
    assert rep1["memories_written"] == 1

    # 3. Simulate crash before mark_processed was committed or re-consolidation of same episode batch
    with _cursor(commit=True, db_path=temp_db) as cur:
        cur.execute("UPDATE execution_episodes SET processed_by_sleep = 0 WHERE id = ?", (ep_id,))

    # 4. Re-run consolidation on identical inputs
    rep2 = consolidator.run(session_id="s_idem")
    assert rep2["episodes_processed"] == 1

    # Verify: No duplicate memories created, evidence_count not incorrectly inflated
    with _cursor(db_path=temp_db) as cur:
        cur.execute("SELECT COUNT(*), evidence_count FROM semantic_memories WHERE scope='payment'")
        row = cur.fetchone()
        assert row[0] == 1, f"Expected 1 unique memory, found {row[0]}"
        assert row[1] == 1, f"Expected evidence_count 1, found {row[1]}"
