"""
Unit tests for causal hypotheses and error analysis operational feedback loop.
"""
from pathlib import Path
import pytest

from agent_sleep import AgentMemory, SleepConsolidator
from agent_sleep.storage.db import (
    save_causal_hypothesis,
    recall_causal_hypotheses,
    ensure_db_initialized,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "causal_test.db"
    ensure_db_initialized(db_file)
    return db_file


def test_save_and_recall_causal_hypothesis(temp_db):
    # Single observation gets cautious initial confidence (0.35)
    save_causal_hypothesis(
        session_id="s1",
        action="connect_sqlite(timeout=0.001)",
        hypothesis="Under concurrent writes, tiny timeout causes database lock",
        effect="OperationalError: database is locked",
        confidence=0.35,
        scope="billing",
        db_path=temp_db,
    )

    recalled = recall_causal_hypotheses("concurrent writes sqlite database", scopes=["billing"], db_path=temp_db)
    assert len(recalled) >= 1
    assert "tiny timeout causes database lock" in recalled[0]["hypothesis"]
    assert recalled[0]["confidence"] == 0.35
    assert recalled[0]["support_count"] == 1

    # Second observation accumulates evidence and increases confidence
    save_causal_hypothesis(
        session_id="s2",
        action="connect_sqlite(timeout=0.001)",
        hypothesis="Under concurrent writes, tiny timeout causes database lock",
        effect="OperationalError: database is locked",
        confidence=0.35,
        scope="billing",
        db_path=temp_db,
    )
    recalled2 = recall_causal_hypotheses("concurrent writes sqlite database", scopes=["billing"], db_path=temp_db)
    assert recalled2[0]["support_count"] == 2
    assert recalled2[0]["confidence"] == 0.50  # 0.35 + 0.15


def test_causal_hypothesis_closed_loop_recall(temp_db):
    memory = AgentMemory(session_id="s1", scope="billing", db_path=temp_db)
    consolidator = SleepConsolidator(scope="billing", db_path=temp_db)

    # Record failure
    memory.record_episode(
        goal="Batch invoice payment status updates concurrently",
        action="naive_writer_thread(sqlite)",
        outcome="failure",
        failure_reason="OperationalError: database is locked",
    )

    # Run sleep consolidation
    report = consolidator.run(session_id="s1")
    assert report["beliefs_revised"] >= 1

    # Recall on future session
    context = memory.recall("Process subscription renewals across worker threads database")
    assert "⚡ [CAUSAL MECHANISMS & TRAPS]" in context
    assert "database is locked" in context

    # Structured recall
    structured = memory.recall_structured("Process subscription renewals database")
    assert len(structured["causal_hypotheses"]) >= 1
    assert structured["causal_hypotheses"][0]["action"] == "naive_writer_thread(sqlite)"
