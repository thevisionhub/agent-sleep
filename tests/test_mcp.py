"""
Unit tests for agent-sleep MCP server adapter.

Skipped automatically when the ``mcp`` package is not installed so that
``pip install . && pytest`` works for contributors without the MCP extra.
"""
from pathlib import Path
import pytest

pytest.importorskip("mcp", reason="mcp package not installed; install agent-sleep[mcp] to run MCP tests")

from agent_sleep.mcp_server import (
    agent_sleep_status,
    agent_sleep_record,
    agent_sleep_consolidate,
    agent_sleep_recall,
)


@pytest.fixture
def temp_db(tmp_path):
    return str(tmp_path / "mcp_test.db")


def test_mcp_status_empty(temp_db):
    res = agent_sleep_status(scope="mcp_test", db_path=temp_db)
    assert res["status"] == "ready"
    assert res["semantic_memories_count"] == 0
    assert res["active_rules_count"] == 0
    assert res["unprocessed_episodes_for_session"] == 0


def test_mcp_record_and_consolidate(temp_db):
    # 1. Record failure
    rec = agent_sleep_record(
        goal="Configure Postgres connection pool for high concurrency",
        action="edit_file('db_pool.py')",
        outcome="failure",
        failure_reason="PoolExhaustedError: max 10 connections reached",
        scope="mcp_test",
        session_id="mcp_session_1",
        db_path=temp_db,
    )
    assert rec["recorded"] is True
    assert rec["episode_id"] is not None

    # Check status before consolidation
    status_before = agent_sleep_status(scope="mcp_test", session_id="mcp_session_1", db_path=temp_db)
    assert status_before["unprocessed_episodes_for_session"] == 1

    # 2. Consolidate
    report = agent_sleep_consolidate(session_id="mcp_session_1", scope="mcp_test", db_path=temp_db)
    assert report["episodes_processed"] == 1
    assert report["memories_written"] == 1

    # Check status after consolidation
    status_after = agent_sleep_status(scope="mcp_test", session_id="mcp_session_1", db_path=temp_db)
    assert status_after["unprocessed_episodes_for_session"] == 0
    assert status_after["semantic_memories_count"] == 1


def test_mcp_recall(temp_db):
    # Record and consolidate
    agent_sleep_record(
        goal="Configure Postgres connection pool for high concurrency",
        action="edit_file('db_pool.py')",
        outcome="failure",
        failure_reason="PoolExhaustedError: max 10 connections reached",
        scope="mcp_test",
        session_id="mcp_session_2",
        db_path=temp_db,
    )
    agent_sleep_consolidate(session_id="mcp_session_2", scope="mcp_test", db_path=temp_db)

    # Recall
    recall_res = agent_sleep_recall(
        query="Postgres database connection pool tuning",
        scope="mcp_test",
        db_path=temp_db,
    )
    assert recall_res["has_memories"] is True
    assert "PoolExhaustedError" in recall_res["context_prompt"]
    assert len(recall_res["memories"]) >= 1
