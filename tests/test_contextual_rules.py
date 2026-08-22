"""
Unit tests for contextual rules, exceptions, and contradiction specialization.
"""
import pytest
from pathlib import Path

from agent_sleep.storage.db import (
    ensure_db_initialized,
    add_candidate_rule,
    recall_rules,
    specialize_rule_with_exception,
    _cursor,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "rules_test.db"
    ensure_db_initialized(db_file)
    return db_file


def test_contextual_rule_condition_and_exception(temp_db):
    add_candidate_rule(
        belief="Configure SQLite WAL mode for concurrency",
        confidence=0.85,
        condition="Local POSIX / Windows disk",
        exception="NFS or network shared filesystem",
        scope="db_service",
        db_path=temp_db,
    )

    recalled = recall_rules("concurrency in SQLite", scopes=["db_service"], db_path=temp_db)
    assert len(recalled) == 1
    assert "Applies: Local POSIX / Windows disk" in recalled[0]
    assert "Except: NFS or network shared filesystem" in recalled[0]


def test_specialize_rule_with_exception(temp_db):
    add_candidate_rule(
        belief="Use multithreading for file downloads",
        confidence=0.70,
        scope="network",
        db_path=temp_db,
    )

    with _cursor(db_path=temp_db) as cur:
        cur.execute("SELECT id FROM candidate_rules WHERE belief LIKE 'Use multithreading%'")
        rule_id = cur.fetchone()["id"]

    res = specialize_rule_with_exception(
        rule_id=rule_id,
        condition="Multiple independent URLs",
        exception="Single rate-limited host",
        db_path=temp_db,
    )
    assert res["status"] == "specialized"

    recalled = recall_rules("file downloads", scopes=["network"], db_path=temp_db)
    assert len(recalled) == 1
    assert "Applies: Multiple independent URLs" in recalled[0]
    assert "Except: Single rate-limited host" in recalled[0]
