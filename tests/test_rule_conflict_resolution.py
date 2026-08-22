"""
Rule Specificity & Conflict Resolution Test Suite.
Tests that specific contextual rules and exceptions take precedence over general rules.
"""
import pytest
from pathlib import Path

from agent_sleep.storage.db import (
    ensure_db_initialized,
    add_candidate_rule,
    recall_rules,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "rule_conflict_test.db"
    ensure_db_initialized(db_file)
    return db_file


def test_specific_contextual_rule_outranks_general_rule(temp_db):
    # General rule
    add_candidate_rule(
        belief="Always use async file IO",
        confidence=0.60,
        scope="io",
        db_path=temp_db,
    )

    # Specific rule with explicit condition and exception
    add_candidate_rule(
        belief="Use synchronous IO with memory mapping",
        confidence=0.85,
        condition="Local SSD with files < 10MB",
        exception="Network NFS mounts",
        scope="io",
        db_path=temp_db,
    )

    recalled = recall_rules("Local SSD file IO", scopes=["io"], db_path=temp_db)
    assert len(recalled) >= 1
    # Specific rule with higher confidence and condition must rank #1
    assert "memory mapping" in recalled[0]
    assert "Applies: Local SSD with files < 10MB" in recalled[0]
