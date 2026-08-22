"""
Unit tests for self-competence model and behavioral policy feedback loop.
"""
from pathlib import Path
import pytest

from agent_sleep import AgentMemory, SelfModel, SleepConsolidator
from agent_sleep.storage.db import ensure_db_initialized, get_domain_competence


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "competence_test.db"
    ensure_db_initialized(db_file)
    return db_file


def test_self_model_policy_derivation(temp_db):
    model = SelfModel(db_path=temp_db)

    # Initial unknown domain defaults to moderate/0.5
    policy = model.get_behavioral_policy("SQL", db_path=temp_db)
    assert policy["level"] == "MODERATE"
    assert policy["verification_intensity"] == "STANDARD"

    # Multiple failures drop competence to LOW
    model.update("SQL", False, db_path=temp_db)
    model.update("SQL", False, db_path=temp_db)
    low_policy = model.get_behavioral_policy("SQL", db_path=temp_db)
    assert low_policy["level"] == "LOW"
    assert low_policy["verification_intensity"] == "STRICT"
    assert low_policy["retry_budget"] == 5
    assert "Mandatory" in low_policy["directive"]

    # Multiple successes raise competence to HIGH
    for _ in range(10):
        model.update("Python", True, db_path=temp_db)
    high_policy = model.get_behavioral_policy("Python", db_path=temp_db)
    assert high_policy["level"] == "HIGH"
    assert high_policy["verification_intensity"] == "LIGHTWEIGHT"
    assert high_policy["autonomy"] == "HIGH"


def test_competence_policy_prompt_injection(temp_db):
    memory = AgentMemory(session_id="s_comp", scope="global", db_path=temp_db)
    consolidator = SleepConsolidator(scope="global", db_path=temp_db)

    # Record repeated failures in SQL domain
    memory.record_episode(
        goal="SELECT * FROM users WHERE active=1 query execution",
        action="execute_raw_sql()",
        outcome="failure",
        failure_reason="SyntaxError in SQL query",
    )
    memory.record_episode(
        goal="UPDATE accounts SET balance=0 database query",
        action="execute_raw_sql()",
        outcome="failure",
        failure_reason="TableNotFound in SQL query",
    )

    consolidator.run(session_id="s_comp")

    # Recall on a SQL task
    context = memory.recall("Write SQL query to fetch user orders")
    assert "🛡 [SELF-MODEL: LOW COMPETENCE IN SQL]" in context
    assert "Mandatory pre-execution validation" in context

    # Structured recall
    structured = memory.recall_structured("Write SQL query to fetch user orders")
    assert structured["domain"] == "SQL"
    assert structured["operational_policy"]["level"] == "LOW"
    assert structured["operational_policy"]["verification_intensity"] == "STRICT"
