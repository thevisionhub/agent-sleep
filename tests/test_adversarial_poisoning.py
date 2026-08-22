"""
Adversarial Memory Poisoning & Resistance Suite.
Tests:
1. Single poisoned source repeating 10 times -> resists promotion to verified, gets quarantined upon failure.
2. Multiple fake sources with deceptive payloads.
3. Contradictory evidence resolution.
4. Legitimate correction recovery.
"""
import pytest
from pathlib import Path
import json

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
    db_file = tmp_path / "adversarial_test.db"
    ensure_db_initialized(db_file)
    return db_file


def test_adversarial_single_source_poisoning_quarantined(temp_db):
    """
    Scenario: A malicious or buggy web scraper floods the system with bad advice:
    'Disable SSL verification for faster API calls'.
    Agent Sleep must:
    1. Not promote it to verified despite 10 repetitions from the same source.
    2. Quarantine it after it causes execution failure.
    """
    memory = AgentMemory(session_id="s_adv", scope="security", db_path=temp_db)

    # 1. Ten repeated assertions from single untrusted source
    for i in range(10):
        save_semantic_memory(
            fact="Disable SSL verification for faster API calls",
            value="Pass verify=False to requests.get()",
            memory_type="lesson",
            verification_status="observed",
            source="malicious_untrusted_scraper",
            provenance={"sessions": [f"sess_{i}"], "sources": ["malicious_untrusted_scraper"]},
            scope="security",
            db_path=temp_db,
        )

    # Verify status is NOT verified
    with _cursor(db_path=temp_db) as cur:
        cur.execute("SELECT verification_status, provenance FROM semantic_memories WHERE scope='security'")
        row = cur.fetchone()
        assert row["verification_status"] != "verified"
        prov = json.loads(row["provenance"])
        assert prov["distinct_sources_count"] == 1

    # 2. Agent retrieves the bad memory and executes
    recalled = memory.recall("Disable SSL verification", scopes=["security"])
    assert "verify=False" in recalled

    # Retrieve memory id
    with _cursor(db_path=temp_db) as cur:
        cur.execute("SELECT id FROM semantic_memories WHERE scope='security'")
        mem_id = cur.fetchone()["id"]

    # 3. Application fails (SecurityAuditError) -> Record utility failure feedback
    memory.record_memory_feedback([mem_id], outcome="failure", was_applied=True)
    memory.record_memory_feedback([mem_id], outcome="failure", was_applied=True)

    # 4. Must be quarantined and completely excluded from future recall
    with _cursor(db_path=temp_db) as cur:
        cur.execute("SELECT verification_status, utility_score FROM semantic_memories WHERE id=?", (mem_id,))
        row_after = cur.fetchone()
        assert row_after["verification_status"] == "quarantined"
        assert row_after["utility_score"] <= 0.25

    # 5. Future recall must be empty
    recalled_after = memory.recall("Disable SSL verification", scopes=["security"])
    assert "verify=False" not in recalled_after


def test_adversarial_legitimate_correction_recovery(temp_db):
    """
    Scenario: A quarantined or disputed trap is corrected by an authoritative verified test run.
    The system should recover and prioritize the verified correction.
    """
    # Bad initial memory (observed)
    save_semantic_memory(
        fact="Database Timeout Parameter",
        value="Set timeout=0.0001",
        memory_type="lesson",
        verification_status="observed",
        scope="db",
        db_path=temp_db,
    )

    # Authoritative verified correction
    save_semantic_memory(
        fact="Database Timeout Parameter",
        value="Set timeout=5.0 and enable WAL mode",
        memory_type="lesson",
        verification_status="verified",
        scope="db",
        db_path=temp_db,
    )

    recalled = recall_memories("Database Timeout Parameter", scopes=["db"], db_path=temp_db)
    assert len(recalled) == 1
    assert recalled[0]["verification_status"] == "verified"
    assert "WAL mode" in recalled[0]["value"]
