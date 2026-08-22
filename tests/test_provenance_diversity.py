"""
Unit tests for provenance diversity:
Verifying that 10 observations from the same source do NOT count as 10 independent sources.
"""
import pytest
from pathlib import Path

from agent_sleep.storage.db import (
    ensure_db_initialized,
    save_semantic_memory,
    save_causal_hypothesis,
    recall_memories,
    recall_causal_hypotheses,
    _cursor,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "prov_test.db"
    ensure_db_initialized(db_file)
    return db_file


def test_single_source_multiple_observations_gated(temp_db):
    # 1. Ten duplicate observations from the exact same source and session
    for _ in range(10):
        save_semantic_memory(
            fact="False API Fact",
            value="Misleading data from single buggy scraper",
            memory_type="lesson",
            verification_status="observed",
            source="buggy_scraper_v1",
            provenance={"sessions": ["session_a"], "sources": ["buggy_scraper_v1"]},
            db_path=temp_db,
        )

    with _cursor(db_path=temp_db) as cur:
        cur.execute("SELECT verification_status, confidence, provenance FROM semantic_memories WHERE fact='False API Fact'")
        row = cur.fetchone()
        assert row["verification_status"] != "verified", "Single source must never self-promote to verified"


def test_independent_sources_promote_memory(temp_db):
    # 1. Observation from source 1
    save_semantic_memory(
        fact="SQLite WAL mode is recommended for concurrency",
        value="Enables multiple readers and one writer without lock collisions",
        memory_type="lesson",
        verification_status="observed",
        source="doc_official",
        provenance={"sessions": ["s1"], "sources": ["doc_official"]},
        db_path=temp_db,
    )

    # 2. Observation from independent source 2
    save_semantic_memory(
        fact="SQLite WAL mode is recommended for concurrency",
        value="Enables multiple readers and one writer without lock collisions",
        memory_type="lesson",
        verification_status="observed",
        source="benchmark_runner",
        provenance={"sessions": ["s2"], "sources": ["benchmark_runner"]},
        db_path=temp_db,
    )

    with _cursor(db_path=temp_db) as cur:
        cur.execute("SELECT verification_status, evidence_count FROM semantic_memories WHERE fact LIKE 'SQLite WAL mode%'")
        row = cur.fetchone()
        assert row["verification_status"] == "repeated"
        assert row["evidence_count"] >= 2
