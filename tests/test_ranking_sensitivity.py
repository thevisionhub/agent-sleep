"""
Ranking Sensitivity & Parameter Robustness Suite.
Validates that retrieval ranking is stable across parameter sweeps
(relevance-heavy, trust-heavy, utility-heavy, balanced).
"""
import pytest
from pathlib import Path

from agent_sleep.storage.db import (
    ensure_db_initialized,
    save_semantic_memory,
    _cursor,
    cosine_similarity,
    deserialize_vector,
)
from agent_sleep.storage.embeddings import embed


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "sensitivity_test.db"
    ensure_db_initialized(db_file)
    return db_file


def test_ranking_parameter_sensitivity_sweep(temp_db):
    """
    Populate a set of benchmark candidate memories with varying epistemic status,
    utility scores, and semantic relevance, then sweep weighting vectors to verify
    ranking monotonicity and stability.
    """
    # 1. Verified memory (high trust, high utility)
    save_semantic_memory(
        fact="Postgres Connection Pool Sizing Strategy",
        value="Use max_connections=20 with pgbouncer",
        memory_type="procedural",
        importance=0.8,
        confidence=0.9,
        utility_score=0.9,
        verification_status="verified",
        scope="infra",
        db_path=temp_db,
    )

    # 2. Observed memory (medium trust, neutral utility)
    save_semantic_memory(
        fact="Postgres Connection Pool Sizing Observation",
        value="Observed default 100 connections without pooler",
        memory_type="lesson",
        importance=0.7,
        confidence=0.6,
        utility_score=0.5,
        verification_status="observed",
        scope="infra",
        db_path=temp_db,
    )

    # 3. Raw unverified speculation (low trust, low utility)
    save_semantic_memory(
        fact="Postgres Connection Pool Speculation",
        value="Maybe set unlimited connections",
        memory_type="lesson",
        importance=0.5,
        confidence=0.4,
        utility_score=0.3,
        verification_status="raw",
        scope="infra",
        db_path=temp_db,
    )

    query = "Postgres Connection Pool Sizing"
    q_vec = embed(query)

    with _cursor(db_path=temp_db) as cur:
        cur.execute("SELECT fact, value, verification_status, utility_score, importance, confidence, embedding FROM semantic_memories")
        rows = [dict(r) for r in cur.fetchall()]

    trust_map = {"verified": 1.0, "repeated": 0.75, "observed": 0.50, "raw": 0.20}

    # Weight configurations to sweep
    sweeps = [
        {"name": "balanced", "base": 0.20, "w_trust": 0.50, "w_util": 0.15, "w_imp": 0.10, "w_conf": 0.05},
        {"name": "trust_heavy", "base": 0.10, "w_trust": 0.70, "w_util": 0.10, "w_imp": 0.05, "w_conf": 0.05},
        {"name": "utility_heavy", "base": 0.15, "w_trust": 0.35, "w_util": 0.35, "w_imp": 0.10, "w_conf": 0.05},
        {"name": "relevance_heavy", "base": 0.40, "w_trust": 0.30, "w_util": 0.15, "w_imp": 0.10, "w_conf": 0.05},
    ]

    for sweep in sweeps:
        scored = []
        for r in rows:
            vec = deserialize_vector(r["embedding"])
            sim = cosine_similarity(q_vec, vec)
            trust = trust_map[r["verification_status"]]
            q_factor = (
                sweep["base"]
                + sweep["w_trust"] * trust
                + sweep["w_util"] * r["utility_score"]
                + sweep["w_imp"] * r["importance"]
                + sweep["w_conf"] * r["confidence"]
            )
            score = sim * q_factor
            scored.append((score, r["verification_status"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Verified must rank #1 and raw must rank #3 across ALL parameter sweeps
        assert scored[0][1] == "verified", f"Sweep {sweep['name']} failed: verified was not #1"
        assert scored[1][1] == "observed", f"Sweep {sweep['name']} failed: observed was not #2"
        assert scored[2][1] == "raw", f"Sweep {sweep['name']} failed: raw was not #3"
