"""
Epistemic provenance, safety gating, and scope isolation test suite.
"""
import pytest
from pathlib import Path
from agent_sleep.storage.db import (
    ensure_db_initialized,
    save_semantic_memory,
    record_memory_utility_feedback,
    _recall_keyword,
    save_causal_hypothesis,
    recall_causal_hypotheses,
    update_competence,
    get_domain_competence,
    get_all_competencies,
    save_episode,
    recall_rules,
    add_candidate_rule,
    _cursor,
)
from agent_sleep.self_model import SelfModel
from agent_sleep.beliefs import BeliefSystem, promote_candidate_rules


def test_utility_feedback_single_source_caps_at_repeated(tmp_path: Path):
    """Single-source repeated success should cap at 'repeated' and not reach 'verified'."""
    db_file = tmp_path / "test_epistemic_single.db"
    ensure_db_initialized(db_path=db_file)

    mid = save_semantic_memory(
        "API convention",
        "Use header X-Auth-Token",
        scope="repo_a",
        source="agent_tool",
        verification_status="observed",
        provenance={"distinct_sources_count": 1, "sources": ["agent_tool"]},
        db_path=db_file,
    )

    # 5 consecutive successful applications from the same single source
    for _ in range(5):
        record_memory_utility_feedback([mid], outcome="success", was_applied=True, db_path=db_file)

    with _cursor(db_path=db_file) as cur:
        cur.execute("SELECT verification_status, success_count FROM semantic_memories WHERE id=?", (mid,))
        row = cur.fetchone()
        assert row["success_count"] == 5
        assert row["verification_status"] == "repeated", "Single source must cap at 'repeated' without independent verification"


def test_utility_feedback_multi_source_promotes_to_verified(tmp_path: Path):
    """Multi-source evidence with 5 successful applications should promote to 'verified'."""
    db_file = tmp_path / "test_epistemic_multi.db"
    ensure_db_initialized(db_path=db_file)

    mid = save_semantic_memory(
        "API convention",
        "Use header X-Auth-Token",
        scope="repo_a",
        source="agent_tool",
        verification_status="observed",
        provenance={"distinct_sources_count": 2, "sources": ["agent_tool", "test_runner"]},
        db_path=db_file,
    )

    for _ in range(5):
        record_memory_utility_feedback([mid], outcome="success", was_applied=True, db_path=db_file)

    with _cursor(db_path=db_file) as cur:
        cur.execute("SELECT verification_status, success_count, utility_score FROM semantic_memories WHERE id=?", (mid,))
        row = cur.fetchone()
        assert row["success_count"] == 5
        assert row["utility_score"] >= 0.75
        assert row["verification_status"] == "verified", "Multi-source evidence must promote to 'verified'"


def test_causal_hypothesis_confidence_scaling_with_diversity(tmp_path: Path):
    """Single source repeated failures cap at 0.65; multi-source diverse failures scale to 0.95."""
    db_file = tmp_path / "test_causal_div.db"
    ensure_db_initialized(db_path=db_file)

    # Single source repeated 10 times
    for i in range(10):
        save_causal_hypothesis(
            session_id=f"sess_{i}",
            action="git merge --ff",
            hypothesis="Fast forward merge failed due to diverged history",
            effect="Conflict",
            confidence=0.35,
            scope="repo_a",
            provenance={"distinct_sources_count": 1, "sources": ["single_tool"]},
            db_path=db_file,
        )

    with _cursor(db_path=db_file) as cur:
        cur.execute("SELECT support_count, confidence FROM causal_hypotheses WHERE action='git merge --ff'")
        row = cur.fetchone()
        assert row["support_count"] == 10
        assert row["confidence"] <= 0.65, "Single source repetition must cap confidence at 0.65"

    # Multi-source diverse evidence
    for i in range(5):
        save_causal_hypothesis(
            session_id=f"sess_multi_{i}",
            action="docker run --privileged",
            hypothesis="Docker permission denied due to seccomp profile",
            effect="Permission Denied",
            confidence=0.35,
            scope="repo_a",
            provenance={"distinct_sources_count": 3, "distinct_environments_count": 2, "sources": ["ci", "local", "k8s"]},
            db_path=db_file,
        )

    with _cursor(db_path=db_file) as cur:
        cur.execute("SELECT support_count, confidence FROM causal_hypotheses WHERE action='docker run --privileged'")
        row = cur.fetchone()
        assert row["confidence"] >= 0.85, "Multi-source diverse evidence must achieve high confidence"


def test_fallback_keyword_recall_excludes_quarantined_and_expired(tmp_path: Path):
    """Keyword fallback recall must exclude quarantined and expired memories."""
    db_file = tmp_path / "test_kw_safety.db"
    ensure_db_initialized(db_path=db_file)

    m1 = save_semantic_memory("PostgreSQL host", "localhost:5432", scope="repo_a", db_path=db_file)
    m2 = save_semantic_memory("PostgreSQL user", "postgres_admin", scope="repo_a", db_path=db_file)
    m3 = save_semantic_memory("PostgreSQL pass", "secret_pass", scope="repo_a", db_path=db_file)

    with _cursor(commit=True, db_path=db_file) as cur:
        cur.execute("UPDATE semantic_memories SET verification_status='quarantined' WHERE id=?", (m2,))
        cur.execute("UPDATE semantic_memories SET verification_status='expired' WHERE id=?", (m3,))

    results = _recall_keyword("PostgreSQL", top_k=10, scopes=["repo_a"], memory_types=None, db_path=db_file)
    recalled_ids = [r["id"] for r in results]

    assert m1 in recalled_ids
    assert m2 not in recalled_ids, "Quarantined memory must be excluded from keyword recall"
    assert m3 not in recalled_ids, "Expired memory must be excluded from keyword recall"


def test_self_competence_scope_isolation(tmp_path: Path):
    """Competence updates in scope_A must not affect scope_B."""
    db_file = tmp_path / "test_self_model_scope.db"
    ensure_db_initialized(db_path=db_file)

    sm_a = SelfModel(scope="repo_a", db_path=db_file)
    sm_b = SelfModel(scope="repo_b", db_path=db_file)

    # Train repo_a on SQL successes
    for _ in range(5):
        sm_a.update("SQL", True, db_path=db_file)

    # Train repo_b on SQL failures
    for _ in range(5):
        sm_b.update("SQL", False, db_path=db_file)

    comp_a = sm_a.get_competence("SQL", db_path=db_file)
    comp_b = sm_b.get_competence("SQL", db_path=db_file)

    assert comp_a > 0.70, f"Expected high competence in repo_a, got {comp_a}"
    assert comp_b < 0.35, f"Expected low competence in repo_b, got {comp_b}"


@pytest.mark.asyncio
async def test_cross_session_rule_promotion(tmp_path: Path):
    """Failures occurring across distinct sessions in the same scope should accumulate and promote a rule."""
    db_file = tmp_path / "test_cross_sess_rules.db"
    ensure_db_initialized(db_path=db_file)

    # Session 1: One failure
    save_episode(
        session_id="session_1",
        goal="Run alembic migration",
        action="alembic upgrade head",
        outcome="failure",
        failure_reason="Target database is not up to date",
        scope="repo_a",
        db_path=db_file,
    )

    # Session 2: One identical failure
    save_episode(
        session_id="session_2",
        goal="Run alembic migration after merge",
        action="alembic upgrade head",
        outcome="failure",
        failure_reason="Target database is not up to date",
        scope="repo_a",
        db_path=db_file,
    )

    from agent_sleep.consolidator import SleepConsolidator
    consolidator = SleepConsolidator(scope="repo_a", db_path=db_file)
    res = consolidator.run("session_2")

    active_rules = recall_rules("alembic upgrade migration", scopes=["repo_a"], db_path=db_file)
    assert len(active_rules) > 0
    assert any("alembic" in r.lower() for r in active_rules)
