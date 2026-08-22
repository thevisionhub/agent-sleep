"""
SQLite persistence for agent-sleep (v0.1.1).

Features:
- Lazy table initialization (no side-effects on package import).
- Stored vector embeddings as BLOBs (eliminates O(N) re-embedding during recall).
- Multi-tier scope isolation (e.g. global, project, repo).
- Epistemic status tracking (observed, inferred, verified).
- Selective semantic rule retrieval.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from agent_sleep.storage.embeddings import (
    cosine_similarity,
    deserialize_vector,
    embed,
    serialize_vector,
)

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).parent.parent / "data" / "agent_sleep.db"

_INIT_LOCK = threading.Lock()
_INITIALIZED_PATHS: set[str] = set()


def get_db_path() -> Path:
    return Path(os.environ.get("AGENT_SLEEP_DB", str(_DEFAULT_DB)))


def _get_conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    target_path = db_path or get_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


@contextmanager
def _cursor(commit: bool = False, db_path: Optional[Path] = None):
    ensure_db_initialized(db_path)
    conn = _get_conn(db_path)
    try:
        yield conn.cursor()
        if commit:
            conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema (Lazy initialization)
# ---------------------------------------------------------------------------

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS execution_episodes (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id           TEXT    NOT NULL,
        scope                TEXT    NOT NULL DEFAULT 'global',
        goal                 TEXT    NOT NULL,
        plan                 TEXT    DEFAULT '',
        action               TEXT    NOT NULL,
        outcome              TEXT    NOT NULL,
        failure_reason       TEXT    DEFAULT '',
        episode_kind         TEXT    DEFAULT 'action',
        emotion_label        TEXT    DEFAULT '',
        prediction_error     REAL    DEFAULT 0.5,
        novelty              REAL    DEFAULT 1.0,
        emotional_weight     REAL    DEFAULT 1.0,
        processed_by_sleep   INTEGER DEFAULT 0,
        timestamp            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS semantic_memories (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        scope                TEXT    NOT NULL DEFAULT 'global',
        memory_type          TEXT    NOT NULL,
        fact                 TEXT    NOT NULL,
        value                TEXT    NOT NULL,
        importance           REAL    DEFAULT 0.5,
        confidence           REAL    DEFAULT 0.8,
        utility_score        REAL    DEFAULT 0.5,
        verification_status  TEXT    DEFAULT 'observed',
        evidence_count       INTEGER DEFAULT 1,
        contradiction_count  INTEGER DEFAULT 0,
        times_retrieved      INTEGER DEFAULT 0,
        times_applied        INTEGER DEFAULT 0,
        success_count        INTEGER DEFAULT 0,
        failure_count        INTEGER DEFAULT 0,
        embedding            BLOB    DEFAULT NULL,
        provenance           TEXT    DEFAULT '{}',
        source               TEXT    DEFAULT '',
        access_count         INTEGER DEFAULT 0,
        last_accessed        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        timestamp            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(scope, memory_type, fact)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS causal_hypotheses (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id          TEXT NOT NULL,
        scope               TEXT NOT NULL DEFAULT 'global',
        action              TEXT NOT NULL,
        hypothesis          TEXT NOT NULL,
        effect              TEXT NOT NULL,
        confidence          REAL DEFAULT 0.35,
        support_count       INTEGER DEFAULT 1,
        contradiction_count INTEGER DEFAULT 0,
        embedding           BLOB DEFAULT NULL,
        timestamp           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS skills (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id     TEXT NOT NULL,
        scope          TEXT NOT NULL DEFAULT 'global',
        skill_name     TEXT NOT NULL,
        description    TEXT DEFAULT '',
        preconditions  TEXT DEFAULT '{}',
        postconditions TEXT DEFAULT '{}',
        procedure      TEXT DEFAULT '[]',
        domain         TEXT DEFAULT 'General',
        status         TEXT DEFAULT 'CANDIDATE',
        generation     INTEGER DEFAULT 1,
        parents        TEXT DEFAULT '[]',
        confidence     REAL DEFAULT 0.5,
        timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(session_id, skill_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS candidate_rules (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        scope               TEXT NOT NULL DEFAULT 'global',
        belief              TEXT NOT NULL,
        status              TEXT DEFAULT 'CANDIDATE',
        confidence          REAL DEFAULT 0.5,
        confirmations       INTEGER DEFAULT 0,
        refutations         INTEGER DEFAULT 0,
        evidence_count      INTEGER DEFAULT 1,
        contradiction_count INTEGER DEFAULT 0,
        times_applied       INTEGER DEFAULT 0,
        embedding           BLOB DEFAULT NULL,
        timestamp           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(scope, belief)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS self_competence (
        domain              TEXT PRIMARY KEY,
        competence          REAL DEFAULT 0.5,
        uncertainty         REAL DEFAULT 0.5,
        historical_accuracy REAL DEFAULT 0.5,
        success_count       INTEGER DEFAULT 1,
        failure_count       INTEGER DEFAULT 1,
        total_episodes      INTEGER DEFAULT 0,
        last_updated        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
]


def ensure_db_initialized(db_path: Optional[Path] = None) -> None:
    """Lazy initialize schema and apply lightweight column migrations."""
    target_path = str((db_path or get_db_path()).resolve())
    if target_path in _INITIALIZED_PATHS:
        return
    with _INIT_LOCK:
        if target_path in _INITIALIZED_PATHS:
            return
        conn = _get_conn(Path(target_path))
        try:
            cur = conn.cursor()
            for ddl in _DDL:
                cur.execute(ddl)
            
            # Migrations for tables created in prior versions
            _migrations = [
                ("execution_episodes", "scope", "TEXT NOT NULL DEFAULT 'global'"),
                ("semantic_memories", "confidence", "REAL DEFAULT 0.8"),
                ("semantic_memories", "utility_score", "REAL DEFAULT 0.5"),
                ("semantic_memories", "verification_status", "TEXT DEFAULT 'observed'"),
                ("semantic_memories", "evidence_count", "INTEGER DEFAULT 1"),
                ("semantic_memories", "contradiction_count", "INTEGER DEFAULT 0"),
                ("semantic_memories", "times_retrieved", "INTEGER DEFAULT 0"),
                ("semantic_memories", "times_applied", "INTEGER DEFAULT 0"),
                ("semantic_memories", "success_count", "INTEGER DEFAULT 0"),
                ("semantic_memories", "failure_count", "INTEGER DEFAULT 0"),
                ("semantic_memories", "embedding", "BLOB DEFAULT NULL"),
                ("semantic_memories", "provenance", "TEXT DEFAULT '{}'"),
                ("semantic_memories", "last_accessed", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ("candidate_rules", "scope", "TEXT NOT NULL DEFAULT 'global'"),
                ("candidate_rules", "condition", "TEXT DEFAULT ''"),
                ("candidate_rules", "exception", "TEXT DEFAULT ''"),
                ("candidate_rules", "evidence_count", "INTEGER DEFAULT 1"),
                ("candidate_rules", "contradiction_count", "INTEGER DEFAULT 0"),
                ("candidate_rules", "times_applied", "INTEGER DEFAULT 0"),
                ("candidate_rules", "embedding", "BLOB DEFAULT NULL"),
                ("causal_hypotheses", "scope", "TEXT NOT NULL DEFAULT 'global'"),
                ("causal_hypotheses", "condition", "TEXT DEFAULT ''"),
                ("causal_hypotheses", "exception", "TEXT DEFAULT ''"),
                ("causal_hypotheses", "provenance", "TEXT DEFAULT '{}'"),
                ("causal_hypotheses", "embedding", "BLOB DEFAULT NULL"),
                ("skills", "scope", "TEXT NOT NULL DEFAULT 'global'"),
                ("self_competence", "success_count", "INTEGER DEFAULT 1"),
                ("self_competence", "failure_count", "INTEGER DEFAULT 1"),
                ("self_competence", "total_episodes", "INTEGER DEFAULT 0"),
            ]
            for tbl, col, col_def in _migrations:
                try:
                    cur.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {col_def}")
                except sqlite3.OperationalError:
                    pass  # column already exists

            # Migration: candidate_rules UNIQUE(belief) → UNIQUE(scope, belief)
            # Detect whether the old global-unique index still exists by checking
            # if inserting the same belief in two different scopes would fail.
            try:
                cur.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='candidate_rules'"
                )
                row = cur.fetchone()
                if row and "belief TEXT NOT NULL UNIQUE" in (row["sql"] or ""):
                    # Old schema detected — rebuild with correct composite unique
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS candidate_rules_new (
                            id             INTEGER PRIMARY KEY AUTOINCREMENT,
                            scope          TEXT NOT NULL DEFAULT 'global',
                            belief         TEXT NOT NULL,
                            status         TEXT DEFAULT 'CANDIDATE',
                            confidence     REAL DEFAULT 0.5,
                            confirmations  INTEGER DEFAULT 0,
                            refutations    INTEGER DEFAULT 0,
                            embedding      BLOB DEFAULT NULL,
                            timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(scope, belief)
                        )
                    """)
                    cur.execute("""
                        INSERT OR IGNORE INTO candidate_rules_new
                            (id, scope, belief, status, confidence, confirmations,
                             refutations, embedding, timestamp)
                        SELECT id, scope, belief, status, confidence, confirmations,
                               refutations, embedding, timestamp
                        FROM candidate_rules
                    """)
                    cur.execute("DROP TABLE candidate_rules")
                    cur.execute("ALTER TABLE candidate_rules_new RENAME TO candidate_rules")
                    logger.info("candidate_rules migrated to UNIQUE(scope, belief)")
            except Exception as _mig_err:
                logger.warning(f"candidate_rules migration failed (non-fatal): {_mig_err}")

            conn.commit()
            _INITIALIZED_PATHS.add(target_path)
        finally:
            conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Explicitly initialize tables (optional helper)."""
    ensure_db_initialized(db_path)


# ---------------------------------------------------------------------------
# Episode store
# ---------------------------------------------------------------------------

def save_episode(
    session_id: str,
    goal: str,
    action: str,
    outcome: str,
    *,
    scope: str = "global",
    plan: str = "",
    failure_reason: str = "",
    episode_kind: str = "action",
    emotion_label: str = "",
    prediction_error: float = 0.5,
    novelty: float = 1.0,
    emotional_weight: float = 1.0,
    db_path: Optional[Path] = None,
) -> int:
    with _cursor(commit=True, db_path=db_path) as cur:
        cur.execute(
            """
            INSERT INTO execution_episodes
            (session_id, scope, goal, plan, action, outcome, failure_reason,
             episode_kind, emotion_label, prediction_error, novelty, emotional_weight)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (session_id, scope, goal, plan, action, outcome, failure_reason,
             episode_kind, emotion_label, prediction_error, novelty, emotional_weight),
        )
        return cur.lastrowid


def get_unprocessed_episodes(session_id: str, scope: Optional[str] = None, db_path: Optional[Path] = None) -> List[dict]:
    """Fetch unprocessed episodes for a session.

    Parameters
    ----------
    session_id : str
    scope : str, optional
        When supplied, only episodes whose scope matches are returned.
        Omit (or pass None) only in contexts where cross-scope access is intentional.
    """
    with _cursor(db_path=db_path) as cur:
        if scope is not None:
            cur.execute(
                "SELECT * FROM execution_episodes WHERE session_id=? AND scope=? AND processed_by_sleep=0 ORDER BY id",
                (session_id, scope),
            )
        else:
            cur.execute(
                "SELECT * FROM execution_episodes WHERE session_id=? AND processed_by_sleep=0 ORDER BY id",
                (session_id,),
            )
        return [dict(r) for r in cur.fetchall()]


def mark_episodes_processed(episode_ids: List[int], db_path: Optional[Path] = None) -> None:
    if not episode_ids:
        return
    with _cursor(commit=True, db_path=db_path) as cur:
        cur.executemany(
            "UPDATE execution_episodes SET processed_by_sleep=1 WHERE id=?",
            [(eid,) for eid in episode_ids],
        )


def get_stale_episodes(max_age_days: float = 14.0, db_path: Optional[Path] = None) -> List[dict]:
    with _cursor(db_path=db_path) as cur:
        cur.execute(
            """
            SELECT * FROM execution_episodes
            WHERE processed_by_sleep=1
              AND timestamp < datetime('now', ?)
            ORDER BY id
            """,
            (f"-{max_age_days} days",),
        )
        return [dict(r) for r in cur.fetchall()]


def delete_episodes(episode_ids: List[int], db_path: Optional[Path] = None) -> None:
    if not episode_ids:
        return
    with _cursor(commit=True, db_path=db_path) as cur:
        cur.executemany(
            "DELETE FROM execution_episodes WHERE id=?",
            [(eid,) for eid in episode_ids],
        )


# ---------------------------------------------------------------------------
# Semantic memory store (with stored vector BLOBs & Provenance)
# ---------------------------------------------------------------------------

def _merge_provenance(old_prov: dict, new_prov: dict, new_source: str = "", new_session: str = "") -> dict:
    prov = dict(old_prov or {})
    if new_prov:
        prov.update(new_prov)
    sessions = set((old_prov or {}).get("sessions", []))
    sources = set((old_prov or {}).get("sources", []))
    episode_ids = set((old_prov or {}).get("source_episodes", []))
    input_hashes = set((old_prov or {}).get("input_hashes", []))

    if new_session:
        sessions.add(new_session)
    if new_source:
        sources.add(new_source)

    for s in (new_prov or {}).get("sessions", []):
        sessions.add(s)
    for src in (new_prov or {}).get("sources", []):
        sources.add(src)
    for ep in (new_prov or {}).get("source_episodes", []):
        episode_ids.add(ep)
    for h in (new_prov or {}).get("input_hashes", []):
        input_hashes.add(h)
    if new_prov and "input_fingerprint" in new_prov:
        input_hashes.add(new_prov["input_fingerprint"])

    prov["sessions"] = sorted(list(sessions))
    prov["sources"] = sorted(list(sources))
    prov["source_episodes"] = sorted(list(episode_ids))
    prov["input_hashes"] = sorted(list(input_hashes))
    
    # Calculate independent sources (diversity): distinct sessions + distinct external sources
    prov["independent_sources_count"] = max(1, len(sessions) + len(sources) - (1 if sessions and sources else 0))
    return prov


def save_semantic_memory(
    fact: str,
    value: str,
    *,
    memory_type: str = "procedural",
    importance: float = 0.7,
    confidence: float = 0.8,
    utility_score: float = 0.5,
    verification_status: str = "observed",
    provenance: Optional[dict] = None,
    source: str = "",
    scope: str = "global",
    embedding: Optional[np.ndarray] = None,
    db_path: Optional[Path] = None,
) -> None:
    if embedding is None:
        try:
            embedding = embed(f"{fact} {value}")
        except Exception:
            embedding = None

    vec_blob = serialize_vector(embedding) if embedding is not None else None
    incoming_prov = provenance or {}

    with _cursor(commit=True, db_path=db_path) as cur:
        cur.execute(
            """
            SELECT id, provenance, evidence_count, verification_status, confidence, utility_score
            FROM semantic_memories
            WHERE scope=? AND memory_type=? AND fact=?
            """,
            (scope, memory_type, fact),
        )
        row = cur.fetchone()

        if row:
            old_prov = {}
            if row["provenance"]:
                try:
                    old_prov = json.loads(row["provenance"])
                except Exception:
                    old_prov = {}

            # Idempotency check: if input_fingerprint was already consolidated, don't duplicate evidence
            incoming_fp = incoming_prov.get("input_fingerprint")
            already_processed = incoming_fp and incoming_fp in old_prov.get("input_hashes", [])

            merged_prov = _merge_provenance(old_prov, incoming_prov, new_source=source)
            indep_sources = merged_prov.get("independent_sources_count", 1)
            
            old_evidence = int(row["evidence_count"] or 1)
            new_evidence = old_evidence if already_processed else old_evidence + 1
            
            # Epistemic lifecycle transition gating:
            # - Must have independent sources (not just 1 bad source repeating) or multiple verified runs
            curr_status = row["verification_status"] or "observed"
            if verification_status == "verified" or curr_status == "verified":
                new_status = "verified"
                new_conf = max(0.90, float(row["confidence"] or confidence))
            elif indep_sources >= 2 or new_evidence >= 2:
                new_status = "repeated"
                new_conf = min(0.95, float(row["confidence"] or 0.75) + 0.05)
            else:
                new_status = curr_status
                new_conf = float(row["confidence"] or confidence)

            new_u = max(float(row["utility_score"] or 0.5), utility_score)

            cur.execute(
                """
                UPDATE semantic_memories
                SET value = ?,
                    importance = MAX(importance, ?),
                    confidence = ?,
                    utility_score = ?,
                    evidence_count = ?,
                    verification_status = ?,
                    embedding = COALESCE(?, embedding),
                    provenance = ?,
                    source = COALESCE(NULLIF(?, ''), source),
                    access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP,
                    timestamp = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (value, importance, new_conf, new_u, new_evidence, new_status, vec_blob, json.dumps(merged_prov), source, row["id"]),
            )
        else:
            init_prov = _merge_provenance({}, incoming_prov, new_source=source)
            cur.execute(
                """
                INSERT INTO semantic_memories 
                (scope, memory_type, fact, value, importance, confidence, utility_score, verification_status,
                 evidence_count, embedding, provenance, source)
                VALUES (?,?,?,?,?,?,?,?,1,?,?,?)
                """,
                (scope, memory_type, fact, value, importance, confidence, utility_score, verification_status, vec_blob, json.dumps(init_prov), source),
            )


def recall_memories(
    query_text: str,
    *,
    top_k: int = 5,
    scopes: Optional[Sequence[str]] = ("global",),
    memory_types: Optional[Sequence[str]] = None,
    min_score: float = 0.05,
    min_similarity: float = 0.10,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """
    Retrieve active semantic memories using pre-computed vector BLOBs and active epistemic ranking:
      final_score = semantic_similarity * importance * confidence * utility_score * trust_multiplier
    Excludes quarantined and expired memories. Tracks access and retrieval counts.
    """
    try:
        q_vec = embed(query_text)
    except Exception:
        return _recall_keyword(query_text, top_k=top_k, scopes=scopes, memory_types=memory_types, db_path=db_path)

    where_clauses = ["verification_status NOT IN ('quarantined', 'expired')"]
    params: list = []

    if scopes:
        placeholders = ",".join("?" * len(scopes))
        where_clauses.append(f"scope IN ({placeholders})")
        params.extend(scopes)

    if memory_types:
        placeholders = ",".join("?" * len(memory_types))
        where_clauses.append(f"memory_type IN ({placeholders})")
        params.extend(memory_types)

    where_sql = f"WHERE {' AND '.join(where_clauses)}"

    with _cursor(db_path=db_path) as cur:
        cur.execute(f"SELECT * FROM semantic_memories {where_sql}", params)
        rows = [dict(r) for r in cur.fetchall()]

    # Epistemic trust multipliers (mathematically active ranking)
    trust_multipliers = {
        "verified": 1.00,
        "repeated": 0.75,
        "observed": 0.50,
        "raw": 0.20,
    }

    scored = []
    recalled_ids = []
    for row in rows:
        vec = deserialize_vector(row.get("embedding"))
        if vec is None:
            vec = embed(f"{row['fact']} {row['value']}")
        sim = cosine_similarity(q_vec, vec)

        status = (row.get("verification_status") or "observed").lower()
        trust = trust_multipliers.get(status, 0.50)
        utility = max(0.10, min(1.0, float(row.get("utility_score") or 0.50)))
        importance = float(row.get("importance") or 0.70)
        conf = float(row.get("confidence") or 0.80)

        # Active ranking score
        score = sim * importance * conf * utility * trust

        if sim >= min_similarity and score >= min_score:
            row_copy = dict(row)
            row_copy.pop("embedding", None)
            row_copy["relevance_score"] = round(score, 3)
            row_copy["epistemic_trust"] = trust
            scored.append((score, row_copy))
            recalled_ids.append(row["id"])

    # Update access counts, retrieval counts, and last_accessed timestamp
    if recalled_ids:
        with _cursor(commit=True, db_path=db_path) as cur:
            placeholders = ",".join("?" * len(recalled_ids))
            cur.execute(
                f"""
                UPDATE semantic_memories 
                SET access_count=access_count+1, 
                    times_retrieved=times_retrieved+1, 
                    last_accessed=CURRENT_TIMESTAMP 
                WHERE id IN ({placeholders})
                """,
                tuple(recalled_ids),
            )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


def record_memory_utility_feedback(
    memory_ids: Sequence[int],
    outcome: str,
    was_applied: bool = True,
    db_path: Optional[Path] = None,
) -> dict:
    """
    Record whether retrieved memories were applied and what happened afterward.
    
    If applied and outcome was successful:
      - Increases utility_score (+0.10)
      - Increments success_count
      - Promotes observed -> repeated / verified on consistent success
    
    If applied and outcome failed:
      - Decreases utility_score (-0.15)
      - Increments failure_count
      - If utility drops below 0.20 and failures exceed successes: quarantines memory
    """
    if not memory_ids:
        return {"updated": 0, "quarantined": 0}

    is_success = "success" in outcome.lower()
    quarantined = 0
    updated = 0

    with _cursor(commit=True, db_path=db_path) as cur:
        for mid in memory_ids:
            cur.execute("SELECT id, utility_score, success_count, failure_count, verification_status FROM semantic_memories WHERE id=?", (mid,))
            row = cur.fetchone()
            if not row:
                continue

            u = float(row["utility_score"] or 0.5)
            s_cnt = int(row["success_count"] or 0)
            f_cnt = int(row["failure_count"] or 0)
            status = row["verification_status"] or "observed"

            if was_applied:
                if is_success:
                    u = min(1.0, u + 0.10)
                    s_cnt += 1
                    if status == "observed" and s_cnt >= 2:
                        status = "repeated"
                    elif s_cnt >= 5:
                        status = "verified"
                else:
                    u = max(0.0, u - 0.15)
                    f_cnt += 1
                    if f_cnt >= 2 and f_cnt > s_cnt and u <= 0.25:
                        status = "quarantined"
                        quarantined += 1

                cur.execute(
                    """
                    UPDATE semantic_memories
                    SET times_applied = times_applied + 1,
                        utility_score = ?,
                        success_count = ?,
                        failure_count = ?,
                        verification_status = ?,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (u, s_cnt, f_cnt, status, mid),
                )
                updated += 1

    return {"updated": updated, "quarantined": quarantined}


def decay_stale_memories(
    max_age_days: float = 30.0,
    min_utility: float = 0.2,
    db_path: Optional[Path] = None,
) -> dict:
    """
    Apply utility decay to unaccessed, unverified memories.
    Memories that fall below min_utility are pruned.
    """
    with _cursor(commit=True, db_path=db_path) as cur:
        # Decay utility by 10% for memories unaccessed past max_age_days
        cur.execute(
            """
            UPDATE semantic_memories
            SET utility_score = utility_score * 0.9,
                importance = importance * 0.95
            WHERE verification_status != 'verified'
              AND last_accessed <= datetime('now', ?)
            """,
            (f"-{max_age_days} days",),
        )
        # Delete memories whose utility fell below threshold
        cur.execute(
            """
            DELETE FROM semantic_memories
            WHERE verification_status != 'verified'
              AND utility_score < ?
              AND last_accessed <= datetime('now', ?)
            """,
            (min_utility, f"-{max_age_days} days"),
        )
        pruned_count = cur.rowcount

    return {"pruned_stale_memories": pruned_count}


def _recall_keyword(query_text: str, *, top_k: int, scopes, memory_types, db_path) -> List[dict]:
    kw = query_text.lower()
    where_clauses = []
    params: list = []

    if scopes:
        placeholders = ",".join("?" * len(scopes))
        where_clauses.append(f"scope IN ({placeholders})")
        params.extend(scopes)

    if memory_types:
        placeholders = ",".join("?" * len(memory_types))
        where_clauses.append(f"memory_type IN ({placeholders})")
        params.extend(memory_types)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with _cursor(db_path=db_path) as cur:
        cur.execute(f"SELECT * FROM semantic_memories {where_sql}", params)
        rows = [dict(r) for r in cur.fetchall()]

    matches = []
    for r in rows:
        if kw in (r["fact"] + " " + r["value"]).lower():
            r_copy = dict(r)
            r_copy.pop("embedding", None)
            matches.append(r_copy)
    return matches[:top_k]


# ---------------------------------------------------------------------------
# Selective rule / belief store
# ---------------------------------------------------------------------------

def add_candidate_rule(
    belief: str,
    confidence: float = 0.5,
    scope: str = "global",
    condition: str = "",
    exception: str = "",
    db_path: Optional[Path] = None,
) -> None:
    vec = embed(f"{belief} {condition} {exception}")
    vec_blob = serialize_vector(vec)

    with _cursor(commit=True, db_path=db_path) as cur:
        cur.execute(
            """
            INSERT INTO candidate_rules (scope, belief, condition, exception, confidence, embedding)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(scope, belief) DO UPDATE SET
                condition=COALESCE(NULLIF(excluded.condition, ''), candidate_rules.condition),
                exception=COALESCE(NULLIF(excluded.exception, ''), candidate_rules.exception),
                confirmations=confirmations+1,
                evidence_count=evidence_count+1,
                confidence=MIN(0.99, confidence+0.05),
                embedding=COALESCE(excluded.embedding, candidate_rules.embedding),
                timestamp=CURRENT_TIMESTAMP
            """,
            (scope, belief, condition, exception, confidence, vec_blob),
        )


def specialize_rule_with_exception(
    rule_id: int,
    condition: str = "",
    exception: str = "",
    counter_evidence: str = "",
    db_path: Optional[Path] = None,
) -> dict:
    """
    Specialize a rule with explicit context and exceptions instead of blanket deletion/quarantine.
    Allows representational nuance: 'Rule X holds in context Y EXCEPT in context Z'.
    """
    with _cursor(commit=True, db_path=db_path) as cur:
        cur.execute("SELECT id, belief, condition, exception FROM candidate_rules WHERE id=?", (rule_id,))
        row = cur.fetchone()
        if not row:
            return {"status": "not_found", "rule_id": rule_id}

        existing_cond = row["condition"] or ""
        existing_exc = row["exception"] or ""

        new_cond = f"{existing_cond}; {condition}".strip("; ") if condition else existing_cond
        new_exc = f"{existing_exc}; {exception}".strip("; ") if exception else existing_exc

        cur.execute(
            """
            UPDATE candidate_rules
            SET condition = ?,
                exception = ?,
                contradiction_count = contradiction_count + 1,
                status = 'SPECIALIZED',
                timestamp = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (new_cond, new_exc, rule_id),
        )
        return {"status": "specialized", "rule_id": rule_id, "condition": new_cond, "exception": new_exc}


def recall_rules(
    query_text: str,
    *,
    top_k: int = 3,
    scopes: Optional[Sequence[str]] = ("global",),
    min_confidence: float = 0.4,
    db_path: Optional[Path] = None,
) -> List[str]:
    """
    Selectively retrieve only the rules most semantically relevant to query_text.
    Includes contextual conditions and exceptions when present.
    """
    try:
        q_vec = embed(query_text)
    except Exception:
        return get_active_rules(scopes=scopes, db_path=db_path)[:top_k]

    where_clauses = ["status IN ('CANDIDATE', 'VERIFIED', 'SPECIALIZED')", "confidence >= ?"]
    params: list = [min_confidence]

    if scopes:
        placeholders = ",".join("?" * len(scopes))
        where_clauses.append(f"scope IN ({placeholders})")
        params.extend(scopes)

    where_sql = f"WHERE {' AND '.join(where_clauses)}"

    with _cursor(db_path=db_path) as cur:
        cur.execute(f"SELECT id, belief, condition, exception, confidence, embedding FROM candidate_rules {where_sql}", params)
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        return []

    scored = []
    for r in rows:
        vec = deserialize_vector(r.get("embedding"))
        if vec is None:
            vec = embed(f"{r['belief']} {r.get('condition','')} {r.get('exception','')}")
        sim = cosine_similarity(q_vec, vec)
        # Score combining relevance and rule confidence
        score = sim * float(r["confidence"])
        if sim > 0.25:  # Relevance threshold
            text = r["belief"]
            if r.get("condition"):
                text += f" [Applies: {r['condition']}]"
            if r.get("exception"):
                text += f" [Except: {r['exception']}]"
            scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [belief for _, belief in scored[:top_k]]


def get_active_rules(scopes: Optional[Sequence[str]] = ("global",), db_path: Optional[Path] = None) -> List[str]:
    where = "WHERE status IN ('CANDIDATE','VERIFIED','SPECIALIZED')"
    params: list = []
    if scopes:
        placeholders = ",".join("?" * len(scopes))
        where += f" AND scope IN ({placeholders})"
        params.extend(scopes)

    with _cursor(db_path=db_path) as cur:
        cur.execute(
            f"SELECT belief, condition, exception FROM candidate_rules {where} ORDER BY confidence DESC LIMIT 10",
            params,
        )
        results = []
        for r in cur.fetchall():
            text = r["belief"]
            if r["condition"]:
                text += f" [Applies: {r['condition']}]"
            if r["exception"]:
                text += f" [Except: {r['exception']}]"
            results.append(text)
        return results


# ---------------------------------------------------------------------------
# Causal hypotheses & Competence
# ---------------------------------------------------------------------------

def save_causal_hypothesis(
    session_id: str,
    action: str,
    hypothesis: str,
    effect: str,
    confidence: float = 0.35,
    scope: str = "global",
    condition: str = "",
    exception: str = "",
    provenance: Optional[dict] = None,
    db_path: Optional[Path] = None,
) -> None:
    """
    Save or update a causal hypothesis with provenance diversity and evidence accumulation.
    Initial observation starts with cautious confidence (0.35).
    Subsequent observations increment support_count and scale confidence with independent sources.
    """
    vec = embed(f"{action} {hypothesis} {effect} {condition} {exception}")
    vec_blob = serialize_vector(vec)
    incoming_prov = provenance or {}

    with _cursor(commit=True, db_path=db_path) as cur:
        cur.execute(
            """
            SELECT id, support_count, confidence, provenance FROM causal_hypotheses
            WHERE scope=? AND action=? AND hypothesis=?
            """,
            (scope, action[:200], hypothesis),
        )
        row = cur.fetchone()
        if row:
            old_prov = {}
            if row["provenance"]:
                try:
                    old_prov = json.loads(row["provenance"])
                except Exception:
                    old_prov = {}

            merged_prov = _merge_provenance(old_prov, incoming_prov, new_session=session_id)
            indep_sources = merged_prov.get("independent_sources_count", 1)

            new_support = int(row["support_count"] or 1) + 1
            new_conf = min(0.95, 0.35 + 0.15 * (indep_sources - 1))

            cur.execute(
                """
                UPDATE causal_hypotheses
                SET support_count = ?,
                    confidence = ?,
                    condition = COALESCE(NULLIF(?, ''), condition),
                    exception = COALESCE(NULLIF(?, ''), exception),
                    provenance = ?,
                    embedding = COALESCE(?, embedding),
                    timestamp = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (new_support, new_conf, condition, exception, json.dumps(merged_prov), vec_blob, row["id"]),
            )
        else:
            init_prov = _merge_provenance({}, incoming_prov, new_session=session_id)
            cur.execute(
                """
                INSERT INTO causal_hypotheses
                (session_id, scope, action, hypothesis, effect, condition, exception, confidence, support_count, provenance, embedding)
                VALUES (?,?,?,?,?,?,?,?,1,?,?)
                """,
                (session_id, scope, action[:200], hypothesis, effect, condition, exception, confidence, json.dumps(init_prov), vec_blob),
            )


def recall_causal_hypotheses(
    query_text: str,
    *,
    top_k: int = 3,
    scopes: Optional[Sequence[str]] = ("global",),
    min_confidence: float = 0.30,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """
    Selectively retrieve causal hypotheses relevant to a given task or action query.
    Closes the loop between error analysis and future planning.
    """
    try:
        q_vec = embed(query_text)
    except Exception:
        q_vec = None

    where_clauses = ["confidence >= ?"]
    params: list = [min_confidence]

    if scopes:
        placeholders = ",".join("?" * len(scopes))
        where_clauses.append(f"scope IN ({placeholders})")
        params.extend(scopes)

    where_sql = f"WHERE {' AND '.join(where_clauses)}"

    with _cursor(db_path=db_path) as cur:
        cur.execute(
            f"SELECT id, action, hypothesis, effect, condition, exception, confidence, support_count, contradiction_count, embedding FROM causal_hypotheses {where_sql}",
            params,
        )
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        return []

    scored = []
    for r in rows:
        sim = 0.0
        if q_vec is not None:
            vec = deserialize_vector(r.get("embedding"))
            if vec is None:
                vec = embed(f"{r['action']} {r['hypothesis']} {r['effect']}")
            sim = cosine_similarity(q_vec, vec)
        else:
            # Fallback keyword match
            kw = query_text.lower()
            text = f"{r['action']} {r['hypothesis']} {r['effect']}".lower()
            if any(word in text for word in kw.split() if len(word) > 3):
                sim = 0.5

        if sim >= 0.15:
            score = sim * float(r["confidence"])
            r_copy = dict(r)
            r_copy.pop("embedding", None)
            r_copy["relevance_score"] = round(score, 3)
            scored.append((score, r_copy))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def update_competence(
    domain: str,
    competence: float,
    historical_accuracy: Optional[float] = None,
    success: Optional[bool] = None,
    db_path: Optional[Path] = None,
) -> None:
    """
    Persist domain competence computed by SelfModel.
    Stores the exact smoothed competence without double-smoothing in SQL.
    Computes Bayesian Beta uncertainty based on observation counts.
    """
    acc = historical_accuracy if historical_accuracy is not None else competence
    s_inc = 1 if success is True else 0
    f_inc = 1 if success is False else 0

    with _cursor(commit=True, db_path=db_path) as cur:
        cur.execute("SELECT success_count, failure_count, total_episodes FROM self_competence WHERE domain=?", (domain,))
        row = cur.fetchone()
        if row:
            s_cnt = int(row["success_count"] or 1) + s_inc
            f_cnt = int(row["failure_count"] or 1) + f_inc
            tot = int(row["total_episodes"] or 0) + 1
        else:
            s_cnt = 1 + s_inc
            f_cnt = 1 + f_inc
            tot = 1

        # Bayesian Beta distribution standard deviation as uncertainty metric:
        # Var(Beta) = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        alpha = float(s_cnt)
        beta = float(f_cnt)
        total_obs = alpha + beta
        var = (alpha * beta) / ((total_obs ** 2) * (total_obs + 1.0))
        std = math.sqrt(var)
        # Normalized uncertainty: drops as total observations increase
        uncertainty = round(min(1.0, std * 4.0), 3)

        cur.execute(
            """
            INSERT INTO self_competence 
                (domain, competence, uncertainty, historical_accuracy, success_count, failure_count, total_episodes, last_updated)
            VALUES (?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(domain) DO UPDATE SET
                competence=excluded.competence,
                uncertainty=excluded.uncertainty,
                historical_accuracy=excluded.historical_accuracy,
                success_count=excluded.success_count,
                failure_count=excluded.failure_count,
                total_episodes=excluded.total_episodes,
                last_updated=CURRENT_TIMESTAMP
            """,
            (domain, competence, uncertainty, acc, s_cnt, f_cnt, tot),
        )


def get_domain_competence(domain: str, db_path: Optional[Path] = None) -> dict:
    """Retrieve tracked competence metrics for a specific domain."""
    with _cursor(db_path=db_path) as cur:
        cur.execute(
            "SELECT domain, competence, uncertainty, historical_accuracy, success_count, failure_count, total_episodes, last_updated FROM self_competence WHERE domain=?",
            (domain,),
        )
        row = cur.fetchone()
        if row:
            return dict(row)
        return {
            "domain": domain,
            "competence": 0.5,
            "uncertainty": 0.5,
            "historical_accuracy": 0.5,
            "success_count": 1,
            "failure_count": 1,
            "total_episodes": 0,
            "last_updated": None,
        }


def get_all_competencies(db_path: Optional[Path] = None) -> Dict[str, float]:
    """Retrieve all tracked domain competence scores."""
    with _cursor(db_path=db_path) as cur:
        cur.execute("SELECT domain, competence FROM self_competence")
        return {r["domain"]: float(r["competence"]) for r in cur.fetchall()}

