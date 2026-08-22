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
        confidence          REAL DEFAULT 0.5,
        support_count       INTEGER DEFAULT 1,
        contradiction_count INTEGER DEFAULT 0,
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
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        scope          TEXT NOT NULL DEFAULT 'global',
        belief         TEXT NOT NULL UNIQUE,
        status         TEXT DEFAULT 'CANDIDATE',
        confidence     REAL DEFAULT 0.5,
        confirmations  INTEGER DEFAULT 0,
        refutations    INTEGER DEFAULT 0,
        embedding      BLOB DEFAULT NULL,
        timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS self_competence (
        domain              TEXT PRIMARY KEY,
        competence          REAL DEFAULT 0.5,
        uncertainty         REAL DEFAULT 0.5,
        historical_accuracy REAL DEFAULT 0.5,
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
                ("semantic_memories", "embedding", "BLOB DEFAULT NULL"),
                ("semantic_memories", "provenance", "TEXT DEFAULT '{}'"),
                ("semantic_memories", "last_accessed", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ("candidate_rules", "scope", "TEXT NOT NULL DEFAULT 'global'"),
                ("candidate_rules", "embedding", "BLOB DEFAULT NULL"),
                ("causal_hypotheses", "scope", "TEXT NOT NULL DEFAULT 'global'"),
                ("skills", "scope", "TEXT NOT NULL DEFAULT 'global'"),
            ]
            for tbl, col, col_def in _migrations:
                try:
                    cur.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {col_def}")
                except sqlite3.OperationalError:
                    pass  # column already exists

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


def get_unprocessed_episodes(session_id: str, db_path: Optional[Path] = None) -> List[dict]:
    with _cursor(db_path=db_path) as cur:
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
    prov_str = json.dumps(provenance or {})

    with _cursor(commit=True, db_path=db_path) as cur:
        cur.execute(
            """
            INSERT INTO semantic_memories 
            (scope, memory_type, fact, value, importance, confidence, utility_score, verification_status, embedding, provenance, source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(scope, memory_type, fact) DO UPDATE SET
                value=excluded.value,
                importance=MAX(importance, excluded.importance),
                confidence=MAX(confidence, excluded.confidence),
                utility_score=MAX(utility_score, excluded.utility_score),
                verification_status=excluded.verification_status,
                embedding=COALESCE(excluded.embedding, embedding),
                provenance=excluded.provenance,
                access_count=access_count+1,
                last_accessed=CURRENT_TIMESTAMP,
                timestamp=CURRENT_TIMESTAMP
            """,
            (scope, memory_type, fact, value, importance, confidence, utility_score, verification_status, vec_blob, prov_str, source),
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
    Retrieve semantic memories using pre-computed vector BLOBs.
    Fast O(1) query embedding + stored dot product. Updates access counts.
    """
    try:
        q_vec = embed(query_text)
    except Exception:
        return _recall_keyword(query_text, top_k=top_k, scopes=scopes, memory_types=memory_types, db_path=db_path)

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

    scored = []
    recalled_ids = []
    for row in rows:
        vec = deserialize_vector(row.get("embedding"))
        if vec is None:
            vec = embed(f"{row['fact']} {row['value']}")
        sim = cosine_similarity(q_vec, vec)
        score = sim * float(row.get("importance", 0.5)) * float(row.get("confidence", 0.8))
        if sim >= min_similarity and score >= min_score:
            row_copy = dict(row)
            row_copy.pop("embedding", None)
            row_copy["relevance_score"] = round(score, 3)
            scored.append((score, row_copy))
            recalled_ids.append(row["id"])

    # Update access counts and last_accessed timestamp
    if recalled_ids:
        with _cursor(commit=True, db_path=db_path) as cur:
            placeholders = ",".join("?" * len(recalled_ids))
            cur.execute(
                f"UPDATE semantic_memories SET access_count=access_count+1, last_accessed=CURRENT_TIMESTAMP WHERE id IN ({placeholders})",
                tuple(recalled_ids),
            )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


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
    db_path: Optional[Path] = None,
) -> None:
    vec = embed(belief)
    vec_blob = serialize_vector(vec)

    with _cursor(commit=True, db_path=db_path) as cur:
        cur.execute(
            """
            INSERT INTO candidate_rules (scope, belief, confidence, embedding)
            VALUES (?,?,?,?)
            ON CONFLICT(belief) DO UPDATE SET
                confirmations=confirmations+1,
                confidence=MIN(0.99, confidence+0.05),
                embedding=COALESCE(excluded.embedding, embedding),
                timestamp=CURRENT_TIMESTAMP
            """,
            (scope, belief, confidence, vec_blob),
        )


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
    Avoids context pollution by not injecting all rules blindly.
    """
    try:
        q_vec = embed(query_text)
    except Exception:
        return get_active_rules(scopes=scopes, db_path=db_path)[:top_k]

    where_clauses = ["status IN ('CANDIDATE', 'VERIFIED')", "confidence >= ?"]
    params: list = [min_confidence]

    if scopes:
        placeholders = ",".join("?" * len(scopes))
        where_clauses.append(f"scope IN ({placeholders})")
        params.extend(scopes)

    where_sql = f"WHERE {' AND '.join(where_clauses)}"

    with _cursor(db_path=db_path) as cur:
        cur.execute(f"SELECT belief, confidence, embedding FROM candidate_rules {where_sql}", params)
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        return []

    scored = []
    for r in rows:
        vec = deserialize_vector(r.get("embedding"))
        if vec is None:
            vec = embed(r["belief"])
        sim = cosine_similarity(q_vec, vec)
        # Score combining relevance and rule confidence
        score = sim * float(r["confidence"])
        if sim > 0.25:  # Relevance threshold
            scored.append((score, r["belief"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [belief for _, belief in scored[:top_k]]


def get_active_rules(scopes: Optional[Sequence[str]] = ("global",), db_path: Optional[Path] = None) -> List[str]:
    where = "WHERE status IN ('CANDIDATE','VERIFIED')"
    params: list = []
    if scopes:
        placeholders = ",".join("?" * len(scopes))
        where += f" AND scope IN ({placeholders})"
        params.extend(scopes)

    with _cursor(db_path=db_path) as cur:
        cur.execute(
            f"SELECT belief FROM candidate_rules {where} ORDER BY confidence DESC LIMIT 10",
            params,
        )
        return [r["belief"] for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Causal hypotheses & Competence
# ---------------------------------------------------------------------------

def save_causal_hypothesis(
    session_id: str,
    action: str,
    hypothesis: str,
    effect: str,
    confidence: float = 0.8,
    scope: str = "global",
    db_path: Optional[Path] = None,
) -> None:
    with _cursor(commit=True, db_path=db_path) as cur:
        cur.execute(
            """
            INSERT INTO causal_hypotheses
            (session_id, scope, action, hypothesis, effect, confidence)
            VALUES (?,?,?,?,?,?)
            """,
            (session_id, scope, action[:200], hypothesis, effect, confidence),
        )


def update_competence(domain: str, accuracy: float, db_path: Optional[Path] = None) -> None:
    with _cursor(commit=True, db_path=db_path) as cur:
        cur.execute(
            """
            INSERT INTO self_competence (domain, competence, historical_accuracy)
            VALUES (?,?,?)
            ON CONFLICT(domain) DO UPDATE SET
                competence=(competence*0.8 + ?*0.2),
                historical_accuracy=(historical_accuracy*0.8 + ?*0.2),
                last_updated=CURRENT_TIMESTAMP
            """,
            (domain, accuracy, accuracy, accuracy, accuracy),
        )
