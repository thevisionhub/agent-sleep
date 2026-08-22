"""
VectorStore Abstraction Layer for agent-sleep.

Provides an extensible interface (BaseVectorStore) enabling pluggable backends:
  - SQLiteVectorStore (default, zero external dependency, pre-computed BLOBs)
  - Extensible to sqlite-vec, FAISS, and pgvector.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from agent_sleep.storage.embeddings import (
    embed,
    cosine_similarity,
    serialize_vector,
    deserialize_vector,
    get_backend_info,
)


class BaseVectorStore(abc.ABC):
    """Abstract interface for agent-sleep vector index storage and retrieval."""

    @abc.abstractmethod
    def upsert(
        self,
        item_id: int,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        scope: str = "global",
    ) -> None:
        """Insert or update a vector with associated metadata."""
        pass

    @abc.abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        scopes: Optional[Sequence[str]] = ("global",),
        min_similarity: float = 0.10,
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search for nearest neighbor vectors.
        Returns a list of (item_id, similarity_score, metadata).
        """
        pass

    @abc.abstractmethod
    def delete(self, item_ids: Sequence[int]) -> None:
        """Delete vectors by their IDs."""
        pass

    @abc.abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return backend driver metadata."""
        pass


class SQLiteVectorStore(BaseVectorStore):
    """
    Standard SQLite vector store implementation using pre-computed BLOBs.
    Zero-external dependency, ACID-compliant, isolated by scope.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path

    def upsert(
        self,
        item_id: int,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        scope: str = "global",
    ) -> None:
        from agent_sleep.storage.db import _cursor
        blob = serialize_vector(vector)
        with _cursor(commit=True, db_path=self.db_path) as cur:
            cur.execute(
                "UPDATE semantic_memories SET embedding=? WHERE id=?",
                (blob, item_id),
            )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        scopes: Optional[Sequence[str]] = ("global",),
        min_similarity: float = 0.10,
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        from agent_sleep.storage.db import _cursor
        where_clauses = ["verification_status NOT IN ('quarantined', 'expired')"]
        params: list = []
        if scopes:
            placeholders = ",".join("?" * len(scopes))
            where_clauses.append(f"scope IN ({placeholders})")
            params.extend(scopes)

        where_sql = f"WHERE {' AND '.join(where_clauses)}"
        with _cursor(db_path=self.db_path) as cur:
            cur.execute(
                f"SELECT id, scope, memory_type, fact, value, importance, confidence, utility_score, verification_status, embedding FROM semantic_memories {where_sql}",
                params,
            )
            rows = [dict(r) for r in cur.fetchall()]

        scored = []
        for r in rows:
            vec = deserialize_vector(r.get("embedding"))
            if vec is None:
                continue
            sim = cosine_similarity(query_vector, vec)
            if sim >= min_similarity:
                meta = {k: v for k, v in r.items() if k != "embedding"}
                scored.append((r["id"], sim, meta))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete(self, item_ids: Sequence[int]) -> None:
        from agent_sleep.storage.db import _cursor
        if not item_ids:
            return
        with _cursor(commit=True, db_path=self.db_path) as cur:
            placeholders = ",".join("?" * len(item_ids))
            cur.execute(f"DELETE FROM semantic_memories WHERE id IN ({placeholders})", tuple(item_ids))

    def get_info(self) -> Dict[str, Any]:
        info = get_backend_info()
        info["store_type"] = "SQLiteBLOB"
        return info


def get_default_vector_store(db_path: Optional[Path] = None) -> BaseVectorStore:
    """Factory for default vector store."""
    return SQLiteVectorStore(db_path=db_path)
