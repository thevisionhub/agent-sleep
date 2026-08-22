"""
Embedding utility for agent-sleep.

Uses sentence-transformers (all-MiniLM-L6-v2) when available.
Falls back to a deterministic hashed bag-of-words representation when
sentence-transformers is not installed, so the library remains lightweight
and usable with zero external ML dependencies.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_BACKEND: str = "none"


def _load_model() -> None:
    global _model, _BACKEND
    if _BACKEND != "none":
        return
    try:
        from sentence_transformers import SentenceTransformer
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu", local_files_only=True)
        except Exception:
            _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        _BACKEND = "sentence_transformers"
        logger.debug("agent-sleep: initialized all-MiniLM-L6-v2 embedding model (CPU)")
    except ImportError:
        _BACKEND = "hashed_bow"
        logger.info("agent-sleep: sentence-transformers not installed; using deterministic hashed bag-of-words fallback")


def _hashed_bow_embed(text: str, dim: int = 384) -> np.ndarray:
    """
    Deterministic hashed bag-of-words vector.
    Computes token frequency hashed over fixed dimensions with length sub-linear scaling.
    """
    tokens = text.lower().replace("_", " ").replace(".", " ").replace("(", " ").replace(")", " ").split()
    vec = np.zeros(dim, dtype=np.float32)
    if not tokens:
        return vec
    for tok in tokens:
        if len(tok) < 2:
            continue
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0 + math.log(1 + len(tok))
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


def embed(text: str) -> np.ndarray:
    """Embed a string into a normalized float32 numpy vector."""
    _load_model()
    if _BACKEND == "sentence_transformers":
        return _model.encode(text, normalize_embeddings=True).astype(np.float32)
    return _hashed_bow_embed(text)


def embed_batch(texts: List[str]) -> List[np.ndarray]:
    """Batch embed strings into normalized float32 numpy vectors."""
    _load_model()
    if _BACKEND == "sentence_transformers":
        vecs = _model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        return [v.astype(np.float32) for v in vecs]
    return [_hashed_bow_embed(t) for t in texts]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two float vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def serialize_vector(vec: np.ndarray) -> bytes:
    """Serialize a numpy float32 vector to bytes for SQLite BLOB storage."""
    return vec.astype(np.float32).tobytes()


def deserialize_vector(data: Optional[bytes], dim: int = 384) -> Optional[np.ndarray]:
    """Deserialize SQLite BLOB back to a numpy float32 vector."""
    if not data:
        return None
    try:
        return np.frombuffer(data, dtype=np.float32)
    except Exception:
        return None
