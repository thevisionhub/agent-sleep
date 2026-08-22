"""
ConceptHierarchy: online, no-backprop abstraction clustering.

Two-level in-memory cluster hierarchy maintained by streaming agent experiences:
- L1 (concrete): "Bash error: directory not found" clusters
- L2 (abstract): "File system errors" mega-clusters

No training, no labels, no hyperparameter search. Just Welford online means
and cosine similarity thresholds, directly on the embedding of whatever the
agent just did.

This closes the loop that most agent memory systems miss: not just "remember X
happened" but "in situations LIKE THIS, how has the agent historically done?"
That is what ConceptHierarchy.query() returns: the mean outcome of all past
situations whose embedding was similar to this one.

Persistence: saves/loads to a .pt file so clusters survive restarts.

Note: This module is intentionally not wired into the consolidation pipeline
in v0.1.1-alpha. It will be integrated in a future release to provide
experience-based competence prediction.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_SAVE_PATH = Path(__file__).parent / "data" / "concept_hierarchy.npz"

# Similarity thresholds for cluster merge decisions:
# L1: tight (0.85) — only very similar situations merge
# L2: looser (0.70) — broader concept grouping
THRESHOLD_L1 = 0.85
THRESHOLD_L2 = 0.70


class ConceptHierarchy:
    """
    Two-level online concept clustering.

    Parameters
    ----------
    latent_dim : int
        Dimension of the embedding vectors. Defaults to 384 (all-MiniLM-L6-v2).
    threshold_l1 : float
        Cosine similarity threshold to merge into an existing L1 cluster.
    threshold_l2 : float
        Cosine similarity threshold to merge into an existing L2 cluster.
    save_path : str, optional
        Where to persist the cluster state. Defaults to data/concept_hierarchy.npz.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        threshold_l1: float = THRESHOLD_L1,
        threshold_l2: float = THRESHOLD_L2,
        save_path: Optional[str] = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.threshold_l1 = threshold_l1
        self.threshold_l2 = threshold_l2
        self.save_path = Path(save_path) if save_path else _DEFAULT_SAVE_PATH

        # L1 state
        self._l1_centroids: list = []     # list of np.ndarray (normalized)
        self._l1_counts: list = []        # int
        self._l1_outcome_sum: list = []   # float
        self._l1_outcome_n: list = []     # int
        self._l1_examples: list = []      # str

        # L2 state
        self._l2_centroids: list = []
        self._l2_counts: list = []

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def add_memory(
        self,
        embedding: np.ndarray,
        outcome: Optional[float] = None,
        example: str = "",
    ) -> None:
        """
        Add one experience to the hierarchy.

        Parameters
        ----------
        embedding : np.ndarray
            The latent embedding for this experience (e.g., from the agent's
            goal/action/state text).
        outcome : float, optional
            The real outcome signal (0=bad, 1=good). None if unknown.
        example : str
            A short human-readable label (the action taken, the goal, etc.)
        """
        z = self._normalize(np.asarray(embedding, dtype=np.float32))

        # ── L1 update ─────────────────────────────────────────────────
        l1_idx = self._best_match(z, self._l1_centroids, self.threshold_l1)
        if l1_idx is None:
            # New L1 cluster
            self._l1_centroids.append(z.copy())
            self._l1_counts.append(1)
            self._l1_outcome_sum.append(outcome if outcome is not None else 0.0)
            self._l1_outcome_n.append(1 if outcome is not None else 0)
            self._l1_examples.append(example)
            l1_idx = len(self._l1_centroids) - 1
        else:
            # Merge into existing L1 cluster (Welford online mean)
            n = self._l1_counts[l1_idx]
            old_c = self._l1_centroids[l1_idx]
            new_c = (old_c * n + z) / (n + 1)
            self._l1_centroids[l1_idx] = self._normalize(new_c)
            self._l1_counts[l1_idx] = n + 1
            if outcome is not None:
                self._l1_outcome_sum[l1_idx] += outcome
                self._l1_outcome_n[l1_idx] += 1
            if example:
                self._l1_examples[l1_idx] = example  # keep most recent label

        # ── L2 update using the updated L1 centroid ────────────────────
        l1_centroid = self._l1_centroids[l1_idx]
        l2_idx = self._best_match(l1_centroid, self._l2_centroids, self.threshold_l2)
        if l2_idx is None:
            self._l2_centroids.append(l1_centroid.copy())
            self._l2_counts.append(1)
        else:
            n = self._l2_counts[l2_idx]
            old_c = self._l2_centroids[l2_idx]
            new_c = (old_c * n + l1_centroid) / (n + 1)
            self._l2_centroids[l2_idx] = self._normalize(new_c)
            self._l2_counts[l2_idx] = n + 1

    def query(self, embedding: np.ndarray, level: int = 1, min_similarity: Optional[float] = None) -> Optional[dict]:
        """
        Find the nearest cluster to this embedding and return what it knows.

        This is the generalization primitive: new situation → "situations like
        this have historically gone well/poorly".

        Parameters
        ----------
        embedding : np.ndarray
        level : int
            1 = L1 (concrete clusters), 2 = L2 (abstract clusters).
        min_similarity : float, optional
            Override matching threshold (defaults to threshold_l1 or threshold_l2).

        Returns
        -------
        dict or None
            None if no cluster is within the similarity threshold.
            For L1: {"level", "similarity", "count", "mean_outcome",
                      "outcome_observations", "example"}
            For L2: {"level", "similarity", "count"}
        """
        z = self._normalize(np.asarray(embedding, dtype=np.float32))
        centroids = self._l1_centroids if level == 1 else self._l2_centroids
        threshold = min_similarity if min_similarity is not None else (self.threshold_l1 if level == 1 else self.threshold_l2)

        idx = self._best_match(z, centroids, threshold)
        if idx is None:
            return None

        result: dict = {
            "level": level,
            "similarity": round(self._cosine(z, centroids[idx]), 3),
            "count": (self._l1_counts if level == 1 else self._l2_counts)[idx],
        }
        if level == 1:
            n = self._l1_outcome_n[idx]
            result["mean_outcome"] = round(self._l1_outcome_sum[idx] / n, 3) if n > 0 else None
            result["outcome_observations"] = n
            result["example"] = self._l1_examples[idx]
        return result

    def get_stats(self) -> dict:
        """Return cluster counts for monitoring."""
        return {
            "l1_clusters": len(self._l1_centroids),
            "l2_clusters": len(self._l2_centroids),
            "l1_total_memories": sum(self._l1_counts),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Persist cluster state to disk."""
        target = Path(path) if path else self.save_path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Truncate examples to 256 chars so they fit in dtype='U256'
            # (avoids dtype=object which requires allow_pickle=True)
            examples_safe = [str(e)[:256] for e in self._l1_examples]
            np.savez_compressed(
                str(target),
                l1_centroids=np.array(self._l1_centroids) if self._l1_centroids else np.empty((0, self.latent_dim)),
                l1_counts=np.array(self._l1_counts, dtype=np.int32),
                l1_outcome_sum=np.array(self._l1_outcome_sum, dtype=np.float32),
                l1_outcome_n=np.array(self._l1_outcome_n, dtype=np.int32),
                l1_examples=np.array(examples_safe, dtype="U256"),
                l2_centroids=np.array(self._l2_centroids) if self._l2_centroids else np.empty((0, self.latent_dim)),
                l2_counts=np.array(self._l2_counts, dtype=np.int32),
            )
            logger.debug(f"ConceptHierarchy saved to {target}")
        except Exception as e:
            logger.warning(f"ConceptHierarchy save failed: {e}")

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load cluster state from disk. No-op if file doesn't exist."""
        target = Path(path) if path else self.save_path
        if not target.exists():
            return
        try:
            data = np.load(str(target), allow_pickle=False)
            self._l1_centroids = list(data["l1_centroids"])
            self._l1_counts = list(data["l1_counts"].astype(int))
            self._l1_outcome_sum = list(data["l1_outcome_sum"].astype(float))
            self._l1_outcome_n = list(data["l1_outcome_n"].astype(int))
            self._l1_examples = [str(e) for e in data["l1_examples"]]
            self._l2_centroids = list(data["l2_centroids"])
            self._l2_counts = list(data["l2_counts"].astype(int))
            logger.debug(f"ConceptHierarchy loaded: {len(self._l1_centroids)} L1, {len(self._l2_centroids)} L2 clusters")
        except ValueError:
            # Legacy file saved with dtype=object requires allow_pickle=True.
            # Refuse to load it (security risk) and start fresh instead.
            logger.warning(
                f"ConceptHierarchy at {target} appears to be a legacy pickle-format file "
                "and cannot be loaded safely. Starting with an empty hierarchy. "
                "Delete the file to suppress this warning."
            )
        except Exception as e:
            logger.warning(f"ConceptHierarchy load failed (starting fresh): {e}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-9)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _best_match(
        self, z: np.ndarray, centroids: list, threshold: float
    ) -> Optional[int]:
        """Return the index of the closest centroid above threshold, or None."""
        if not centroids:
            return None
        best_sim = threshold - 1e-9
        best_idx = None
        for i, c in enumerate(centroids):
            sim = self._cosine(z, c)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx


def get_hierarchy(db_path: Optional[Union[str, Path]] = None) -> ConceptHierarchy:
    """Retrieve or initialize the concept hierarchy for the active project/db."""
    if db_path:
        p = Path(db_path)
        save_path = p.parent / "concept_hierarchy.npz"
    else:
        env_db = os.environ.get("AGENT_SLEEP_DB")
        if env_db:
            save_path = Path(env_db).parent / "concept_hierarchy.npz"
        else:
            save_path = Path.cwd() / ".agent_sleep" / "concept_hierarchy.npz"
    h = ConceptHierarchy(save_path=save_path)
    h.load()
    return h


# Module-level default singleton
global_hierarchy = ConceptHierarchy()
global_hierarchy.load()

