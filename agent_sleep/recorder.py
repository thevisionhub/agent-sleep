"""
AgentMemory — public-facing recorder and recall interface (v0.1.1).

Improvements in v0.1.1:
- Selective rule retrieval (only rules semantically relevant to the task).
- Scope isolation (e.g., project_id/repo_id namespaces).
- Epistemic status formatting (observed, inferred, verified).
- Explicit db_path support.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

from agent_sleep.storage.db import (
    ensure_db_initialized,
    recall_memories,
    recall_rules,
    save_episode,
)

logger = logging.getLogger(__name__)


class AgentMemory:
    """
    High-level memory interface for AI agents.

    Parameters
    ----------
    session_id : str
        The current execution run/session ID.
    scope : str, default "global"
        The namespace for this memory (e.g. project name, repo name, organization).
        When recalling, memories from both this scope and 'global' are retrieved.
    db_path : str or Path, optional
        Custom path for SQLite database file. Defaults to AGENT_SLEEP_DB env var.
    """

    def __init__(
        self,
        session_id: str,
        scope: str = "global",
        db_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.session_id = session_id
        self.scope = scope
        self.db_path = Path(db_path) if db_path else None

    def initialize(self) -> None:
        """Explicitly ensure the database schema is initialized."""
        ensure_db_initialized(self.db_path)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_episode(
        self,
        goal: str,
        action: str,
        outcome: str,
        *,
        scope: Optional[str] = None,
        plan: str = "",
        failure_reason: str = "",
        episode_kind: str = "action",
        emotion_label: str = "",
        prediction_error: float = 0.5,
        novelty: float = 1.0,
        emotional_weight: Optional[float] = None,
    ) -> int:
        """
        Record one step of an agent's execution.
        """
        is_failure = "fail" in outcome.lower() or outcome.lower() == "rejected"
        ew = emotional_weight if emotional_weight is not None else (1.5 if is_failure else 1.0)
        target_scope = scope or self.scope

        return save_episode(
            session_id=self.session_id,
            scope=target_scope,
            goal=goal,
            action=action,
            outcome=outcome,
            plan=plan,
            failure_reason=failure_reason,
            episode_kind=episode_kind,
            emotion_label=emotion_label,
            prediction_error=prediction_error,
            novelty=novelty,
            emotional_weight=ew,
            db_path=self.db_path,
        )

    def record_task_verdict(
        self,
        goal: str,
        passed: bool,
        failure_reason: str = "",
        scope: Optional[str] = None,
    ) -> int:
        """
        Record an authoritative, externally verified whole-task outcome (e.g. from tests).
        """
        outcome = "success" if passed else "failure"
        return self.record_episode(
            goal=goal,
            action="verified by automated grading",
            outcome=outcome,
            failure_reason=failure_reason,
            episode_kind="task_verdict",
            prediction_error=0.9 if not passed else 0.1,
            emotional_weight=2.0 if not passed else 1.0,
            scope=scope or self.scope,
        )

    # ------------------------------------------------------------------
    # Recall (Selective & Scope-Aware)
    # ------------------------------------------------------------------

    def recall(
        self,
        task: str,
        *,
        top_k: int = 5,
        top_k_rules: int = 3,
        include_rules: bool = True,
        scopes: Optional[Sequence[str]] = None,
        memory_types: Optional[Sequence[str]] = None,
    ) -> str:
        """
        Retrieve relevant memories & rules for a task.

        Selectively injects only high-confidence, semantically relevant rules
        and memories to prevent prompt pollution.
        """
        active_scopes = list(scopes) if scopes else (
            [self.scope] if self.scope == "global" else [self.scope, "global"]
        )

        memories = recall_memories(
            task,
            top_k=top_k,
            scopes=active_scopes,
            memory_types=memory_types,
            db_path=self.db_path,
        )

        rules = recall_rules(
            task,
            top_k=top_k_rules,
            scopes=active_scopes,
            db_path=self.db_path,
        ) if include_rules else []

        if not memories and not rules:
            return ""

        lines = ["[MEMORY CONTEXT]"]

        if memories:
            lines.append("Relevant past experience:")
            for m in memories:
                status = m.get("verification_status", "observed")
                tag = {
                    "procedural": "✓ [PROCEDURAL]",
                    "lesson": "⚠ [LESSON]",
                    "compressed_episodic": "📦 [SUMMARY]",
                }.get(m.get("memory_type", ""), "·")
                ver_badge = f"({status})" if status == "verified" else ""
                lines.append(f"  {tag} {m['fact']} {ver_badge}: {m['value']}")

        if rules:
            lines.append("Applicable behavioral rules:")
            for rule in rules:
                lines.append(f"  ▶ {rule}")

        lines.append("[END MEMORY CONTEXT]")
        return "\n".join(lines)

    def recall_structured(
        self,
        task: str,
        *,
        top_k: int = 5,
        top_k_rules: int = 3,
        scopes: Optional[Sequence[str]] = None,
    ) -> dict:
        """Return structured dictionary for custom prompt formatting."""
        active_scopes = list(scopes) if scopes else (
            [self.scope] if self.scope == "global" else [self.scope, "global"]
        )
        memories = recall_memories(task, top_k=top_k, scopes=active_scopes, db_path=self.db_path)
        rules = recall_rules(task, top_k=top_k_rules, scopes=active_scopes, db_path=self.db_path)
        return {
            "memories": memories,
            "rules": rules,
            "scopes": active_scopes,
        }
