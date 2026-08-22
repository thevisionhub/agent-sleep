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
    recall_causal_hypotheses,
    recall_memories,
    recall_rules,
    record_memory_utility_feedback,
    save_episode,
)
from agent_sleep.self_model import SelfModel, infer_domain
from agent_sleep.hierarchy import HIERARCHY_RECALL_THRESHOLD, get_hierarchy
from agent_sleep.storage.embeddings import embed

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

    def record_memory_feedback(
        self,
        memory_ids: Sequence[int],
        outcome: str,
        was_applied: bool = True,
        is_causal_contributor: bool = False,
    ) -> dict:
        """
        Record whether retrieved memories were applied and what happened afterward.
        Closes the utility feedback loop so the memory system learns which knowledge helps.
        """
        return record_memory_utility_feedback(
            memory_ids=memory_ids,
            outcome=outcome,
            was_applied=was_applied,
            is_causal_contributor=is_causal_contributor,
            db_path=self.db_path,
        )

    # ------------------------------------------------------------------
    # Recall (Selective, Scope-Aware & Closed-Loop)
    # ------------------------------------------------------------------

    def recall(
        self,
        task: str,
        *,
        top_k: int = 5,
        top_k_rules: int = 3,
        include_rules: bool = True,
        include_causal: bool = True,
        include_competence: bool = True,
        include_cluster: bool = True,
        scopes: Optional[Sequence[str]] = None,
        memory_types: Optional[Sequence[str]] = None,
    ) -> str:
        """
        Retrieve relevant memories, rules, causal hypotheses, and competence policies.

        Selectively injects high-confidence, operational context to actively guide
        agent decision making, tool parameters, and verification intensity.
        """
        domain = infer_domain(task)
        active_scopes = list(scopes) if scopes else list({self.scope, domain, "global"})

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

        causal = recall_causal_hypotheses(
            task,
            top_k=2,
            scopes=active_scopes,
            db_path=self.db_path,
        ) if include_causal else []

        self_model = SelfModel(db_path=self.db_path)
        policy = self_model.get_behavioral_policy(domain, db_path=self.db_path) if include_competence else None

        cluster_info = None
        if include_cluster:
            try:
                hierarchy = get_hierarchy(db_path=self.db_path)
                vec = embed(task)
                cluster_info = hierarchy.query(vec, level=1, min_similarity=HIERARCHY_RECALL_THRESHOLD)
            except Exception:
                cluster_info = None

        if not memories and not rules and not causal and (policy is None or policy["level"] == "MODERATE") and not cluster_info:
            return ""

        lines = ["[MEMORY CONTEXT]"]

        # 1. Operational Policy / Self-Competence Directive
        if include_competence and policy and policy["level"] != "MODERATE":
            lines.append(f"🛡 [SELF-MODEL: {policy['level']} COMPETENCE IN {domain.upper()}]")
            lines.append(f"  Directive: {policy['directive']}")

        # 2. Semantic Memories
        if memories:
            lines.append("Relevant past experience:")
            for m in memories:
                status = m.get("verification_status", "observed")
                tag = {
                    "procedural": "✓ [PROCEDURAL]",
                    "lesson": "⚠ [LESSON]",
                    "compressed_episodic": "📦 [SUMMARY]",
                }.get(m.get("memory_type", ""), "·")
                ver_badge = f"({status})" if status in ("verified", "repeated") else ""
                lines.append(f"  {tag} {m['fact']} {ver_badge}: {m['value']}")

        # 3. Causal Hypotheses
        if causal:
            lines.append("⚡ [CAUSAL MECHANISMS & TRAPS]:")
            for c in causal:
                lines.append(f"  • {c['hypothesis']}")

        # 4. Applicable behavioral rules
        if rules:
            lines.append("Applicable behavioral rules:")
            for rule in rules:
                lines.append(f"  ▶ {rule}")

        # 5. Concept Hierarchy Cluster Track Record
        if cluster_info and cluster_info.get("mean_outcome") is not None:
            mo = cluster_info["mean_outcome"]
            obs = cluster_info["outcome_observations"]
            ex = cluster_info.get("example") or "Similar task pattern"
            rec_str = cluster_info.get("recommendation_strength", "MODERATE")
            lines.append(f"📊 [CLUSTER TRACK RECORD ({rec_str})] '{ex}': historical success rate {mo:.0%} across {obs} prior attempts.")

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
        """Return structured dictionary for custom prompt formatting and programmatic agent control."""
        domain = infer_domain(task)
        active_scopes = list(scopes) if scopes else list({self.scope, domain, "global"})
        
        subsystem_status: dict = {
            "memories": "ok",
            "rules": "ok",
            "causal": "ok",
            "self_model": "ok",
            "hierarchy": "ok",
        }

        from agent_sleep.storage.embeddings import get_backend_info
        backend_info = get_backend_info()

        try:
            memories = recall_memories(task, top_k=top_k, scopes=active_scopes, db_path=self.db_path)
            subsystem_status["memories"] = "SUCCESS" if memories else "NO_DATA"
        except Exception as e:
            memories = []
            subsystem_status["memories"] = f"FAILED: {e}"

        try:
            rules = recall_rules(task, top_k=top_k_rules, scopes=active_scopes, db_path=self.db_path)
            subsystem_status["rules"] = "SUCCESS" if rules else "NO_DATA"
        except Exception as e:
            rules = []
            subsystem_status["rules"] = f"FAILED: {e}"

        try:
            causal = recall_causal_hypotheses(task, top_k=3, scopes=active_scopes, db_path=self.db_path)
            subsystem_status["causal"] = "SUCCESS" if causal else "NO_DATA"
        except Exception as e:
            causal = []
            subsystem_status["causal"] = f"FAILED: {e}"

        try:
            self_model = SelfModel(db_path=self.db_path)
            policy = self_model.get_behavioral_policy(domain, db_path=self.db_path)
            subsystem_status["self_model"] = "SUCCESS"
        except Exception as e:
            policy = {"competence": 0.5, "uncertainty": 0.5, "level": "MODERATE", "directive": "Standard execution."}
            subsystem_status["self_model"] = f"FAILED: {e}"

        cluster_info = None
        try:
            hierarchy = get_hierarchy(db_path=self.db_path)
            vec = embed(task)
            cluster_info = hierarchy.query(vec, level=1, min_similarity=HIERARCHY_RECALL_THRESHOLD)
            subsystem_status["hierarchy"] = "SUCCESS" if cluster_info is not None else "NO_DATA"
        except Exception as e:
            cluster_info = None
            subsystem_status["hierarchy"] = f"FAILED: {e}"

        return {
            "memories": memories,
            "rules": rules,
            "causal_hypotheses": causal,
            "domain": domain,
            "self_competence": policy["competence"],
            "operational_policy": policy,
            "cluster_insight": cluster_info,
            "subsystem_status": subsystem_status,
            "embedding_backend": backend_info,
            "scopes": active_scopes,
        }

    def specialize_rule(self, rule_id: int, condition: str = "", exception: str = "") -> dict:
        """Specialize a rule with explicit context conditions or exceptions."""
        from agent_sleep.storage.db import specialize_rule_with_exception
        return specialize_rule_with_exception(rule_id, condition=condition, exception=exception, db_path=self.db_path)
