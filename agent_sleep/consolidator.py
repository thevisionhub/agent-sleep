"""
SleepConsolidator: 8-stage offline learning pipeline (v0.1.1).
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SleepConsolidator:
    """
    Offline sleep consolidation engine.

    Parameters
    ----------
    llm_fn : callable, optional
        Used for HOW-memory distillation and failure-pattern abstraction.
    on_progress : callable, optional
        Progress callback: (stage, message, percent).
    mode : str
        "MICRO" (top 2), "NORMAL" (top 10), "DEEP" (all).
    scope : str, default "global"
        The target namespace/project scope for this consolidation.
    db_path : Path or str, optional
        Custom SQLite database path.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        on_progress: Optional[Callable[[str, str, int], None]] = None,
        mode: str = "NORMAL",
        scope: str = "global",
        db_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.llm_fn = llm_fn
        self.on_progress = on_progress
        self.mode = mode
        self.scope = scope
        self.db_path = Path(db_path) if db_path else None

    def _emit(self, stage: str, message: str, percent: int = 0) -> None:
        logger.info(f"[SLEEP {stage}] ({percent}%) {message}")
        if self.on_progress:
            try:
                self.on_progress(stage, message, percent)
            except Exception:
                pass

    def run(self, session_id: str) -> dict:
        """Synchronous entry point with modern async runtime safety."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._run_async(session_id))
                return future.result()
        else:
            return asyncio.run(self._run_async(session_id))

    async def run_async(self, session_id: str) -> dict:
        return await self._run_async(session_id)

    async def _run_async(self, session_id: str) -> dict:
        import hashlib
        import uuid
        from agent_sleep.storage.db import (
            get_unprocessed_episodes,
            mark_episodes_processed,
        )

        start = time.time()
        run_id = uuid.uuid4().hex[:12]
        stage_status: Dict[str, str] = {
            "compression": "SKIPPED",
            "replay": "PENDING",
            "distillation": "PENDING",
            "how_distillation": "SKIPPED" if self.llm_fn is None else "PENDING",
            "rules": "PENDING",
            "causal": "PENDING",
            "self_reflection": "PENDING",
            "hierarchy": "PENDING",
            "finalize": "PENDING",
            "decay": "PENDING",
        }

        stats: Dict[str, Any] = {
            "consolidation_run_id": run_id,
            "session_id": session_id,
            "scope": self.scope,
            "episodes_processed": 0,
            "memories_written": 0,
            "rules_promoted": 0,
            "beliefs_revised": 0,
            "domains_updated": 0,
            "stage_status": stage_status,
            "compression_report": {},
            "duration_seconds": 0.0,
        }

        self._emit("INIT", f"Starting sleep consolidation (run_id={run_id}, mode={self.mode}, scope={self.scope})", 0)

        # ── Stage 0: Episodic Compression ─────────────────────────────
        self._emit("COMPRESSION", "Compressing stale episodes", 5)
        try:
            from agent_sleep.episodic import compress_stale_episodes
            def _get_stale(max_age_days):
                from agent_sleep.storage.db import get_stale_episodes
                return get_stale_episodes(max_age_days=max_age_days, db_path=self.db_path)
            def _del_eps(ids):
                from agent_sleep.storage.db import delete_episodes
                return delete_episodes(ids, db_path=self.db_path)
            def _save_mem(fact, value, **kwargs):
                from agent_sleep.storage.db import save_semantic_memory
                return save_semantic_memory(fact, value, db_path=self.db_path, **kwargs)

            compression_report = compress_stale_episodes(
                scope=self.scope,
                get_stale_fn=_get_stale,
                delete_fn=_del_eps,
                save_fn=_save_mem,
            )
            stats["compression_report"] = compression_report
            stage_status["compression"] = "SUCCESS" if compression_report.get("clusters_compressed") else "NO_DATA"
            if compression_report.get("clusters_compressed"):
                self._emit(
                    "COMPRESSION",
                    f"Compressed {compression_report['episodes_compressed']} episodes "
                    f"into {compression_report['clusters_compressed']} summaries",
                    8,
                )
        except Exception as e:
            stage_status["compression"] = f"FAILED: {e}"
            logger.warning(f"Episodic compression failed (non-fatal): {e}")

        # ── Fetch episodes ─────────────────────────────────────────────
        episodes_raw = get_unprocessed_episodes(session_id, scope=self.scope, db_path=self.db_path)
        if not episodes_raw:
            self._emit("DONE", "No unprocessed episodes. Nothing to consolidate.", 100)
            stats["duration_seconds"] = round(time.time() - start, 2)
            stage_status["replay"] = "NO_DATA"
            return stats

        self._emit("REPLAY", f"Fetched {len(episodes_raw)} unprocessed episodes", 10)
        stage_status["replay"] = "SUCCESS"

        # ── Stage 1: Priority Replay & Deterministic Fingerprinting ─────
        prioritized = self._prioritize(episodes_raw)
        stats["episodes_processed"] = len(prioritized)

        # Deterministic input fingerprint for crash-consistency & idempotency
        sorted_ep_ids = sorted(ep.get("id", 0) for ep in prioritized)
        input_fingerprint = hashlib.sha256(f"{session_id}:{self.scope}:{sorted_ep_ids}".encode("utf-8")).hexdigest()[:16]

        def _bound_save(fact, value, **kwargs):
            from agent_sleep.storage.db import save_semantic_memory
            prov = kwargs.pop("provenance", {}) or {}
            prov["consolidation_run_id"] = run_id
            prov["input_fingerprint"] = input_fingerprint
            prov["sessions"] = [session_id]
            prov["source_episodes"] = sorted_ep_ids
            target_scope = kwargs.pop("scope", self.scope)
            return save_semantic_memory(fact, value, provenance=prov, scope=target_scope, db_path=self.db_path, **kwargs)

        # ── Stage 2: Episodic → Semantic Distillation ─────────────────
        self._emit("DISTILLATION", "Distilling episodes into semantic memories", 20)
        try:
            from agent_sleep.episodic import consolidate_episodes
            consolidation_report = consolidate_episodes(prioritized, save_fn=_bound_save, scope=self.scope)
            stats["memories_written"] += consolidation_report["written"]
            stage_status["distillation"] = "SUCCESS" if consolidation_report["written"] > 0 else "NO_DATA"
            self._emit(
                "DISTILLATION",
                f"Wrote {consolidation_report['written']} memories "
                f"({consolidation_report['procedural']} procedural, "
                f"{consolidation_report['lessons']} lessons)",
                35,
            )
        except Exception as e:
            stage_status["distillation"] = f"FAILED: {e}"
            logger.warning(f"Episodic distillation failed (non-fatal): {e}")
            logger.debug("Distillation traceback:", exc_info=True)

        # ── Stage 3: HOW Memory Distillation (LLM-backed) ─────────────
        if self.llm_fn is not None:
            self._emit("HOW_DISTILLATION", "Distilling successful trajectories into HOW memories", 45)
            try:
                from agent_sleep.episodic import distill_how_memories
                how_report = await distill_how_memories(
                    prioritized, llm_fn=self.llm_fn, save_fn=_bound_save, scope=self.scope
                )
                stats["memories_written"] += how_report["how_written"]
                stage_status["how_distillation"] = "SUCCESS" if how_report["how_written"] > 0 else "NO_DATA"
            except Exception as e:
                stage_status["how_distillation"] = f"FAILED: {e}"
                logger.warning(f"HOW distillation failed (non-fatal): {e}")

        # ── Stage 4: Rule Promotion ────────────────────────────────────
        self._emit("RULE_PROMOTION", "Promoting behavioral rules from repeated failures", 60)
        try:
            from agent_sleep.beliefs import BeliefSystem, promote_candidate_rules
            def _add_rule(belief, conf=0.5, scope="global"):
                from agent_sleep.storage.db import add_candidate_rule
                return add_candidate_rule(belief, confidence=conf, scope=scope, db_path=self.db_path)
            def _get_rules(scopes=None):
                from agent_sleep.storage.db import get_active_rules
                return get_active_rules(scopes=scopes, db_path=self.db_path)

            belief_system = BeliefSystem(add_rule_fn=_add_rule, get_rules_fn=_get_rules, scope=self.scope)
            rule_report = await promote_candidate_rules(
                prioritized,
                belief_system=belief_system,
                llm_fn=self.llm_fn,
                scope=self.scope,
            )
            stats["rules_promoted"] = rule_report["rules_promoted"]
            stage_status["rules"] = "SUCCESS" if rule_report["rules_promoted"] > 0 else "NO_DATA"
        except Exception as e:
            stage_status["rules"] = f"FAILED: {e}"
            logger.warning(f"Rule promotion failed (non-fatal): {e}")

        # ── Stage 5: Error Analysis (Causal hypotheses with cautious initial confidence) ───
        self._emit("ERROR_ANALYSIS", "Analyzing failure episodes for causal mechanisms", 70)
        beliefs_revised = 0
        try:
            for ep in prioritized:
                outcome = (ep.get("outcome") or "").strip().lower()
                if outcome not in ("failure", "rejected"):
                    continue
                try:
                    from agent_sleep.storage.db import save_causal_hypothesis
                    action = str(ep.get("action") or "Unknown Action")[:200]
                    reason = ep.get("failure_reason") or "unknown failure"
                    goal = ep.get("goal") or "Task Execution"
                    save_causal_hypothesis(
                        session_id=session_id,
                        action=action,
                        hypothesis=f"When attempting '{goal[:100]}', action '{action[:80]}' failed due to: {reason[:120]}",
                        effect=f"Execution Failed: {reason[:80]}",
                        confidence=0.35,  # Cautious initial confidence (evidence accumulation)
                        scope=ep.get("scope", self.scope),
                        provenance={
                            "consolidation_run_id": run_id,
                            "input_fingerprint": input_fingerprint,
                            "sessions": [session_id],
                            "sources": [self.scope],
                        },
                        db_path=self.db_path,
                    )
                    beliefs_revised += 1
                except Exception as e:
                    logger.warning(f"Causal hypothesis save failed: {e}")
            stats["beliefs_revised"] = beliefs_revised
            stage_status["causal"] = "SUCCESS" if beliefs_revised > 0 else "NO_DATA"
        except Exception as e:
            stage_status["causal"] = f"FAILED: {e}"

        # ── Stage 6: Self-Reflection (competence tracking) ────────────
        self._emit("SELF_REFLECTION", "Updating self-competence model", 80)
        try:
            from agent_sleep.self_model import run_self_reflection
            reflection_report = run_self_reflection(prioritized, db_path=self.db_path)
            stats["domains_updated"] = reflection_report["domains_updated"]
            stage_status["self_reflection"] = "SUCCESS"
        except Exception as e:
            stage_status["self_reflection"] = f"FAILED: {e}"
            logger.warning(f"Self-reflection failed (non-fatal): {e}")

        # ── Stage 6.5: Concept Hierarchy Clustering ────────────────────
        try:
            from agent_sleep.hierarchy import get_hierarchy
            from agent_sleep.storage.embeddings import embed
            hierarchy = get_hierarchy(db_path=self.db_path)
            for ep in prioritized:
                goal = ep.get("goal") or ""
                action = ep.get("action") or ""
                outcome_str = (ep.get("outcome") or "").strip().lower()
                outcome_val = 1.0 if outcome_str == "success" else 0.0
                ep_text = f"{goal} {action}"
                vec = embed(ep_text)
                hierarchy.add_memory(vec, outcome=outcome_val, example=goal[:80])
            hierarchy.save()
            stage_status["hierarchy"] = "SUCCESS"
        except Exception as e:
            stage_status["hierarchy"] = f"FAILED: {e}"
            logger.debug(f"Concept hierarchy clustering skipped: {e}")

        # ── Stage 7: Mark episodes as processed ───────────────────────
        self._emit("FINALIZE", "Marking episodes as processed", 90)
        try:
            episode_ids = [ep["id"] for ep in prioritized if ep.get("id")]
            mark_episodes_processed(episode_ids, db_path=self.db_path)
            stage_status["finalize"] = "SUCCESS"
        except Exception as e:
            stage_status["finalize"] = f"FAILED: {e}"
            logger.warning(f"mark_episodes_processed failed: {e}")

        # ── Stage 8: Memory Utility Decay & Forgetting ────────────────
        self._emit("DECAY", "Applying utility decay to unaccessed memories", 95)
        try:
            from agent_sleep.storage.db import decay_stale_memories
            decay_report = decay_stale_memories(db_path=self.db_path)
            stats["pruned_stale_memories"] = decay_report.get("pruned_stale_memories", 0)
            stage_status["decay"] = "SUCCESS"
        except Exception as e:
            stage_status["decay"] = f"FAILED: {e}"
            logger.warning(f"Memory decay failed (non-fatal): {e}")

        stats["duration_seconds"] = round(time.time() - start, 2)
        self._emit(
            "DONE",
            f"Sleep complete (run {run_id}): {stats['memories_written']} memories, "
            f"{stats['rules_promoted']} rules ({stats['duration_seconds']}s)",
            100,
        )
        return stats

    def _prioritize(self, episodes: list) -> list:
        prioritized = []
        for ep in episodes:
            try:
                prediction_error = float(ep.get("prediction_error") or 0.5)
                novelty = float(ep.get("novelty") or 1.0)
                outcome = (ep.get("outcome") or "").lower()
                is_failure = "fail" in outcome or outcome == "rejected"
                ew = ep.get("emotional_weight")
                emotional_weight = float(ew) if ew else (1.5 if is_failure else 1.0)
                ep["_priority"] = prediction_error * novelty * emotional_weight
                prioritized.append(ep)
            except Exception as e:
                logger.warning(f"Priority calculation failed for episode {ep.get('id')}: {e}")

        prioritized.sort(key=lambda x: x["_priority"], reverse=True)
        mode_limits = {"MICRO": 2, "NORMAL": 10, "DEEP": None}
        limit = mode_limits.get(self.mode)
        return prioritized[:limit] if limit else prioritized
