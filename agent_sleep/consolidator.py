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
        from agent_sleep.storage.db import (
            get_unprocessed_episodes,
            mark_episodes_processed,
        )

        start = time.time()
        stats: Dict[str, Any] = {
            "episodes_processed": 0,
            "memories_written": 0,
            "rules_promoted": 0,
            "beliefs_revised": 0,
            "domains_updated": 0,
            "compression_report": {},
            "duration_seconds": 0.0,
        }

        self._emit("INIT", f"Starting sleep consolidation (mode={self.mode}, scope={self.scope})", 0)

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
            if compression_report.get("clusters_compressed"):
                self._emit(
                    "COMPRESSION",
                    f"Compressed {compression_report['episodes_compressed']} episodes "
                    f"into {compression_report['clusters_compressed']} summaries",
                    8,
                )
        except Exception as e:
            logger.warning(f"Episodic compression failed (non-fatal): {e}")

        # ── Fetch episodes ─────────────────────────────────────────────
        episodes_raw = get_unprocessed_episodes(session_id, db_path=self.db_path)
        if not episodes_raw:
            self._emit("DONE", "No unprocessed episodes. Nothing to consolidate.", 100)
            stats["duration_seconds"] = round(time.time() - start, 2)
            return stats

        self._emit("REPLAY", f"Fetched {len(episodes_raw)} unprocessed episodes", 10)

        # ── Stage 1: Priority Replay ───────────────────────────────────
        prioritized = self._prioritize(episodes_raw)
        stats["episodes_processed"] = len(prioritized)

        def _bound_save(fact, value, **kwargs):
            from agent_sleep.storage.db import save_semantic_memory
            return save_semantic_memory(fact, value, db_path=self.db_path, **kwargs)

        # ── Stage 2: Episodic → Semantic Distillation ─────────────────
        self._emit("DISTILLATION", "Distilling episodes into semantic memories", 20)
        try:
            from agent_sleep.episodic import consolidate_episodes
            consolidation_report = consolidate_episodes(prioritized, save_fn=_bound_save, scope=self.scope)
            stats["memories_written"] += consolidation_report["written"]
            self._emit(
                "DISTILLATION",
                f"Wrote {consolidation_report['written']} memories "
                f"({consolidation_report['procedural']} procedural, "
                f"{consolidation_report['lessons']} lessons)",
                35,
            )
        except Exception as e:
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
            except Exception as e:
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
        except Exception as e:
            logger.warning(f"Rule promotion failed (non-fatal): {e}")

        # ── Stage 5: Error Analysis (Causal hypotheses) ───────────────
        self._emit("ERROR_ANALYSIS", "Analyzing failure episodes", 70)
        beliefs_revised = 0
        for ep in prioritized:
            outcome = (ep.get("outcome") or "").strip().lower()
            if outcome not in ("failure", "rejected"):
                continue
            try:
                from agent_sleep.storage.db import save_causal_hypothesis
                action = str(ep.get("action") or "Unknown Action")[:200]
                reason = ep.get("failure_reason") or "unknown failure"
                save_causal_hypothesis(
                    session_id, action, reason, "Failed Execution", 0.8,
                    scope=ep.get("scope", self.scope), db_path=self.db_path,
                )
                beliefs_revised += 1
            except Exception as e:
                logger.warning(f"Causal hypothesis save failed: {e}")
        stats["beliefs_revised"] = beliefs_revised

        # ── Stage 6: Self-Reflection (competence tracking) ────────────
        self._emit("SELF_REFLECTION", "Updating self-competence model", 80)
        try:
            from agent_sleep.self_model import run_self_reflection
            reflection_report = run_self_reflection(prioritized)
            stats["domains_updated"] = reflection_report["domains_updated"]
        except Exception as e:
            logger.warning(f"Self-reflection failed (non-fatal): {e}")

        # ── Stage 7: Mark episodes as processed ───────────────────────
        self._emit("FINALIZE", "Marking episodes as processed", 90)
        try:
            episode_ids = [ep["id"] for ep in prioritized if ep.get("id")]
            mark_episodes_processed(episode_ids, db_path=self.db_path)
        except Exception as e:
            logger.warning(f"mark_episodes_processed failed: {e}")

        # ── Stage 8: Memory Utility Decay & Forgetting ────────────────
        self._emit("DECAY", "Applying utility decay to unaccessed memories", 95)
        try:
            from agent_sleep.storage.db import decay_stale_memories
            decay_report = decay_stale_memories(db_path=self.db_path)
            stats["pruned_stale_memories"] = decay_report.get("pruned_stale_memories", 0)
        except Exception as e:
            logger.warning(f"Memory decay failed (non-fatal): {e}")

        stats["duration_seconds"] = round(time.time() - start, 2)
        self._emit(
            "DONE",
            f"Sleep complete: {stats['memories_written']} memories, "
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
