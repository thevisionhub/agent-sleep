"""
Episodic -> Semantic consolidation (v0.1.1).

Converts raw execution_episodes into deduplicated semantic memories.
Grounding first, abstraction second.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MIN_GOAL_CHARS = 20
MAX_GOAL_KEY_CHARS = 140
MAX_ACTION_CHARS = 200
MAX_HOW_VALUE_CHARS = 700
MAX_TRAJECTORY_STEPS = 12
COMPRESSION_SIMILARITY_THRESHOLD = 0.75
MIN_COMPRESSION_CLUSTER_SIZE = 5


def _summarize_goal(goal_raw) -> str:
    g = goal_raw
    if isinstance(goal_raw, str):
        try:
            g = json.loads(goal_raw)
        except (ValueError, TypeError):
            g = goal_raw
    if isinstance(g, dict):
        content = g.get("content") or g.get("goal") or g.get("task") or g.get("objective") or ""
    else:
        content = str(g)
    return " ".join(str(content).split())


def _goal_key(episode: dict) -> str:
    goal = _summarize_goal(episode.get("goal"))
    if len(goal) < MIN_GOAL_CHARS:
        return ""
    return goal if len(goal) <= MAX_GOAL_KEY_CHARS else goal[: MAX_GOAL_KEY_CHARS - 3] + "..."


def distill_episode(episode: dict) -> Optional[Tuple[str, str, str, float, float, str]]:
    """
    Distill one episode into a tuple:
    (memory_type, fact, value, importance, confidence, verification_status)
    """
    goal = _summarize_goal(episode.get("goal"))
    if len(goal) < MIN_GOAL_CHARS:
        return None

    goal_key = goal if len(goal) <= MAX_GOAL_KEY_CHARS else goal[: MAX_GOAL_KEY_CHARS - 3] + "..."
    outcome = (episode.get("outcome") or "").strip().lower()
    action = (episode.get("action") or "").strip()
    emotion = (episode.get("emotion_label") or "").strip()
    emotion_suffix = f" Emotion: {emotion}." if emotion else ""
    is_verdict = episode.get("episode_kind") == "task_verdict"

    if outcome == "success":
        if is_verdict:
            return (
                "procedural",
                f"Task done before (verified): {goal_key}",
                f"Verified correct by automated grading.{emotion_suffix}",
                0.85,
                0.95,
                "verified",
            )
        value = (f"Achieved before. A step that was involved: {action[:MAX_ACTION_CHARS]}"
                 if action else "This kind of task was achieved before.")
        return ("procedural", f"Task done before: {goal_key}", value + emotion_suffix, 0.7, 0.8, "observed")

    if outcome in ("failure", "rejected"):
        reason = (episode.get("failure_reason") or "").strip() or "no reason recorded"
        if is_verdict:
            return (
                "lesson",
                f"Task FAILED (verified): {goal_key}",
                f"Reason: {reason[:MAX_ACTION_CHARS]}.{emotion_suffix}",
                0.85,
                0.95,
                "verified",
            )
        value = f"A previous attempt failed: {reason[:MAX_ACTION_CHARS]}"
        if action:
            value += f" Failing action was: {action[:MAX_ACTION_CHARS]}"
        return ("lesson", f"Caution on task: {goal_key}", value + emotion_suffix, 0.65, 0.75, "observed")

    return None


def distill_procedures(episodes: list) -> list:
    by_goal: Dict[str, list] = {}
    for ep in episodes:
        if (ep.get("outcome") or "").strip().lower() != "success":
            continue
        if ep.get("episode_kind") == "task_verdict":
            continue
        key = _goal_key(ep)
        action = (ep.get("action") or "").strip()
        if not key or not action:
            continue
        steps = by_goal.setdefault(key, [])
        if action not in steps:
            steps.append(action)

    procedures = []
    for key, steps in by_goal.items():
        if len(steps) < 2:
            continue
        listing = "; ".join(f"{i+1}. {s[:MAX_ACTION_CHARS]}" for i, s in enumerate(steps[:6]))
        procedures.append((
            "procedural",
            f"Procedure that worked for: {key}",
            f"Steps in order: {listing}",
            0.75,
            0.85,
            "inferred",
        ))
    return procedures


def consolidate_episodes(
    episodes: list,
    save_fn: Optional[Callable] = None,
    scope: str = "global",
) -> dict:
    if save_fn is None:
        from agent_sleep.storage.db import save_semantic_memory
        save_fn = save_semantic_memory

    written = skipped = procedural = lessons = sequences = 0

    for ep in episodes:
        distilled = distill_episode(ep)
        if not distilled:
            skipped += 1
            continue
        mtype, fact, value, importance, confidence, ver_status = distilled
        ep_scope = ep.get("scope") or scope
        prov = {
            "session_id": ep.get("session_id", ""),
            "episode_id": ep.get("id"),
            "episode_kind": ep.get("episode_kind", "action"),
            "timestamp": ep.get("timestamp", ""),
        }
        try:
            save_fn(
                fact,
                value,
                memory_type=mtype,
                importance=importance,
                confidence=confidence,
                verification_status=ver_status,
                provenance=prov,
                scope=ep_scope,
                source="episodic_consolidation",
            )
            written += 1
            procedural += mtype == "procedural"
            lessons += mtype == "lesson"
        except Exception as e:
            logger.warning(f"Episode consolidation skipped one episode: {e}")
            skipped += 1

    for mtype, fact, value, importance, confidence, ver_status in distill_procedures(episodes):
        try:
            save_fn(
                fact,
                value,
                memory_type=mtype,
                importance=importance,
                confidence=confidence,
                verification_status=ver_status,
                scope=scope,
                source="episodic_consolidation",
            )
            written += 1
            sequences += 1
        except Exception as e:
            logger.warning(f"Procedure consolidation skipped one goal: {e}")

    return {
        "written": written,
        "skipped": skipped,
        "procedural": procedural,
        "lessons": lessons,
        "sequences": sequences,
    }


def _group_success_trajectories(episodes: list) -> dict:
    groups: Dict[str, dict] = {}
    for ep in sorted(episodes, key=lambda e: e.get("id") or 0):
        key = _goal_key(ep)
        if not key:
            continue
        g = groups.setdefault(key, {"steps": [], "verified": False, "succeeded": False, "scope": ep.get("scope", "global")})
        outcome = (ep.get("outcome") or "").strip().lower()
        if ep.get("episode_kind") == "task_verdict":
            if outcome == "success":
                g["verified"] = True
                g["succeeded"] = True
            continue
        action = (ep.get("action") or "").strip()
        if not action:
            continue
        line = action[:MAX_ACTION_CHARS]
        if outcome == "success":
            g["succeeded"] = True
        elif outcome in ("failure", "rejected"):
            reason = (ep.get("failure_reason") or "").strip()
            line += f"  [FAILED: {reason[:120]}]" if reason else "  [FAILED]"
        g["steps"].append(line)
    return {k: g for k, g in groups.items() if g["succeeded"] and g["steps"]}


def _fallback_how_value(steps: list) -> str:
    listing = "; ".join(f"{i+1}. {s}" for i, s in enumerate(steps[:6]))
    return f"Steps that worked: {listing}"[:MAX_HOW_VALUE_CHARS]


async def distill_how_memories(episodes: list, llm_fn=None, save_fn=None, scope: str = "global") -> dict:
    if save_fn is None:
        from agent_sleep.storage.db import save_semantic_memory
        save_fn = save_semantic_memory

    groups = _group_success_trajectories(episodes)
    written = llm_failures = 0

    for goal_key, g in groups.items():
        steps = g["steps"][:MAX_TRAJECTORY_STEPS]
        value = None

        if llm_fn is not None:
            try:
                value = await _distill_how_with_llm(goal_key, steps, g["verified"], llm_fn)
            except Exception as e:
                logger.warning(f"How-distillation LLM pass failed for '{goal_key[:60]}': {e}")
                llm_failures += 1

        if not value:
            value = _fallback_how_value(steps)

        ep_scope = g.get("scope") or scope
        try:
            save_fn(
                f"How to: {goal_key}",
                value[:MAX_HOW_VALUE_CHARS],
                memory_type="procedural",
                importance=0.85 if g["verified"] else 0.7,
                confidence=0.9 if g["verified"] else 0.8,
                verification_status="verified" if g["verified"] else "inferred",
                scope=ep_scope,
                source="how_distillation",
            )
            written += 1
        except Exception as e:
            logger.warning(f"How-distillation save failed for '{goal_key[:60]}': {e}")

    return {"how_written": written, "llm_failures": llm_failures, "groups": len(groups)}


async def _distill_how_with_llm(goal_key: str, steps: list, verified: bool, llm_fn) -> str:
    trajectory = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    messages = [
        {"role": "system", "content": (
            "You distill reusable knowledge from a coding agent's completed task. "
            "In 2-4 sentences, describe: (1) the approach used, "
            "(2) any codebase conventions discovered, "
            "(3) any pitfalls hit and how they were resolved. "
            "Be concrete and actionable. Do not restate the goal.")},
        {"role": "user", "content": (
            f"Task ({'verified correct by grader' if verified else 'completed'}):\n{goal_key}\n\n"
            f"Actual trajectory:\n{trajectory}")},
    ]
    response = await llm_fn(messages) if hasattr(llm_fn, "__call__") else llm_fn(messages)
    if isinstance(response, str):
        return response.strip()
    if hasattr(response, "choices"):
        return response.choices[0].message.content.strip()
    if isinstance(response, dict):
        return (response.get("content") or response.get("text") or "").strip()
    return str(response).strip()


def compress_stale_episodes(
    max_age_days: float = 14.0,
    min_cluster_size: int = MIN_COMPRESSION_CLUSTER_SIZE,
    similarity_threshold: float = COMPRESSION_SIMILARITY_THRESHOLD,
    get_stale_fn=None,
    delete_fn=None,
    save_fn=None,
    embed_fn=None,
    scope: str = "global",
) -> dict:
    if get_stale_fn is None:
        from agent_sleep.storage.db import get_stale_episodes
        get_stale_fn = get_stale_episodes
    if delete_fn is None:
        from agent_sleep.storage.db import delete_episodes
        delete_fn = delete_episodes
    if save_fn is None:
        from agent_sleep.storage.db import save_semantic_memory
        save_fn = save_semantic_memory

    candidates = get_stale_fn(max_age_days=max_age_days)
    if not candidates:
        return {"episodes_scanned": 0, "clusters_compressed": 0, "episodes_compressed": 0}

    if embed_fn is None:
        from agent_sleep.storage.embeddings import embed as embed_fn, cosine_similarity
    else:
        from agent_sleep.storage.embeddings import cosine_similarity

    embeddings = {}
    for ep in candidates:
        goal = _summarize_goal(ep.get("goal"))
        if goal:
            try:
                embeddings[ep["id"]] = embed_fn(goal)
            except Exception:
                pass

    clusters: list = []
    for ep in candidates:
        vec = embeddings.get(ep["id"])
        if vec is None:
            clusters.append({"seed": None, "members": [ep]})
            continue
        placed = False
        for cluster in clusters:
            if cluster["seed"] is None:
                continue
            if cosine_similarity(vec, cluster["seed"]) >= similarity_threshold:
                cluster["members"].append(ep)
                placed = True
                break
        if not placed:
            clusters.append({"seed": vec, "members": [ep]})

    compressed_clusters = 0
    compressed_episodes = 0

    for cluster in clusters:
        members = cluster["members"]
        if len(members) < min_cluster_size:
            continue

        goal_texts = [_summarize_goal(m.get("goal")) for m in members]
        representative = max(goal_texts, key=len) if goal_texts else "past episodes"
        goal_key = (representative if len(representative) <= MAX_GOAL_KEY_CHARS
                    else representative[: MAX_GOAL_KEY_CHARS - 3] + "...")

        success_count = sum(1 for m in members
                           if (m.get("outcome") or "").strip().lower() == "success")
        reasons = Counter(
            (m.get("failure_reason") or "").strip()
            for m in members
            if (m.get("outcome") or "").strip().lower() != "success"
            and (m.get("failure_reason") or "").strip()
        )
        common_reason = reasons.most_common(1)[0][0] if reasons else None
        value = f"{len(members)} similar episodes; {success_count} succeeded, {len(members) - success_count} failed."
        if common_reason:
            value += f" Most common failure: {common_reason[:MAX_ACTION_CHARS]}."

        importance = min(0.9, 0.5 + 0.02 * len(members))

        try:
            save_fn(
                f"Compressed history: {goal_key}",
                value,
                memory_type="compressed_episodic",
                importance=importance,
                confidence=0.85,
                verification_status="inferred",
                scope=scope,
                source="episodic_compression",
            )
        except Exception as e:
            logger.warning(f"Compression save failed (originals kept): {e}")
            continue

        delete_fn([m["id"] for m in members])
        compressed_clusters += 1
        compressed_episodes += len(members)

    return {
        "episodes_scanned": len(candidates),
        "clusters_compressed": compressed_clusters,
        "episodes_compressed": compressed_episodes,
    }
