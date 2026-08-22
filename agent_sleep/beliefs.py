"""
BeliefSystem: candidate behavioral rules extracted from repeated failures (v0.1.1).
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

MAX_RULES_PER_CONSOLIDATION = 2
MIN_PATTERN_COUNT = 2


class BeliefSystem:
    def __init__(
        self,
        add_rule_fn: Optional[Callable] = None,
        get_rules_fn: Optional[Callable] = None,
        scope: str = "global",
    ) -> None:
        if add_rule_fn is None:
            from agent_sleep.storage.db import add_candidate_rule
            add_rule_fn = add_candidate_rule
        if get_rules_fn is None:
            from agent_sleep.storage.db import get_active_rules
            get_rules_fn = get_active_rules

        self._add = add_rule_fn
        self._get = get_rules_fn
        self.scope = scope

    def add_candidate_rule(self, belief: str, confidence: float = 0.5, scope: Optional[str] = None) -> None:
        self._add(belief, confidence, scope=scope or self.scope)

    def get_rules(self, scopes: Optional[Sequence[str]] = None) -> List[str]:
        return self._get(scopes=scopes or (self.scope, "global") if self.scope != "global" else ("global",))


async def promote_candidate_rules(
    episodes: list,
    belief_system: Optional[BeliefSystem] = None,
    llm_fn: Optional[Callable] = None,
    scope: str = "global",
) -> dict:
    if belief_system is None:
        belief_system = BeliefSystem(scope=scope)

    patterns: Dict[tuple, dict] = {}
    for ep in episodes:
        outcome = (ep.get("outcome") or "").strip().lower()
        if outcome not in ("failure", "rejected") or ep.get("episode_kind") == "task_verdict":
            continue
        action = (ep.get("action") or "").strip()
        reason = " ".join((ep.get("failure_reason") or "").split()).strip()
        if not action or not reason:
            continue
        tool = action.split("(", 1)[0].strip()[:40] or "unknown_tool"
        key = (tool, reason[:60].lower())
        patterns.setdefault(key, {"tool": tool, "reason": reason, "count": 0, "scope": ep.get("scope", scope)})
        patterns[key]["count"] += 1

    existing_texts = belief_system.get_rules()
    promoted = 0

    for p in sorted(patterns.values(), key=lambda x: -x["count"]):
        if promoted >= MAX_RULES_PER_CONSOLIDATION:
            break
        if p["count"] < MIN_PATTERN_COUNT:
            continue
        if any(p["reason"][:50].lower() in t.lower() for t in existing_texts):
            continue

        rule_text = (
            f"Known failure mode of {p['tool']} (seen {p['count']}x): "
            f"{p['reason'][:160]} -- check for this before repeating the action."
        )

        if llm_fn is not None:
            abstracted = await _abstract_failure_pattern(p["tool"], p["reason"], p["count"], llm_fn)
            if abstracted:
                rule_text += f" GENERAL PATTERN: {abstracted}"

        # Contradiction check against existing rules
        contradiction_detected = False
        for ex in existing_texts:
            if p["tool"].lower() in ex.lower():
                # Opposing polarity check (e.g. required vs prohibited)
                if ("always" in ex.lower() and "avoid" in rule_text.lower()) or \
                   ("enable" in ex.lower() and "disable" in rule_text.lower()):
                    contradiction_detected = True
                    rule_text = f"[SPECIALIZED] {rule_text} (supersedes conflicting prior guidance: '{ex[:80]}...')"
                    break

        belief_system.add_candidate_rule(
            rule_text,
            confidence=0.50 if contradiction_detected else 0.60,
            scope=p.get("scope", scope)
        )
        promoted += 1
        logger.info(f"Promoted candidate rule: {rule_text[:140]}")

    return {"patterns_seen": len(patterns), "rules_promoted": promoted}


async def _abstract_failure_pattern(tool: str, reason: str, count: int, llm_fn: Callable) -> Optional[str]:
    try:
        messages = [
            {"role": "system", "content": (
                "An agent's tool call failed repeatedly with a similar error. "
                "In ONE concise sentence, state the GENERAL PATTERN behind the failure "
                "(not the specific file/value/argument from this instance) "
                "and in ONE sentence give short, general, actionable GUIDANCE to avoid it.")},
            {"role": "user", "content": f"Tool: {tool}\nSeen {count} times. Error:\n{reason[:400]}"},
        ]
        response = await llm_fn(messages) if hasattr(llm_fn, "__call__") else llm_fn(messages)
        if isinstance(response, str):
            text = response.strip()
        elif hasattr(response, "choices"):
            text = response.choices[0].message.content.strip()
        elif isinstance(response, dict):
            text = (response.get("content") or response.get("text") or "").strip()
        else:
            text = str(response).strip()
        return text[:360] if text else None
    except Exception as e:
        logger.debug(f"Failure-pattern abstraction skipped: {e}")
        return None
