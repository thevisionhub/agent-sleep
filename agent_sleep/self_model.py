"""
Self-model: tracks the agent's own competence per domain.

After consolidation the agent knows not just WHAT it learned, but HOW GOOD
it is at different kinds of tasks. This awareness lets it hedge on uncertain
steps ("I have low accuracy in SQL queries") and be confident on proven ones.

Competence is tracked as an exponential moving average of binary outcomes
(success=1.0, failure=0.0) over time, per domain string.

Domains are coarse categories inferred from the task description:
"Python", "SQL", "file_system", "API_calls", "General", etc.
The domain classifier is a simple keyword heuristic by default;
replace it with an embedding-based classifier for better accuracy.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "Python": ["python", ".py", "def ", "class ", "import ", "pytest"],
    "SQL": ["sql", "select ", "insert ", "update ", "delete ", "join", "database"],
    "JavaScript": ["javascript", "typescript", ".js", ".ts", "node", "npm", "react"],
    "file_system": ["read_file", "write_file", "open(", "os.path", "pathlib", "mkdir", "listdir"],
    "API_calls": ["requests.", "httpx", "curl", "http", "endpoint", "api", "webhook"],
    "shell": ["subprocess", "bash", "shell", "terminal", "command", "exec("],
    "git": ["git ", "commit", "branch", "merge", "clone", "pull", "push"],
}

_EMA_ALPHA = 0.2  # exponential moving average decay


def infer_domain(text: str) -> str:
    """Infer a task domain from its text description (keyword heuristic)."""
    text_lower = text.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return domain
    return "General"


class SelfModel:
    """
    Tracks agent competence per task domain.

    In-memory state, persisted via db.update_competence on each update.
    """

    def __init__(self) -> None:
        self._competence: Dict[str, float] = {}

    def update(self, domain: str, success: bool) -> float:
        """
        Update competence for a domain after one outcome.

        Returns the new competence score (0-1).
        """
        outcome = 1.0 if success else 0.0
        current = self._competence.get(domain, 0.5)
        new_value = current * (1 - _EMA_ALPHA) + outcome * _EMA_ALPHA
        self._competence[domain] = new_value

        try:
            from agent_sleep.storage.db import update_competence
            update_competence(domain, new_value)
        except Exception as e:
            logger.warning(f"Could not persist competence update: {e}")

        logger.debug(f"Self-model: {domain} competence {current:.2f} -> {new_value:.2f}")
        return new_value

    def get_competence(self, domain: str) -> float:
        """Return the current competence score for a domain (0-1)."""
        return self._competence.get(domain, 0.5)

    def get_summary(self) -> dict:
        """Return all tracked domains and their competence scores."""
        return dict(self._competence)

    def confidence_statement(self, domain: str) -> str:
        """
        Return a natural-language confidence statement for prompt injection.
        """
        c = self.get_competence(domain)
        if c >= 0.85:
            return f"High confidence in {domain} tasks (historical accuracy: {c:.0%})."
        if c >= 0.65:
            return f"Moderate confidence in {domain} tasks (historical accuracy: {c:.0%})."
        return f"Low confidence in {domain} tasks (historical accuracy: {c:.0%}). Double-check this work."


def run_self_reflection(episodes: list) -> dict:
    """
    Process a batch of consolidated episodes to update the self-model.

    Returns a summary of competence updates.
    """
    self_model = SelfModel()
    updates: Dict[str, list] = {}

    for ep in episodes:
        goal = ep.get("goal") or ""
        action = ep.get("action") or ""
        outcome = (ep.get("outcome") or "").strip().lower()
        if outcome not in ("success", "failure"):
            continue

        domain = infer_domain(goal + " " + action)
        success = outcome == "success"
        new_score = self_model.update(domain, success)
        updates.setdefault(domain, []).append(new_score)

    return {
        "domains_updated": len(updates),
        "competence_summary": {
            domain: round(scores[-1], 3)
            for domain, scores in updates.items()
        },
    }
