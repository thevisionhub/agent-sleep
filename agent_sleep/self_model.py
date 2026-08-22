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
    Tracks agent competence per task domain and derives operational behavioral policies.

    In-memory state, persisted via db.update_competence on each update.
    Scope-isolated by default.
    """

    def __init__(self, scope: str = "global", db_path: Optional[Path] = None) -> None:
        self.scope = scope
        self.db_path = db_path
        self._competence: Dict[str, float] = {}
        self._load_from_db()

    def _load_from_db(self) -> None:
        try:
            from agent_sleep.storage.db import get_all_competencies
            self._competence = get_all_competencies(scope=self.scope, db_path=self.db_path)
        except Exception as e:
            logger.debug(f"Could not load initial competencies from DB: {e}")

    def update(self, domain: str, success: bool, scope: Optional[str] = None, db_path: Optional[Path] = None) -> float:
        """
        Update competence for a domain after one outcome.

        Returns the new competence score (0-1).
        """
        target_scope = scope or self.scope
        target_db = db_path or self.db_path
        outcome = 1.0 if success else 0.0
        current = self.get_competence(domain, scope=target_scope, db_path=target_db)
        new_value = current * (1 - _EMA_ALPHA) + outcome * _EMA_ALPHA
        self._competence[domain] = new_value

        try:
            from agent_sleep.storage.db import update_competence
            update_competence(
                domain,
                competence=new_value,
                historical_accuracy=new_value,
                success=success,
                scope=target_scope,
                db_path=target_db,
            )
        except Exception as e:
            logger.warning(f"Could not persist competence update: {e}")

        logger.debug(f"Self-model [{target_scope}]: {domain} competence {current:.2f} -> {new_value:.2f}")
        return new_value

    def get_competence(self, domain: str, scope: Optional[str] = None, db_path: Optional[Path] = None) -> float:
        """Return the current competence score for a domain (0-1)."""
        target_scope = scope or self.scope
        target_db = db_path or self.db_path
        if domain in self._competence:
            return self._competence[domain]
        try:
            from agent_sleep.storage.db import get_domain_competence
            record = get_domain_competence(domain, scope=target_scope, db_path=target_db)
            score = float(record.get("competence", 0.5))
            self._competence[domain] = score
            return score
        except Exception:
            return 0.5

    def get_domain_metrics(self, domain: str, scope: Optional[str] = None, db_path: Optional[Path] = None) -> dict:
        """Return full domain metrics including uncertainty and episode counts."""
        target_scope = scope or self.scope
        target_db = db_path or self.db_path
        try:
            from agent_sleep.storage.db import get_domain_competence
            return get_domain_competence(domain, scope=target_scope, db_path=target_db)
        except Exception:
            return {
                "scope": target_scope,
                "domain": domain,
                "competence": self.get_competence(domain, scope=target_scope, db_path=target_db),
                "uncertainty": 0.5,
                "historical_accuracy": 0.5,
                "success_count": 1,
                "failure_count": 1,
                "total_episodes": 0,
            }

    def get_behavioral_policy(self, domain: str, scope: Optional[str] = None, db_path: Optional[Path] = None) -> dict:
        """
        Derive an actionable operational policy based on self-competence and Bayesian uncertainty.
        Provides decision support for the host agent's verification intensity and retry budget.
        """
        target_scope = scope or self.scope
        metrics = self.get_domain_metrics(domain, scope=target_scope, db_path=db_path)
        c = float(metrics.get("competence", 0.5))
        unc = float(metrics.get("uncertainty", 0.5))
        total_ep = int(metrics.get("total_episodes", 0))

        # Derive policy based on competence score and uncertainty
        if total_ep == 0:
            level = "MODERATE"
            directive = f"Novel domain '{domain}' with no prior history. Standard verification and test checks recommended."
            ver_intensity = "STANDARD"
            retry_budget = 3
            autonomy = "BALANCED"
        elif c >= 0.75 and unc <= 0.40:
            level = "HIGH"
            directive = f"High historical competence in {domain} ({c:.0%}). Fast-path execution permitted."
            ver_intensity = "LIGHTWEIGHT"
            retry_budget = 2
            autonomy = "HIGH"
        elif c < 0.50:
            level = "LOW"
            directive = f"Low historical competence in {domain} ({c:.0%}). Mandatory pre-execution validation, defensive error handling, and test verification before completion."
            ver_intensity = "STRICT"
            retry_budget = 5
            autonomy = "CAUTIOUS"
        else:
            level = "MODERATE"
            directive = f"Moderate competence in {domain} ({c:.0%}). Standard error handling and testing required."
            ver_intensity = "STANDARD"
            retry_budget = 3
            autonomy = "BALANCED"

        return {
            "scope": target_scope,
            "domain": domain,
            "competence": round(c, 2),
            "uncertainty": round(unc, 2),
            "sample_size": total_ep,
            "level": level,
            "verification_intensity": ver_intensity,
            "retry_budget": retry_budget,
            "autonomy": autonomy,
            "directive": directive,
        }

    def get_summary(self) -> dict:
        """Return all tracked domains and their competence scores."""
        return dict(self._competence)

    def confidence_statement(self, domain: str, scope: Optional[str] = None, db_path: Optional[Path] = None) -> str:
        """
        Return a natural-language confidence directive for prompt injection.
        """
        target_scope = scope or self.scope
        policy = self.get_behavioral_policy(domain, scope=target_scope, db_path=db_path)
        return f"🛡 [SELF-MODEL: {policy['level']} COMPETENCE IN {domain.upper()}] Policy: {policy['directive']}"


def run_self_reflection(episodes: list, scope: str = "global", db_path: Optional[Path] = None) -> dict:
    """
    Process a batch of consolidated episodes to update the self-model.

    Returns a summary of competence updates.
    """
    self_model = SelfModel(scope=scope, db_path=db_path)
    updates: Dict[str, list] = {}

    for ep in episodes:
        goal = ep.get("goal") or ""
        action = ep.get("action") or ""
        ep_scope = ep.get("scope") or scope
        outcome = (ep.get("outcome") or "").strip().lower()
        if outcome not in ("success", "failure"):
            continue

        domain = infer_domain(goal + " " + action)
        success = outcome == "success"
        new_score = self_model.update(domain, success, scope=ep_scope, db_path=db_path)
        updates.setdefault(domain, []).append(new_score)

    return {
        "domains_updated": len(updates),
        "competence_summary": {
            domain: round(scores[-1], 3)
            for domain, scores in updates.items()
        },
    }
