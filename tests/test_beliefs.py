"""
Tests for the belief promotion system.
"""
import pytest
import asyncio
from agent_sleep.beliefs import BeliefSystem, promote_candidate_rules


def _ep(**kwargs):
    defaults = {
        "goal": "Refactor the authentication module to use JWT tokens",
        "action": "edit_file('auth.py')",
        "outcome": "failure",
        "failure_reason": "SyntaxError: invalid syntax at line 42",
        "episode_kind": "action",
    }
    defaults.update(kwargs)
    return defaults


def _run(coro):
    return asyncio.run(coro)


def test_no_rules_promoted_for_single_failure():
    """A pattern seen only once should NOT be promoted."""
    episodes = [_ep()]
    saved = []

    class FakeBeliefSystem:
        beliefs = {}
        def add_candidate_rule(self, belief, confidence=0.5, **kwargs):
            saved.append(belief)
        def get_rules(self, *args, **kwargs):
            return []

    report = _run(promote_candidate_rules(episodes, belief_system=FakeBeliefSystem()))
    assert report["rules_promoted"] == 0
    assert len(saved) == 0


def test_rule_promoted_for_repeated_failure():
    """A pattern seen 2+ times should be promoted."""
    action = "edit_file('auth.py')"
    reason = "SyntaxError: invalid syntax at line 42"
    episodes = [
        _ep(action=action, failure_reason=reason),
        _ep(action=action, failure_reason=reason),
    ]
    saved = []

    class FakeBeliefSystem:
        beliefs = {}
        def add_candidate_rule(self, belief, confidence=0.5, **kwargs):
            saved.append(belief)
        def get_rules(self, *args, **kwargs):
            return []

    report = _run(promote_candidate_rules(episodes, belief_system=FakeBeliefSystem()))
    assert report["rules_promoted"] == 1
    assert len(saved) == 1
    assert "SyntaxError" in saved[0]
    assert "seen 2x" in saved[0]


def test_max_two_rules_per_cycle():
    """At most 2 rules should be promoted per consolidation cycle."""
    episodes = []
    for i in range(3):
        for _ in range(2):
            episodes.append(_ep(
                action=f"tool_{i}",
                failure_reason=f"Error type {i}: something went wrong at position {i}"
            ))

    saved = []

    class FakeBeliefSystem:
        beliefs = {}
        def add_candidate_rule(self, belief, confidence=0.5, **kwargs):
            saved.append(belief)
        def get_rules(self, *args, **kwargs):
            return []

    report = _run(promote_candidate_rules(episodes, belief_system=FakeBeliefSystem()))
    assert report["rules_promoted"] <= 2
    assert len(saved) <= 2


def test_no_dedup_of_existing_rule():
    """Should not add a rule that's already in the belief system."""
    reason = "SyntaxError: invalid syntax at line 42"
    episodes = [
        _ep(failure_reason=reason),
        _ep(failure_reason=reason),
    ]
    saved = []

    class FakeBeliefSystem:
        beliefs = {}
        def add_candidate_rule(self, belief, confidence=0.5):
            saved.append(belief)
        def get_rules(self):
            # Return a rule that already covers this failure
            return [f"Known failure mode: SyntaxError: invalid syntax at line 42 -- check for this"]

    report = _run(promote_candidate_rules(episodes, belief_system=FakeBeliefSystem()))
    assert report["rules_promoted"] == 0


def test_verdict_rows_excluded_from_rule_promotion():
    """task_verdict rows should never contribute to rule promotion."""
    episodes = [
        _ep(outcome="failure", failure_reason="SomeError", episode_kind="task_verdict"),
        _ep(outcome="failure", failure_reason="SomeError", episode_kind="task_verdict"),
    ]
    saved = []

    class FakeBeliefSystem:
        beliefs = {}
        def add_candidate_rule(self, belief, confidence=0.5):
            saved.append(belief)
        def get_rules(self):
            return []

    report = _run(promote_candidate_rules(episodes, belief_system=FakeBeliefSystem()))
    assert report["rules_promoted"] == 0


def test_contradiction_specialization():
    """Conflicting rule should be marked as specialized rather than blindly added."""
    action = "run_query(sql)"
    reason = "avoid raw string concatenation in queries"
    episodes = [
        _ep(action=action, failure_reason=reason),
        _ep(action=action, failure_reason=reason),
    ]
    saved = []

    class FakeBeliefSystem:
        beliefs = {}
        def add_candidate_rule(self, belief, confidence=0.5, **kwargs):
            saved.append(belief)
        def get_rules(self, *args, **kwargs):
            return ["Always use run_query for database execution"]

    report = _run(promote_candidate_rules(episodes, belief_system=FakeBeliefSystem()))
    assert report["rules_promoted"] == 1
    assert len(saved) == 1
    assert "[SPECIALIZED]" in saved[0]
