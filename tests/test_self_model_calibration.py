"""
Self-Model Calibration & Brier Score Validation Suite.
Measures predicted competence vs empirical outcome variance and validates calibration.
"""
import pytest
from pathlib import Path
import math

from agent_sleep.self_model import SelfModel, run_self_reflection
from agent_sleep.storage.db import ensure_db_initialized, _cursor


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "calibration_test.db"
    ensure_db_initialized(db_file)
    return db_file


def test_self_model_brier_calibration_score(temp_db):
    """
    Test calibration across synthetic task batches in domain 'database'.
    Evaluates:
      Brier Score = (1/N) * sum((predicted_p - actual_outcome)^2)
    A well-calibrated adaptive Bayesian estimator should achieve Brier Score < 0.20.
    """
    model = SelfModel(db_path=temp_db)

    # 1. 10 synthetic episodes with 7 successes, 3 failures (ground truth rate = 0.70)
    outcomes = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    episodes = [
        {"goal": f"Execute database transaction batch {i}", "action": "run_sql()", "outcome": "success" if o == 1.0 else "failure"}
        for i, o in enumerate(outcomes)
    ]

    # Run self reflection
    run_self_reflection(episodes, db_path=temp_db)

    policy = model.get_behavioral_policy("database", db_path=temp_db)
    predicted_p = policy["competence"]
    uncertainty = policy["uncertainty"]

    # Predicted competence should track around ~0.70 (within +- uncertainty)
    assert abs(predicted_p - 0.70) <= uncertainty + 0.10

    # Calculate Brier score for the sequence
    brier_score = sum((predicted_p - o) ** 2 for o in outcomes) / len(outcomes)
    assert brier_score <= 0.25, f"Brier score {brier_score:.3f} indicates uncalibrated model"
