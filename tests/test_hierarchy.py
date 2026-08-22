"""
Unit tests for ConceptHierarchy online clustering and closed-loop recall integration.
"""
from pathlib import Path
import numpy as np
import pytest

from agent_sleep import AgentMemory, ConceptHierarchy, SleepConsolidator
from agent_sleep.hierarchy import get_hierarchy
from agent_sleep.storage.embeddings import embed


@pytest.fixture
def temp_npz(tmp_path):
    return str(tmp_path / "hierarchy.npz")


def test_concept_hierarchy_clustering(temp_npz):
    ch = ConceptHierarchy(save_path=temp_npz)

    v1 = embed("Bash error: directory not found during build")
    v2 = embed("Bash error: file not found during compile")
    v3 = embed("Postgres database connection timeout on port 5432")

    # Add failure observations
    ch.add_memory(v1, outcome=0.0, example="Bash directory missing")
    ch.add_memory(v2, outcome=0.0, example="Bash file missing")
    ch.add_memory(v3, outcome=1.0, example="Postgres connection OK")

    assert ch.get_stats()["l1_total_memories"] == 3

    # Query similar bash task
    q_v = embed("Bash error: path does not exist")
    res = ch.query(q_v, level=1, min_similarity=0.40)
    assert res is not None
    assert res["outcome_observations"] >= 1
    assert res["mean_outcome"] is not None

    # Save and reload
    ch.save()
    ch2 = ConceptHierarchy(save_path=temp_npz)
    ch2.load()
    assert ch2.get_stats()["l1_total_memories"] == 3


def test_hierarchy_sleep_closed_loop(tmp_path):
    db_file = tmp_path / "hier_db.db"
    memory = AgentMemory(session_id="s_hier", scope="global", db_path=db_file)
    consolidator = SleepConsolidator(scope="global", db_path=db_file)

    # Record multiple episodes
    memory.record_episode(
        goal="Run bash script to initialize docker environment",
        action="execute_shell_script('init.sh')",
        outcome="failure",
        failure_reason="Command not found",
    )
    memory.record_episode(
        goal="Run bash script to start background services",
        action="execute_shell_script('start.sh')",
        outcome="failure",
        failure_reason="Permission denied",
    )

    consolidator.run(session_id="s_hier")

    # Query structured memory
    structured = memory.recall_structured("Run bash script to deploy container")
    assert structured["cluster_insight"] is not None
    assert structured["cluster_insight"]["outcome_observations"] >= 1
