"""
Multi-agent concurrency, simultaneous consolidation, and thread-safety test suite.
"""
import concurrent.futures
import threading
import time
from pathlib import Path
import pytest

from agent_sleep import AgentMemory, SleepConsolidator
from agent_sleep.storage.db import (
    ensure_db_initialized,
    save_episode,
    save_semantic_memory,
    recall_memories,
    get_all_competencies,
    _cursor,
)


def test_concurrent_agent_episode_writes(tmp_path: Path):
    """Multiple agents recording episodes simultaneously to the same SQLite database."""
    db_file = tmp_path / "concurrent_episodes.db"
    ensure_db_initialized(db_path=db_file)

    num_agents = 8
    episodes_per_agent = 25

    def agent_worker(agent_id: int):
        memory = AgentMemory(
            session_id=f"agent_session_{agent_id}",
            scope=f"repo_{agent_id % 3}",
            db_path=db_file,
        )
        for i in range(episodes_per_agent):
            memory.record_episode(
                goal=f"Task {i} by Agent {agent_id}",
                action=f"execute_step_{i}()",
                outcome="success" if i % 2 == 0 else "failure",
                failure_reason="SimulatedTimeout" if i % 2 != 0 else "",
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as executor:
        futures = [executor.submit(agent_worker, i) for i in range(num_agents)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    with _cursor(db_path=db_file) as cur:
        cur.execute("SELECT COUNT(*) FROM execution_episodes")
        total_episodes = cur.fetchone()[0]
        assert total_episodes == num_agents * episodes_per_agent


def test_concurrent_consolidation_and_recall(tmp_path: Path):
    """One thread consolidating episodes while another thread recalls memories."""
    db_file = tmp_path / "concurrent_consolidate.db"
    ensure_db_initialized(db_path=db_file)

    # Pre-populate some memories
    for i in range(10):
        save_semantic_memory(
            f"Fact {i}",
            f"Value {i} for postgresql database pooling",
            scope="repo_a",
            db_path=db_file,
        )

    # Insert episodes to consolidate
    for i in range(10):
        save_episode(
            session_id="sess_concur",
            goal="Refactor postgresql connection pool",
            action="init_pool()",
            outcome="success",
            scope="repo_a",
            db_path=db_file,
        )

    errors = []

    def consolidator_worker():
        try:
            consolidator = SleepConsolidator(scope="repo_a", db_path=db_file)
            report = consolidator.run(session_id="sess_concur")
            assert report["episodes_processed"] == 10
        except Exception as e:
            errors.append(f"Consolidator error: {e}")

    def recall_worker(worker_id: int):
        try:
            for _ in range(20):
                recalled = recall_memories("postgresql database pooling", scopes=["repo_a"], db_path=db_file)
                assert len(recalled) > 0
                time.sleep(0.001)
        except Exception as e:
            errors.append(f"Recall worker {worker_id} error: {e}")

    threads = [
        threading.Thread(target=consolidator_worker),
        threading.Thread(target=recall_worker, args=(1,)),
        threading.Thread(target=recall_worker, args=(2,)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Encountered concurrency errors: {errors}"
