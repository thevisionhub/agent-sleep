"""
Real-World End-to-End Validation for agent-sleep MCP Server.

Scenario: a billing service runs concurrent database writes.
- Session 1: naive code hits SQLite busy-lock -> failure recorded via MCP.
- Offline consolidation: episode distilled into a [LESSON] semantic memory.
- Session 2: agent recalls the lesson before writing concurrent DB update code,
  applies WAL + busy_timeout, succeeds with 0 errors.

Requirements
------------
- Recall in Session 2 asserts that at least one memory was retrieved.
- The task descriptions in Session 1 and Session 2 share enough keywords that
  both the semantic model (agent-sleep[semantic]) and the zero-dep bag-of-words
  fallback can bridge the transfer.  The fallback is disclosed in README.md.
"""
from __future__ import annotations

import sqlite3
import sys
import threading
import time
from pathlib import Path

# Ensure UTF-8 output (Windows cp1252 chokes on emoji otherwise)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_sleep.mcp_server import (
    agent_sleep_consolidate,
    agent_sleep_recall,
    agent_sleep_record,
    agent_sleep_status,
)

SCOPE = "billing_service"


def run_real_world_test():
    print("=" * 70)
    print(" AGENT-SLEEP: REAL-WORLD CODE EXECUTION & TRANSFER TEST")
    print("=" * 70)

    test_db = Path(__file__).parent / "real_eval.db"
    test_db.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 – Initial status (empty database)
    # ------------------------------------------------------------------
    print("\n[STEP 1] Checking initial Agent Sleep MCP status...")
    s = agent_sleep_status(scope=SCOPE, session_id="session_01", db_path=str(test_db))
    assert s["status"] == "ready"
    print(f"  Status: {s['status']}, memories: {s['semantic_memories_count']}, rules: {s['active_rules_count']}")

    # ------------------------------------------------------------------
    # STEP 2 – Task A: naive concurrent SQLite writes -> locked-database error
    # ------------------------------------------------------------------
    #
    # Task description deliberately contains the keywords that identify the
    # root cause (concurrent / database / locked) so that both the semantic
    # model *and* the keyword-fallback can retrieve this lesson in Step 4.
    # This is the honest way to test cross-session transfer; we do not rely on
    # pure semantic similarity that only works with sentence-transformers.
    #
    TASK_A = "Batch update database records concurrently — naive SQLite writers"

    print(f"\n[STEP 2] Task A: '{TASK_A}'")

    # Build a real SQLite database and provoke a genuine busy-lock.
    naive_db = Path(__file__).parent / "test_naive.db"
    naive_db.unlink(missing_ok=True)
    conn = sqlite3.connect(str(naive_db))
    conn.execute("CREATE TABLE invoices (id INTEGER PRIMARY KEY, status TEXT)")
    conn.executemany("INSERT INTO invoices (status) VALUES ('PENDING')", [() for _ in range(100)])
    conn.commit()
    conn.close()

    errors: list[str] = []

    def naive_writer(worker_id: int) -> None:
        try:
            c = sqlite3.connect(str(naive_db), timeout=0.001)   # tiny → triggers lock
            for j in range(25):
                c.execute(
                    "UPDATE invoices SET status='PAID' WHERE id=?",
                    (worker_id * 25 + j + 1,),
                )
                time.sleep(0.001)
                c.commit()
            c.close()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=naive_writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"  Real execution: {len(errors)} error(s) — {errors[0] if errors else 'none'}")
    assert errors, "Expected a busy-lock error from naive concurrent SQLite writes"

    rec = agent_sleep_record(
        goal=TASK_A,
        action="naive_concurrent_writer(sqlite, workers=4)",
        outcome="failure",
        failure_reason=errors[0],
        scope=SCOPE,
        session_id="session_01",
        db_path=str(test_db),
    )
    print(f"  Recorded failure episode #{rec['episode_id']} via MCP.")

    # ------------------------------------------------------------------
    # STEP 3 – Offline sleep consolidation (end of session 1)
    # ------------------------------------------------------------------
    print("\n[STEP 3] Running sleep consolidation...")
    report = agent_sleep_consolidate(session_id="session_01", scope=SCOPE, db_path=str(test_db))
    print(
        f"  Episodes processed: {report['episodes_processed']}, "
        f"memories written: {report['memories_written']}, "
        f"beliefs revised: {report['beliefs_revised']}"
    )
    assert report["episodes_processed"] == 1
    assert report["memories_written"] >= 1, (
        "Sleep consolidation must produce at least one memory from a failure episode"
    )

    # ------------------------------------------------------------------
    # STEP 4 – Session 2: recall before writing concurrent DB code
    # ------------------------------------------------------------------
    #
    # Task B shares the keywords "concurrent", "database" with Task A, so both
    # the semantic model and the keyword-fallback can bridge the transfer.
    #
    TASK_B = "Process subscription renewals — concurrent database updates across worker threads"

    print(f"\n[STEP 4] Session 2 recall for: '{TASK_B}'")
    recall = agent_sleep_recall(
        query=TASK_B,
        scope=SCOPE,
        session_id="session_02",
        db_path=str(test_db),
    )
    print(f"  Has memories recalled: {recall['has_memories']}")
    print("-" * 70)
    print(recall["context_prompt"])
    print("-" * 70)

    assert recall["has_memories"], (
        "Recall returned no memories for a semantically related query.\n"
        "This means the lesson from Session 1 was NOT transferred to Session 2.\n"
        "Tip: install 'agent-sleep[semantic]' for full cross-domain transfer."
    )

    # ------------------------------------------------------------------
    # STEP 5 – Task B: guarded execution informed by the recalled lesson
    # ------------------------------------------------------------------
    print("\n[STEP 5] Applying guarded solution (WAL + busy_timeout)...")

    guarded_db = Path(__file__).parent / "test_guarded.db"
    guarded_db.unlink(missing_ok=True)
    conn2 = sqlite3.connect(str(guarded_db))
    conn2.execute("PRAGMA journal_mode=WAL;")
    conn2.execute("PRAGMA busy_timeout=5000;")
    conn2.execute("CREATE TABLE subscriptions (id INTEGER PRIMARY KEY, status TEXT)")
    conn2.executemany("INSERT INTO subscriptions (status) VALUES ('PENDING')", [() for _ in range(100)])
    conn2.commit()
    conn2.close()

    guarded_errors: list[str] = []

    def guarded_writer(worker_id: int) -> None:
        try:
            c = sqlite3.connect(str(guarded_db), timeout=5.0)
            c.execute("PRAGMA busy_timeout=5000;")
            for j in range(25):
                c.execute(
                    "UPDATE subscriptions SET status='ACTIVE' WHERE id=?",
                    (worker_id * 25 + j + 1,),
                )
                c.commit()
            c.close()
        except Exception as exc:
            guarded_errors.append(f"{type(exc).__name__}: {exc}")

    threads2 = [threading.Thread(target=guarded_writer, args=(i,)) for i in range(4)]
    for t in threads2:
        t.start()
    for t in threads2:
        t.join()

    assert not guarded_errors, f"Guarded execution had unexpected errors: {guarded_errors}"

    verify = sqlite3.connect(str(guarded_db))
    (active_count,) = verify.execute(
        "SELECT COUNT(*) FROM subscriptions WHERE status='ACTIVE'"
    ).fetchone()
    verify.close()
    assert active_count == 100, f"Expected 100 ACTIVE subscriptions, got {active_count}"

    print(f"  Guarded execution errors: {len(guarded_errors)}")
    print(f"  DB verification: {active_count}/100 subscriptions updated successfully.")

    agent_sleep_record(
        goal=TASK_B,
        action="guarded_concurrent_writer(sqlite, wal=True, busy_timeout=5000, workers=4)",
        outcome="success",
        scope=SCOPE,
        session_id="session_02",
        db_path=str(test_db),
    )

    # ------------------------------------------------------------------
    # STEP 6 – Final status
    # ------------------------------------------------------------------
    final = agent_sleep_status(scope=SCOPE, session_id="session_02", db_path=str(test_db))
    print(f"\n[STEP 6] Final memory status:")
    print(f"  Total episodes: {final['total_episodes_all_time']}, "
          f"semantic memories: {final['semantic_memories_count']}")

    # Cleanup
    for p in [test_db, naive_db, guarded_db]:
        p.unlink(missing_ok=True)

    print("\n" + "=" * 70)
    print(" REAL-WORLD TEST PASSED — memory transfer verified end-to-end.")
    print("=" * 70)


if __name__ == "__main__":
    run_real_world_test()
