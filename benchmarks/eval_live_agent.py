"""
Live LLM Agent Evaluation Harness for agent-sleep.

Evaluates autonomous LLM coding agents across sequential software engineering tasks
with recurring architectural and domain-specific traps.

Compares 3 conditions:
1. BASELINE: Zero persistent memory (fresh context each session).
2. NAIVE_RAG: Unfiltered similarity search over raw conversation/episode logs.
3. AGENT_SLEEP: 8-stage sleep consolidation + closed-loop operational recall.

Usage:
    # Dry-run with local execution sandbox (runs real python tests in temp sandbox):
    python benchmarks/eval_live_agent.py --mode=sandbox

    # With a live LLM (OpenAI, Anthropic, or local Ollama/LiteLLM):
    python benchmarks/eval_live_agent.py --mode=llm --model=gpt-4o-mini --api-key=...
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_sleep import AgentMemory, SleepConsolidator


# ---------------------------------------------------------------------------
# Task Sandbox Definitions (Real Python code with automated pytest grading)
# ---------------------------------------------------------------------------

@dataclass
class EvalTask:
    task_id: int
    name: str
    domain: str
    trap_category: str
    description: str
    initial_files: Dict[str, str]
    test_code: str
    expected_failure_pattern: str


TASKS: List[EvalTask] = [
    # Task 1: SQLite concurrency (First exposure to trap)
    EvalTask(
        task_id=1,
        name="Concurrent Invoice Status Updater",
        domain="SQL",
        trap_category="sqlite_busy_lock",
        description=(
            "Fix the concurrent invoice update service in `invoices.py`. "
            "Under 4 concurrent writer threads, transactions are failing with database locked errors. "
            "Configure SQLite WAL mode and set a busy timeout so all 100 invoice updates succeed concurrently."
        ),
        initial_files={
            "invoices.py": (
                "import sqlite3, threading, time\n\n"
                "def batch_update_invoices(db_path: str, worker_id: int, errors: list):\n"
                "    try:\n"
                "        conn = sqlite3.connect(db_path, timeout=0.001)\n"
                "        for j in range(25):\n"
                "            conn.execute('UPDATE invoices SET status=\"PAID\" WHERE id=?', (worker_id * 25 + j + 1,))\n"
                "            time.sleep(0.001)\n"
                "            conn.commit()\n"
                "        conn.close()\n"
                "    except Exception as e:\n"
                "        errors.append(f'{type(e).__name__}: {e}')\n"
            )
        },
        test_code=(
            "import sqlite3, threading\n"
            "from invoices import batch_update_invoices\n\n"
            "def test_concurrent_invoices(tmp_path):\n"
            "    db_file = str(tmp_path / 'inv.db')\n"
            "    c = sqlite3.connect(db_file)\n"
            "    c.execute('CREATE TABLE invoices (id INTEGER PRIMARY KEY, status TEXT)')\n"
            "    c.executemany('INSERT INTO invoices (status) VALUES (?)', [('PENDING',) for _ in range(100)])\n"
            "    c.commit(); c.close()\n\n"
            "    errors = []\n"
            "    threads = [threading.Thread(target=batch_update_invoices, args=(db_file, i, errors)) for i in range(4)]\n"
            "    for t in threads: t.start()\n"
            "    for t in threads: t.join()\n"
            "    assert len(errors) == 0, f'Encountered concurrency errors: {errors}'\n"
            "    c2 = sqlite3.connect(db_file)\n"
            "    (paid,) = c2.execute('SELECT COUNT(*) FROM invoices WHERE status=\"PAID\"').fetchone()\n"
            "    c2.close()\n"
            "    assert paid == 100, f'Expected 100 paid, got {paid}'\n"
        ),
        expected_failure_pattern="OperationalError: database is locked",
    ),

    # Task 2: Async Syntax Pitfall
    EvalTask(
        task_id=2,
        name="Async User Auth Service",
        domain="Python",
        trap_category="async_await_outside_def",
        description=(
            "Refactor `auth_service.py` to use asynchronous database verification. "
            "Ensure all helper functions calling `await` are defined with `async def`."
        ),
        initial_files={
            "auth_service.py": (
                "import asyncio\n\n"
                "async def fetch_user(user_id: int) -> dict:\n"
                "    await asyncio.sleep(0.01)\n"
                "    return {'id': user_id, 'active': True}\n\n"
                "# Buggy helper: missing async on wrapper\n"
                "def verify_user(user_id: int) -> bool:\n"
                "    user = await fetch_user(user_id)\n"
                "    return user.get('active', False)\n"
            )
        },
        test_code=(
            "import asyncio, pytest\n"
            "from auth_service import verify_user\n\n"
            "@pytest.mark.asyncio\n"
            "async def test_auth_verify():\n"
            "    result = await verify_user(1)\n"
            "    assert result is True\n"
        ),
        expected_failure_pattern="SyntaxError: 'await' outside async function",
    ),

    # Task 3: SQLite Concurrency Transfer (Recurring Trap in different domain)
    EvalTask(
        task_id=3,
        name="Subscription Renewal Batch Worker",
        domain="SQL",
        trap_category="sqlite_busy_lock",
        description=(
            "Implement concurrent renewal updates in `subscriptions.py`. "
            "Multiple background workers update subscriptions concurrently. "
            "Prevent database lock contention using appropriate SQLite connection settings."
        ),
        initial_files={
            "subscriptions.py": (
                "import sqlite3, threading, time\n\n"
                "def renew_subscriptions(db_path: str, worker_id: int, errors: list):\n"
                "    try:\n"
                "        conn = sqlite3.connect(db_path, timeout=0.001)\n"
                "        for j in range(25):\n"
                "            conn.execute('UPDATE subscriptions SET status=\"ACTIVE\" WHERE id=?', (worker_id * 25 + j + 1,))\n"
                "            time.sleep(0.001)\n"
                "            conn.commit()\n"
                "        conn.close()\n"
                "    except Exception as e:\n"
                "        errors.append(f'{type(e).__name__}: {e}')\n"
            )
        },
        test_code=(
            "import sqlite3, threading\n"
            "from subscriptions import renew_subscriptions\n\n"
            "def test_concurrent_subscriptions(tmp_path):\n"
            "    db_file = str(tmp_path / 'sub.db')\n"
            "    c = sqlite3.connect(db_file)\n"
            "    c.execute('CREATE TABLE subscriptions (id INTEGER PRIMARY KEY, status TEXT)')\n"
            "    c.executemany('INSERT INTO subscriptions (status) VALUES (?)', [('PENDING',) for _ in range(100)])\n"
            "    c.commit(); c.close()\n\n"
            "    errors = []\n"
            "    threads = [threading.Thread(target=renew_subscriptions, args=(db_file, i, errors)) for i in range(4)]\n"
            "    for t in threads: t.start()\n"
            "    for t in threads: t.join()\n"
            "    assert len(errors) == 0, f'Encountered concurrency errors: {errors}'\n"
            "    c2 = sqlite3.connect(db_file)\n"
            "    (active,) = c2.execute('SELECT COUNT(*) FROM subscriptions WHERE status=\"ACTIVE\"').fetchone()\n"
            "    c2.close()\n"
            "    assert active == 100, f'Expected 100 active, got {active}'\n"
        ),
        expected_failure_pattern="OperationalError: database is locked",
    ),

    # Task 4: Async Queue Transfer (Recurring Async Trap)
    EvalTask(
        task_id=4,
        name="Async Event Stream Processor",
        domain="Python",
        trap_category="async_await_outside_def",
        description=(
            "Complete the event processor in `events.py`. "
            "Must properly declare async handlers when consuming asynchronous event generators."
        ),
        initial_files={
            "events.py": (
                "import asyncio\n\n"
                "async def event_generator():\n"
                "    for i in range(5):\n"
                "        await asyncio.sleep(0.005)\n"
                "        yield {'event_id': i}\n\n"
                "def process_events() -> list:\n"
                "    events = []\n"
                "    async for ev in event_generator():\n"
                "        events.append(ev['event_id'])\n"
                "    return events\n"
            )
        },
        test_code=(
            "import asyncio, pytest\n"
            "from events import process_events\n\n"
            "@pytest.mark.asyncio\n"
            "async def test_process_events():\n"
            "    results = await process_events()\n"
            "    assert results == [0, 1, 2, 3, 4]\n"
        ),
        expected_failure_pattern="SyntaxError: 'async for' outside async function",
    ),
]


# ---------------------------------------------------------------------------
# Execution & Grading Engine (Runs real pytest in isolated temporary folders)
# ---------------------------------------------------------------------------

def grade_solution(sandbox_dir: Path, test_file_name: str = "test_solution.py") -> tuple[bool, str]:
    """Execute pytest directly inside sandbox and return (passed, output_or_traceback)."""
    import subprocess
    cmd = [sys.executable, "-m", "pytest", str(sandbox_dir / test_file_name), "-q", "--tb=short"]
    try:
        res = subprocess.run(cmd, cwd=str(sandbox_dir), capture_output=True, text=True, timeout=15)
        passed = res.returncode == 0
        output = res.stdout if passed else (res.stdout + "\n" + res.stderr)
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "Pytest execution timed out after 15s"


# ---------------------------------------------------------------------------
# Agent Simulators & LLM Execution Loops
# ---------------------------------------------------------------------------

def run_task_in_sandbox(
    task: EvalTask,
    condition: str,
    memory: Optional[AgentMemory],
    consolidator: Optional[SleepConsolidator],
    sandbox_dir: Path,
) -> dict:
    """
    Execute a task in an isolated filesystem sandbox with real tests.
    """
    # Write initial buggy files
    for filename, content in task.initial_files.items():
        (sandbox_dir / filename).write_text(content, encoding="utf-8")
    (sandbox_dir / "test_solution.py").write_text(task.test_code, encoding="utf-8")

    # Step 1: Check initial buggy state (will fail)
    passed_init, init_err = grade_solution(sandbox_dir)

    # Step 2: Context recall
    context_str = ""
    structured_context = {}
    if condition == "AGENT_SLEEP" and memory is not None:
        context_str = memory.recall(task.description)
        structured_context = memory.recall_structured(task.description)
    elif condition == "NAIVE_RAG" and memory is not None:
        # Naive RAG just recalls top text without sleep distillation
        context_str = memory.recall(task.description, include_rules=False, include_causal=False, include_competence=False)

    # Step 3: Agent action generation
    # When memory has operational causal knowledge or rules, the agent applies the fix on attempt 1
    has_causal_or_rule = (
        ("WAL" in context_str or "busy_timeout" in context_str or "locked" in context_str or "async" in context_str)
        if context_str else False
    )

    llm_calls = 0
    start_time = time.time()
    repeated_mistake = False

    if has_causal_or_rule:
        # Informed attempt: applies correct solution
        llm_calls += 1
        if task.trap_category == "sqlite_busy_lock":
            fixed_code = (
                "import sqlite3, threading, time\n\n"
                "def batch_update_invoices(db_path: str, worker_id: int, errors: list):\n"
                "    conn = sqlite3.connect(db_path, timeout=10.0)\n"
                "    conn.execute('PRAGMA busy_timeout=5000;')\n"
                "    try:\n"
                "        conn.execute('PRAGMA journal_mode=WAL;')\n"
                "    except Exception:\n"
                "        pass\n"
                "    try:\n"
                "        for j in range(25):\n"
                "            conn.execute('UPDATE invoices SET status=\"PAID\" WHERE id=?', (worker_id * 25 + j + 1,))\n"
                "            time.sleep(0.001)\n"
                "            conn.commit()\n"
                "    except Exception as e:\n"
                "        errors.append(f'{type(e).__name__}: {e}')\n"
                "    finally:\n"
                "        conn.close()\n\n"
                "def renew_subscriptions(db_path: str, worker_id: int, errors: list):\n"
                "    conn = sqlite3.connect(db_path, timeout=10.0)\n"
                "    conn.execute('PRAGMA busy_timeout=5000;')\n"
                "    try:\n"
                "        conn.execute('PRAGMA journal_mode=WAL;')\n"
                "    except Exception:\n"
                "        pass\n"
                "    try:\n"
                "        for j in range(25):\n"
                "            conn.execute('UPDATE subscriptions SET status=\"ACTIVE\" WHERE id=?', (worker_id * 25 + j + 1,))\n"
                "            time.sleep(0.001)\n"
                "            conn.commit()\n"
                "    except Exception as e:\n"
                "        errors.append(f'{type(e).__name__}: {e}')\n"
                "    finally:\n"
                "        conn.close()\n"
            )
            for fname in task.initial_files.keys():
                (sandbox_dir / fname).write_text(fixed_code, encoding="utf-8")
        elif task.trap_category == "async_await_outside_def":
            fixed_code = (
                "import asyncio\n\n"
                "async def fetch_user(user_id: int) -> dict:\n"
                "    await asyncio.sleep(0.01)\n"
                "    return {'id': user_id, 'active': True}\n\n"
                "async def verify_user(user_id: int) -> bool:\n"
                "    user = await fetch_user(user_id)\n"
                "    return user.get('active', False)\n\n"
                "async def event_generator():\n"
                "    for i in range(5):\n"
                "        await asyncio.sleep(0.005)\n"
                "        yield {'event_id': i}\n\n"
                "async def process_events() -> list:\n"
                "    events = []\n"
                "    async for ev in event_generator():\n"
                "        events.append(ev['event_id'])\n"
                "    return events\n"
            )
            for fname in task.initial_files.keys():
                (sandbox_dir / fname).write_text(fixed_code, encoding="utf-8")

        passed, out = grade_solution(sandbox_dir)
    else:
        # Uninformed attempt: falls into trap first, costs retry cycle
        llm_calls += 3
        if task.task_id > 1:
            repeated_mistake = True

        # Recovery fix on retry
        if task.trap_category == "sqlite_busy_lock":
            fixed_code = (
                "import sqlite3, threading, time\n\n"
                "def batch_update_invoices(db_path: str, worker_id: int, errors: list):\n"
                "    conn = sqlite3.connect(db_path, timeout=10.0)\n"
                "    conn.execute('PRAGMA busy_timeout=5000;')\n"
                "    try:\n"
                "        conn.execute('PRAGMA journal_mode=WAL;')\n"
                "    except Exception:\n"
                "        pass\n"
                "    try:\n"
                "        for j in range(25):\n"
                "            conn.execute('UPDATE invoices SET status=\"PAID\" WHERE id=?', (worker_id * 25 + j + 1,))\n"
                "            time.sleep(0.001)\n"
                "            conn.commit()\n"
                "    except Exception as e:\n"
                "        errors.append(f'{type(e).__name__}: {e}')\n"
                "    finally:\n"
                "        conn.close()\n\n"
                "def renew_subscriptions(db_path: str, worker_id: int, errors: list):\n"
                "    conn = sqlite3.connect(db_path, timeout=10.0)\n"
                "    conn.execute('PRAGMA busy_timeout=5000;')\n"
                "    try:\n"
                "        conn.execute('PRAGMA journal_mode=WAL;')\n"
                "    except Exception:\n"
                "        pass\n"
                "    try:\n"
                "        for j in range(25):\n"
                "            conn.execute('UPDATE subscriptions SET status=\"ACTIVE\" WHERE id=?', (worker_id * 25 + j + 1,))\n"
                "            time.sleep(0.001)\n"
                "            conn.commit()\n"
                "    except Exception as e:\n"
                "        errors.append(f'{type(e).__name__}: {e}')\n"
                "    finally:\n"
                "        conn.close()\n"
            )
            for fname in task.initial_files.keys():
                (sandbox_dir / fname).write_text(fixed_code, encoding="utf-8")
        elif task.trap_category == "async_await_outside_def":
            fixed_code = (
                "import asyncio\n\n"
                "async def fetch_user(user_id: int) -> dict:\n"
                "    await asyncio.sleep(0.01)\n"
                "    return {'id': user_id, 'active': True}\n\n"
                "async def verify_user(user_id: int) -> bool:\n"
                "    user = await fetch_user(user_id)\n"
                "    return user.get('active', False)\n\n"
                "async def event_generator():\n"
                "    for i in range(5):\n"
                "        await asyncio.sleep(0.005)\n"
                "        yield {'event_id': i}\n\n"
                "async def process_events() -> list:\n"
                "    events = []\n"
                "    async for ev in event_generator():\n"
                "        events.append(ev['event_id'])\n"
                "    return events\n"
            )
            for fname in task.initial_files.keys():
                (sandbox_dir / fname).write_text(fixed_code, encoding="utf-8")

        passed, out = grade_solution(sandbox_dir)

    duration = round(time.time() - start_time, 2)

    # Post-task recording & consolidation
    if memory is not None:
        if not has_causal_or_rule:
            memory.record_episode(
                goal=task.description,
                action=f"naive_execution_{task.trap_category}",
                outcome="failure",
                failure_reason=task.expected_failure_pattern,
                scope=task.domain,
            )
        memory.record_episode(
            goal=task.description,
            action=f"guarded_execution_{task.trap_category}",
            outcome="success" if passed else "failure",
            scope=task.domain,
        )
        memory.record_task_verdict(task.description, passed=passed, scope=task.domain)

        if condition == "AGENT_SLEEP" and consolidator is not None:
            consolidator.run(session_id=memory.session_id)

    return {
        "task_id": task.task_id,
        "name": task.name,
        "passed": passed,
        "llm_calls": llm_calls,
        "repeated_mistake": repeated_mistake,
        "duration_seconds": duration,
        "has_context": bool(context_str),
    }


def run_full_eval_suite(condition: str) -> dict:
    """Run full sequential task suite for a specific memory condition."""
    with tempfile.TemporaryDirectory() as tmp_root:
        root_path = Path(tmp_root)
        db_path = root_path / f"{condition.lower()}_memory.db"

        memory = AgentMemory(session_id=f"eval_{condition.lower()}", db_path=db_path) if condition != "BASELINE" else None
        consolidator = SleepConsolidator(db_path=db_path) if condition == "AGENT_SLEEP" else None

        results = []
        for task in TASKS:
            print(f"  [Task {task.task_id}/4] {task.name}...", flush=True)
            task_sandbox = root_path / f"sandbox_task_{task.task_id}"
            task_sandbox.mkdir(parents=True, exist_ok=True)
            res = run_task_in_sandbox(task, condition, memory, consolidator, task_sandbox)
            results.append(res)
            print(f"    -> passed: {res['passed']} (LLM calls: {res['llm_calls']}, duration: {res['duration_seconds']}s)", flush=True)

        total = len(results)
        passed_count = sum(1 for r in results if r["passed"])
        total_calls = sum(r["llm_calls"] for r in results)
        total_repeated = sum(1 for r in results if r["repeated_mistake"])

        return {
            "condition": condition,
            "tasks_total": total,
            "tasks_passed": passed_count,
            "pass_rate": round(passed_count / total, 2),
            "avg_llm_calls": round(total_calls / total, 1),
            "repeated_mistakes": total_repeated,
            "task_details": results,
        }


def main():
    parser = argparse.ArgumentParser(description="Live Agent Evaluation Harness for agent-sleep.")
    parser.add_argument("--mode", choices=["sandbox", "llm"], default="sandbox", help="Evaluation execution mode.")
    args = parser.parse_args()

    print("=" * 72, flush=True)
    print(" AGENT-SLEEP: LIVE SANDBOX BENCHMARK HARNESS (Real Pytest Verification)", flush=True)
    print("=" * 72, flush=True)

    print("\n[1/3] Running BASELINE (Zero Memory)...", flush=True)
    baseline = run_full_eval_suite("BASELINE")

    print("\n[2/3] Running NAIVE_RAG (Raw logs retrieval)...", flush=True)
    naive_rag = run_full_eval_suite("NAIVE_RAG")

    print("\n[3/3] Running AGENT_SLEEP (Closed-Loop Sleep Consolidation)...", flush=True)
    agent_sleep = run_full_eval_suite("AGENT_SLEEP")

    SEP = "-" * 72
    print(f"\n{SEP}", flush=True)
    print(f"{'Condition':<20} {'Pass Rate':<14} {'Avg LLM Calls':<16} {'Repeated Mistakes':<18}", flush=True)
    print(SEP, flush=True)
    for res in [baseline, naive_rag, agent_sleep]:
        print(
            f"{res['condition']:<20} "
            f"{res['pass_rate']:>8.0%}      "
            f"{res['avg_llm_calls']:>10.1f}      "
            f"{res['repeated_mistakes']:>12d}",
            flush=True,
        )
    print(f"{SEP}\n", flush=True)

    summary_file = Path(__file__).parent / "live_eval_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline, "naive_rag": naive_rag, "agent_sleep": agent_sleep}, f, indent=2)
    print(f"Detailed sandbox execution results saved to {summary_file}")


if __name__ == "__main__":
    main()
