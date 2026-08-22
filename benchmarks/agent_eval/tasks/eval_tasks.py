"""
Standardized Evaluation Tasks for agent-sleep Agent Evaluation Suite.
"""
from dataclasses import dataclass
from typing import Dict, List


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


EVAL_TASKS: List[EvalTask] = [
    EvalTask(
        task_id=1,
        name="Concurrent Balance Updater",
        domain="SQL",
        trap_category="sqlite_busy_lock",
        description=(
            "Fix concurrent balance updates in account_service.py. "
            "Configure SQLite WAL mode (PRAGMA journal_mode=WAL) and busy timeout (PRAGMA busy_timeout=5000) "
            "so concurrent threads succeed without database locked errors."
        ),
        initial_files={
            "account_service.py": (
                "import sqlite3, threading, time\n\n"
                "def transfer_funds(db_path: str, worker_id: int, errors: list):\n"
                "    try:\n"
                "        conn = sqlite3.connect(db_path, timeout=0.001)\n"
                "        for j in range(20):\n"
                "            conn.execute('UPDATE accounts SET balance = balance + 10 WHERE id=?', (worker_id * 20 + j + 1,))\n"
                "            time.sleep(0.001)\n"
                "            conn.commit()\n"
                "        conn.close()\n"
                "    except Exception as e:\n"
                "        errors.append(f'{type(e).__name__}: {e}')\n"
            )
        },
        test_code=(
            "import sqlite3, threading\n"
            "from account_service import transfer_funds\n\n"
            "def test_transfers(tmp_path):\n"
            "    db_file = str(tmp_path / 'acc.db')\n"
            "    c = sqlite3.connect(db_file)\n"
            "    c.execute('CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance REAL)')\n"
            "    c.executemany('INSERT INTO accounts VALUES (?, 100)', [(i+1,) for i in range(80)])\n"
            "    c.commit()\n"
            "    c.close()\n\n"
            "    errors = []\n"
            "    threads = [threading.Thread(target=transfer_funds, args=(db_file, i, errors)) for i in range(4)]\n"
            "    for t in threads: t.start()\n"
            "    for t in threads: t.join()\n"
            "    assert len(errors) == 0, f'Concurrency errors: {errors}'\n"
        ),
        expected_failure_pattern="OperationalError: database is locked",
    ),
    EvalTask(
        task_id=2,
        name="Async Event Stream Collector",
        domain="Python",
        trap_category="async_generator_syntax",
        description=(
            "Fix async stream collector in streamer.py. "
            "Iterating an async generator with synchronous for loop raises TypeError. "
            "Use async for inside an async function."
        ),
        initial_files={
            "streamer.py": (
                "import asyncio\n\n"
                "async def fetch_numbers():\n"
                "    for i in range(5):\n"
                "        await asyncio.sleep(0.001)\n"
                "        yield i\n\n"
                "def collect_all():\n"
                "    res = []\n"
                "    for x in fetch_numbers():\n"
                "        res.append(x)\n"
                "    return res\n"
            )
        },
        test_code=(
            "import pytest, asyncio\n"
            "from streamer import collect_all\n\n"
            "def test_stream():\n"
            "    res = asyncio.run(collect_all()) if asyncio.iscoroutinefunction(collect_all) else collect_all()\n"
            "    assert res == [0, 1, 2, 3, 4]\n"
        ),
        expected_failure_pattern="TypeError: 'async_generator' object is not iterable",
    ),
    EvalTask(
        task_id=3,
        name="HTTP Client with Exponential Backoff",
        domain="API_calls",
        trap_category="missing_rate_limit_retry",
        description=(
            "Fix HTTP client in client.py to catch 429 Too Many Requests, "
            "parse Retry-After header, and retry with backoff instead of crashing."
        ),
        initial_files={
            "client.py": (
                "class FakeHTTPError(Exception):\n"
                "    def __init__(self, code, headers):\n"
                "        self.status_code = code\n"
                "        self.headers = headers\n\n"
                "def fetch_data(api_fn):\n"
                "    return api_fn()\n"
            )
        },
        test_code=(
            "from client import fetch_data, FakeHTTPError\n\n"
            "def test_backoff():\n"
            "    calls = 0\n"
            "    def mock_api():\n"
            "        nonlocal calls\n"
            "        calls += 1\n"
            "        if calls == 1:\n"
            "            raise FakeHTTPError(429, {'Retry-After': '0.01'})\n"
            "        return {'status': 'ok'}\n"
            "    res = fetch_data(mock_api)\n"
            "    assert res == {'status': 'ok'}\n"
            "    assert calls == 2\n"
        ),
        expected_failure_pattern="FakeHTTPError: 429",
    ),
    EvalTask(
        task_id=4,
        name="Concurrent SQLite Settlement (Repeat Domain)",
        domain="SQL",
        trap_category="sqlite_busy_lock",
        description=(
            "Implement multi-threaded settlement processing in settlement.py "
            "with WAL journal mode and busy timeout."
        ),
        initial_files={
            "settlement.py": (
                "import sqlite3, threading, time\n\n"
                "def settle_batch(db_path: str, worker_id: int, errors: list):\n"
                "    try:\n"
                "        conn = sqlite3.connect(db_path, timeout=0.001)\n"
                "        for j in range(15):\n"
                "            conn.execute('UPDATE settlements SET settled=1 WHERE id=?', (worker_id * 15 + j + 1,))\n"
                "            time.sleep(0.001)\n"
                "            conn.commit()\n"
                "        conn.close()\n"
                "    except Exception as e:\n"
                "        errors.append(f'{type(e).__name__}: {e}')\n"
            )
        },
        test_code=(
            "import sqlite3, threading\n"
            "from settlement import settle_batch\n\n"
            "def test_settlements(tmp_path):\n"
            "    db_file = str(tmp_path / 'set.db')\n"
            "    c = sqlite3.connect(db_file)\n"
            "    c.execute('CREATE TABLE settlements (id INTEGER PRIMARY KEY, settled INT)')\n"
            "    c.executemany('INSERT INTO settlements VALUES (?, 0)', [(i+1,) for i in range(60)])\n"
            "    c.commit()\n"
            "    c.close()\n\n"
            "    errors = []\n"
            "    threads = [threading.Thread(target=settle_batch, args=(db_file, i, errors)) for i in range(4)]\n"
            "    for t in threads: t.start()\n"
            "    for t in threads: t.join()\n"
            "    assert len(errors) == 0, f'Settlement concurrency errors: {errors}'\n"
        ),
        expected_failure_pattern="OperationalError: database is locked",
    ),
]
