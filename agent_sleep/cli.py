"""
agent-sleep CLI — inspect and configure your agent's memory.

Subcommands
-----------
  init    Print the mcp_config.json snippet for your platform and MCP client.
  show    Dump stored memories and rules for a scope as a readable table.
  reset   Clear all memories, rules, and episodes for a scope.

Examples
--------
  agent-sleep init
  agent-sleep show
  agent-sleep show --scope myproject
  agent-sleep reset --scope myproject
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Smart defaults — mirrors what the MCP server uses
# ---------------------------------------------------------------------------

def _default_db() -> Path:
    """Project-local DB: .agent_sleep/memory.db in cwd."""
    return Path.cwd() / ".agent_sleep" / "memory.db"


def _default_scope() -> str:
    return Path.cwd().name


# ---------------------------------------------------------------------------
# init — print mcp_config.json snippets
# ---------------------------------------------------------------------------

def _cmd_init(args) -> None:
    db_path = _default_db()
    py = sys.executable.replace("\\", "/")

    snippet_antigravity = {
        "mcpServers": {
            "agent-sleep": {
                "command": py,
                "args": ["-m", "agent_sleep.mcp_server"],
                "env": {
                    "AGENT_SLEEP_DB": str(db_path).replace("\\", "/")
                }
            }
        }
    }

    snippet_uvx = {
        "mcpServers": {
            "agent-sleep": {
                "command": "uvx",
                "args": ["--from", "git+https://github.com/thevisionhub/agent-sleep.git", "agent-sleep-mcp"],
                "env": {
                    "AGENT_SLEEP_DB": str(db_path).replace("\\", "/")
                }
            }
        }
    }

    system = platform.system()
    if system == "Darwin":
        claude_config = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        antigravity_config = Path.home() / ".gemini" / "config" / "mcp_config.json"
    elif system == "Windows":
        claude_config = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Claude" / "claude_desktop_config.json"
        antigravity_config = Path.home() / ".gemini" / "config" / "mcp_config.json"
    else:
        claude_config = Path.home() / ".config" / "claude" / "claude_desktop_config.json"
        antigravity_config = Path.home() / ".gemini" / "config" / "mcp_config.json"

    SEP = "-" * 60

    print(f"\n{SEP}")
    print("  agent-sleep MCP configuration")
    print(f"  Detected OS : {system}")
    print(f"  Project DB  : {db_path}")
    print(f"{SEP}\n")

    print("==  Option A -- uvx (zero-install, recommended for new users)  ==\n")
    print(f"Paste into: {claude_config}\n")
    print(json.dumps(snippet_uvx, indent=2))

    print(f"\n==  Option B -- local Python (use after: pip install -e .)  ==\n")
    print(f"Paste into: {antigravity_config}  (Antigravity / Cursor)\n")
    print(json.dumps(snippet_antigravity, indent=2))

    print(f"""
{SEP}
  Tip: set AGENT_SLEEP_DB to any path you prefer.
  The directory will be created automatically on first use.

  After adding the config, restart your MCP client, then ask
  your agent:  "record that we're using Python 3.11 for this project"
{SEP}
""")


# ---------------------------------------------------------------------------
# show — dump memories and rules in a readable table
# ---------------------------------------------------------------------------

def _fmt_row(cols: list[str], widths: list[int]) -> str:
    return "  ".join(str(c)[:w].ljust(w) for c, w in zip(cols, widths))


def _cmd_show(args) -> None:
    db_path = Path(args.db) if args.db else _default_db()
    scope = args.scope or _default_scope()

    if not db_path.exists():
        print(f"No memory database found at {db_path}")
        print(f"(Start using agent-sleep with scope='{scope}' to create one.)")
        return

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agent_sleep.storage.db import ensure_db_initialized, _cursor

    try:
        ensure_db_initialized(db_path)
    except Exception as e:
        print(f"Error opening database: {e}")
        return

    with _cursor(db_path=db_path) as cur:
        cur.execute(
            "SELECT memory_type, fact, value, importance, confidence, access_count "
            "FROM semantic_memories WHERE scope IN (?, 'global') ORDER BY importance DESC",
            (scope,)
        )
        memories = [dict(r) for r in cur.fetchall()]

        cur.execute(
            "SELECT belief, confidence, status, confirmations FROM candidate_rules "
            "WHERE scope IN (?, 'global') AND status != 'REFUTED' ORDER BY confidence DESC",
            (scope,)
        )
        rules = [dict(r) for r in cur.fetchall()]

        cur.execute("SELECT COUNT(*) FROM execution_episodes WHERE processed_by_sleep=0")
        pending = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM execution_episodes")
        total_ep = cur.fetchone()[0]

    SEP = "─" * 80
    print(f"\n{SEP}")
    print(f"  agent-sleep memory  |  scope: {scope}  |  db: {db_path}")
    print(f"{SEP}")
    print(f"  Episodes : {total_ep} total, {pending} pending consolidation")
    print(f"  Memories : {len(memories)}   Rules: {len(rules)}")
    print(SEP)

    if memories:
        print("\n  SEMANTIC MEMORIES\n")
        header = _fmt_row(["TYPE", "FACT", "VALUE", "IMP", "CONF", "HITS"], [12, 35, 35, 5, 5, 4])
        print("  " + header)
        print("  " + "─" * len(header))
        for m in memories:
            row = _fmt_row([
                m["memory_type"],
                m["fact"],
                m["value"],
                f"{m['importance']:.2f}",
                f"{m['confidence']:.2f}",
                str(m["access_count"]),
            ], [12, 35, 35, 5, 5, 4])
            print("  " + row)
    else:
        print("\n  No semantic memories stored yet.")

    if rules:
        print("\n  BEHAVIORAL RULES\n")
        for r in rules:
            status_marker = "✓" if r["status"] == "VERIFIED" else "·"
            print(f"  {status_marker} [{r['confidence']:.2f}] {r['belief']}")
    else:
        print("\n  No behavioral rules promoted yet.")

    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# reset — clear a scope
# ---------------------------------------------------------------------------

def _cmd_reset(args) -> None:
    db_path = Path(args.db) if args.db else _default_db()
    scope = args.scope or _default_scope()

    if not db_path.exists():
        print(f"Nothing to reset — no database at {db_path}")
        return

    if not args.yes:
        confirm = input(f"Reset scope '{scope}' in {db_path}? This deletes all memories, rules, and episodes. [y/N] ")
        if confirm.strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agent_sleep.storage.db import ensure_db_initialized, _cursor

    ensure_db_initialized(db_path)
    with _cursor(commit=True, db_path=db_path) as cur:
        cur.execute("DELETE FROM semantic_memories WHERE scope=?", (scope,))
        mem_del = cur.rowcount
        cur.execute("DELETE FROM candidate_rules WHERE scope=?", (scope,))
        rule_del = cur.rowcount
        cur.execute("DELETE FROM execution_episodes WHERE scope=?", (scope,))
        ep_del = cur.rowcount

# ---------------------------------------------------------------------------
# status — inspect memory health, epistemic state, and embedding backend
# ---------------------------------------------------------------------------

def _cmd_status(args) -> None:
    db_path = Path(args.db) if args.db else _default_db()
    scope = args.scope or _default_scope()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agent_sleep.storage.embeddings import get_backend_info
    from agent_sleep.storage.db import ensure_db_initialized, _cursor

    backend = get_backend_info()

    if not db_path.exists():
        print(f"\n[agent-sleep status]")
        print(f"  Database : {db_path} (not created yet)")
        print(f"  Scope    : {scope}")
        print(f"  Backend  : {backend['backend']} (mode: {backend['mode']})")
        return

    ensure_db_initialized(db_path)
    with _cursor(db_path=db_path) as cur:
        cur.execute("SELECT verification_status, COUNT(*) as cnt FROM semantic_memories WHERE scope IN (?, 'global') GROUP BY verification_status", (scope,))
        epistemic = {r["verification_status"]: r["cnt"] for r in cur.fetchall()}

        cur.execute("SELECT COUNT(*) FROM candidate_rules WHERE scope IN (?, 'global') AND status != 'REFUTED'", (scope,))
        rules_cnt = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM causal_hypotheses WHERE scope IN (?, 'global')", (scope,))
        causal_cnt = cur.fetchone()[0]

        cur.execute("SELECT domain, competence, uncertainty FROM self_competence")
        competence = [dict(r) for r in cur.fetchall()]

        cur.execute("SELECT COUNT(*) FROM execution_episodes WHERE processed_by_sleep=0")
        pending_ep = cur.fetchone()[0]

    SEP = "─" * 70
    print(f"\n{SEP}")
    print(f"  agent-sleep System Status  |  scope: {scope}")
    print(f"{SEP}")
    print(f"  Embedding Backend : {backend['backend']} (mode: {backend['mode']}, dim: {backend['dimension']})")
    if backend.get("is_fallback"):
        print(f"  Note              : {backend.get('note')}")
    print(f"  Database Path     : {db_path}")
    print(f"  Pending Episodes  : {pending_ep}")
    print(f"  Active Rules      : {rules_cnt}")
    print(f"  Causal Hypotheses : {causal_cnt}")
    print(f"  Epistemic Memory States:")
    for st in ["verified", "repeated", "observed", "raw", "quarantined"]:
        print(f"    - {st:<12}: {epistemic.get(st, 0)}")
    if competence:
        print(f"  Self-Competence Model:")
        for c in competence:
            print(f"    - {c['domain']:<12}: {c['competence']:.0%} (uncertainty: ±{c['uncertainty']:.2f})")
    print(f"{SEP}\n")


# ---------------------------------------------------------------------------
# benchmark — canonical evaluation across memory conditions
# ---------------------------------------------------------------------------

def _cmd_benchmark(args) -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from benchmarks.agent_eval.runner import run_benchmark_condition

    print("=" * 72)
    print(" AGENT-SLEEP: CANONICAL EVALUATION BENCHMARK SUITE")
    print("=" * 72)

    conditions = ["BASELINE", "NAIVE_RAG", "AGENT_SLEEP"]
    results = []
    for cond in conditions:
        print(f"  Evaluating condition: {cond}...")
        res = run_benchmark_condition(cond)
        results.append(res)

    SEP = "-" * 72
    print(f"\n{SEP}")
    print(f"{'Condition':<18} {'Pass Rate':<12} {'Avg Calls':<14} {'Repeated Mistakes':<18} {'Memory Usefulness':<18}")
    print(SEP)
    for res in results:
        print(
            f"{res['condition']:<18} "
            f"{res['pass_rate']:>7.0%}      "
            f"{res['avg_llm_calls']:>8.1f}      "
            f"{res['repeated_mistakes']:>14d}      "
            f"{res.get('memory_usefulness_rate', 0.0):>14.0%}"
        )
    print(f"{SEP}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent-sleep",
        description="Inspect, configure, and benchmark your agent's persistent memory.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    sub.add_parser("init", help="Print the mcp_config.json snippet for your platform.")

    # status
    p_stat = sub.add_parser("status", help="Inspect embedding backend and memory health.")
    p_stat.add_argument("--scope", default="", help="Scope to inspect (default: current dir name).")
    p_stat.add_argument("--db", default="", help="Path to SQLite DB (default: .agent_sleep/memory.db).")

    # show
    p_show = sub.add_parser("show", help="Display stored memories and rules.")
    p_show.add_argument("--scope", default="", help="Scope to inspect (default: current dir name).")
    p_show.add_argument("--db", default="", help="Path to SQLite DB (default: .agent_sleep/memory.db).")

    # benchmark
    sub.add_parser("benchmark", help="Run the canonical 3-condition evaluation benchmark.")

    # reset
    p_reset = sub.add_parser("reset", help="Clear all data for a scope.")
    p_reset.add_argument("--scope", default="", help="Scope to clear (default: current dir name).")
    p_reset.add_argument("--db", default="", help="Path to SQLite DB (default: .agent_sleep/memory.db).")
    p_reset.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt.")

    args = parser.parse_args()
    if args.command == "init":
        _cmd_init(args)
    elif args.command == "status":
        _cmd_status(args)
    elif args.command == "show":
        _cmd_show(args)
    elif args.command == "benchmark":
        _cmd_benchmark(args)
    elif args.command == "reset":
        _cmd_reset(args)


if __name__ == "__main__":
    main()

