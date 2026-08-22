"""
agent-sleep MCP Server Adapter.

Exposes Agent Sleep memory recording, consolidation, and selective recall
as standard Model Context Protocol (MCP) tools for AI agents (Antigravity, Claude, Cursor, etc.).

All internal logging is directed to sys.stderr to preserve the purity
of sys.stdout for MCP JSON-RPC protocol transport.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure logging strictly outputs to stderr (stdout is used for MCP protocol)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[agent-sleep-mcp] %(levelname)s: %(message)s",
)
logger = logging.getLogger("agent_sleep.mcp")

from agent_sleep.recorder import AgentMemory
from agent_sleep.consolidator import SleepConsolidator
from agent_sleep.storage.db import (
    ensure_db_initialized,
    _cursor,
    get_unprocessed_episodes,
)


# ---------------------------------------------------------------------------
# Smart defaults — zero-config for first-time users
# ---------------------------------------------------------------------------

def _resolve_db(db_path: Optional[str]) -> Optional[Path]:
    """
    Resolve db_path. If None, use .agent_sleep/memory.db in the server's cwd.
    Falls back to AGENT_SLEEP_DB env var if set (highest priority).
    """
    env_path = os.environ.get("AGENT_SLEEP_DB")
    if env_path:
        return Path(env_path)
    if db_path:
        return Path(db_path)
    # Project-local default: keeps each project's memory isolated by default
    local = Path.cwd() / ".agent_sleep" / "memory.db"
    local.parent.mkdir(parents=True, exist_ok=True)
    return local


def _resolve_scope(scope: str) -> str:
    """Use the current directory name as default scope when 'auto' is requested."""
    if scope in ("auto", ""):
        return Path.cwd().name
    return scope


def _sanitize_mcp_output(obj: Any) -> Any:
    """Recursively convert numpy types, custom objects to native Python primitives for clean JSON-RPC serialization."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _sanitize_mcp_output(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_mcp_output(x) for x in obj]
    return obj

# ---------------------------------------------------------------------------
# MCP Server Definition
# ---------------------------------------------------------------------------

try:
    from mcp.server import MCPServer
    app = MCPServer(
        name="agent-sleep",
        version="0.1.2-alpha",
        instructions=(
            "Agent Sleep persistent cognitive memory system. "
            "Use agent_sleep_recall before tasks to retrieve past lessons and rules. "
            "Use agent_sleep_record during execution to capture tool outcomes or failures. "
            "Use agent_sleep_consolidate after finishing a session to distill episodes into permanent procedural memory."
        ),
    )
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
        app = FastMCP("agent-sleep")
    except ImportError:
        raise ImportError(
            "The 'mcp' package is required to run the MCP server. "
            "Install it via: pip install 'mcp[cli]'"
        )


@app.tool(
    name="agent_sleep_recall",
    description=(
        "Retrieve relevant procedural memories, past failure lessons, and behavioral rules "
        "for the current task. Call this before planning or executing unfamiliar actions."
    ),
)
def agent_sleep_recall(
    query: str,
    scope: str = "auto",
    top_k: int = 5,
    min_score: float = 0.05,
    session_id: str = "antigravity",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve semantically relevant context for a given goal or query.

    Parameters:
    - query: Task description, tool error, or goal to search memories for.
    - scope: Project/repo namespace. Use 'auto' (default) to infer from cwd.
    - top_k: Maximum number of memories to return (default: 5).
    - min_score: Minimum relevance threshold (default: 0.05).
    - session_id: Current session name (default: 'antigravity').
    - db_path: Path to SQLite database. Defaults to .agent_sleep/memory.db in cwd.
    """
    resolved_scope = _resolve_scope(scope)
    resolved_db = _resolve_db(db_path)
    logger.info(f"Recall request for query: '{query[:60]}' [scope={resolved_scope}]")
    mem = AgentMemory(session_id=session_id, scope=resolved_scope, db_path=resolved_db)
    structured = mem.recall_structured(query, top_k=top_k)
    formatted_prompt = mem.recall(query, top_k=top_k)

    return _sanitize_mcp_output({
        "query": query,
        "scope": resolved_scope,
        "has_memories": bool(
            structured["memories"]
            or structured["rules"]
            or structured.get("causal_hypotheses")
        ),
        "context_prompt": formatted_prompt,
        "memories": structured["memories"],
        "rules": structured["rules"],
        "causal_hypotheses": structured.get("causal_hypotheses", []),
        "operational_policy": structured.get("operational_policy", {}),
        "cluster_insight": structured.get("cluster_insight"),
    })


@app.tool(
    name="agent_sleep_record",
    description=(
        "Record an agent execution episode (action, outcome, and optional failure reason). "
        "Call this whenever a tool fails, a bug is fixed, or an important milestone is reached."
    ),
)
def agent_sleep_record(
    goal: str,
    action: str,
    outcome: str,
    failure_reason: str = "",
    scope: str = "auto",
    session_id: str = "antigravity",
    episode_kind: str = "action",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Record an execution episode into memory.

    Parameters:
    - goal: The overarching goal of the task.
    - action: The specific tool call or action executed.
    - outcome: 'success', 'failure', or 'partial'.
    - failure_reason: Error message, traceback, or reason for failure.
    - scope: Project/repo namespace. Use 'auto' (default) to infer from cwd.
    - session_id: Current execution session ID (default: 'antigravity').
    - episode_kind: 'action' or 'task_verdict' (default: 'action').
    - db_path: Path to SQLite database. Defaults to .agent_sleep/memory.db in cwd.
    """
    resolved_scope = _resolve_scope(scope)
    resolved_db = _resolve_db(db_path)
    logger.info(f"Record episode for goal: '{goal[:60]}' [outcome={outcome}]")
    mem = AgentMemory(session_id=session_id, scope=resolved_scope, db_path=resolved_db)
    episode_id = mem.record_episode(
        goal=goal,
        action=action,
        outcome=outcome,
        failure_reason=failure_reason,
        episode_kind=episode_kind,
    )
    return _sanitize_mcp_output({
        "recorded": True,
        "episode_id": episode_id,
        "session_id": session_id,
        "scope": resolved_scope,
        "outcome": outcome,
    })


@app.tool(
    name="agent_sleep_consolidate",
    description=(
        "Run the 8-stage offline Sleep Consolidation pipeline to distill unanalyzed episodes "
        "into lasting procedural memories, behavioral rules, and competence updates."
    ),
)
def agent_sleep_consolidate(
    session_id: str = "antigravity",
    scope: str = "auto",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Consolidate raw episodes into semantic memory and behavioral rules.

    Parameters:
    - session_id: Session ID whose episodes should be consolidated.
    - scope: Namespace for consolidated knowledge. Use 'auto' (default) to infer from cwd.
    - db_path: Path to SQLite database. Defaults to .agent_sleep/memory.db in cwd.
    """
    resolved_scope = _resolve_scope(scope)
    resolved_db = _resolve_db(db_path)
    logger.info(f"Starting sleep consolidation for session: '{session_id}' [scope={resolved_scope}]")
    consolidator = SleepConsolidator(db_path=resolved_db, scope=resolved_scope)
    report = consolidator.run(session_id=session_id)
    logger.info(f"Consolidation complete: {report}")
    return _sanitize_mcp_output(report)


@app.tool(
    name="agent_sleep_status",
    description=(
        "Return the current status and statistics of the Agent Sleep memory database, "
        "including number of stored memories, active rules, and pending episodes."
    ),
)
def agent_sleep_status(
    scope: str = "auto",
    session_id: str = "antigravity",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inspect database status and memory counts.

    Parameters:
    - scope: Namespace to inspect. Use 'auto' (default) to infer from cwd.
    - session_id: Session ID to inspect pending episodes for (default: 'antigravity').
    - db_path: Path to SQLite database. Defaults to .agent_sleep/memory.db in cwd.
    """
    resolved_scope = _resolve_scope(scope)
    resolved_db = _resolve_db(db_path)
    ensure_db_initialized(resolved_db)

    from agent_sleep.storage.embeddings import get_backend_info
    backend_info = get_backend_info()

    with _cursor(db_path=resolved_db) as cur:
        cur.execute("SELECT COUNT(*) FROM semantic_memories WHERE scope IN (?, 'global')", (resolved_scope,))
        mem_count = cur.fetchone()[0]

        cur.execute("SELECT verification_status, COUNT(*) as cnt FROM semantic_memories WHERE scope IN (?, 'global') GROUP BY verification_status", (resolved_scope,))
        epistemic = {r["verification_status"]: r["cnt"] for r in cur.fetchall()}

        cur.execute("SELECT COUNT(*) FROM candidate_rules WHERE scope IN (?, 'global') AND status != 'REFUTED'", (resolved_scope,))
        rule_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM causal_hypotheses WHERE scope IN (?, 'global')", (resolved_scope,))
        causal_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM execution_episodes WHERE session_id=? AND processed_by_sleep=0", (session_id,))
        unprocessed_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM execution_episodes")
        total_episodes = cur.fetchone()[0]

    return _sanitize_mcp_output({
        "status": "ready",
        "scope": resolved_scope,
        "session_id": session_id,
        "semantic_memories_count": mem_count,
        "epistemic_status_breakdown": epistemic,
        "active_rules_count": rule_count,
        "causal_hypotheses_count": causal_count,
        "unprocessed_episodes_for_session": unprocessed_count,
        "total_episodes_all_time": total_episodes,
        "embedding_backend": backend_info,
    })


@app.tool(
    name="agent_sleep_feedback",
    description=(
        "Record outcome attribution feedback for retrieved memories. "
        "Allows the cognitive memory system to update utility scores and promote or quarantine memories "
        "based on verifiable evidence records or outcome booleans."
    ),
)
def agent_sleep_feedback(
    memory_ids: List[int],
    outcome: str = "success",
    was_applied: bool = True,
    is_causal_contributor: bool = False,
    error_signature: Optional[str] = None,
    evidence_record: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Record utility feedback for retrieved memories.

    Parameters:
    - memory_ids: List of integer memory IDs retrieved and evaluated.
    - outcome: 'success', 'failure', or error description.
    - was_applied: Whether the memories were actively enacted in the task (default: True).
    - is_causal_contributor: Whether the memory was the primary cause of task resolution (default: False).
    - error_signature: Optional error pattern when execution failed despite memory application.
    - evidence_record: Optional structured evidence dict with fields (retrieved, explicitly_referenced, action_changed, outcome, causal_confidence).
    - db_path: Path to SQLite database. Defaults to .agent_sleep/memory.db in cwd.
    """
    from agent_sleep.storage.db import record_memory_utility_feedback
    resolved_db = _resolve_db(db_path)
    report = record_memory_utility_feedback(
        memory_ids=memory_ids,
        outcome=outcome,
        was_applied=was_applied,
        is_causal_contributor=is_causal_contributor,
        error_signature=error_signature,
        evidence_record=evidence_record,
        db_path=resolved_db,
    )
    return _sanitize_mcp_output(report)


@app.tool(
    name="agent_sleep_specialize_rule",
    description=(
        "Specialize an existing behavioral rule with specific conditions and exceptions "
        "when environmental contradictions or edge cases are encountered."
    ),
)
def agent_sleep_specialize_rule(
    rule_id: int,
    condition: str = "",
    exception: str = "",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Specialize a rule with explicit context and exceptions.

    Parameters:
    - rule_id: ID of the candidate rule to specialize.
    - condition: Specific environment or context where the rule holds.
    - exception: Specific context or environment where the rule does NOT apply.
    - db_path: Path to SQLite database. Defaults to .agent_sleep/memory.db in cwd.
    """
    from agent_sleep.storage.db import specialize_rule_with_exception
    resolved_db = _resolve_db(db_path)
    return _sanitize_mcp_output(specialize_rule_with_exception(
        rule_id=rule_id,
        condition=condition,
        exception=exception,
        db_path=resolved_db,
    ))


def main():
    """Main entrypoint for running the stdio MCP server."""
    logger.info("Starting agent-sleep MCP stdio server...")
    if hasattr(app, "run"):
        app.run()
    else:
        import asyncio
        asyncio.run(app.run_stdio_async())


if __name__ == "__main__":
    main()
