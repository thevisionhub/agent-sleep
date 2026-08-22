"""
Live LLM Agent Evaluation Benchmark Runner for agent-sleep.

Evaluates real or simulated LLM agents across sequential coding tasks with recurring architectural traps:
  - Compares: NO_MEMORY vs RAW_TRANSCRIPT vs VECTOR_RAG vs AGENT_SLEEP
  - Measures: Zero-shot Pass Rate, Retry Calls, Repeated Mistakes, Tokens, and Memory Usefulness
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_sleep import AgentMemory, SleepConsolidator
from agent_sleep.storage.embeddings import get_backend_info
from benchmarks.agent_eval.tasks.eval_tasks import EVAL_TASKS, EvalTask
from benchmarks.llm_eval.providers import BaseLLMProvider, get_llm_provider
from benchmarks.llm_eval.agent_loop import run_llm_agent_on_task


def run_llm_evaluation_condition(
    condition: str,
    provider: BaseLLMProvider,
    tasks: List[EvalTask],
) -> Dict[str, Any]:
    """Evaluate one experimental condition using the specified LLM provider."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_root:
        root_path = Path(tmp_root)
        db_path = root_path / f"{condition.lower()}_bench.db"
        memory = AgentMemory(session_id=f"eval_{condition.lower()}", scope="global", db_path=db_path) if condition != "NO_MEMORY" else None
        consolidator = SleepConsolidator(scope="global", db_path=db_path) if condition == "AGENT_SLEEP" else None

        task_results = []
        for task in tasks:
            sandbox_dir = root_path / f"sandbox_task_{task.task_id}"
            sandbox_dir.mkdir(parents=True, exist_ok=True)
            res = run_llm_agent_on_task(
                task=task,
                condition=condition,
                provider=provider,
                memory=memory,
                consolidator=consolidator,
                sandbox_dir=sandbox_dir,
            )
            task_results.append(res)

        gc.collect()

        total = len(task_results)
        zero_shot_passed = sum(1 for r in task_results if r["first_attempt_passed"])
        final_passed = sum(1 for r in task_results if r["final_passed"])
        total_calls = sum(r["llm_calls"] for r in task_results)
        repeated_traps = sum(1 for r in task_results if r["repeated_mistake"])
        useful_memories = sum(1 for r in task_results if r["memory_useful"])

        return {
            "condition": condition,
            "total_tasks": total,
            "zero_shot_pass_rate": round(zero_shot_passed / total, 4) if total else 0.0,
            "final_pass_rate": round(final_passed / total, 4) if total else 0.0,
            "avg_llm_calls": round(total_calls / total, 2) if total else 0.0,
            "repeated_traps": repeated_traps,
            "memory_usefulness_rate": round(useful_memories / total, 4) if total else 0.0,
            "token_usage": provider.get_token_usage(),
            "task_details": task_results,
        }


def main():
    parser = argparse.ArgumentParser(description="Run Live LLM Agent Benchmark for agent-sleep.")
    parser.add_argument("--provider", default="mock", choices=["mock", "openai", "anthropic"], help="LLM Provider to use.")
    parser.add_argument("--model", default=None, help="Model name (e.g. gpt-4o-mini, claude-3-5-haiku-20241022).")
    parser.add_argument("--api-key", default=None, help="API key override (or read from OPENAI_API_KEY/ANTHROPIC_API_KEY).")
    parser.add_argument("--base-url", default=None, help="Custom Base URL for OpenAI/Ollama/vLLM.")
    args = parser.parse_args()

    backend_info = get_backend_info()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print("=" * 85)
    print(" AGENT-SLEEP: LIVE LLM AGENT EVALUATION BENCHMARK")
    print(f" Provider: {args.provider.upper()} | Model: {args.model or 'default'} | Backend: {backend_info['backend']}")
    print(f" Timestamp: {timestamp}")
    print("=" * 85)

    conditions = ["NO_MEMORY", "RAW_TRANSCRIPT", "VECTOR_RAG", "AGENT_SLEEP"]
    results = []

    for cond in conditions:
        print(f"  Evaluating Condition: {cond:<18}...", end=" ", flush=True)
        provider = get_llm_provider(
            provider_name=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        t0 = time.time()
        res = run_llm_evaluation_condition(cond, provider, EVAL_TASKS)
        elapsed = time.time() - t0
        print(f"Done ({elapsed:.2f}s) | Zero-Shot: {res['zero_shot_pass_rate']:.0%}, Calls: {res['avg_llm_calls']:.1f}, Repeated Traps: {res['repeated_traps']}")
        results.append(res)

    SEP = "-" * 85
    print(f"\n{SEP}")
    print(f"{'Condition':<18} {'Zero-Shot Pass':<16} {'Avg Calls':<12} {'Repeated Traps':<16} {'Tokens':<12}")
    print(SEP)
    for r in results:
        tot_tok = r.get("token_usage", {}).get("total_tokens", 0)
        print(
            f"{r['condition']:<18} "
            f"{r['zero_shot_pass_rate']:>11.1%}      "
            f"{r['avg_llm_calls']:>8.1f}    "
            f"{r['repeated_traps']:>10d}        "
            f"{tot_tok:>8d}"
        )
    print(f"{SEP}\n")

    report = {
        "metadata": {
            "benchmark": "agent-sleep Live LLM Agent Evaluation",
            "timestamp": timestamp,
            "provider": args.provider,
            "model": args.model or "default",
            "embedding_backend": backend_info,
            "task_count": len(EVAL_TASKS),
        },
        "results": results,
    }

    out_file = Path(__file__).parent / "live_results.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved live LLM evaluation results to: {out_file}")


if __name__ == "__main__":
    main()
