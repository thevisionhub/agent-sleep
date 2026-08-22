"""
Evaluation Metrics Calculator for agent-sleep Benchmark.
"""
from typing import Any, Dict, List


def compute_eval_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_tasks = len(results)
    if total_tasks == 0:
        return {}
    passed = sum(1 for r in results if r.get("passed", False))
    repeated_mistakes = sum(1 for r in results if r.get("repeated_mistake", False))
    total_llm_calls = sum(r.get("llm_calls", 0) for r in results)
    total_duration = sum(r.get("duration_seconds", 0.0) for r in results)
    useful_memories = sum(1 for r in results if r.get("memory_useful", False))

    return {
        "total_tasks": total_tasks,
        "passed_tasks": passed,
        "pass_rate": round(passed / total_tasks, 3),
        "repeated_mistakes": repeated_mistakes,
        "repeated_mistake_rate": round(repeated_mistakes / total_tasks, 3),
        "avg_llm_calls": round(total_llm_calls / total_tasks, 2),
        "total_duration_seconds": round(total_duration, 2),
        "memory_usefulness_rate": round(useful_memories / total_tasks, 3),
    }
