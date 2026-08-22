# agent-sleep Empirical Evaluation Protocol & Reproducibility Guide

## Evaluation Methodology & Scientific Framing

This benchmark provides a **controlled sandbox evaluation of memory-driven agent-control dynamics using deterministic task policies**. It systematically measures how persistent memory retrieval, offline consolidation, epistemic filtering, and causal rule abstraction impact error reduction and execution efficiency across sequential software engineering tasks.

### 6-Way Experimental Ablation Conditions
1. **`NO_MEMORY` (Baseline Amnesia)**: Fresh agent context on every session. Zero cross-session memory.
2. **`RAW_TRANSCRIPT` (Unconsolidated Context)**: Raw dumps of prior session logs without abstraction, deduplication, or sleep consolidation.
3. **`VECTOR_RAG` (Naive Semantic Search)**: Standard top-k cosine similarity retrieval over unconsolidated episode records without cognitive distillation.
4. **`AGENT_SLEEP_CORE` (Episodic Sleep Distillation)**: Offline sleep consolidation pipeline extracting procedural lessons and distilled execution memories.
5. **`AGENT_SLEEP_EPISTEMIC` (Core + Epistemic Lifecycle & Provenance)**: Core pipeline plus epistemic lifecycle gating (`observed` → `repeated` → `verified`), multi-source provenance validation, and quarantine suppression.
6. **`AGENT_SLEEP_FULL` (Full Cognitive Architecture)**: Epistemic lifecycle + utility learning + causal hypothesis tracking + contextual rule specificity + Bayesian self-model + concept hierarchy track record.

### Standardized 8-Task Evaluation Suite
- **Task 1**: Concurrent Balance Updater (`SQL` - SQLite concurrency busy lock trap)
- **Task 2**: Async Event Stream Collector (`Python` - async generator iteration trap)
- **Task 3**: HTTP Client with Exponential Backoff (`API_calls` - 429 rate limit backoff trap)
- **Task 4**: Concurrent SQLite Settlement (`SQL` - Transfer domain evaluation)
- **Task 5**: Async Event Batch Processor (`Python` - Transfer domain evaluation)
- **Task 6**: HTTP Webhook Dispatcher with Retry (`API_calls` - Transfer domain evaluation)
- **Task 7**: PostgreSQL Connection Pool Init (`SQL` - Clean execution baseline)
- **Task 8**: SQLite In-Memory Fast Cache (`SQL` - Contextual rule exception evaluation)

### Measured Metrics
- **Zero-Shot Task Pass Rate**: Percentage of tasks passing on the very first execution attempt without hitting traps.
- **Average LLM Calls / Task**: Number of LLM reasoning/retry turns required to achieve passing execution.
- **Repeated Mistake Count**: Number of times a known trap was encountered again after prior occurrence in the session history.
- **Memory Usefulness Rate**: Percentage of tasks where retrieved memory context was actively enacted and yielded verified success.

### Running the Canonical Benchmark
```bash
python benchmarks/agent_eval/runner.py
```
Outputs and synchronizes execution metrics directly to [`benchmarks/agent_eval/results.json`](results.json).
