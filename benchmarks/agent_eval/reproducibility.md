# agent-sleep Empirical Evaluation Protocol & Reproducibility Guide

## Evaluation Methodology

This benchmark provides a standardized, controlled evaluation of autonomous coding agents across sequential software engineering tasks with recurring domain-specific traps.

### Experimental Conditions
1. **BASELINE (Zero Persistent Memory)**: Fresh agent context on every session. No cross-session memory.
2. **NAIVE_RAG (Raw Transcript / Vector Search)**: Retrieves raw text chunks from prior session logs without abstraction, deduplication, or sleep consolidation.
3. **AGENT_SLEEP (Closed-Loop Sleep Consolidation)**: 8-stage offline consolidation + selective semantic recall (semantic memories, causal mechanisms, self-model competence, concept hierarchy track record).

### Experimental Controls
- **Identical Task Suite**: 4-6 software engineering tasks evaluated via automated pytest sandboxes.
- **Identical Execution Sandbox**: Isolated temporary directory per task run with clean filesystem state.
- **Identical Grading Criterion**: Authoritative pytest execution in child process.
- **Controlled Repetitions**: Repeated runs to measure variance across conditions.

### Measured Metrics
- **Task Pass Rate**: Percentage of tasks passing all verification tests.
- **Repeated Mistake Rate**: Frequency of repeating an identical failure mode encountered in earlier sessions.
- **Average LLM Calls / Tool Steps**: Call overhead to reach a passing solution.
- **Memory Utility Attribution**: Proportion of retrieved memories that directly contributed to successful test execution.
