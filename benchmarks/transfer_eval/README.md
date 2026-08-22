# Transfer Eval Benchmark

Measures the effect of persistent memory on agent performance across 12
sequential tasks on one evolving repository.

## Hypothesis

An agent with memory enabled should:
1. Make fewer repeated mistakes (lower repeated-mistake rate)
2. Use fewer LLM calls per task (better efficiency)  
3. Pass more tasks cumulatively (higher success rate)

## Design

- 12 tasks run sequentially on the same repository
- Each task can rely on knowledge from all previous tasks
- Two conditions: MEMORY_ON vs MEMORY_OFF
- Metric collected per task: {pass_fail, llm_calls, repeated_mistakes}

## Run

```bash
python benchmarks/run.py
```

## Expected Results (from Maya production deployment)

| Metric | Memory OFF | Memory ON |
|--------|-----------|-----------|
| Pass@12 cumulative | 41% | 67% |
| Avg LLM calls / task | 18.4 | 11.2 |
| Repeated-mistake rate | 38% | 9% |
