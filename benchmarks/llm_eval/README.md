# 🧪 agent-sleep Live LLM Agent Evaluation Benchmark

This benchmark evaluates **real autonomous LLM agents** executing sequential software engineering tasks across multiple memory architectures:

1. **`NO_MEMORY`**: Standard stateless baseline.
2. **`RAW_TRANSCRIPT`**: Unconsolidated transcript log injection.
3. **`VECTOR_RAG`**: Naive semantic vector search.
4. **`AGENT_SLEEP`**: Full cognitive architecture (Episodic recording + Offline Sleep Consolidation + Epistemic Filtering + Causal Hypotheses + Bayesian Competence + Specificity Rules).

---

## Running with Offline Simulated Agent (Zero-API CI Mode)

```bash
python -m benchmarks.llm_eval.runner --provider mock
```

## Running with Real LLM APIs

### 1. OpenAI (`gpt-4o`, `gpt-4o-mini`)
```bash
export OPENAI_API_KEY="your-api-key"
python -m benchmarks.llm_eval.runner --provider openai --model gpt-4o-mini
```

### 2. Anthropic (`claude-3-5-sonnet`, `claude-3-5-haiku`)
```bash
export ANTHROPIC_API_KEY="your-api-key"
python -m benchmarks.llm_eval.runner --provider anthropic --model claude-3-5-haiku-20241022
```

### 3. Local / Self-Hosted Models via Ollama / vLLM / LiteLLM
```bash
python -m benchmarks.llm_eval.runner --provider openai --model llama3.1 --base-url http://localhost:11434/v1
```

---

## Tracked Metrics

- **Zero-Shot Pass Rate**: Proportion of tasks solved on the very first attempt without encountering errors or retries.
- **Average LLM Calls / Task**: Number of reasoning and code generation steps required to achieve passing tests.
- **Repeated Traps**: Number of times a known trap was repeated on transfer tasks.
- **Token Usage & Overhead**: Total prompt tokens and completion tokens consumed per condition.
- **Memory Usefulness**: Percentage of tasks where retrieved memory context was actively enacted and produced successful test execution.
