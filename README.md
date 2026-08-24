# 🧠 agent-sleep

**Persistent Experience Consolidation & Decision Support for AI Agents.**

A lightweight, framework-agnostic Python library and MCP server that provides persistent experience consolidation and decision-support signals that a host agent can use to adapt across sessions — inspired by how the biological brain uses sleep cycles to consolidate waking experiences into lasting procedural rules and lessons.

---

## The Problem: "Agent Amnesia"

Every modern AI agent framework (LangChain, AutoGen, CrewAI, OpenAI Assistants) suffers from **Agent Amnesia**:
* Every new chat or subagent run starts completely from scratch.
* When an agent hits an error or discovers a codebase convention on Monday, it repeats the **exact same mistake** on Tuesday.
* Vector DBs (RAG) only search static documents — they **do not learn from runtime experience**.

---

## MCP Quick Start — 10 seconds

> **This is the primary usage path.** `agent-sleep` ships as an MCP server, so any agent that supports MCP (Antigravity, Claude Desktop, Cursor, Cline) can use it without writing any code.

### Step 1 — Install and generate your config

```bash
# Option A: Install from GitHub with MCP extras
pip install "agent-sleep[mcp] @ git+https://github.com/thevisionhub/agent-sleep.git"

# Option B: Local editable install (if cloned)
cd path/to/agent-sleep
pip install -e ".[mcp]"

# Option C: Zero-install via uvx (if uv is installed)
uvx --from "git+https://github.com/thevisionhub/agent-sleep.git" agent-sleep-mcp

# Generate your MCP configuration snippet:
agent-sleep init
```

`agent-sleep init` auto-detects your OS and prints the JSON snippet to paste into your MCP client's config file. No hand-editing required.

### Step 2 — Paste the config snippet

The `init` command prints exactly what to paste and where. Example output for Claude Desktop on macOS:

```json
{
  "mcpServers": {
    "agent-sleep": {
      "command": "uvx",
      "args": ["agent-sleep-mcp"]
    }
  }
}
```

Paste that into `~/Library/Application Support/Claude/claude_desktop_config.json`, restart Claude, and you're done.

### Step 3 — Ask your agent to use it

```
"Before we start, check your memory for anything relevant to this task."
"Record that we use pytest fixtures — not unittest — in this project."
"Run a sleep consolidation so you remember today's lessons next session."
```

Memory is automatically stored in `.agent_sleep/memory.db` in your project directory (gitignored by default).

---

## Inspect what's stored — CLI

You don't need to go through an LLM to see what your agent has learned:

```bash
# See all memories and rules for the current project
agent-sleep show

# Clear a project's memory (with confirmation prompt)
agent-sleep reset

# Target a specific scope or DB
agent-sleep show --scope my_api --db /path/to/memory.db
```

---

## How It Works: The 3-Phase Pipeline

```
          [ ONLINE EXECUTION PHASE ]
            Agent executes tool calls
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  1. EPISODIC RECORDING                       │
│     memory.record_episode(...)               │  Fast, minimal overhead.
│     Records goal, action, outcome, errors.   │  Stores execution events.
└──────────────────────┬───────────────────────┘
                       │
             (Session ends / Agent idle)
                       │
                       ▼
          [ OFFLINE SLEEP CONSOLIDATION ]
┌──────────────────────────────────────────────┐
│  2. SLEEP CONSOLIDATOR (8-Stage Pipeline)    │
│     SleepConsolidator.run(session_id)        │
│                                              │
│     • Priority Replay (prediction error)     │
│     • Deterministic Episodic Distillation    │  Grounding first:
│     • Procedural Recipe Extraction           │  distills facts & lessons
│     • How-Memory Trajectory Abstraction      │  before optional LLM
│     • Behavioral Rule Promotion (seen ≥2x)   │  generalization passes.
│     • Epistemic Status (observed vs verified)│
│     • Episodic Compression over time         │
│     • Self-Competence EMA Tracking           │
└──────────────────────┬───────────────────────┘
                       │
              (Next session / New task)
                       │
                       ▼
          [ ONLINE SELECTIVE RECALL ]
┌──────────────────────────────────────────────┐
│  3. SELECTIVE SEMANTIC RECALL                │
│     memory.recall(new_task)                  │  Pre-computed vector BLOBs.
│     Returns only relevant lessons & rules    │  Prevents prompt dilution.
│     filtered by project scope & relevance.   │
└──────────────────────────────────────────────┘
```

---

## Key Features (v0.1.2-alpha)

* **Pre-Computed Vector BLOBs**: Embeds the query once and compares it against pre-computed stored vectors, eliminating repeated text embedding during recall.
* **Epistemic Memory Lifecycle**: Tracks memory progression through stages (`RAW` → `OBSERVED` → `REPEATED` → `VERIFIED` → `ACTIVE`), automatically quarantining contradictory or high-failure memories.
* **Verifiable Causal Attribution & Utility Feedback**: Evaluates whether retrieved memories actually helped future execution via structured evidence records (`retrieval` → `action change` → `outcome attribution`).
* **Evidence Diversity Causal Hypotheses**: Distills recurring failures into causal mechanisms using evidence diversity scaling across independent sources and environments.
* **Bayesian Self-Competence Model**: Estimates domain competence and Bayesian Beta-distribution uncertainty across composite domains to provide adaptive decision support (verification intensity, retry budgets) for host agents.
* **First-Class Rule Specificity Engine**: Resolves rule conflicts through hierarchical precedence (`specific verified` > `general verified` > `specific candidate` > `general candidate`) and dynamic exception suppression.
* **Scope & Project Isolation**: Multi-tier namespaces (`scope="repo_a"`, `scope="global"`). Project-specific knowledge is strictly isolated, while universal idioms and tool failure modes can optionally be shared via `global`.
* **Zero Mandatory Heavy Dependencies**: Works out-of-the-box using standard SQLite and a deterministic hashed bag-of-words fallback. Seamlessly upgrades to `sentence-transformers` (`all-MiniLM-L6-v2`) when installed.

---

## Benchmarks & Evaluation

### 1. Controlled Transfer Simulation (`benchmarks/run.py`)
Evaluates memory consolidation, vector retrieval, and knowledge transfer across 12 sequential software tasks with recurring architectural traps:

| Metric | Memory OFF | Memory ON | Improvement |
|:---|:---:|:---:|:---:|
| **Pass Rate (Pass@12)** | 67% | **92%** | **+25 percentage points** |
| **Avg LLM Calls / Task** | 14.7 | **8.5** | **-42% (fewer calls)** |
| **Repeated Mistakes** | 8 | **2** | **-75% (fewer mistakes)** |

*Note: The controlled transfer simulation evaluates the deterministic cognitive-control dynamics of memory retrieval and trap avoidance.*

### 2. Canonical 6-Way Ablation Benchmark (`benchmarks/agent_eval/runner.py`)
Controlled sandbox evaluation of memory-driven agent-control dynamics across 8 standardized software engineering tasks:

| Experimental Condition | Pass Rate (Zero-Shot) | Avg LLM Calls / Task | Repeated Traps | Memory Useful Rate |
|:---|:---:|:---:|:---:|:---:|
| `NO_MEMORY` (Baseline Amnesia) | 12.5% | 3.6 | 4 | 0.0% |
| `RAW_TRANSCRIPT` (Unconsolidated) | 12.5% | 3.6 | 4 | 0.0% |
| `VECTOR_RAG` (Naive Semantic) | 12.5% | 3.6 | 4 | 0.0% |
| `AGENT_SLEEP_CORE` (Episodic Distillation) | 25.0% | 2.9 | 2 | 12.5% |
| `AGENT_SLEEP_EPISTEMIC` (Core + Provenance) | 37.5% | 2.5 | 1 | 25.0% |
| **`AGENT_SLEEP_FULL` (Full Cognitive Architecture)** | **75.0%** | **1.4** | **0** | **75.0%** |

```bash
python benchmarks/agent_eval/runner.py
```

> [!NOTE]
> **Scientific & Backend Disclosure**:
> - The sandbox benchmark evaluates agent control dynamics, token efficiency, and error avoidance under controlled test suites.
> - **Embedding Backends**: High-precision vector similarity relies on `sentence-transformers` (`all-MiniLM-L6-v2`). When dependencies are absent, the library automatically falls back to a deterministic hashed bag-of-words embedding.
> - Full reproducibility protocols and metric logs are documented in [`benchmarks/agent_eval/results.json`](benchmarks/agent_eval/results.json).

---

## Python Library Usage

> If you prefer to drive the memory system from your own agent code rather than via MCP, the Python API is fully supported.

```python
from agent_sleep import AgentMemory, SleepConsolidator

# 1. Initialize memory scoped to your project/repo
memory = AgentMemory(session_id="session_01", scope="payment_service")

# 2. Record actions and outcomes during your agent's loop
memory.record_episode(
    goal="Refactor payment processor to async",
    action="edit_file('processor.py', ...)",
    outcome="failure",
    failure_reason="SyntaxError: 'await' outside async function",
)

# 3. Trigger sleep consolidation when idle or at session end
consolidator = SleepConsolidator(scope="payment_service")
report = consolidator.run(session_id="session_01")
# -> {'episodes_processed': 1, 'memories_written': 1, 'rules_promoted': 0, ...}

# 4. Next session: recall relevant context before executing
context = memory.recall("Add Stripe webhook handler")
print(context)
# [MEMORY CONTEXT]
# Relevant past experience:
#   ⚠ [LESSON] Caution on task: Refactor payment processor to async:
#     A previous attempt failed: SyntaxError: 'await' outside async function.
# [END MEMORY CONTEXT]
```

---

## Installation

### From GitHub (Latest Alpha with MCP):
```bash
pip install "agent-sleep[mcp] @ git+https://github.com/thevisionhub/agent-sleep.git"
```

### With full semantic embeddings (`sentence-transformers`):
```bash
pip install "agent-sleep[all] @ git+https://github.com/thevisionhub/agent-sleep.git"
```

### Editable install for local development:
```bash
git clone https://github.com/thevisionhub/agent-sleep.git
cd agent-sleep
pip install -e ".[all]"
```

---

## MCP Tools Reference

| Tool | When to call |
|:---|:---|
| `agent_sleep_recall` | **Before** planning or executing any non-trivial task — retrieves lessons, rules, causal traps, and self-competence directives |
| `agent_sleep_record` | **During** execution — after each tool failure or milestone |
| `agent_sleep_consolidate` | **After** a session ends or when the agent is idle |
| `agent_sleep_status` | Anytime — inspects memory health, epistemic breakdowns, and pending episodes |
| `agent_sleep_feedback` | **After** applying retrieved knowledge — records causal outcome attribution and updates utility scores |
| `agent_sleep_specialize_rule` | When discovering exceptions or boundary conditions for existing rules |

All tools default `scope` to the current working directory name and `db_path` to `.agent_sleep/memory.db` in the project root. No configuration required for the common case.

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Get Discovered — Registry Listings

Submitting `agent-sleep` to MCP registries takes about 5 minutes each and is the fastest way to reach developers looking for memory tools:

- **[Smithery](https://smithery.ai/submit)** — paste the GitHub URL, add a short description, done.
- **[modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)** — open a PR adding an entry to the README under "Community Servers".
- **Cursor** — also surfaces MCP servers; check [their current docs](https://docs.cursor.com) for the latest submission process.

---

## License

MIT License — free for personal, commercial, and research use.
