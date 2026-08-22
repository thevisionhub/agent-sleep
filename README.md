# 🧠 agent-sleep

**Episodic Memory → Offline Sleep Consolidation → Selective Semantic Recall** for AI agents.

A lightweight, framework-agnostic Python library that gives AI agents persistent, cross-session memory — inspired by how the biological brain uses sleep cycles to consolidate waking experiences into lasting procedural rules and lessons.

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
# Option A: zero-install (recommended)
uvx agent-sleep-mcp

# Option B: install first, then run the init helper
pip install "agent-sleep[mcp]"
agent-sleep init     # prints the correct config snippet for your platform
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

## Key Features (v0.1.1-alpha)

* **Pre-Computed Vector BLOBs**: Embeds the query once and compares it against pre-computed stored vectors, eliminating repeated text embedding during recall.
* **Selective Rule Retrieval**: Injects only high-confidence rules semantically relevant to the current task — zero prompt clutter.
* **Scope & Project Isolation**: Multi-tier namespaces (`scope="repo_a"`, `scope="global"`). Prevents cross-project rule bleed.
* **Epistemic Tracking**: Distinguishes `[OBSERVED]` step lessons from `[VERIFIED]` test-runner verdicts with provenance metadata.
* **Memory Decay & Forgetting**: Automatically decays utility for stale, unverified memories that receive zero recall activity over time.
* **Zero Mandatory Heavy Dependencies**: Works out-of-the-box using standard SQLite and a deterministic hashed bag-of-words fallback. Seamlessly upgrades to `sentence-transformers` (`all-MiniLM-L6-v2`) when installed.

---

## Benchmark: Transfer Eval

We evaluate `agent-sleep` on a reproducible 12-task sequential software suite containing recurring architectural traps (async syntax errors, SQLite busy locks, missing rate-limit headers, SQL injection vulnerabilities):

```bash
python benchmarks/run.py
```

### Measured Results:

| Metric | Memory OFF | Memory ON | Improvement |
|:---|:---:|:---:|:---:|
| **Pass Rate (Pass@12)** | 67% | **92%** | **+25 percentage points** |
| **Avg LLM Calls / Task** | 14.7 | **8.5** | **-42% (fewer calls)** |
| **Repeated Mistakes** | 8 | **2** | **-75% (fewer mistakes)** |
| **Repeated Mistake Rate** | 67% | **17%** | **-50 percentage points** |

> [!NOTE]
> **Benchmark note**: The included Transfer Eval suite is a deterministic simulation designed to validate the memory-consolidation, vector retrieval, and knowledge transfer pipeline across sequential tasks. It is not yet a benchmark of autonomous real-world LLM agent performance.

> [!IMPORTANT]
> **Embedding mode**: By default, `agent-sleep` uses a lightweight hashed bag-of-words fallback for recall. This handles same-domain keyword overlap well but will not catch cross-domain semantic transfer (e.g., "invoice locking" → "subscription threading") without the optional `sentence-transformers` model. Install `pip install "agent-sleep[semantic]"` to enable full semantic similarity recall. The benchmark and MCP integration tests use the semantic model.

*Full evaluation harness and task definitions in `benchmarks/run.py`.*

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

### Quick install with MCP support:
```bash
pip install "agent-sleep[mcp]"
```

### With full semantic embeddings (recommended):
```bash
pip install "agent-sleep[all]"
```

### From GitHub (latest alpha):
```bash
pip install git+https://github.com/srishiva/agent-sleep.git
```

### Editable install for development:
```bash
git clone https://github.com/srishiva/agent-sleep.git
cd agent-sleep
pip install -e ".[all]"
```

---

## MCP Tools Reference

| Tool | When to call |
|:---|:---|
| `agent_sleep_recall` | **Before** planning or executing any non-trivial task |
| `agent_sleep_record` | **During** execution — after each tool failure or success milestone |
| `agent_sleep_consolidate` | **After** a session ends or when the agent is idle |
| `agent_sleep_status` | Anytime — to inspect memory health and pending episodes |

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
