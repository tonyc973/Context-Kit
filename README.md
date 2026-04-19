<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-vector%20store-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/OpenAI-powered-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  <a href="https://github.com/tonyc973/Context-Kit/actions/workflows/test.yml"><img src="https://github.com/tonyc973/Context-Kit/actions/workflows/test.yml/badge.svg" alt="Tests" /></a>
  <img src="https://img.shields.io/badge/tests-44%20passing-brightgreen" alt="Tests" />
</p>

# Context-Kit

**Give your AI agents a brain that remembers.**

Context-Kit is a memory management library that gives LLM-powered agents persistent, structured memory across conversations. It organizes knowledge into seven specialized memory stores — inspired by how human cognition works — and automatically retrieves the right context at the right time.

---

## Why Context-Kit?

Most LLM agents are stateless. Every conversation starts from zero. Context-Kit fixes that.

- **Persistent memory** — your agent remembers users, facts, and past interactions across sessions
- **Structured recall** — seven purpose-built memory types, not one giant blob
- **Automatic context injection** — relevant memories are retrieved and woven into prompts
- **Smart summarization** — conversations are compressed when they grow too long, so nothing is lost
- **Token-aware** — respects your model's context window, never overflows

---

## The 7 Memory Types

| Store | Purpose | Example |
|-------|---------|---------|
| **Episodic** | Specific past interactions & events | *"User asked about async patterns on Tuesday"* |
| **Semantic** | General facts & knowledge | *"Python was created by Guido van Rossum"* |
| **Procedural** | Instructions & how-to knowledge | *"When deploying, always run migrations first"* |
| **Working** | Short-term session context | *"Currently debugging the auth module"* |
| **Entity** | Info about people, places, things | *"Alice is a backend engineer at Acme Corp"* |
| **Summary** | Compressed conversation history | *"Previous session covered API design and testing"* |
| **Buffer** | Sliding window of recent messages | Raw message history for the current conversation |

---

## Installation

```bash
git clone https://github.com/tonyc973/Context-Kit.git
cd Context-Kit
pip install -e .
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Quickstart

```python
from contextkit import Agent

# Create an agent with persistent memory
agent = Agent(system_prompt="You are a helpful assistant with long-term memory.")

# Teach it some facts
agent.memory.save("semantic", "The user's name is Alice.")
agent.memory.save("entity", "Alice is a software engineer at Acme Corp.")
agent.memory.save("episodic", "Last session, Alice asked about Python async patterns.")

# Chat — relevant memories are automatically retrieved
reply = agent.chat("Hi! What were we talking about last time?")
print(reply)
# → "Hey Alice! Last time we were discussing Python async patterns..."
```

---

## Auto Memory Extraction

Don't want to manually call `memory.save()`? Let the LLM extract memories for you:

```python
from contextkit import MemoryExtractor, MemoryManager

memory = MemoryManager()
extractor = MemoryExtractor(memory)

conversation = [
    {"role": "user", "content": "Hi, I'm Alice, a data scientist at Acme Corp. I love Python."},
    {"role": "assistant", "content": "Nice to meet you Alice!"},
]

result = extractor.extract_and_save(conversation)
print(f"Saved {result.total} memories")
print(f"  Entities: {result.entities}")
print(f"  Facts: {result.facts}")
# → Entities stored in entity store, facts in semantic, etc.
```

---

## Observability

See exactly which memories got injected into each chat turn:

```python
agent.chat("Tell me about Alice", debug=True)
# ────────────────────────────────────────────────────────
# QUERY: Tell me about Alice
# REPLY: Alice is a data scientist at Acme Corp...
# INPUT TOKENS: 142
# SUMMARIZED: False
# TOOL CALLS: 0
# RETRIEVED MEMORIES:
#   [entity] (1)
#     • Alice: data scientist at Acme Corp
#   [semantic] (1)
#     • Alice loves Python
# ────────────────────────────────────────────────────────

# Or access it programmatically
ctx = agent.last_context
print(ctx.input_tokens, ctx.retrieved_memories)
```

---

## Using Components Independently

Context-Kit is modular. Use only what you need:

```python
from contextkit import MemoryManager, ContextBuilder
from contextkit.config import Config

config = Config(model_name="gpt-4o", max_context_tokens=8192)
memory = MemoryManager(config=config)

# Save and query memories directly
memory.save("semantic", "FastAPI uses Starlette under the hood.")
results = memory.query("semantic", "What framework does FastAPI use?")

# Build context-enriched prompts for your own LLM calls
builder = ContextBuilder(memory, config)
messages = builder.build(
    query="Tell me about FastAPI",
    system_prompt="You are a Python expert.",
    messages=[{"role": "user", "content": "How does FastAPI work?"}],
)
```

---

## Configuration

All settings can be configured via environment variables or the `Config` object:

```python
from contextkit.config import Config

config = Config(
    openai_api_key="sk-...",            # or set OPENAI_API_KEY env var
    model_name="gpt-4o",               # LLM model for the agent
    embedding_model="text-embedding-3-small",  # embedding model for search
    chromadb_data_dir=".contextkit_data",       # or set CONTEXTKIT_CHROMADB_DIR env var
    max_context_tokens=8192,            # context window budget
    summary_threshold=0.8,             # summarize when 80% of budget is used
)
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                     Agent                       │
│  ┌──────────────┐  ┌────────────────────────┐   │
│  │ OpenAI Chat  │  │   ContextBuilder       │   │
│  │              │◄─┤  - token counting      │   │
│  │              │  │  - context assembly     │   │
│  │              │  │  - summary triggering   │   │
│  └──────────────┘  └───────────┬────────────┘   │
│                                │                │
│                    ┌───────────▼────────────┐    │
│                    │    MemoryManager       │    │
│                    │                        │    │
│  ┌─────────┬──────┴──┬──────┬───────┬──────┤    │
│  │Episodic │Semantic │Proc. │Entity │ ...  │    │
│  └────┬────┴────┬────┴──┬───┴───┬───┴──┬───┘    │
│       └─────────┴───────┴───────┴──────┘        │
│                    ChromaDB                     │
└─────────────────────────────────────────────────┘
```

---

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

All tests use ChromaDB in-memory mode and mock OpenAI calls — **no API key needed, no network, no cost**.

---

## License

MIT

---

<p align="center">
  Built by <a href="https://github.com/tonyc973">tonyc973</a>
</p>
