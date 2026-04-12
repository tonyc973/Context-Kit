# Claude Code prompts for contextkit

Put CONTEXTKIT_SPEC.md in your repo root. Then paste these one at a time.

---

## Prompt 1 — Build the whole thing

```
Read CONTEXTKIT_SPEC.md.

Build this library. Start with pyproject.toml, then stores.py (all 7 stores),
then manager.py (MemoryManager), then context_builder.py, then agent.py and
tools.py. Write config.py for settings (OpenAI API key from env var, model name,
ChromaDB data dir, embedding model name).

Export MemoryManager, ContextBuilder, and Agent from __init__.py.

Write tests in tests/ that use ChromaDB in-memory mode and mock OpenAI calls.
Test: saving and reading from each store, building context, and the
summarization trigger (stuff enough tokens to exceed 80% threshold).

Write examples/quickstart.py matching the usage example in the spec.

Run all tests and make sure they pass.
```

---

## Prompt 2 — Polish

```
Add a README.md with:
- One paragraph explaining what contextkit does
- Install instructions (pip install -e .)
- The quickstart code from the spec
- A short explanation of the 7 memory types (just copy the table)

Then run ruff check and fix any lint issues. Run tests once more.
```

---

## That's it

Two prompts. You'll have a working library in ~10 minutes.

If something breaks, just say "fix it" — Claude Code will figure it out.
