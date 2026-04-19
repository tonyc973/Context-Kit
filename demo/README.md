# Memory Bot — Context-Kit Demo

A live Streamlit demo showing Context-Kit's memory system in action.

## Features

- 💬 **Chat interface** — talk to an agent that remembers you across sessions
- 📊 **Live memory inspector** — watch all 7 memory stores update in real time
- 🔍 **Per-turn observability** — see exactly which memories were retrieved for each response, plus token counts
- ✨ **Auto memory extraction** — the bot automatically pulls entities, facts, events, and procedures from conversation
- 💾 **Persistent memory** — close the tab, come back, everything is still there

## Run locally

```bash
pip install -e ".[demo]"
export OPENAI_API_KEY="sk-..."
streamlit run demo/app.py
```

Then open http://localhost:8501.

## Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo, set `demo/app.py` as the entry point
4. Add `OPENAI_API_KEY` in the app secrets
5. Deploy — you get a public URL instantly

## Screenshots

The UI has:
- **Sidebar**: API config, memory inspector showing all 7 stores with counts and recent entries
- **Main**: chat interface with a "Memory trace" expander under each reply showing retrieved memories and token usage

## What's in the memory?

After a few turns, click through the sidebar expanders to see what the bot has learned:

- **Entity** — people, places, things you mentioned
- **Semantic** — general facts you stated
- **Episodic** — specific events you described
- **Procedural** — instructions or how-tos
- **Working** — ephemeral short-term context
- **Summary** — compressed past conversations
- **Buffer** — raw recent messages

This is how Context-Kit turns a stateless LLM into something that feels like it *knows* you.
