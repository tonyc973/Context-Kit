"""Memory Bot — a Streamlit demo for Context-Kit.

A chat interface backed by Context-Kit's seven memory stores, with a live
memory inspector and per-turn observability panel.

Run locally:
    streamlit run demo/app.py

Set your OpenAI API key via environment variable (OPENAI_API_KEY) or in the
sidebar input.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from contextkit import Agent, MemoryExtractor, MemoryManager
from contextkit.config import Config

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Memory Bot — powered by Context-Kit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parent.parent / ".contextkit_demo_data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

STORE_ICONS = {
    "episodic": "📅",
    "semantic": "📚",
    "procedural": "🛠️",
    "working": "⚡",
    "entity": "👤",
    "summary": "📝",
    "buffer": "💬",
}

STORE_DESCRIPTIONS = {
    "episodic": "Specific past interactions & events",
    "semantic": "General facts & knowledge",
    "procedural": "Instructions & how-to knowledge",
    "working": "Short-term session context",
    "entity": "People, places & things",
    "summary": "Compressed conversation history",
    "buffer": "Recent raw messages",
}


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "extractor" not in st.session_state:
        st.session_state.extractor = None
    if "auto_extract" not in st.session_state:
        st.session_state.auto_extract = True
    if "use_routing" not in st.session_state:
        st.session_state.use_routing = True


def build_agent(api_key: str, model: str, use_routing: bool) -> Agent:
    """Create an agent that persists memory to the local data directory."""
    config = Config(
        openai_api_key=api_key,
        model_name=model,
        chromadb_data_dir=str(DATA_DIR),
        max_context_tokens=8192,
    )
    memory = MemoryManager(config=config)
    agent = Agent(
        config=config,
        memory=memory,
        use_routing=use_routing,
        system_prompt=(
            "You are Memory Bot — a helpful assistant with long-term memory. "
            "Remember facts about the user and their preferences. "
            "Use what you know about them to give personalized responses."
        ),
    )
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuration & memory inspector
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    st.sidebar.title("🧠 Memory Bot")
    st.sidebar.caption("Powered by Context-Kit")

    with st.sidebar.expander("⚙️  Configuration", expanded=not st.session_state.agent):
        env_key = os.environ.get("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API Key",
            value=env_key,
            type="password",
            help="Your key is only used in this session. Set OPENAI_API_KEY env var to auto-fill.",
        )
        model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            index=0,
        )
        st.session_state.auto_extract = st.toggle(
            "Auto-extract memories",
            value=st.session_state.auto_extract,
            help="After each turn, use the LLM to automatically extract entities, facts, events, and procedures into the right stores.",
        )
        st.session_state.use_routing = st.toggle(
            "Smart query routing",
            value=st.session_state.use_routing,
            help="Use a fast LLM to classify each query and only search the relevant memory stores. Cuts token costs and improves precision.",
        )

        if st.button("🔌 Connect", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please provide an OpenAI API key.")
            else:
                st.session_state.agent = build_agent(
                    api_key, model, st.session_state.use_routing
                )
                st.session_state.extractor = MemoryExtractor(
                    st.session_state.agent.memory,
                    config=st.session_state.agent.config,
                )
                st.success("Connected!")
                st.rerun()

    if st.session_state.agent is None:
        st.sidebar.info("👆 Connect to start chatting")
        return

    # Memory inspector
    st.sidebar.divider()
    st.sidebar.subheader("📊 Memory Inspector")

    memory = st.session_state.agent.memory
    stores = {
        "episodic": memory.episodic,
        "semantic": memory.semantic,
        "procedural": memory.procedural,
        "working": memory.working,
        "entity": memory.entity,
        "summary": memory.summary,
        "buffer": memory.buffer,
    }

    for name, store in stores.items():
        count = store.count()
        icon = STORE_ICONS[name]
        with st.sidebar.expander(
            f"{icon} **{name.title()}** — {count}",
            expanded=False,
        ):
            st.caption(STORE_DESCRIPTIONS[name])
            if count == 0:
                st.caption("*(empty)*")
            else:
                # Show up to 5 most recent entries
                records = store.query(text="", n_results=min(count, 5))
                for r in records:
                    st.markdown(f"- {r.content}")

    st.sidebar.divider()
    col1, col2 = st.sidebar.columns(2)
    if col1.button("🗑️  Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent._history = []
        st.rerun()
    if col2.button("💥 Wipe memory", use_container_width=True, type="secondary"):
        import shutil
        shutil.rmtree(DATA_DIR, ignore_errors=True)
        DATA_DIR.mkdir(exist_ok=True)
        st.session_state.agent = None
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────────────────

def render_memory_trace(ctx) -> None:
    """Render the observability panel for a single turn."""
    with st.expander(
        f"🔍 Memory trace — {ctx.input_tokens} tokens, "
        f"{sum(len(v) for v in ctx.retrieved_memories.values())} memories retrieved"
    ):
        col1, col2, col3 = st.columns(3)
        col1.metric("Input tokens", ctx.input_tokens)
        col2.metric("Summarized?", "Yes" if ctx.summarized else "No")
        col3.metric("Tool calls", ctx.tool_calls_made)

        if ctx.routed_to is not None:
            routed_icons = " ".join(
                f"{STORE_ICONS.get(s, '•')} `{s}`" for s in ctx.routed_to
            )
            st.markdown(f"**🎯 Routed to:** {routed_icons}")

        if not ctx.retrieved_memories:
            st.caption("No memories retrieved for this turn.")
            return

        st.markdown("**Retrieved memories:**")
        for store_name, records in ctx.retrieved_memories.items():
            icon = STORE_ICONS.get(store_name, "•")
            st.markdown(f"{icon} **{store_name.title()}** ({len(records)})")
            for r in records:
                st.markdown(f"  - {r.content}")


def render_chat() -> None:
    st.title("💬 Chat with Memory Bot")
    st.caption(
        "Tell me about yourself — I'll remember. Every conversation enriches my memory of you."
    )

    if st.session_state.agent is None:
        st.info("👈 Configure your OpenAI key in the sidebar to start.")
        st.markdown("### What makes this demo special?")
        st.markdown(
            """
            - **Seven specialized memory stores** inspired by cognitive science
            - **Live memory inspector** in the sidebar — watch what the bot learns
            - **Per-turn observability** — see exactly which memories get retrieved
            - **Automatic extraction** — the bot distills facts from conversation
            - **Persistent memory** — close the tab, come back, it still remembers
            """
        )
        return

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("ctx"):
                render_memory_trace(msg["ctx"])

    # New input
    if prompt := st.chat_input("Say something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = st.session_state.agent.chat(prompt)
                    ctx = st.session_state.agent.last_context
                except Exception as e:
                    st.error(f"Error: {e}")
                    return

                st.markdown(reply)
                render_memory_trace(ctx)

                # Auto-extract in the background
                if st.session_state.auto_extract and st.session_state.extractor:
                    try:
                        recent = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": reply},
                        ]
                        result = st.session_state.extractor.extract_and_save(recent)
                        if result.total > 0:
                            st.caption(
                                f"✨ Auto-extracted {result.total} memories "
                                f"({len(result.entities)} entities, "
                                f"{len(result.facts)} facts, "
                                f"{len(result.events)} events, "
                                f"{len(result.procedures)} procedures)"
                            )
                    except Exception as e:
                        st.caption(f"(Auto-extract skipped: {e})")

        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "ctx": ctx}
        )
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    init_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
