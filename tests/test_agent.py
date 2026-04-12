"""Tests for Agent with mocked OpenAI calls."""

from unittest.mock import MagicMock

import pytest

from contextkit.agent import Agent
from contextkit.config import Config
from contextkit.manager import MemoryManager


def _make_mock_response(content="Hello!", finish_reason="stop", tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture()
def agent(chroma_client, unique_prefix):
    config = Config(max_context_tokens=200, summary_threshold=0.8)
    memory = MemoryManager(
        config=config, chroma_client=chroma_client, collection_prefix=unique_prefix
    )
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = _make_mock_response("Hi there!")
    return Agent(
        config=config,
        memory=memory,
        openai_client=mock_openai,
    )


def test_chat_returns_reply(agent):
    reply = agent.chat("Hello")
    assert reply == "Hi there!"


def test_chat_stores_messages_in_buffer(agent):
    agent.chat("Hello")
    recent = agent.memory.recent_messages(n=10)
    assert len(recent) == 2  # user + assistant
    roles = [r.metadata["role"] for r in recent]
    assert "user" in roles
    assert "assistant" in roles


def test_chat_builds_history(agent):
    agent.chat("First message")
    agent.chat("Second message")
    assert len(agent._history) == 4  # 2 user + 2 assistant


def test_summarization_trigger(agent):
    """Stuff enough tokens to exceed the 80% threshold and trigger summarization."""
    agent.config.max_context_tokens = 100
    agent.config.summary_threshold = 0.8

    summary_resp = _make_mock_response("Summary of conversation so far.")
    normal_resp = _make_mock_response("Got it!")
    agent._openai.chat.completions.create.side_effect = [
        normal_resp,  # first chat
        normal_resp,  # second chat
        summary_resp,  # summarization call
        normal_resp,  # third chat (after summarization)
    ]

    long_msg = " ".join(f"word{i}" for i in range(60))
    agent.chat(long_msg)
    agent.chat(long_msg)
    agent.chat("short follow-up")

    # Verify summary was saved
    summaries = agent.memory.summary.query(text="conversation", n_results=5)
    assert len(summaries) > 0
