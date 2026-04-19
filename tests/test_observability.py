"""Tests for Agent observability (last_context, debug mode)."""

from unittest.mock import MagicMock

import pytest

from contextkit.agent import Agent, ChatContext
from contextkit.config import Config
from contextkit.manager import MemoryManager


def _make_mock_response(content="reply", finish_reason="stop", tool_calls=None):
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
    config = Config()
    memory = MemoryManager(
        config=config, chroma_client=chroma_client, collection_prefix=unique_prefix
    )
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = _make_mock_response("hello!")
    return Agent(config=config, memory=memory, openai_client=mock_openai)


def test_last_context_populated_after_chat(agent):
    agent.chat("What's my name?")
    assert isinstance(agent.last_context, ChatContext)
    assert agent.last_context.query == "What's my name?"
    assert agent.last_context.reply == "hello!"
    assert agent.last_context.input_tokens > 0


def test_last_context_includes_retrieved_memories(agent):
    agent.memory.save("semantic", "The user's name is Alice")
    agent.memory.save("entity", "Alice is a data scientist")
    agent.chat("Tell me about myself")

    retrieved = agent.last_context.retrieved_memories
    assert "semantic" in retrieved or "entity" in retrieved
    assert sum(len(v) for v in retrieved.values()) > 0


def test_debug_mode_prints(agent, capsys):
    agent.memory.save("semantic", "The sky is blue")
    agent.chat("What color is the sky?", debug=True)
    captured = capsys.readouterr()
    assert "QUERY:" in captured.out
    assert "REPLY:" in captured.out
    assert "INPUT TOKENS:" in captured.out


def test_pretty_output_format(agent):
    agent.chat("hi")
    pretty = agent.last_context.pretty()
    assert "QUERY:" in pretty
    assert "REPLY:" in pretty


def test_context_builder_tracks_last_retrieved(agent):
    agent.memory.save("semantic", "Fact one")
    agent.chat("query one")
    assert isinstance(agent.context_builder.last_retrieved, dict)
