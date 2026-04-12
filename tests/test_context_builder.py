"""Tests for ContextBuilder including the summarization trigger."""

import pytest

from contextkit.config import Config
from contextkit.context_builder import ContextBuilder
from contextkit.manager import MemoryManager


@pytest.fixture()
def builder(chroma_client, unique_prefix):
    config = Config(max_context_tokens=100, summary_threshold=0.8)
    memory = MemoryManager(
        config=config, chroma_client=chroma_client, collection_prefix=unique_prefix
    )
    return ContextBuilder(memory, config)


def test_build_with_no_memories(builder):
    messages = builder.build(query="hello", system_prompt="You are helpful.")
    assert messages[0]["role"] == "system"
    assert "You are helpful" in messages[0]["content"]


def test_build_with_memories(builder):
    builder.memory.save("semantic", "Python was created by Guido van Rossum")
    messages = builder.build(query="Who created Python?", system_prompt="Be concise.")
    system_content = messages[0]["content"]
    assert "Guido" in system_content
    assert "Semantic Memory" in system_content


def test_build_includes_passed_messages(builder):
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    messages = builder.build(query="Hi", messages=history)
    user_msgs = [m for m in messages if m["role"] == "user"]
    assert len(user_msgs) == 1


def test_needs_summary_false(builder):
    messages = [{"role": "user", "content": "short"}]
    assert builder.needs_summary(messages) is False


def test_needs_summary_true(builder):
    """Stuff enough tokens to exceed the 80% threshold of 100 tokens."""
    long_content = " ".join(f"word{i}" for i in range(100))
    messages = [{"role": "user", "content": long_content}]
    assert builder.needs_summary(messages) is True


def test_count_tokens(builder):
    count = builder.count_tokens("Hello world")
    assert isinstance(count, int)
    assert count > 0
