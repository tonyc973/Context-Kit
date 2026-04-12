"""Tests for MemoryManager."""

import pytest

from contextkit.config import Config
from contextkit.manager import MemoryManager


@pytest.fixture()
def manager(chroma_client, unique_prefix):
    return MemoryManager(
        config=Config(), chroma_client=chroma_client, collection_prefix=unique_prefix
    )


def test_save_and_query_each_store(manager):
    store_names = ["episodic", "semantic", "procedural", "working", "entity", "summary"]
    for name in store_names:
        manager.save(name, f"Test content for {name}")
        results = manager.query(name, "Test content", n_results=1)
        assert len(results) == 1
        assert name in results[0].content


def test_add_message_and_recent(manager):
    manager.add_message("user", "Hello")
    manager.add_message("assistant", "Hi!")
    recent = manager.recent_messages(n=10)
    assert len(recent) == 2


def test_query_all(manager):
    manager.save("episodic", "Met Alice at the conference")
    manager.save("entity", "Alice is a software engineer")
    results = manager.query_all("Alice")
    assert "episodic" in results
    assert "entity" in results
