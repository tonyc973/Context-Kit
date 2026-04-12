"""Tests for all seven memory stores."""

import pytest

from contextkit.stores import (
    BufferStore,
    EntityStore,
    EpisodicStore,
    ProceduralStore,
    SemanticStore,
    SummaryStore,
    WorkingStore,
)


STORE_CLASSES = [
    EpisodicStore,
    SemanticStore,
    ProceduralStore,
    WorkingStore,
    EntityStore,
    SummaryStore,
    BufferStore,
]


@pytest.mark.parametrize("store_cls", STORE_CLASSES, ids=lambda c: c.store_name)
def test_save_and_query(chroma_client, unique_prefix, store_cls):
    store = store_cls(chroma_client, prefix=unique_prefix)
    rid = store.save("The capital of France is Paris", metadata={"topic": "geography"})
    assert isinstance(rid, str)
    assert store.count() == 1

    results = store.query(text="France capital", n_results=1)
    assert len(results) == 1
    assert "Paris" in results[0].content


@pytest.mark.parametrize("store_cls", STORE_CLASSES, ids=lambda c: c.store_name)
def test_delete(chroma_client, unique_prefix, store_cls):
    store = store_cls(chroma_client, prefix=unique_prefix)
    rid = store.save("temporary data")
    assert store.count() == 1
    store.delete(rid)
    assert store.count() == 0


@pytest.mark.parametrize("store_cls", STORE_CLASSES, ids=lambda c: c.store_name)
def test_query_empty_store(chroma_client, unique_prefix, store_cls):
    store = store_cls(chroma_client, prefix=unique_prefix)
    results = store.query(text="anything", n_results=5)
    assert results == []


def test_buffer_push_and_recent(chroma_client, unique_prefix):
    buf = BufferStore(chroma_client, prefix=unique_prefix)
    buf.push("user", "Hello")
    buf.push("assistant", "Hi there!")
    buf.push("user", "How are you?")

    recent = buf.recent(n=2)
    assert len(recent) == 2
    assert recent[-1].content == "How are you?"
