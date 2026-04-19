"""Tests for per-store update semantics: dedup, replacement, TTL, eviction."""

import time

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


# ─────────────────────────────────────────────────────────────────────────────
# Entity — name-based dedup
# ─────────────────────────────────────────────────────────────────────────────


def test_entity_dedup_same_name_updates(chroma_client, unique_prefix):
    store = EntityStore(chroma_client, prefix=unique_prefix)
    id1 = store.save("Alice: junior engineer at Acme")
    id2 = store.save("Alice: senior engineer at Acme")

    assert id1 == id2, "same-name saves should map to the same record"
    assert store.count() == 1
    assert store.last_operation.operation == "updated"
    assert "junior" in store.last_operation.previous_content


def test_entity_dedup_is_case_insensitive(chroma_client, unique_prefix):
    store = EntityStore(chroma_client, prefix=unique_prefix)
    id1 = store.save("alice: description one")
    id2 = store.save("Alice: description two")
    id3 = store.save("ALICE: description three")

    assert id1 == id2 == id3
    assert store.count() == 1


def test_entity_different_names_create_separate_records(chroma_client, unique_prefix):
    store = EntityStore(chroma_client, prefix=unique_prefix)
    store.save("Alice: engineer")
    store.save("Bob: designer")

    assert store.count() == 2


def test_entity_dedup_can_be_disabled(chroma_client, unique_prefix):
    store = EntityStore(chroma_client, prefix=unique_prefix, dedup_enabled=False)
    store.save("Alice: v1")
    store.save("Alice: v2")

    assert store.count() == 2


def test_entity_handles_content_without_colon(chroma_client, unique_prefix):
    store = EntityStore(chroma_client, prefix=unique_prefix)
    id1 = store.save("Alice the engineer")
    id2 = store.save("Alice the painter")

    # Both start with "Alice" → same canonical name
    assert id1 == id2
    assert store.count() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Semantic — similarity-based replacement
# ─────────────────────────────────────────────────────────────────────────────


def test_semantic_replaces_near_duplicate(chroma_client, unique_prefix):
    store = SemanticStore(
        chroma_client, prefix=unique_prefix, similarity_threshold=0.5
    )
    id1 = store.save("Python is a programming language")
    id2 = store.save("Python is a programming language")  # identical

    assert id1 == id2
    assert store.count() == 1
    assert store.last_operation.operation == "updated"


def test_semantic_keeps_distinct_facts(chroma_client, unique_prefix):
    store = SemanticStore(
        chroma_client, prefix=unique_prefix, similarity_threshold=0.99
    )
    store.save("Python is a programming language")
    store.save("Coffee grows in tropical regions")

    assert store.count() == 2


def test_semantic_first_save_is_created(chroma_client, unique_prefix):
    store = SemanticStore(chroma_client, prefix=unique_prefix)
    store.save("A novel fact")

    assert store.last_operation.operation == "created"
    assert store.count() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Procedural — similarity dedup
# ─────────────────────────────────────────────────────────────────────────────


def test_procedural_dedup_identical(chroma_client, unique_prefix):
    store = ProceduralStore(
        chroma_client, prefix=unique_prefix, similarity_threshold=0.5
    )
    id1 = store.save("Always write tests before code")
    id2 = store.save("Always write tests before code")

    assert id1 == id2
    assert store.count() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Working — TTL
# ─────────────────────────────────────────────────────────────────────────────


def test_working_ttl_expires_entries(chroma_client, unique_prefix):
    store = WorkingStore(
        chroma_client, prefix=unique_prefix, default_ttl_seconds=0.1
    )
    store.save("short-lived context")
    assert store.count() == 1

    time.sleep(0.15)
    # Query triggers cleanup
    results = store.query(text="anything", n_results=5)
    assert results == []
    assert store.count() == 0


def test_working_non_expired_entries_survive(chroma_client, unique_prefix):
    store = WorkingStore(
        chroma_client, prefix=unique_prefix, default_ttl_seconds=10.0
    )
    store.save("still valid")
    results = store.query(text="still", n_results=5)
    assert len(results) == 1


def test_working_custom_ttl_overrides_default(chroma_client, unique_prefix):
    store = WorkingStore(
        chroma_client, prefix=unique_prefix, default_ttl_seconds=10.0
    )
    store.save("short", ttl_seconds=0.05)
    time.sleep(0.1)
    store.cleanup_expired()
    assert store.count() == 0


def test_working_cleanup_returns_count(chroma_client, unique_prefix):
    store = WorkingStore(
        chroma_client, prefix=unique_prefix, default_ttl_seconds=0.05
    )
    store.save("a")
    store.save("b")
    time.sleep(0.1)
    deleted = store.cleanup_expired()
    assert deleted == 2


# ─────────────────────────────────────────────────────────────────────────────
# Buffer — sliding window
# ─────────────────────────────────────────────────────────────────────────────


def test_buffer_enforces_max_size(chroma_client, unique_prefix):
    store = BufferStore(chroma_client, prefix=unique_prefix, max_size=3)
    for i in range(6):
        store.push("user", f"message {i}")

    assert store.count() == 3
    recent = store.recent(n=10)
    # Oldest two evicted — we should see messages 3, 4, 5
    contents = [r.content for r in recent]
    assert contents == ["message 3", "message 4", "message 5"]


def test_buffer_under_limit_keeps_everything(chroma_client, unique_prefix):
    store = BufferStore(chroma_client, prefix=unique_prefix, max_size=50)
    for i in range(5):
        store.push("user", f"msg {i}")
    assert store.count() == 5


# ─────────────────────────────────────────────────────────────────────────────
# Episodic & Summary — append-only
# ─────────────────────────────────────────────────────────────────────────────


def test_episodic_is_append_only(chroma_client, unique_prefix):
    store = EpisodicStore(chroma_client, prefix=unique_prefix)
    store.save("Met Alice on Monday")
    store.save("Met Alice on Monday")  # exact duplicate

    assert store.count() == 2, "episodic should never dedup — history is append-only"


def test_summary_is_append_only(chroma_client, unique_prefix):
    store = SummaryStore(chroma_client, prefix=unique_prefix)
    store.save("Summary A")
    store.save("Summary A")
    assert store.count() == 2


# ─────────────────────────────────────────────────────────────────────────────
# BaseStore: update + find_similar + distance on query results
# ─────────────────────────────────────────────────────────────────────────────


def test_explicit_update_preserves_id(chroma_client, unique_prefix):
    store = EpisodicStore(chroma_client, prefix=unique_prefix)
    rid = store.save("original content")
    store.update(rid, "revised content")

    assert store.count() == 1
    assert store.last_operation.operation == "updated"
    assert store.last_operation.previous_content == "original content"


def test_query_populates_distance(chroma_client, unique_prefix):
    store = SemanticStore(
        chroma_client, prefix=unique_prefix, similarity_threshold=0.99
    )
    store.save("the quick brown fox")
    results = store.query(text="the quick brown fox", n_results=1)

    assert len(results) == 1
    assert results[0].distance is not None
    assert 0.0 <= results[0].similarity <= 1.0


def test_find_similar_filters_by_threshold(chroma_client, unique_prefix):
    store = SemanticStore(
        chroma_client, prefix=unique_prefix, similarity_threshold=0.99
    )
    store.save("completely unrelated content")

    # Very high threshold, unrelated query → no matches
    assert store.find_similar("totally different topic", min_similarity=0.99) == []

    # Very low threshold → returns something
    assert store.find_similar("anything", min_similarity=0.0) != []


# ─────────────────────────────────────────────────────────────────────────────
# Integration: MemoryExtractor + dedup (extracting Alice twice → 1 record)
# ─────────────────────────────────────────────────────────────────────────────


def test_extractor_benefits_from_entity_dedup(chroma_client, unique_prefix):
    """When the extractor saves the same entity twice, we get one record."""
    from contextkit.config import Config
    from contextkit.manager import MemoryManager

    memory = MemoryManager(
        config=Config(), chroma_client=chroma_client, collection_prefix=unique_prefix
    )
    memory.save("entity", "Alice: junior engineer")
    memory.save("entity", "Alice: senior engineer")
    memory.save("entity", "Bob: designer")

    assert memory.entity.count() == 2  # Alice (deduped) + Bob
