"""Seven memory stores with distinct update semantics, backed by ChromaDB.

Each store implements the update strategy best suited to its cognitive role:

    Store          Update strategy
    ─────────      ────────────────────────────────────────────
    Episodic       Append-only (events are historical facts)
    Semantic       Similarity-based replacement (new wins)
    Procedural     Similarity-based dedup (procedures are unique)
    Entity         Name-based dedup (same name = same entity)
    Working        TTL — auto-expire after a time budget
    Summary        Append-only (lossless summary history)
    Buffer         Sliding window — evict oldest past max_size

The BaseStore provides the shared primitives: save, query, update, delete,
find_similar, and last_operation for observability.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import chromadb


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MemoryRecord:
    """A single memory entry. `distance` is populated when returned by a query."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    distance: float | None = None  # cosine distance from query (0 = identical)

    @property
    def similarity(self) -> float | None:
        """Cosine similarity in [0, 1]. None when not from a query."""
        if self.distance is None:
            return None
        # Clamp to [0, 1] — ChromaDB can return tiny negative distances for
        # identical vectors due to floating-point rounding.
        return min(1.0, max(0.0, 1.0 - self.distance))


@dataclass
class StoreOperation:
    """Describes what a save() call actually did — for observability."""

    record_id: str
    operation: str  # "created" | "updated" | "skipped"
    previous_content: str | None = None
    reason: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Base store
# ─────────────────────────────────────────────────────────────────────────────


class BaseStore:
    """Shared primitives for all memory stores."""

    store_name: str = "base"

    def __init__(self, client: chromadb.ClientAPI, prefix: str = "") -> None:
        collection_name = f"{prefix}{self.store_name}" if prefix else self.store_name
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.last_operation: StoreOperation | None = None

    # ── Core writes ────────────────────────────────────────────────────────

    def save(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        record_id: str | None = None,
    ) -> str:
        """Default save: always insert a new record. Subclasses may override to
        dedup, replace, or attach extra metadata.
        """
        rid = record_id or uuid.uuid4().hex
        meta = {**(metadata or {}), "timestamp": time.time()}
        kwargs: dict[str, Any] = {
            "ids": [rid],
            "documents": [content],
            "metadatas": [meta],
        }
        if embedding is not None:
            kwargs["embeddings"] = [embedding]
        self._collection.upsert(**kwargs)
        self.last_operation = StoreOperation(record_id=rid, operation="created")
        return rid

    def update(
        self,
        record_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Replace the content of an existing record, preserving its ID."""
        previous = self._get_content(record_id)
        meta = {**(metadata or {}), "timestamp": time.time()}
        self._collection.update(
            ids=[record_id], documents=[content], metadatas=[meta]
        )
        self.last_operation = StoreOperation(
            record_id=record_id,
            operation="updated",
            previous_content=previous,
        )

    def delete(self, record_id: str) -> None:
        self._collection.delete(ids=[record_id])

    # ── Reads ──────────────────────────────────────────────────────────────

    def query(
        self,
        text: str | None = None,
        embedding: list[float] | None = None,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        """Vector-similarity query. Returns records with `distance` populated."""
        total = self._collection.count()
        if total == 0:
            return []
        kwargs: dict[str, Any] = {"n_results": min(n_results, total)}
        if embedding is not None:
            kwargs["query_embeddings"] = [embedding]
        elif text is not None:
            kwargs["query_texts"] = [text]
        else:
            kwargs["query_texts"] = [""]
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        records: list[MemoryRecord] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0] or []
        dists = results.get("distances", [[]])[0] or []
        for i, doc_id in enumerate(ids):
            records.append(
                MemoryRecord(
                    id=doc_id,
                    content=docs[i] if i < len(docs) else "",
                    metadata=metas[i] if i < len(metas) else {},
                    distance=dists[i] if i < len(dists) else None,
                )
            )
        return records

    def find_similar(
        self, text: str, min_similarity: float = 0.85, n_results: int = 3
    ) -> list[MemoryRecord]:
        """Return records with cosine similarity ≥ `min_similarity` to `text`."""
        candidates = self.query(text=text, n_results=n_results)
        return [
            r for r in candidates
            if r.similarity is not None and r.similarity >= min_similarity
        ]

    def count(self) -> int:
        return self._collection.count()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_content(self, record_id: str) -> str | None:
        try:
            result = self._collection.get(ids=[record_id], include=["documents"])
            docs = result.get("documents") or []
            return docs[0] if docs else None
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Specialized stores
# ─────────────────────────────────────────────────────────────────────────────


class EpisodicStore(BaseStore):
    """Specific past events. Append-only — history doesn't get rewritten."""

    store_name = "episodic"


class SemanticStore(BaseStore):
    """General facts. New facts that are highly similar to existing ones replace them.

    Rationale: if the user corrects or refines a previously-stated fact, we want
    the newest version to win. Vector similarity tolerates paraphrasing, so
    "Alice lives in Paris" and "Alice's home is in Paris" merge naturally.
    """

    store_name = "semantic"

    def __init__(
        self,
        client: chromadb.ClientAPI,
        prefix: str = "",
        similarity_threshold: float = 0.85,
    ) -> None:
        super().__init__(client, prefix)
        self._similarity_threshold = similarity_threshold

    def save(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        record_id: str | None = None,
    ) -> str:
        # Search for a near-duplicate
        if self.count() > 0:
            similar = self.find_similar(
                content, min_similarity=self._similarity_threshold, n_results=1
            )
            if similar:
                existing = similar[0]
                self.update(existing.id, content, metadata)
                self.last_operation = StoreOperation(
                    record_id=existing.id,
                    operation="updated",
                    previous_content=existing.content,
                    reason=f"similarity={existing.similarity:.2f} ≥ "
                           f"{self._similarity_threshold:.2f}",
                )
                return existing.id
        return super().save(content, metadata, embedding, record_id)


class ProceduralStore(BaseStore):
    """Rules and how-tos. Highly similar procedures are merged."""

    store_name = "procedural"

    def __init__(
        self,
        client: chromadb.ClientAPI,
        prefix: str = "",
        similarity_threshold: float = 0.88,
    ) -> None:
        super().__init__(client, prefix)
        self._similarity_threshold = similarity_threshold

    def save(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        record_id: str | None = None,
    ) -> str:
        if self.count() > 0:
            similar = self.find_similar(
                content, min_similarity=self._similarity_threshold, n_results=1
            )
            if similar:
                existing = similar[0]
                self.update(existing.id, content, metadata)
                self.last_operation = StoreOperation(
                    record_id=existing.id,
                    operation="updated",
                    previous_content=existing.content,
                    reason=f"similarity={existing.similarity:.2f} ≥ "
                           f"{self._similarity_threshold:.2f}",
                )
                return existing.id
        return super().save(content, metadata, embedding, record_id)


class EntityStore(BaseStore):
    """People, places, things. Deduped by canonical name (case-insensitive).

    The name is parsed from the content: the part before the first colon, or
    the first word if no colon is present. "Alice: junior engineer" and
    "Alice: senior engineer" are treated as the same Alice — the latter wins.
    """

    store_name = "entity"

    def __init__(
        self,
        client: chromadb.ClientAPI,
        prefix: str = "",
        dedup_enabled: bool = True,
    ) -> None:
        super().__init__(client, prefix)
        self._dedup_enabled = dedup_enabled

    @staticmethod
    def _canonical_name(content: str) -> str:
        if ":" in content:
            name = content.split(":", 1)[0]
        else:
            tokens = content.split()
            name = tokens[0] if tokens else content
        return name.strip().lower()

    def save(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        record_id: str | None = None,
    ) -> str:
        meta = {**(metadata or {})}
        if self._dedup_enabled:
            name = self._canonical_name(content)
            meta["entity_name"] = name
            existing_id = self._find_by_name(name)
            if existing_id is not None:
                previous = self._get_content(existing_id)
                self._collection.update(
                    ids=[existing_id],
                    documents=[content],
                    metadatas=[{**meta, "timestamp": time.time()}],
                )
                self.last_operation = StoreOperation(
                    record_id=existing_id,
                    operation="updated",
                    previous_content=previous,
                    reason=f"entity_name='{name}' already exists",
                )
                return existing_id
        return super().save(content, meta, embedding, record_id)

    def _find_by_name(self, name: str) -> str | None:
        try:
            result = self._collection.get(
                where={"entity_name": name}, limit=1, include=[]
            )
            ids = result.get("ids") or []
            return ids[0] if ids else None
        except Exception:
            return None


class WorkingStore(BaseStore):
    """Short-term session memory. Entries auto-expire after a TTL."""

    store_name = "working"

    def __init__(
        self,
        client: chromadb.ClientAPI,
        prefix: str = "",
        default_ttl_seconds: float = 3600.0,
    ) -> None:
        super().__init__(client, prefix)
        self._default_ttl = default_ttl_seconds

    def save(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        record_id: str | None = None,
        ttl_seconds: float | None = None,
    ) -> str:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        meta = {**(metadata or {}), "expires_at": time.time() + ttl}
        return super().save(content, meta, embedding, record_id)

    def query(
        self,
        text: str | None = None,
        embedding: list[float] | None = None,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        self.cleanup_expired()
        return super().query(text, embedding, n_results, where)

    def cleanup_expired(self) -> int:
        """Delete entries whose expires_at is in the past. Returns count deleted."""
        if self._collection.count() == 0:
            return 0
        result = self._collection.get(include=["metadatas"])
        now = time.time()
        expired_ids = [
            rid
            for rid, meta in zip(result.get("ids", []), result.get("metadatas", []) or [])
            if meta and meta.get("expires_at", float("inf")) < now
        ]
        if expired_ids:
            self._collection.delete(ids=expired_ids)
        return len(expired_ids)


class SummaryStore(BaseStore):
    """Compressed conversation history. Append-only — summaries are immutable."""

    store_name = "summary"


class BufferStore(BaseStore):
    """Sliding window of recent raw messages, capped at max_size."""

    store_name = "buffer"
    _seq: int = 0

    def __init__(
        self,
        client: chromadb.ClientAPI,
        prefix: str = "",
        max_size: int = 50,
    ) -> None:
        super().__init__(client, prefix)
        self._max_size = max_size

    def push(self, role: str, content: str, **extra_meta: Any) -> str:
        BufferStore._seq += 1
        rid = self.save(
            content,
            metadata={"role": role, "seq": BufferStore._seq, **extra_meta},
        )
        self._enforce_window()
        return rid

    def _enforce_window(self) -> None:
        total = self._collection.count()
        if total <= self._max_size:
            return
        overflow = total - self._max_size
        all_records = self._collection.get(include=["metadatas"])
        pairs = sorted(
            zip(all_records.get("ids", []), all_records.get("metadatas", []) or []),
            key=lambda p: (p[1] or {}).get("seq", 0),
        )
        to_delete = [rid for rid, _ in pairs[:overflow]]
        if to_delete:
            self._collection.delete(ids=to_delete)

    def recent(self, n: int = 20) -> list[MemoryRecord]:
        total = self._collection.count()
        if total == 0:
            return []
        results = self._collection.get(
            limit=total,
            include=["documents", "metadatas"],
        )
        records = []
        ids = results.get("ids", [])
        docs = results.get("documents", []) or []
        metas = results.get("metadatas", []) or []
        for i, doc_id in enumerate(ids):
            records.append(
                MemoryRecord(
                    id=doc_id,
                    content=docs[i] if i < len(docs) else "",
                    metadata=metas[i] if i < len(metas) else {},
                )
            )
        records.sort(
            key=lambda r: (r.metadata.get("seq", 0), r.metadata.get("timestamp", 0))
        )
        return records[-n:]
