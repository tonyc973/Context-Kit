"""Seven memory stores backed by ChromaDB collections."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import chromadb


@dataclass
class MemoryRecord:
    """A single memory entry."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class BaseStore:
    """Base class for all memory stores."""

    store_name: str = "base"

    def __init__(self, client: chromadb.ClientAPI, prefix: str = "") -> None:
        collection_name = f"{prefix}{self.store_name}" if prefix else self.store_name
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def save(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        record_id: str | None = None,
    ) -> str:
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
        return rid

    def query(
        self,
        text: str | None = None,
        embedding: list[float] | None = None,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        kwargs: dict[str, Any] = {"n_results": min(n_results, self._collection.count() or 1)}
        if self._collection.count() == 0:
            return []
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
        for i, doc_id in enumerate(results["ids"][0]):
            records.append(
                MemoryRecord(
                    id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
            )
        return records

    def count(self) -> int:
        return self._collection.count()

    def delete(self, record_id: str) -> None:
        self._collection.delete(ids=[record_id])


class EpisodicStore(BaseStore):
    """Stores specific past interactions and events."""

    store_name = "episodic"


class SemanticStore(BaseStore):
    """Stores general facts and knowledge."""

    store_name = "semantic"


class ProceduralStore(BaseStore):
    """Stores instructions and how-to knowledge."""

    store_name = "procedural"


class WorkingStore(BaseStore):
    """Short-term memory for the current session."""

    store_name = "working"


class EntityStore(BaseStore):
    """Stores information about specific entities (people, places, things)."""

    store_name = "entity"


class SummaryStore(BaseStore):
    """Stores compressed summaries of past conversations."""

    store_name = "summary"


class BufferStore(BaseStore):
    """Sliding-window buffer of recent raw messages."""

    store_name = "buffer"

    _seq: int = 0

    def push(self, role: str, content: str, **extra_meta: Any) -> str:
        BufferStore._seq += 1
        return self.save(
            content, metadata={"role": role, "seq": BufferStore._seq, **extra_meta}
        )

    def recent(self, n: int = 20) -> list[MemoryRecord]:
        total = self._collection.count()
        if total == 0:
            return []
        results = self._collection.get(
            limit=total,
            include=["documents", "metadatas"],
        )
        records = []
        for i, doc_id in enumerate(results["ids"]):
            records.append(
                MemoryRecord(
                    id=doc_id,
                    content=results["documents"][i],
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                )
            )
        # Sort by sequence number (reliable), falling back to timestamp
        records.sort(key=lambda r: (r.metadata.get("seq", 0), r.metadata.get("timestamp", 0)))
        return records[-n:]
