"""MemoryManager coordinates all seven memory stores."""

from __future__ import annotations

from typing import Any

import chromadb

from .config import Config
from .stores import (
    BufferStore,
    EntityStore,
    EpisodicStore,
    MemoryRecord,
    ProceduralStore,
    SemanticStore,
    SummaryStore,
    WorkingStore,
)


class MemoryManager:
    """Unified interface for reading and writing across all memory stores."""

    def __init__(
        self,
        config: Config | None = None,
        chroma_client: chromadb.ClientAPI | None = None,
        collection_prefix: str = "",
    ) -> None:
        self.config = config or Config()
        self._client = chroma_client or chromadb.PersistentClient(
            path=self.config.chromadb_data_dir
        )

        self.episodic = EpisodicStore(self._client, prefix=collection_prefix)
        self.semantic = SemanticStore(self._client, prefix=collection_prefix)
        self.procedural = ProceduralStore(self._client, prefix=collection_prefix)
        self.working = WorkingStore(self._client, prefix=collection_prefix)
        self.entity = EntityStore(self._client, prefix=collection_prefix)
        self.summary = SummaryStore(self._client, prefix=collection_prefix)
        self.buffer = BufferStore(self._client, prefix=collection_prefix)

        self._stores = {
            "episodic": self.episodic,
            "semantic": self.semantic,
            "procedural": self.procedural,
            "working": self.working,
            "entity": self.entity,
            "summary": self.summary,
            "buffer": self.buffer,
        }

    def save(
        self,
        store_name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        store = self._stores[store_name]
        return store.save(content, metadata=metadata)

    def query(
        self,
        store_name: str,
        text: str,
        n_results: int = 5,
    ) -> list[MemoryRecord]:
        store = self._stores[store_name]
        return store.query(text=text, n_results=n_results)

    def query_all(
        self,
        text: str,
        n_results: int = 3,
        stores: list[str] | None = None,
    ) -> dict[str, list[MemoryRecord]]:
        """Query memory stores and return results keyed by store name.

        Args:
            text: The query text.
            n_results: Max results per store.
            stores: Optional list of store names to query. If None, query all
                non-empty stores (default). Used by QueryRouter to do intent-
                based routing.
        """
        targets = stores if stores is not None else list(self._stores.keys())
        return {
            name: self._stores[name].query(text=text, n_results=n_results)
            for name in targets
            if name in self._stores and self._stores[name].count() > 0
        }

    def add_message(self, role: str, content: str) -> str:
        """Push a message into the buffer store."""
        return self.buffer.push(role, content)

    def recent_messages(self, n: int = 20) -> list[MemoryRecord]:
        return self.buffer.recent(n)
