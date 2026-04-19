"""ContextBuilder assembles context from memory stores into an LLM prompt."""

from __future__ import annotations

import tiktoken

from .config import Config
from .manager import MemoryManager
from .stores import MemoryRecord


class ContextBuilder:
    """Builds a context string from memory stores, respecting token limits."""

    def __init__(
        self,
        memory: MemoryManager,
        config: Config | None = None,
    ) -> None:
        self.memory = memory
        self.config = config or memory.config
        self._encoder = tiktoken.encoding_for_model("gpt-4o")
        self.last_retrieved: dict[str, list[MemoryRecord]] = {}

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    @property
    def token_budget(self) -> int:
        return self.config.max_context_tokens

    def needs_summary(self, messages: list[dict[str, str]]) -> bool:
        """Return True if total message tokens exceed the summary threshold."""
        total = sum(self.count_tokens(m.get("content", "")) for m in messages)
        return total > self.token_budget * self.config.summary_threshold

    def build(
        self,
        query: str,
        system_prompt: str = "",
        messages: list[dict[str, str]] | None = None,
        n_results: int = 3,
    ) -> list[dict[str, str]]:
        """Build a list of chat messages enriched with relevant memory context."""
        sections: list[str] = []

        # Gather relevant memories from all stores
        results = self.memory.query_all(query, n_results=n_results)
        self.last_retrieved = {k: v for k, v in results.items() if v}
        for store_name, records in results.items():
            if records:
                section = self._format_section(store_name, records)
                sections.append(section)

        # Build the system message
        memory_block = "\n\n".join(sections)
        system_parts = []
        if system_prompt:
            system_parts.append(system_prompt)
        if memory_block:
            system_parts.append(
                f"# Relevant memories\n\n{memory_block}"
            )

        built: list[dict[str, str]] = []
        if system_parts:
            built.append({"role": "system", "content": "\n\n".join(system_parts)})

        if messages:
            built.extend(messages)

        return built

    def _format_section(
        self, store_name: str, records: list[MemoryRecord]
    ) -> str:
        lines = [f"## {store_name.title()} Memory"]
        for rec in records:
            lines.append(f"- {rec.content}")
        return "\n".join(lines)
