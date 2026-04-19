"""Automatic memory extraction from conversations using an LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from openai import OpenAI

from .config import Config
from .manager import MemoryManager

EXTRACTION_PROMPT = """Extract memories from the following conversation.

Return a strict JSON object with these keys:
- entities: list of {"name": str, "description": str} for people/places/things mentioned
- facts: list of strings — general facts or knowledge stated (for semantic memory)
- events: list of strings — specific events that occurred (for episodic memory)
- procedures: list of strings — instructions or how-to knowledge (for procedural memory)

Rules:
- Only extract information that is genuinely worth remembering across sessions.
- Skip greetings, small talk, and filler.
- If a category has nothing, return an empty list for that key.
- Output JSON only, no prose, no markdown fences."""


@dataclass
class ExtractionResult:
    """Summary of what was extracted and saved."""

    entities: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.entities) + len(self.facts) + len(self.events) + len(self.procedures)


class MemoryExtractor:
    """Extracts structured memories from conversation text and saves them."""

    def __init__(
        self,
        memory: MemoryManager,
        config: Config | None = None,
        openai_client: OpenAI | None = None,
    ) -> None:
        self.memory = memory
        self.config = config or memory.config
        self._openai = openai_client or OpenAI(api_key=self.config.openai_api_key)

    def extract_and_save(
        self, messages: list[dict[str, str]]
    ) -> ExtractionResult:
        """Extract memories from a list of chat messages and persist them."""
        conversation = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages
        )

        response = self._openai.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": conversation},
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {}

        result = ExtractionResult()

        for entity in data.get("entities", []) or []:
            if isinstance(entity, dict) and entity.get("name"):
                text = f"{entity['name']}: {entity.get('description', '')}".strip(": ")
                self.memory.save("entity", text)
                result.entities.append(text)

        for fact in data.get("facts", []) or []:
            if isinstance(fact, str) and fact.strip():
                self.memory.save("semantic", fact)
                result.facts.append(fact)

        for event in data.get("events", []) or []:
            if isinstance(event, str) and event.strip():
                self.memory.save("episodic", event)
                result.events.append(event)

        for proc in data.get("procedures", []) or []:
            if isinstance(proc, str) and proc.strip():
                self.memory.save("procedural", proc)
                result.procedures.append(proc)

        return result
