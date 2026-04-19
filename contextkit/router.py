"""QueryRouter — LLM-based intent classifier that picks which memory stores to search."""

from __future__ import annotations

import json

from openai import OpenAI

from .config import Config

ALL_STORES = ["episodic", "semantic", "procedural", "entity", "working", "summary"]

ROUTING_PROMPT = """You are a memory query router. Given a user query, decide which memory stores are relevant to search.

Memory stores:
- episodic: specific past events and interactions ("yesterday's meeting", "last time we talked")
- semantic: general facts and knowledge ("Python is a language", "FastAPI is fast")
- procedural: instructions, rules, how-to knowledge ("always write tests", "deploy with...")
- entity: people, places, organizations, products ("Alice", "Acme Corp")
- working: short-term current-session context
- summary: compressed past conversations

Rules:
- Be selective. Only include stores genuinely likely to contain relevant info.
- If the query clearly targets one type (e.g. "who did I mention?") return only that one.
- If the query is broad ("tell me about myself"), return multiple.
- Never return an empty list — if unsure, default to ["semantic", "entity"].

Return strict JSON: {"stores": ["name1", "name2", ...]}. No prose, no markdown fences."""


class QueryRouter:
    """Uses a fast LLM to classify queries and pick relevant memory stores."""

    def __init__(
        self,
        config: Config | None = None,
        openai_client: OpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self.config = config or Config()
        self._openai = openai_client or OpenAI(api_key=self.config.openai_api_key)
        # Default to a cheap+fast model for routing, regardless of main agent model
        self.model = model or "gpt-4o-mini"
        self.last_decision: list[str] = []

    def route(self, query: str) -> list[str]:
        """Return the list of store names relevant to the query.

        On any failure (bad JSON, API error, empty result), falls back to all stores
        so the user's query is never silently starved of context.
        """
        try:
            response = self._openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ROUTING_PROMPT},
                    {"role": "user", "content": f"Query: {query}"},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            stores = data.get("stores", []) or []
            # Keep only valid store names
            valid = [s for s in stores if s in ALL_STORES]
            if not valid:
                valid = ALL_STORES.copy()
            self.last_decision = valid
            return valid
        except Exception:
            # Never let a routing failure break the chat turn
            self.last_decision = ALL_STORES.copy()
            return self.last_decision
