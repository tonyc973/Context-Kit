"""Tool definitions that an Agent can call to manage memory."""

from __future__ import annotations

from typing import Any

from .manager import MemoryManager

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save a piece of information to a specific memory store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store": {
                        "type": "string",
                        "enum": [
                            "episodic",
                            "semantic",
                            "procedural",
                            "working",
                            "entity",
                            "summary",
                            "buffer",
                        ],
                        "description": "Which memory store to save to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to memorize.",
                    },
                },
                "required": ["store", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search memory stores for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "store": {
                        "type": "string",
                        "enum": [
                            "episodic",
                            "semantic",
                            "procedural",
                            "working",
                            "entity",
                            "summary",
                            "buffer",
                        ],
                        "description": "Optionally limit search to one store.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def handle_tool_call(
    memory: MemoryManager, name: str, arguments: dict[str, Any]
) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "save_memory":
        rid = memory.save(arguments["store"], arguments["content"])
        return f"Saved to {arguments['store']} (id={rid})"

    if name == "search_memory":
        store_name = arguments.get("store")
        query = arguments["query"]
        if store_name:
            records = memory.query(store_name, query)
        else:
            all_results = memory.query_all(query)
            records = [r for recs in all_results.values() for r in recs]
        if not records:
            return "No memories found."
        return "\n".join(f"- [{r.id}] {r.content}" for r in records)

    return f"Unknown tool: {name}"
