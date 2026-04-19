"""Agent wraps an OpenAI chat model with automatic memory management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .config import Config
from .context_builder import ContextBuilder
from .manager import MemoryManager
from .router import QueryRouter
from .stores import MemoryRecord
from .tools import TOOL_DEFINITIONS, handle_tool_call


@dataclass
class ChatContext:
    """Observability record for a single chat turn."""

    query: str = ""
    reply: str = ""
    retrieved_memories: dict[str, list[MemoryRecord]] = field(default_factory=dict)
    input_tokens: int = 0
    summarized: bool = False
    tool_calls_made: int = 0
    routed_to: list[str] | None = None

    def pretty(self) -> str:
        """Human-readable summary of what happened this turn."""
        lines = [
            "─" * 60,
            f"QUERY: {self.query}",
            f"REPLY: {self.reply[:120]}{'…' if len(self.reply) > 120 else ''}",
            f"INPUT TOKENS: {self.input_tokens}",
            f"SUMMARIZED: {self.summarized}",
            f"TOOL CALLS: {self.tool_calls_made}",
        ]
        if self.routed_to is not None:
            lines.append(f"ROUTED TO: {', '.join(self.routed_to)}")
        if self.retrieved_memories:
            lines.append("RETRIEVED MEMORIES:")
            for store, records in self.retrieved_memories.items():
                lines.append(f"  [{store}] ({len(records)})")
                for r in records:
                    snippet = r.content[:80].replace("\n", " ")
                    lines.append(f"    • {snippet}")
        else:
            lines.append("RETRIEVED MEMORIES: (none)")
        lines.append("─" * 60)
        return "\n".join(lines)


class Agent:
    """A conversational agent backed by memory stores."""

    def __init__(
        self,
        config: Config | None = None,
        memory: MemoryManager | None = None,
        system_prompt: str = "You are a helpful assistant with long-term memory.",
        openai_client: OpenAI | None = None,
        chroma_client: Any = None,
        use_routing: bool = False,
    ) -> None:
        self.config = config or Config()
        self.memory = memory or MemoryManager(
            config=self.config, chroma_client=chroma_client
        )
        self._openai = openai_client or OpenAI(api_key=self.config.openai_api_key)
        self.router: QueryRouter | None = None
        if use_routing:
            self.router = QueryRouter(
                config=self.config, openai_client=self._openai
            )
        self.context_builder = ContextBuilder(
            self.memory, self.config, router=self.router
        )
        self.system_prompt = system_prompt
        self._history: list[dict[str, str]] = []
        self.last_context: ChatContext = ChatContext()

    def chat(self, user_message: str, debug: bool = False) -> str:
        """Send a message and get a response, with automatic memory management.

        Args:
            user_message: The user's input.
            debug: If True, print a trace of retrieved memories and token usage.
        """
        ctx = ChatContext(query=user_message)

        # Store in buffer
        self.memory.add_message("user", user_message)
        self._history.append({"role": "user", "content": user_message})

        # Check if summarization is needed
        if self.context_builder.needs_summary(self._history):
            self._summarize_history()
            ctx.summarized = True

        # Build context-enriched messages
        messages = self.context_builder.build(
            query=user_message,
            system_prompt=self.system_prompt,
            messages=self._history,
        )
        ctx.retrieved_memories = dict(self.context_builder.last_retrieved)
        ctx.routed_to = self.context_builder.last_routed_to
        ctx.input_tokens = sum(
            self.context_builder.count_tokens(m.get("content", "") or "")
            for m in messages
        )

        # Call OpenAI with tool support
        response = self._openai.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        choice = response.choices[0]

        # Handle tool calls
        while choice.finish_reason == "tool_calls":
            ctx.tool_calls_made += len(choice.message.tool_calls or [])
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = handle_tool_call(self.memory, tc.function.name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

            response = self._openai.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
            choice = response.choices[0]

        reply = choice.message.content or ""
        self.memory.add_message("assistant", reply)
        self._history.append({"role": "assistant", "content": reply})

        ctx.reply = reply
        self.last_context = ctx

        if debug:
            print(ctx.pretty())

        return reply

    def _summarize_history(self) -> None:
        """Compress history into a summary and reset the conversation window."""
        if len(self._history) < 4:
            return

        to_summarize = self._history[:-2]
        kept = self._history[-2:]

        text_block = "\n".join(
            f"{m['role']}: {m['content']}" for m in to_summarize
        )

        response = self._openai.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this conversation concisely, preserving key facts.",
                },
                {"role": "user", "content": text_block},
            ],
        )

        summary_text = response.choices[0].message.content or ""
        self.memory.save("summary", summary_text)
        self._history = [
            {"role": "system", "content": f"Previous conversation summary: {summary_text}"},
            *kept,
        ]
