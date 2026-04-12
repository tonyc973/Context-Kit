"""Agent wraps an OpenAI chat model with automatic memory management."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .config import Config
from .context_builder import ContextBuilder
from .manager import MemoryManager
from .tools import TOOL_DEFINITIONS, handle_tool_call


class Agent:
    """A conversational agent backed by memory stores."""

    def __init__(
        self,
        config: Config | None = None,
        memory: MemoryManager | None = None,
        system_prompt: str = "You are a helpful assistant with long-term memory.",
        openai_client: OpenAI | None = None,
        chroma_client: Any = None,
    ) -> None:
        self.config = config or Config()
        self.memory = memory or MemoryManager(
            config=self.config, chroma_client=chroma_client
        )
        self.context_builder = ContextBuilder(self.memory, self.config)
        self.system_prompt = system_prompt
        self._openai = openai_client or OpenAI(api_key=self.config.openai_api_key)
        self._history: list[dict[str, str]] = []

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, with automatic memory management."""
        # Store in buffer
        self.memory.add_message("user", user_message)
        self._history.append({"role": "user", "content": user_message})

        # Check if summarization is needed
        if self.context_builder.needs_summary(self._history):
            self._summarize_history()

        # Build context-enriched messages
        messages = self.context_builder.build(
            query=user_message,
            system_prompt=self.system_prompt,
            messages=self._history,
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
        return reply

    def _summarize_history(self) -> None:
        """Compress history into a summary and reset the conversation window."""
        if len(self._history) < 4:
            return

        # Take all but the last 2 messages to summarize
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
