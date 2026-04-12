"""Quickstart example for contextkit."""

from contextkit import Agent, ContextBuilder, MemoryManager
from contextkit.config import Config

# Configure — picks up OPENAI_API_KEY from environment
config = Config(
    model_name="gpt-4o",
    max_context_tokens=8192,
)

# Initialize the memory manager and agent
agent = Agent(config=config, system_prompt="You are a helpful assistant with long-term memory.")

# Save some facts to memory
agent.memory.save("semantic", "The user's name is Alice.")
agent.memory.save("entity", "Alice is a software engineer at Acme Corp.")
agent.memory.save("procedural", "When greeting Alice, mention her current project.")
agent.memory.save("episodic", "Last session, Alice asked about Python async patterns.")

# Chat — the agent automatically retrieves relevant memories
reply = agent.chat("Hi! What were we talking about last time?")
print(reply)

# You can also use MemoryManager and ContextBuilder independently
memory = agent.memory
builder = ContextBuilder(memory, config)

# Build context-enriched messages for your own LLM calls
messages = builder.build(
    query="What does Alice do?",
    system_prompt="Answer based on what you know about the user.",
    messages=[{"role": "user", "content": "What do you know about me?"}],
)
print("\nBuilt messages:")
for msg in messages:
    print(f"  [{msg['role']}] {msg['content'][:80]}...")
