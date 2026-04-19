"""contextkit — Memory-augmented context management for LLM agents."""

from .agent import Agent, ChatContext
from .context_builder import ContextBuilder
from .extractor import ExtractionResult, MemoryExtractor
from .manager import MemoryManager

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "ChatContext",
    "ContextBuilder",
    "ExtractionResult",
    "MemoryExtractor",
    "MemoryManager",
]
