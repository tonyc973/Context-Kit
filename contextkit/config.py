"""Configuration for contextkit."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Settings for contextkit, populated from environment variables and defaults."""

    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    model_name: str = "gpt-4o"
    chromadb_data_dir: str = field(
        default_factory=lambda: os.environ.get(
            "CONTEXTKIT_CHROMADB_DIR", ".contextkit_data"
        )
    )
    embedding_model: str = "text-embedding-3-small"
    max_context_tokens: int = 8192
    summary_threshold: float = 0.8
