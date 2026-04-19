"""Configuration for contextkit."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Settings for contextkit, populated from environment variables and defaults."""

    # Credentials & models
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    model_name: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # Storage
    chromadb_data_dir: str = field(
        default_factory=lambda: os.environ.get(
            "CONTEXTKIT_CHROMADB_DIR", ".contextkit_data"
        )
    )

    # Context window management
    max_context_tokens: int = 8192
    summary_threshold: float = 0.8

    # Per-store update behavior
    entity_dedup_enabled: bool = True
    semantic_similarity_threshold: float = 0.85
    procedural_similarity_threshold: float = 0.88
    working_default_ttl_seconds: float = 3600.0
    buffer_max_size: int = 50
