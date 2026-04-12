"""Shared test fixtures."""

import uuid

import chromadb
import pytest


@pytest.fixture()
def chroma_client():
    """Return an ephemeral ChromaDB client with all collections cleared."""
    client = chromadb.EphemeralClient()
    # Clear any existing collections
    for col in client.list_collections():
        client.delete_collection(col.name)
    return client


@pytest.fixture()
def unique_prefix():
    """Return a unique prefix for collection names to avoid test interference."""
    return uuid.uuid4().hex[:8] + "_"
