"""Tests for the MemoryExtractor."""

import json
from unittest.mock import MagicMock

import pytest

from contextkit.config import Config
from contextkit.extractor import MemoryExtractor
from contextkit.manager import MemoryManager


def _mock_extraction_response(payload: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture()
def extractor(chroma_client, unique_prefix):
    config = Config()
    memory = MemoryManager(
        config=config, chroma_client=chroma_client, collection_prefix=unique_prefix
    )
    mock_openai = MagicMock()
    return MemoryExtractor(memory=memory, config=config, openai_client=mock_openai)


def test_extract_all_categories(extractor):
    extractor._openai.chat.completions.create.return_value = _mock_extraction_response(
        {
            "entities": [
                {"name": "Alice", "description": "software engineer at Acme"},
                {"name": "Acme Corp", "description": "a startup"},
            ],
            "facts": ["Python is a programming language"],
            "events": ["Alice joined Acme in 2024"],
            "procedures": ["Run tests before committing"],
        }
    )

    messages = [
        {"role": "user", "content": "Hi, I'm Alice and I work at Acme Corp."},
        {"role": "assistant", "content": "Nice to meet you Alice."},
    ]
    result = extractor.extract_and_save(messages)

    assert len(result.entities) == 2
    assert len(result.facts) == 1
    assert len(result.events) == 1
    assert len(result.procedures) == 1
    assert result.total == 5

    # Confirm they landed in the right stores
    assert extractor.memory.entity.count() == 2
    assert extractor.memory.semantic.count() == 1
    assert extractor.memory.episodic.count() == 1
    assert extractor.memory.procedural.count() == 1


def test_extract_handles_empty_categories(extractor):
    extractor._openai.chat.completions.create.return_value = _mock_extraction_response(
        {"entities": [], "facts": [], "events": [], "procedures": []}
    )

    result = extractor.extract_and_save([{"role": "user", "content": "hi"}])
    assert result.total == 0


def test_extract_handles_malformed_json(extractor):
    msg = MagicMock()
    msg.content = "not json"
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    extractor._openai.chat.completions.create.return_value = resp

    result = extractor.extract_and_save([{"role": "user", "content": "hi"}])
    assert result.total == 0


def test_extract_skips_invalid_entries(extractor):
    extractor._openai.chat.completions.create.return_value = _mock_extraction_response(
        {
            "entities": [{"description": "no name"}, {"name": "Bob", "description": "x"}],
            "facts": ["", "valid fact"],
            "events": [],
            "procedures": [],
        }
    )

    result = extractor.extract_and_save([{"role": "user", "content": "..."}])
    assert len(result.entities) == 1
    assert len(result.facts) == 1
