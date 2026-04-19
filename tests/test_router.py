"""Tests for QueryRouter and its integration with ContextBuilder / Agent."""

import json
from unittest.mock import MagicMock

import pytest

from contextkit.agent import Agent
from contextkit.config import Config
from contextkit.context_builder import ContextBuilder
from contextkit.manager import MemoryManager
from contextkit.router import ALL_STORES, QueryRouter


def _mock_router_response(stores: list[str]) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps({"stores": stores})
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture()
def router():
    mock_openai = MagicMock()
    return QueryRouter(config=Config(), openai_client=mock_openai)


def test_router_returns_valid_stores(router):
    router._openai.chat.completions.create.return_value = _mock_router_response(
        ["entity", "semantic"]
    )
    result = router.route("Who is Alice?")
    assert result == ["entity", "semantic"]
    assert router.last_decision == ["entity", "semantic"]


def test_router_filters_invalid_store_names(router):
    router._openai.chat.completions.create.return_value = _mock_router_response(
        ["entity", "bogus_store", "semantic"]
    )
    result = router.route("query")
    assert result == ["entity", "semantic"]


def test_router_fallback_on_empty_result(router):
    router._openai.chat.completions.create.return_value = _mock_router_response([])
    result = router.route("vague query")
    assert set(result) == set(ALL_STORES)


def test_router_fallback_on_error(router):
    router._openai.chat.completions.create.side_effect = RuntimeError("API down")
    result = router.route("query")
    assert set(result) == set(ALL_STORES)


def test_router_fallback_on_malformed_json(router):
    msg = MagicMock()
    msg.content = "not json at all"
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    router._openai.chat.completions.create.return_value = resp

    result = router.route("query")
    assert set(result) == set(ALL_STORES)


def test_manager_respects_stores_filter(chroma_client, unique_prefix):
    memory = MemoryManager(
        config=Config(), chroma_client=chroma_client, collection_prefix=unique_prefix
    )
    memory.save("entity", "Alice is a person")
    memory.save("semantic", "Python is a language")
    memory.save("procedural", "Always write tests")

    # Unrestricted — all three show up
    all_results = memory.query_all("anything")
    assert "entity" in all_results
    assert "semantic" in all_results
    assert "procedural" in all_results

    # Filtered — only the ones we asked for
    filtered = memory.query_all("anything", stores=["entity", "semantic"])
    assert "entity" in filtered
    assert "semantic" in filtered
    assert "procedural" not in filtered


def test_context_builder_uses_router(chroma_client, unique_prefix):
    memory = MemoryManager(
        config=Config(), chroma_client=chroma_client, collection_prefix=unique_prefix
    )
    memory.save("entity", "Alice")
    memory.save("procedural", "Write tests first")

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = _mock_router_response(["entity"])
    router = QueryRouter(config=Config(), openai_client=mock_openai)

    builder = ContextBuilder(memory, Config(), router=router)
    builder.build(query="Who is Alice?")

    assert builder.last_routed_to == ["entity"]
    assert "entity" in builder.last_retrieved
    assert "procedural" not in builder.last_retrieved


def test_context_builder_without_router_queries_all(chroma_client, unique_prefix):
    memory = MemoryManager(
        config=Config(), chroma_client=chroma_client, collection_prefix=unique_prefix
    )
    memory.save("entity", "Alice")
    memory.save("procedural", "Write tests first")

    builder = ContextBuilder(memory, Config())
    builder.build(query="Tell me everything")

    assert builder.last_routed_to is None
    assert "entity" in builder.last_retrieved
    assert "procedural" in builder.last_retrieved


def test_agent_with_routing_captures_decision(chroma_client, unique_prefix):
    config = Config()
    memory = MemoryManager(
        config=config, chroma_client=chroma_client, collection_prefix=unique_prefix
    )
    memory.save("entity", "Alice")

    mock_openai = MagicMock()
    # Two different responses: one for routing, one for the chat completion
    routing_resp = _mock_router_response(["entity"])
    chat_msg = MagicMock()
    chat_msg.content = "Alice is a person."
    chat_msg.tool_calls = None
    chat_choice = MagicMock()
    chat_choice.message = chat_msg
    chat_choice.finish_reason = "stop"
    chat_resp = MagicMock()
    chat_resp.choices = [chat_choice]
    mock_openai.chat.completions.create.side_effect = [routing_resp, chat_resp]

    agent = Agent(
        config=config,
        memory=memory,
        openai_client=mock_openai,
        use_routing=True,
    )
    agent.chat("Who is Alice?")

    assert agent.last_context.routed_to == ["entity"]
