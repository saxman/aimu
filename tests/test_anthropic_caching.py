"""Mock-only tests for opt-in Anthropic prompt caching (P1-A).

Covers the `cache_prompt` flag injecting `cache_control` ephemeral markers on the system
prompt and tools (via the two format adapters), the off-by-default behavior, an
end-to-end request-payload assertion, and cache-token surfacing in usage_from_anthropic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from aimu.models import AnthropicModel  # noqa: E402
from aimu.models._internal.usage import usage_from_anthropic  # noqa: E402

_MODEL = AnthropicModel.CLAUDE_SONNET_4_6
_EPHEMERAL = {"type": "ephemeral"}


def _bare_anthropic(*, cache_prompt=False, tools=None):
    """Build an AnthropicClient without running __init__ (no SDK/key needed)."""
    from aimu.models.providers.anthropic import AnthropicClient

    client = object.__new__(AnthropicClient)
    client.model = _MODEL
    client.model_kwargs = None
    client._system_message = None
    client.default_generate_kwargs = {}
    client.messages = []
    client.tools = tools or []
    client.last_thinking = ""
    client.last_usage = None
    client.concurrent_tool_calls = False
    client.tool_context_deps = None
    client.cache_prompt = cache_prompt
    return client


def _openai_tool(name="lookup"):
    return {"type": "function", "function": {"name": name, "description": "d", "parameters": {"type": "object"}}}


# ---------------------------------------------------------------------------
# System-prompt marker (via _openai_messages_to_anthropic)
# ---------------------------------------------------------------------------


def test_system_prompt_marked_when_cache_prompt_on():
    client = _bare_anthropic(cache_prompt=True)
    system, _ = client._openai_messages_to_anthropic([{"role": "system", "content": "You are helpful."}])
    assert system == [{"type": "text", "text": "You are helpful.", "cache_control": _EPHEMERAL}]


def test_system_prompt_plain_string_when_cache_prompt_off():
    client = _bare_anthropic(cache_prompt=False)
    system, _ = client._openai_messages_to_anthropic([{"role": "system", "content": "You are helpful."}])
    assert system == "You are helpful."


def test_no_system_block_when_no_system_message_even_with_cache():
    client = _bare_anthropic(cache_prompt=True)
    system, _ = client._openai_messages_to_anthropic([{"role": "user", "content": "hi"}])
    assert system == ""  # empty -> request site sends anthropic.NOT_GIVEN


# ---------------------------------------------------------------------------
# Tools marker (via _openai_tools_to_anthropic)
# ---------------------------------------------------------------------------


def test_only_last_tool_marked_when_cache_prompt_on():
    client = _bare_anthropic(cache_prompt=True)
    out = client._openai_tools_to_anthropic([_openai_tool("a"), _openai_tool("b")])
    assert "cache_control" not in out[0]
    assert out[-1]["cache_control"] == _EPHEMERAL


def test_tools_unmarked_when_cache_prompt_off():
    client = _bare_anthropic(cache_prompt=False)
    out = client._openai_tools_to_anthropic([_openai_tool("a")])
    assert "cache_control" not in out[0]


def test_empty_tools_no_error_with_cache():
    client = _bare_anthropic(cache_prompt=True)
    assert client._openai_tools_to_anthropic([]) == []


# ---------------------------------------------------------------------------
# End-to-end: markers reach the messages.create request payload
# ---------------------------------------------------------------------------


def test_chat_request_carries_cache_control(monkeypatch):
    import aimu

    @aimu.tool
    def lookup(query: str) -> str:
        """Look something up."""
        return query

    client = _bare_anthropic(cache_prompt=True, tools=[lookup])
    client._system_message = "You are helpful."

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hello")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )

    client._client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    client._chat("hi", generate_kwargs={"max_tokens": 16})

    assert captured["system"][0]["cache_control"] == _EPHEMERAL
    assert captured["tools"][-1]["cache_control"] == _EPHEMERAL


# ---------------------------------------------------------------------------
# usage_from_anthropic: surface cache tokens
# ---------------------------------------------------------------------------


def test_usage_surfaces_cache_tokens_when_present():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=100,
            cache_read_input_tokens=200,
        )
    )
    usage = usage_from_anthropic(response)
    assert usage["cache_creation_input_tokens"] == 100
    assert usage["cache_read_input_tokens"] == 200
    assert usage["total_tokens"] == 15


def test_usage_unchanged_when_cache_fields_absent():
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=5))
    usage = usage_from_anthropic(response)
    assert usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


# ---------------------------------------------------------------------------
# Async client inherits the cache-marking adapters
# ---------------------------------------------------------------------------


def test_async_client_inherits_cache_marking():
    from aimu.aio.providers.anthropic import AsyncAnthropicClient

    client = object.__new__(AsyncAnthropicClient)
    client.cache_prompt = True
    system, _ = client._openai_messages_to_anthropic([{"role": "system", "content": "Sys"}])
    assert system == [{"type": "text", "text": "Sys", "cache_control": _EPHEMERAL}]
