"""TOOL_REQUESTED: every call in a round is announced before any of them runs (mock-only).

Without it a streamed caller learns of a tool call only from its TOOL_CALLING result, so a call that
takes a minute shows nothing for a minute. Both drivers are covered on every dispatch path they have
(sequential, concurrent, a streaming tool, a refused call), driven through the engine directly after
staging the assistant tool-call message a provider would have stored.
"""

from __future__ import annotations

import pytest

from aimu.agents._tool_loop import _ToolLoop
from aimu.aio._tool_loop import _AsyncToolLoop
from aimu.models import StreamChunk, StreamingContentType
from aimu.tools import tool
from helpers import MockModelClient
from helpers_aio import MockAsyncModelClient

REQUESTED = StreamingContentType.TOOL_REQUESTED
CALLED = StreamingContentType.TOOL_CALLING


@tool
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


@tool
def counter(n: int):
    """A sync streaming tool."""
    yield StreamChunk(StreamingContentType.GENERATING, f"counting {n}")
    return str(n)


@tool
async def async_counter(n: int):
    """An async streaming tool."""
    yield StreamChunk(StreamingContentType.GENERATING, f"counting {n}")


def _stage(client, calls):
    client.messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": arguments}, "id": f"call-{i}"}
                for i, (name, arguments) in enumerate(calls)
            ],
        }
    )


def _sync_chunks(tools, calls, **loop_kwargs):
    client = MockModelClient([])
    _stage(client, calls)
    return list(_ToolLoop(client, tools, **loop_kwargs)._dispatch_streamed(3))


async def _async_chunks(tools, calls, **loop_kwargs):
    client = MockAsyncModelClient([])
    _stage(client, calls)
    return [chunk async for chunk in _AsyncToolLoop(client, tools, **loop_kwargs)._dispatch_streamed(3)]


def _shape(chunks):
    return [
        (chunk.phase, chunk.content.get("id") if isinstance(chunk.content, dict) else chunk.content) for chunk in chunks
    ]


TWO_ADDS = [("add", {"a": 1, "b": 2}), ("add", {"a": 3, "b": 4})]


@pytest.mark.parametrize("concurrent", [False, True])
def test_sync_announces_every_call_before_any_runs(concurrent):
    chunks = _sync_chunks([add], TWO_ADDS, concurrent_tool_calls=concurrent)
    assert _shape(chunks) == [(REQUESTED, "call-0"), (REQUESTED, "call-1"), (CALLED, "call-0"), (CALLED, "call-1")]


@pytest.mark.parametrize("concurrent", [False, True])
async def test_async_announces_every_call_before_any_runs(concurrent):
    chunks = await _async_chunks([add], TWO_ADDS, concurrent_tool_calls=concurrent)
    assert _shape(chunks) == [(REQUESTED, "call-0"), (REQUESTED, "call-1"), (CALLED, "call-0"), (CALLED, "call-1")]


async def test_the_announcement_carries_the_call_and_no_result():
    chunks = await _async_chunks([add], [("add", {"a": 1, "b": 2})])
    requested, called = chunks
    assert requested.content == {"name": "add", "arguments": {"a": 1, "b": 2}, "id": "call-0"}
    assert requested.iteration == 3
    assert requested.is_tool_request() and not requested.is_tool_call()
    assert called.content == {"name": "add", "arguments": {"a": 1, "b": 2}, "response": "3", "id": "call-0"}


def test_sync_streaming_tool_chunks_sit_between_its_announcement_and_its_result():
    chunks = _sync_chunks([counter], [("counter", {"n": 2})])
    assert _shape(chunks) == [
        (REQUESTED, "call-0"),
        (StreamingContentType.GENERATING, "counting 2"),
        (CALLED, "call-0"),
    ]


async def test_async_streaming_tool_chunks_sit_between_its_announcement_and_its_result():
    chunks = await _async_chunks([async_counter], [("async_counter", {"n": 2})])
    assert _shape(chunks) == [
        (REQUESTED, "call-0"),
        (StreamingContentType.GENERATING, "counting 2"),
        (CALLED, "call-0"),
    ]


def test_sync_a_refused_call_is_announced_too():
    """The announcement precedes approval, so a refused call still has one, followed by its refusal."""
    chunks = _sync_chunks([add], [("add", {"a": 1, "b": 2})], tool_approval=lambda name, arguments: False)
    assert _shape(chunks) == [(REQUESTED, "call-0"), (CALLED, "call-0")]
    assert chunks[1].content["response"] == "Tool 'add' was not approved."


async def test_async_a_refused_call_is_announced_too():
    chunks = await _async_chunks([add], [("add", {"a": 1, "b": 2})], tool_approval=lambda name, arguments: False)
    assert _shape(chunks) == [(REQUESTED, "call-0"), (CALLED, "call-0")]
    assert chunks[1].content["response"] == "Tool 'add' was not approved."


async def test_an_invented_tool_name_is_announced_and_answered():
    chunks = await _async_chunks([add], [("nope", {})])
    assert _shape(chunks) == [(REQUESTED, "call-0"), (CALLED, "call-0")]
