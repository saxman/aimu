"""Mid-run steering on the async surface: messages handed to a loop already running."""

from __future__ import annotations

import pytest

from aimu.aio import Agent
from aimu.models import StreamingContentType
from tests.helpers_aio import MockAsyncModelClient


class ListSteering:
    """A steering source backed by a plain list, with one cursor per reader."""

    def __init__(self, messages=None):
        self.messages = list(messages or [])

    def reader(self):
        seen = 0

        def drain():
            nonlocal seen
            pending = self.messages[seen:]
            seen = len(self.messages)
            return list(pending)

        return drain


def a_tool() -> str:
    """A tool the mock client can be told to call."""
    return "tool result"


async def collect(stream):
    return [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_a_message_pending_at_a_tool_round_becomes_that_rounds_user_message():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])
    steering = ListSteering(["use the other file"])

    await collect(await agent.run("start", stream=True, steering=steering))

    assert {"role": "user", "content": "use the other file"} in client.messages


@pytest.mark.asyncio
async def test_a_steered_round_opens_with_a_steering_chunk_carrying_the_text():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, steering=ListSteering(["use the other file"])))

    steering_chunks = [c for c in chunks if c.phase == StreamingContentType.STEERING]
    assert [c.content for c in steering_chunks] == [{"text": "use the other file"}]


@pytest.mark.asyncio
async def test_a_steering_message_is_not_tagged_as_a_loop_injection():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    await collect(await agent.run("start", stream=True, steering=ListSteering(["stop that"])))

    steered = [m for m in client.messages if m.get("content") == "stop that"]
    assert steered and all("provenance" not in m for m in steered)


@pytest.mark.asyncio
async def test_whitespace_only_steering_is_not_delivered():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    await collect(await agent.run("start", stream=True, steering=ListSteering(["   ", ""])))

    assert [m for m in client.messages if m["role"] == "user"] == [{"role": "user", "content": "start"}]


@pytest.mark.asyncio
async def test_a_raising_steering_source_does_not_end_the_run():
    class Exploding:
        def reader(self):
            def drain():
                raise RuntimeError("the host's mailbox is broken")

            return drain

    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, steering=Exploding()))

    assert any(c.phase == StreamingContentType.GENERATING and c.content == "done" for c in chunks)
