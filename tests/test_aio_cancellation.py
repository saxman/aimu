"""Cooperative cancellation: RunHandle + the partial-message snapshot on cancel (mock-only)."""

from __future__ import annotations

import asyncio

import pytest

from aimu.aio import Agent, RunHandle
from helpers_aio import MockAsyncModelClient


class _BlockingClient(MockAsyncModelClient):
    """Records the user turn, signals it has started, then blocks until cancelled."""

    def __init__(self):
        super().__init__([])
        self.started = asyncio.Event()

    async def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
        self.messages.append({"role": "user", "content": user_message})
        self.started.set()
        await asyncio.Event().wait()  # never set -> hangs until the task is cancelled


async def test_run_handle_cancel_raises_and_marks_cancelled():
    client = _BlockingClient()
    agent = Agent(client, name="a")
    handle = RunHandle.start(agent.run("hi"))
    await client.started.wait()

    assert handle.cancel() is True
    with pytest.raises(asyncio.CancelledError):
        await handle.result()
    assert handle.cancelled is True and handle.done is True


async def test_cancel_captures_partial_messages_for_resume():
    client = _BlockingClient()
    agent = Agent(client, name="a")
    handle = RunHandle.start(agent.run("remember this"))
    await client.started.wait()
    handle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle.result()

    # The live messages (what restore() reads) hold the partial turn...
    assert any(m.get("content") == "remember this" for m in agent.model_client.messages)
    # ...and the post-run snapshot (the `messages` property) was captured in the finally.
    assert [m.get("content") for m in agent.messages["a"]] == ["remember this"]


async def test_run_handle_normal_completion():
    agent = Agent(MockAsyncModelClient(["done"]), name="b")
    handle = RunHandle.start(agent.run("x"))
    assert await handle.result() == "done"
    assert handle.done is True and handle.cancelled is False


async def test_run_handle_awaitable_directly():
    agent = Agent(MockAsyncModelClient(["yo"]), name="c")
    assert await RunHandle.start(agent.run("x")) == "yo"


async def test_streamed_cancel_captures_partial_messages():
    """Cancelling the consumer of a streamed run closes the generator; its finally still snapshots."""
    client = _BlockingClient()
    agent = Agent(client, name="d")

    async def consume():
        async for _ in await agent.run("streamed task", stream=True):
            pass

    handle = RunHandle.start(consume())
    await client.started.wait()
    handle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle.result()

    assert any(m.get("content") == "streamed task" for m in agent.model_client.messages)
    assert agent.messages["d"]  # snapshot populated via the generator's finally
