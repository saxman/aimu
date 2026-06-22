"""Async A2A interop tests (aimu.aio.a2a). Skipped unless the 'a2a' extra is installed.

Serves an async runner via uvicorn in a background thread and drives it with the native
async RemoteAgent (no anyio portal), covering both the request/response and streaming paths.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager

import pytest

pytest.importorskip("a2a", reason="requires the 'a2a' extra")

from aimu.aio import Chain as AsyncChain  # noqa: E402
from aimu.aio.agent import AsyncRunner  # noqa: E402
from aimu.aio.a2a import A2AConnectionError, RemoteAgent, build_a2a_app  # noqa: E402
from aimu.models import StreamingContentType  # noqa: E402


class _AsyncEchoRunner(AsyncRunner):
    def __init__(self, name="echo", system_message="Echo the input back."):
        self.name = name
        self.system_message = system_message

    async def run(self, task, generate_kwargs=None, stream=False, images=None):
        return f"echo: {task}"

    @property
    def messages(self):
        return {self.name: []}


@contextmanager
def _serve(app):
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.02)
        assert server.started, "uvicorn server did not start in time"
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


async def test_async_remote_agent_round_trip_and_composition():
    app = build_a2a_app(_AsyncEchoRunner(name="greeter", system_message="Greet."), url="http://placeholder/")
    with _serve(app) as url:
        remote = await RemoteAgent.connect(url)
        try:
            assert isinstance(remote, AsyncRunner)
            assert remote.name == "greeter"
            assert await remote.run("hi") == "echo: hi"

            # Composes as an async tool.
            fn = remote.as_tool()
            assert fn.__tool_is_async__ is True
            assert fn.__tool_spec__["function"]["description"] == "Greet."
            assert await fn("again") == "echo: again"

            # Composes as an async workflow step.
            chain = AsyncChain(agents=[remote])
            assert await chain.run("chained") == "echo: chained"
        finally:
            await remote.aclose()


async def test_async_remote_agent_streaming():
    app = build_a2a_app(_AsyncEchoRunner(name="greeter"), url="http://placeholder/")
    with _serve(app) as url:
        remote = await RemoteAgent.connect(url)
        try:
            chunks = []
            stream = await remote.run("stream me", stream=True)
            async for chunk in stream:
                chunks.append(chunk)
            generating = [c for c in chunks if c.phase == StreamingContentType.GENERATING]
            assert "".join(c.content for c in generating) == "echo: stream me"
            assert chunks[-1].phase == StreamingContentType.DONE
        finally:
            await remote.aclose()


async def test_async_connect_to_dead_url_raises():
    with pytest.raises(A2AConnectionError):
        await RemoteAgent.connect("http://127.0.0.1:1/")
