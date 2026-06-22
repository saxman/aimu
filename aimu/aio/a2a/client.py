"""Async ``RemoteAgent``: consume a remote A2A agent as an :class:`AsyncRunner`.

Async twin of :class:`aimu.agents.a2a.RemoteAgent`. Where the sync client drives the
``a2a-sdk`` async client through an anyio portal, this one uses it natively (no portal),
exactly as ``aimu.aio.MCPClient`` uses FastMCP's async ``Client`` directly. This is also
the surface that supports real incremental streaming via ``message/stream``.
"""

from __future__ import annotations

import warnings
from typing import Any, AsyncIterator, Optional, Union

import httpx
from a2a.client import A2ACardResolver, A2AClient

from aimu.agents.a2a._card import make_send_request, make_stream_request, result_to_text
from aimu.agents.a2a.client import A2AConnectionError, _response_text
from aimu.agents.base import MessageHistory
from aimu.aio.agent import AsyncRunner
from aimu.models.base import StreamChunk, StreamingContentType

DEFAULT_AGENT_CARD_PATH = "/.well-known/agent-card.json"


class RemoteAgent(AsyncRunner):
    """A remote A2A agent presented as a local asynchronous ``AsyncRunner``.

    Construct via the async :meth:`connect`::

        remote = await RemoteAgent.connect("http://localhost:9000")
        print(await remote.run("Summarise the news"))
        async for chunk in await remote.run("Summarise", stream=True):
            ...
    """

    def __init__(self, client: A2AClient, httpx_client: httpx.AsyncClient, name: str, card: Any):
        self._client = client
        self._httpx_client = httpx_client
        self.name = name
        self.card = card
        self.system_message = getattr(card, "description", None)
        self._last_messages: list[dict] = []

    @classmethod
    async def connect(
        cls,
        url: str,
        *,
        name: Optional[str] = None,
        agent_card_path: str = DEFAULT_AGENT_CARD_PATH,
        timeout: float = 60.0,
    ) -> "RemoteAgent":
        """Resolve the remote agent card at ``url`` and return a connected ``RemoteAgent``."""
        httpx_client = httpx.AsyncClient(timeout=timeout)
        try:
            resolver = A2ACardResolver(httpx_client, base_url=url, agent_card_path=agent_card_path)
            card = await resolver.get_agent_card()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                # Send to the URL we connected to (url=), not the card's advertised url.
                client = A2AClient(httpx_client, agent_card=card, url=url)
        except Exception as exc:
            await httpx_client.aclose()
            raise A2AConnectionError(f"failed to connect to A2A agent at {url!r}: {exc}") from exc
        return cls(client, httpx_client, name or card.name, card)

    async def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        """Send ``task`` to the remote agent; return its text (or a chunk stream)."""
        if stream:
            return self._run_streamed(task)
        try:
            response = await self._client.send_message(make_send_request(task))
        except Exception as exc:
            raise A2AConnectionError(f"A2A send_message to {self.name!r} failed: {exc}") from exc
        text = _response_text(response)
        self._last_messages = [
            {"role": "user", "content": task},
            {"role": "assistant", "content": text},
        ]
        return text

    async def _run_streamed(self, task: str) -> AsyncIterator[StreamChunk]:
        collected: list[str] = []
        try:
            async for response in self._client.send_message_streaming(make_stream_request(task)):
                root = response.root
                error = getattr(root, "error", None)
                if error is not None:
                    raise A2AConnectionError(f"remote A2A agent returned an error: {error}")
                text = result_to_text(root.result)
                if text:
                    collected.append(text)
                    yield StreamChunk(StreamingContentType.GENERATING, text, agent=self.name)
        except A2AConnectionError:
            raise
        except Exception as exc:
            raise A2AConnectionError(f"A2A message/stream to {self.name!r} failed: {exc}") from exc
        self._last_messages = [
            {"role": "user", "content": task},
            {"role": "assistant", "content": "".join(collected)},
        ]
        yield StreamChunk(StreamingContentType.DONE, "", agent=self.name)

    @property
    def messages(self) -> MessageHistory:
        return {self.name: self._last_messages}

    async def aclose(self) -> None:
        await self._httpx_client.aclose()
