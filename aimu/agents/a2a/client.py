"""Consume a remote A2A agent as a local :class:`~aimu.agents.base.Runner`.

``RemoteAgent`` is the agent-level analog of :class:`aimu.tools.MCPClient`: it wraps a
cross-process A2A agent behind AIMU's uniform ``Runner`` interface, so a remote agent
drops straight into a ``Chain`` / ``Router`` / ``OrchestratorAgent`` worker list, or — via
``Runner.as_tool()`` — into any local ``Agent``'s tool set, with no A2A-specific wiring.

The ``a2a-sdk`` client is async; like :class:`MCPClient` this sync wrapper drives it
through an anyio blocking portal so the public surface stays synchronous.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, Union

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageResponse
from anyio.from_thread import start_blocking_portal

from aimu.agents.base import MessageHistory, Runner
from aimu.models.base import StreamChunk, StreamingContentType

from ._card import make_send_request, result_to_text

DEFAULT_AGENT_CARD_PATH = "/.well-known/agent-card.json"


class A2AConnectionError(RuntimeError):
    """Raised when a :class:`RemoteAgent` fails to connect to or call a remote A2A agent."""


def _response_text(response: SendMessageResponse) -> str:
    """Extract plain text from a ``SendMessageResponse``; raise on a JSON-RPC error."""
    root = response.root
    error = getattr(root, "error", None)
    if error is not None:
        raise A2AConnectionError(f"remote A2A agent returned an error: {error}")
    return result_to_text(root.result)


class RemoteAgent(Runner):
    """A remote A2A agent presented as a local synchronous ``Runner``.

    Construct via :meth:`connect` (it resolves the remote agent card first)::

        remote = RemoteAgent.connect("http://localhost:9000")
        print(remote.run("Summarise the news"))

        # Composes like any other runner:
        chain = Chain(agents=[local_agent, remote])
        local = Agent(client, tools=[remote.as_tool()])

    Holds a live async connection (an ``httpx.AsyncClient`` driven through an anyio
    portal); keep the instance alive for the connection's lifetime and let it be garbage
    collected (or call :meth:`close`) to tear the portal down.
    """

    def __init__(self, portal_cm, portal, client: A2AClient, httpx_client: httpx.AsyncClient, name: str, card: Any):
        self._portal_cm = portal_cm
        self._portal = portal
        self._client = client
        self._httpx_client = httpx_client
        self.name = name
        self.card = card
        # Surface the remote card's description as system_message so the inherited
        # Runner.as_tool() produces a meaningful tool description for this remote agent.
        self.system_message = getattr(card, "description", None)
        self._last_messages: list[dict] = []

    @classmethod
    def connect(
        cls,
        url: str,
        *,
        name: Optional[str] = None,
        agent_card_path: str = DEFAULT_AGENT_CARD_PATH,
        timeout: float = 60.0,
    ) -> "RemoteAgent":
        """Resolve the remote agent card at ``url`` and return a connected ``RemoteAgent``.

        ``name`` defaults to the remote agent card's name. Connection failures raise
        :class:`A2AConnectionError` with the original exception chained.
        """
        portal_cm = start_blocking_portal(backend="asyncio")
        try:
            portal = portal_cm.__enter__()
            client, httpx_client, card = portal.call(cls._aconnect, url, agent_card_path, timeout)
        except Exception as exc:
            try:
                portal_cm.__exit__(None, None, None)
            except Exception:
                pass
            raise A2AConnectionError(f"failed to connect to A2A agent at {url!r}: {exc}") from exc
        return cls(portal_cm, portal, client, httpx_client, name or card.name, card)

    @staticmethod
    async def _aconnect(url: str, agent_card_path: str, timeout: float):
        import warnings

        httpx_client = httpx.AsyncClient(timeout=timeout)
        resolver = A2ACardResolver(httpx_client, base_url=url, agent_card_path=agent_card_path)
        card = await resolver.get_agent_card()
        with warnings.catch_warnings():
            # A2AClient is the simplest 0.3.x entry point; silence its ClientFactory deprecation.
            warnings.simplefilter("ignore", DeprecationWarning)
            # Send requests to the URL we connected to (url=), not the card's advertised url,
            # so a card behind a proxy / advertising an internal address still works.
            client = A2AClient(httpx_client, agent_card=card, url=url)
        return client, httpx_client, card

    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Send ``task`` to the remote agent and return its text response.

        ``stream=True`` yields the full response as a single ``GENERATING`` chunk followed
        by ``DONE``; incremental token streaming over the portal is not supported on the
        sync surface (use ``aimu.aio.a2a.RemoteAgent`` for real streaming).
        """
        text = self._send(task)
        self._last_messages = [
            {"role": "user", "content": task},
            {"role": "assistant", "content": text},
        ]
        if stream:
            return iter(
                [
                    StreamChunk(StreamingContentType.GENERATING, text, agent=self.name),
                    StreamChunk(StreamingContentType.DONE, "", agent=self.name),
                ]
            )
        return text

    def _send(self, task: str) -> str:
        try:
            response = self._portal.call(self._client.send_message, make_send_request(task))
        except Exception as exc:
            raise A2AConnectionError(f"A2A send_message to {self.name!r} failed: {exc}") from exc
        return _response_text(response)

    @property
    def messages(self) -> MessageHistory:
        return {self.name: self._last_messages}

    def close(self) -> None:
        try:
            self._portal.call(self._httpx_client.aclose)
        except Exception:
            pass
        try:
            self._portal_cm.__exit__(None, None, None)
        except Exception:
            pass

    def __del__(self):
        self.close()

    def __deepcopy__(self, memo):
        # Holds a live async connection context; it cannot be duplicated.
        memo[id(self)] = self
        return self
