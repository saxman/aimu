"""Async MCP client: thin parallel to ``aimu.tools.MCPClient``.

Uses FastMCP's native async ``Client`` directly. No anyio portal; the caller
already has an event loop. Construction signature mirrors the sync wrapper so
users see the same shape on both surfaces.
"""

from __future__ import annotations

from typing import Optional

from fastmcp import Client, FastMCP

from aimu.tools.client import MCPConnectionError
from aimu.tools.mcp_format import mcp_tools_to_openai


class MCPClient:
    """Async wrapper around a FastMCP ``Client``.

    Use the ``connect()`` classmethod factory to construct + connect in one ``await``::

        mcp = await MCPClient.connect(server=my_fastmcp_server)
        try:
            tools = await mcp.get_tools()
            result = await mcp.call_tool("foo", {"x": 1})
        finally:
            await mcp.aclose()
    """

    def __init__(
        self,
        *,
        config: Optional[dict] = None,
        server: Optional[FastMCP] = None,
        file: Optional[str] = None,
    ):
        sources = [s is not None for s in (config, server, file)]
        if sum(sources) != 1:
            raise MCPConnectionError(
                f"MCPClient requires exactly one of: config=, server=, or file=. Got {sum(sources)} source(s)."
            )
        self._transport = config if config is not None else (server if server is not None else file)
        self._client: Optional[Client] = None

    @classmethod
    async def connect(
        cls,
        *,
        config: Optional[dict] = None,
        server: Optional[FastMCP] = None,
        file: Optional[str] = None,
    ) -> MCPClient:
        """Construct and connect in one ``await``. Returns the live instance."""
        instance = cls(config=config, server=server, file=file)
        try:
            instance._client = Client(instance._transport)
            await instance._client.__aenter__()
        except Exception as exc:
            raise MCPConnectionError(f"failed to connect MCP transport {instance._transport!r}: {exc}") from exc
        return instance

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    @property
    def client(self) -> Optional[Client]:
        return self._client

    async def ping(self) -> list:
        """Verify the connection is alive by listing tools."""
        try:
            return await self.list_tools()
        except Exception as exc:
            raise MCPConnectionError(f"MCP ping failed: {exc}") from exc

    async def call_tool(self, tool_name: str, params: dict):
        if self._client is None:
            raise MCPConnectionError("MCPClient not connected. Use `await MCPClient.connect(...)`.")
        try:
            return await self._client.call_tool(tool_name, params)
        except Exception as exc:
            raise MCPConnectionError(f"MCP call_tool({tool_name!r}) failed: {exc}") from exc

    async def list_tools(self):
        if self._client is None:
            raise MCPConnectionError("MCPClient not connected. Use `await MCPClient.connect(...)`.")
        return await self._client.list_tools()

    async def get_tools(self) -> list[dict]:
        """Return tools in OpenAI function-calling format."""
        return mcp_tools_to_openai(await self.list_tools())

    async def as_tools(self) -> list:
        """Return this server's tools as async ``@tool``-style callables.

        Async mirror of :meth:`aimu.tools.MCPClient.as_tools`. Each callable is an
        ``async def`` that awaits :meth:`call_tool` and returns the result's text content;
        it carries ``__tool_spec__``, ``__tool_is_async__ = True``, and
        ``__tool_is_streaming__ = False``, so it drops into ``client.tools`` /
        ``aio.Agent(tools=...)`` and the async dispatcher awaits it directly::

            mcp = await aio.MCPClient.connect(server=my_server)
            agent = aio.Agent(client, tools=await mcp.as_tools())

        The list is a snapshot (one ``list_tools()`` round-trip); call again to refresh.
        """
        from aimu.tools.mcp_format import mcp_content_to_text

        def _make(spec: dict):
            name = spec["function"]["name"]

            async def _call(**kwargs):
                return mcp_content_to_text(await self.call_tool(name, kwargs))

            _call.__name__ = name
            _call.__qualname__ = name
            _call.__doc__ = spec["function"].get("description") or name
            _call.__tool_spec__ = spec
            _call.__tool_is_async__ = True
            _call.__tool_is_streaming__ = False
            return _call

        return [_make(spec) for spec in await self.get_tools()]

    async def aclose(self) -> None:
        """Close the underlying connection. Idempotent."""
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            finally:
                self._client = None
