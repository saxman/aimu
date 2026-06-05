from typing import Optional

from anyio.from_thread import start_blocking_portal
from fastmcp import Client, FastMCP


class MCPConnectionError(RuntimeError):
    """Raised when an :class:`MCPClient` fails to establish or use a connection."""


class _ToolResponse:
    """Wraps FastMCP call_tool list result to preserve the .content API."""

    def __init__(self, content: list):
        self.content = content


class MCPClient:
    """Synchronous wrapper around an async FastMCP Client.

    Uses anyio's ``start_blocking_portal()`` to run the FastMCP Client in a background
    thread with a properly initialized anyio event loop.

    Pass *exactly one* of ``config``, ``server``, or ``file``. Connection errors are
    re-raised as :class:`MCPConnectionError` with the original exception chained.
    """

    def __init__(self, config: Optional[dict] = None, server: Optional[FastMCP] = None, file: Optional[str] = None):
        sources = [s is not None for s in (config, server, file)]
        if sum(sources) != 1:
            raise MCPConnectionError(
                f"MCPClient requires exactly one of: config=, server=, or file=. Got {sum(sources)} source(s)."
            )
        self._transport = config if config is not None else (server if server is not None else file)

        self._portal_cm = start_blocking_portal(backend="asyncio")
        try:
            self._portal = self._portal_cm.__enter__()
            self._portal.call(self._connect)
        except Exception as exc:
            try:
                self._portal_cm.__exit__(None, None, None)
            except Exception:
                pass
            raise MCPConnectionError(f"failed to connect MCP transport {self._transport!r}: {exc}") from exc

    async def _connect(self):
        self._client = Client(self._transport)
        await self._client.__aenter__()

    def __del__(self):
        try:
            self._portal.call(self._client.__aexit__, None, None, None)
        except Exception:
            pass
        try:
            self._portal_cm.__exit__(None, None, None)
        except Exception:
            pass

    def __deepcopy__(self, memo):
        # MCPClient holds a live async connection context; it cannot be duplicated.
        memo[id(self)] = self
        return self

    @property
    def client(self):
        return self._client

    def ping(self) -> list:
        """Verify the connection is alive by listing tools. Returns the tool list.

        Raises :class:`MCPConnectionError` if the connection has been closed or the
        server is unreachable.
        """
        try:
            return self.list_tools()
        except Exception as exc:
            raise MCPConnectionError(f"MCP ping failed: {exc}") from exc

    def call_tool(self, tool_name: str, params: dict):
        try:
            result = self._portal.call(self._client.call_tool, tool_name, params)
        except Exception as exc:
            raise MCPConnectionError(f"MCP call_tool({tool_name!r}) failed: {exc}") from exc
        content = result.content if hasattr(result, "content") else result
        return _ToolResponse(content)

    def list_tools(self):
        return self._portal.call(self._client.list_tools)

    def get_tools(self) -> list[dict]:
        from .mcp_format import mcp_tools_to_openai

        return mcp_tools_to_openai(self.list_tools())

    def as_tools(self) -> list:
        """Return this server's tools as ``@tool``-style callables.

        Each callable closes over this client, invokes :meth:`call_tool` cross-process,
        and returns the result's text content as a string. The callables carry
        ``__tool_spec__`` (OpenAI format), ``__tool_is_async__ = False``, and
        ``__tool_is_streaming__ = False``, so they drop straight into ``client.tools`` or
        ``Agent(tools=...)`` and dispatch through the same path as ``@tool`` functions —
        no ``model_client.mcp_client`` reference needed::

            mcp = MCPClient(server=my_server)
            client.tools = builtin.web + mcp.as_tools()

        The tool list is a snapshot taken now (one ``list_tools()`` round-trip); call
        ``as_tools()`` again to pick up server-side changes. Keep a reference to this
        ``MCPClient`` (or to the returned callables, which hold one) for the lifetime of
        the connection.
        """
        from .mcp_format import mcp_content_to_text

        def _make(spec: dict):
            name = spec["function"]["name"]

            def _call(**kwargs):
                return mcp_content_to_text(self.call_tool(name, kwargs))

            _call.__name__ = name
            _call.__qualname__ = name
            _call.__doc__ = spec["function"].get("description") or name
            _call.__tool_spec__ = spec
            _call.__tool_is_async__ = False
            _call.__tool_is_streaming__ = False
            return _call

        return [_make(spec) for spec in self.get_tools()]
