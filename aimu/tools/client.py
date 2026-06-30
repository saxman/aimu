from typing import Optional

from anyio.from_thread import start_blocking_portal
from fastmcp import Client, FastMCP


class MCPConnectionError(RuntimeError):
    """Raised when an :class:`MCPClient` fails to establish or use a connection."""


def _build_transport(
    config: Optional[dict],
    server: Optional["FastMCP"],
    file: Optional[str],
    url: Optional[str],
    auth=None,
    headers: Optional[dict] = None,
):
    """Validate the exactly-one-of source contract; return ``(transport, client_auth)``.

    Shared by the sync and async ``MCPClient``. A ``url`` with a string/absent ``auth`` is expanded
    into a single-server ``mcpServers`` config so FastMCP infers SSE vs streamable-HTTP and applies
    ``auth``/``headers`` in one path (``client_auth`` is ``None`` there). A **non-string** ``auth`` (a
    FastMCP ``OAuth`` / ``httpx.Auth`` provider object) is returned as ``client_auth`` and the bare
    ``url`` as the transport, so the caller passes it to ``Client(url, auth=provider)``. ``auth`` /
    ``headers`` without ``url`` is an error.
    """
    sources = [s is not None for s in (config, server, file, url)]
    if sum(sources) != 1:
        raise MCPConnectionError(
            f"MCPClient requires exactly one of: config=, server=, file=, or url=. Got {sum(sources)} source(s)."
        )
    if url is None and (auth is not None or headers is not None):
        raise MCPConnectionError("auth= and headers= apply only to a remote url=.")
    if url is not None and auth is not None and not isinstance(auth, str):
        # A provider object (e.g. a configured OAuth) is passed to the Client directly.
        if headers is not None:
            raise MCPConnectionError("headers= cannot be combined with a non-string auth provider object.")
        return url, auth
    if url is not None:
        server_config: dict = {"url": url}
        if headers is not None:
            server_config["headers"] = headers
        if auth is not None:
            server_config["auth"] = auth
        return {"mcpServers": {"server": server_config}}, None
    return (config if config is not None else (server if server is not None else file)), None


class _ToolResponse:
    """Wraps FastMCP call_tool list result to preserve the .content API."""

    def __init__(self, content: list):
        self.content = content


class MCPClient:
    """Synchronous wrapper around an async FastMCP Client.

    Uses anyio's ``start_blocking_portal()`` to run the FastMCP Client in a background
    thread with a properly initialized anyio event loop.

    Pass *exactly one* of ``config``, ``server``, ``file``, or ``url`` (a remote HTTP/SSE
    server). ``auth`` (a bearer-token string, the literal ``"oauth"``, or a configured FastMCP
    ``OAuth`` / ``httpx.Auth`` provider object) and ``headers`` apply only with ``url``; a provider
    object is passed straight to the FastMCP ``Client`` (and can't be combined with ``headers``).
    Connection errors are re-raised as :class:`MCPConnectionError` with the original exception chained.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        server: Optional[FastMCP] = None,
        file: Optional[str] = None,
        url: Optional[str] = None,
        auth=None,
        headers: Optional[dict] = None,
    ):
        self._transport, self._client_auth = _build_transport(config, server, file, url, auth, headers)

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
        self._client = Client(self._transport, auth=self._client_auth)
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
        ``Agent(tools=...)`` and dispatch through the same path as ``@tool`` functions,
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
