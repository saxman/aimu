from unittest.mock import MagicMock, patch

import requests

from aimu import paths
from aimu.tools.client import MCPClient
from aimu.tools.servers import mcp

from fastmcp import FastMCP

MCP_SERVERS_FILE = str(paths.package / "tools" / "servers.py")


def _response_text(response) -> str:
    return response.content[0].text  # type: ignore[union-attr]


def test_mcp_client_with_config():
    config = {
        "mcpServers": {"echo": {"command": "python", "args": [MCP_SERVERS_FILE]}},
    }

    client = MCPClient(config=config)
    tools = client.list_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0


def test_mcp_client_with_file():
    client = MCPClient(file=MCP_SERVERS_FILE)
    tools = client.list_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0


def test_mcp_client_with_server():
    mcp = FastMCP("TestTools")

    @mcp.tool()
    def echo(echo_string: str) -> str:
        return echo_string

    client = MCPClient(server=mcp)
    tools = client.list_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0

    response = client.call_tool("echo", {"echo_string": "test"})

    assert _response_text(response) == "test"

    # call the tools a second time to ensure the server is running
    response = client.call_tool("echo", {"echo_string": "test again"})

    assert _response_text(response) == "test again"


def test_none_type_tool_response():
    mcp = FastMCP("TestTools")

    @mcp.tool()
    def none_response() -> None:
        return None

    client = MCPClient(server=mcp)
    tools = client.list_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0

    response = client.call_tool("none_response", {})

    assert len(response.content) == 0


def _mock_response(json_data=None, text=None, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.raise_for_status = MagicMock()
    if json_data is not None:
        mock.json.return_value = json_data
    if text is not None:
        mock.text = text
    return mock


def test_search_returns_results():
    fake_results = {
        "results": [
            {"title": "Python Docs", "url": "https://docs.python.org", "content": "Official Python docs."},
            {"title": "PyPI", "url": "https://pypi.org", "content": "Python package index."},
        ]
    }
    client = MCPClient(server=mcp)
    with patch("aimu.tools.servers.requests.get", return_value=_mock_response(json_data=fake_results)):
        response = client.call_tool("search", {"query": "python", "num_results": 2})

    text = _response_text(response)
    assert "Python Docs" in text
    assert "https://docs.python.org" in text
    assert "1." in text and "2." in text


def test_search_no_results():
    client = MCPClient(server=mcp)
    with patch("aimu.tools.servers.requests.get", return_value=_mock_response(json_data={"results": []})):
        response = client.call_tool("search", {"query": "xyzzy"})

    assert _response_text(response) == "No results found."


def test_search_request_error():
    client = MCPClient(server=mcp)
    with patch("aimu.tools.servers.requests.get", side_effect=requests.RequestException("timeout")):
        response = client.call_tool("search", {"query": "python"})

    assert "Error contacting SearXNG" in _response_text(response)


def test_get_webpage_returns_text():
    html = "<html><head><title>Test</title></head><body><h1>Hello</h1><p>World</p></body></html>"
    client = MCPClient(server=mcp)
    with patch("aimu.tools.servers.requests.get", return_value=_mock_response(text=html)):
        response = client.call_tool("get_webpage", {"url": "https://example.com"})

    text = _response_text(response)
    assert "Hello" in text
    assert "World" in text
    assert "<" not in text  # HTML tags stripped


def test_get_webpage_strips_script_and_style():
    html = "<html><body><script>alert('x')</script><style>.a{}</style><p>Visible</p></body></html>"
    client = MCPClient(server=mcp)
    with patch("aimu.tools.servers.requests.get", return_value=_mock_response(text=html)):
        response = client.call_tool("get_webpage", {"url": "https://example.com"})

    text = _response_text(response)
    assert "Visible" in text
    assert "alert" not in text
    assert ".a" not in text


def test_get_webpage_request_error():
    client = MCPClient(server=mcp)
    with patch("aimu.tools.servers.requests.get", side_effect=requests.RequestException("connection refused")):
        response = client.call_tool("get_webpage", {"url": "https://example.com"})

    assert "Error fetching page" in _response_text(response)


def test_multiple_tool_calls_client_reused():
    mcp = FastMCP("TestTools")

    @mcp.tool()
    def none_response() -> None:
        return None

    client = MCPClient(server=mcp)
    tools = client.list_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0

    client.call_tool("none_response", {})
    mcp_client = client.client

    client.call_tool("none_response", {})

    assert mcp_client == client.client
