from aimu.tools.client import MCPClient
from aimu import paths

from fastmcp import FastMCP

MCP_SERVERS_FILE = str(paths.package / "tools" / "servers.py")


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

    assert response.content[0].text == "test"

    # call the tools a second time to ensure the server is running
    response = client.call_tool("echo", {"echo_string": "test again"})

    assert response.content[0].text == "test again"


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
