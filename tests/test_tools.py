from aimu.tools import MCPClient
from aimu import paths

from fastmcp import FastMCP


def test_mcp_client_with_config():
    config = {
        "mcpServers": {"echo": {"command": "python", "args": [str(paths.tests / "mcp_test_server.py")]}},
    }

    client = MCPClient(config=config)

    assert client.server is None
    assert client.config == config

    tools = client.list_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0


def test_mcp_client_with_file():
    client = MCPClient(file=str(paths.tests / "mcp_test_server.py"))

    assert client.server is None
    assert client.config is None

    tools = client.list_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0


def test_mcp_client_with_server():
    mcp = FastMCP("AIMU Tools")

    @mcp.tool()
    def echo(echo_string: str) -> str:
        return echo_string

    client = MCPClient(server=mcp)

    assert isinstance(client.server, FastMCP)
    assert client.config is None

    tools = client.list_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0

    response = client.call_tool("echo", {"echo_string": "test"})

    assert response.content[0].text == "test"
