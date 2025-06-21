from aimu.tools import MCPClient
from aimu import paths


def test_mcp_client_empty_init():
    client = MCPClient()

    assert client.client is not None
    assert client.config is None

    tools = client.list_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0


def test_mcp_client_with_config():
    config = {
        "mcpServers": {"echo": {"command": "python", "args": [str(paths.package / "tools.py")]}},
    }

    client = MCPClient(config=config)

    assert client.client is None
    assert client.config == config

    tools = client.list_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0


def test_mcp_client_with_file():
    client = MCPClient(file=str(paths.package / "tools.py"))

    assert client.client is None
    assert client.config is None

    tools = client.list_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0
