import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from aimu import paths
from aimu.tools.client import MCPClient
from aimu.tools.mcp import mcp

from fastmcp import FastMCP

# Use -m to avoid naming conflict: files named mcp.py shadow the `mcp` package
# when Python adds the script's directory to sys.path on direct execution.
ECHO_SERVER_FILE = str(Path(__file__).parent / "echo_server.py")


def _response_text(response) -> str:
    return response.content[0].text  # type: ignore[union-attr]


def test_mcp_client_with_config():
    config = {
        "mcpServers": {"aimu": {"command": "python", "args": ["-m", "aimu.tools.mcp"]}},
    }

    client = MCPClient(config=config)
    tools = client.list_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0


def test_mcp_client_with_config_module():
    config = {
        "mcpServers": {
            "aimu": {"command": "python", "args": ["-m", "aimu.tools.mcp"]},
        }
    }

    client = MCPClient(config=config)
    tools = client.list_tools()

    assert isinstance(tools, list)
    assert len(tools) > 0


def test_mcp_client_with_file():
    # Uses a helper script not named mcp.py to avoid shadowing the `mcp` package.
    client = MCPClient(file=ECHO_SERVER_FILE)
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


_UNSET = object()


def _mock_response(json_data=_UNSET, text=None, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.raise_for_status = MagicMock()
    if json_data is not _UNSET:
        mock.json.return_value = json_data
    if text is not None:
        mock.text = text
    return mock


FAKE_GEO_RESPONSE = {
    "results": [{"name": "London", "country": "United Kingdom", "latitude": 51.50853, "longitude": -0.12574}]
}

FAKE_WEATHER_RESPONSE = {
    "current": {
        "weather_code": 1,
        "temperature_2m": 20.0,
        "apparent_temperature": 18.0,
        "relative_humidity_2m": 60,
        "wind_speed_10m": 15.0,
    }
}


def test_get_weather_returns_data():
    client = MCPClient(server=mcp)
    with patch(
        "aimu.tools.mcp.requests.get",
        side_effect=[
            _mock_response(json_data=FAKE_GEO_RESPONSE),
            _mock_response(json_data=FAKE_WEATHER_RESPONSE),
        ],
    ):
        response = client.call_tool("get_weather", {"location": "London"})

    text = _response_text(response)
    assert "Mainly clear" in text
    assert "20.0" in text
    assert "London" in text
    assert "United Kingdom" in text


def test_get_weather_location_not_found():
    client = MCPClient(server=mcp)
    with patch("aimu.tools.mcp.requests.get", return_value=_mock_response(json_data={"results": None})):
        response = client.call_tool("get_weather", {"location": "unknownxyz"})

    assert "Location not found" in _response_text(response)


def test_get_weather_request_error():
    client = MCPClient(server=mcp)
    with patch("aimu.tools.mcp.requests.get", side_effect=requests.RequestException("timeout")):
        response = client.call_tool("get_weather", {"location": "London"})

    assert "Error fetching weather" in _response_text(response)


def test_get_weather_coordinates():
    client = MCPClient(server=mcp)
    with patch("aimu.tools.mcp.requests.get", return_value=_mock_response(json_data=FAKE_WEATHER_RESPONSE)):
        response = client.call_tool("get_weather", {"location": "51.5,-0.12"})

    text = _response_text(response)
    assert "20.0" in text
    assert "°C" in text


def test_get_weather_real_call():
    client = MCPClient(server=mcp)
    response = client.call_tool("get_weather", {"location": "London"})

    text = _response_text(response)
    if text.startswith("Error"):
        pytest.skip(f"Open-Meteo unavailable: {text}")
    assert "London" in text
    assert "°C" in text


def test_search_returns_results():
    fake_results = {
        "results": [
            {"title": "Python Docs", "url": "https://docs.python.org", "content": "Official Python docs."},
            {"title": "PyPI", "url": "https://pypi.org", "content": "Python package index."},
        ]
    }
    client = MCPClient(server=mcp)
    with patch("aimu.tools.mcp.requests.get", return_value=_mock_response(json_data=fake_results)):
        response = client.call_tool("search", {"query": "python", "num_results": 2})

    text = _response_text(response)
    assert "Python Docs" in text
    assert "https://docs.python.org" in text
    assert "1." in text and "2." in text


def test_search_no_results():
    client = MCPClient(server=mcp)
    with patch("aimu.tools.mcp.requests.get", return_value=_mock_response(json_data={"results": []})):
        response = client.call_tool("search", {"query": "xyzzy"})

    assert _response_text(response) == "No results found."


def test_search_request_error():
    client = MCPClient(server=mcp)
    with patch("aimu.tools.mcp.requests.get", side_effect=requests.RequestException("timeout")):
        response = client.call_tool("search", {"query": "python"})

    assert "Error contacting SearXNG" in _response_text(response)


def test_get_webpage_returns_text():
    html = "<html><head><title>Test</title></head><body><h1>Hello</h1><p>World</p></body></html>"
    client = MCPClient(server=mcp)
    with patch("aimu.tools.mcp.requests.get", return_value=_mock_response(text=html)):
        response = client.call_tool("get_webpage", {"url": "https://example.com"})

    text = _response_text(response)
    assert "Hello" in text
    assert "World" in text
    assert "<" not in text  # HTML tags stripped


def test_get_webpage_strips_script_and_style():
    html = "<html><body><script>alert('x')</script><style>.a{}</style><p>Visible</p></body></html>"
    client = MCPClient(server=mcp)
    with patch("aimu.tools.mcp.requests.get", return_value=_mock_response(text=html)):
        response = client.call_tool("get_webpage", {"url": "https://example.com"})

    text = _response_text(response)
    assert "Visible" in text
    assert "alert" not in text
    assert ".a" not in text


def test_get_webpage_request_error():
    client = MCPClient(server=mcp)
    with patch("aimu.tools.mcp.requests.get", side_effect=requests.RequestException("connection refused")):
        response = client.call_tool("get_webpage", {"url": "https://example.com"})

    assert "Error fetching page" in _response_text(response)


def test_list_directory_returns_entries():
    client = MCPClient(server=mcp)
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "file.txt").write_text("hello")
        Path(tmp, "subdir").mkdir()
        response = client.call_tool("list_directory", {"path": tmp})

    text = _response_text(response)
    assert "subdir/" in text
    assert "file.txt" in text


def test_list_directory_dirs_before_files():
    client = MCPClient(server=mcp)
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "aaa.txt").write_text("x")
        Path(tmp, "zzz").mkdir()
        response = client.call_tool("list_directory", {"path": tmp})

    lines = _response_text(response).splitlines()
    assert lines[0] == "zzz/"
    assert lines[1] == "aaa.txt"


def test_list_directory_missing_path():
    client = MCPClient(server=mcp)
    response = client.call_tool("list_directory", {"path": "/nonexistent/path/xyz"})
    assert "does not exist" in _response_text(response)


def test_list_directory_on_file():
    client = MCPClient(server=mcp)
    with tempfile.NamedTemporaryFile() as f:
        response = client.call_tool("list_directory", {"path": f.name})
    assert "Not a directory" in _response_text(response)


def test_read_file_returns_content():
    client = MCPClient(server=mcp)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("line one\nline two\nline three")
        name = f.name
    response = client.call_tool("read_file", {"path": name})
    text = _response_text(response)
    assert "line one" in text
    assert "line three" in text


def test_read_file_truncates_at_max_lines():
    client = MCPClient(server=mcp)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(str(i) for i in range(10)))
        name = f.name
    response = client.call_tool("read_file", {"path": name, "max_lines": 3})
    text = _response_text(response)
    assert "truncated" in text
    assert "0" in text and "2" in text
    assert "9" not in text


def test_read_file_missing_path():
    client = MCPClient(server=mcp)
    response = client.call_tool("read_file", {"path": "/nonexistent/file.txt"})
    assert "does not exist" in _response_text(response)


def test_read_file_on_directory():
    client = MCPClient(server=mcp)
    with tempfile.TemporaryDirectory() as tmp:
        response = client.call_tool("read_file", {"path": tmp})
    assert "Not a file" in _response_text(response)


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
