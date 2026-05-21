"""
Tests for the @tool decorator and Python-function tool dispatch through Agent.
"""

from aimu.agents import Agent
from aimu.tools import tool
from helpers import MockModelClient


# ---------------------------------------------------------------------------
# Decorator: spec generation
# ---------------------------------------------------------------------------


def test_tool_decorator_attaches_spec():
    @tool
    def my_tool(word: str) -> int:
        """Count things."""
        return len(word)

    assert hasattr(my_tool, "__tool_spec__")
    spec = my_tool.__tool_spec__
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "my_tool"
    assert spec["function"]["description"] == "Count things."


def test_tool_decorator_parses_parameters():
    @tool
    def f(word: str, count: int, ratio: float, flag: bool):
        """Take various types."""

    props = f.__tool_spec__["function"]["parameters"]["properties"]
    assert props["word"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["ratio"]["type"] == "number"
    assert props["flag"]["type"] == "boolean"


def test_tool_decorator_required_vs_optional():
    @tool
    def f(needed: str, opt: str = "default"):
        """Has one required and one optional arg."""

    required = f.__tool_spec__["function"]["parameters"]["required"]
    assert required == ["needed"]


def test_tool_decorator_first_paragraph_is_description():
    @tool
    def f(x: str):
        """Short summary.

        Longer details that should not be in the description.
        """

    assert f.__tool_spec__["function"]["description"] == "Short summary."


def test_tool_decorator_function_still_callable():
    @tool
    def double(n: int) -> int:
        """Double a number."""
        return n * 2

    assert double(5) == 10


# ---------------------------------------------------------------------------
# Dispatch: Python tool wired through Agent + BaseModelClient._handle_tool_calls
# ---------------------------------------------------------------------------


def test_agent_python_tool_dispatched():
    """Agent passes Python tools to model_client; _handle_tool_calls invokes them by name."""
    calls = []

    @tool
    def record(value: str) -> str:
        """Record a value."""
        calls.append(value)
        return f"recorded:{value}"

    client = MockModelClient([])
    client.tools = [record]

    client._handle_tool_calls(
        [{"name": "record", "arguments": {"value": "hello"}}],
        tools=[record.__tool_spec__],
    )

    assert calls == ["hello"]
    tool_messages = [m for m in client.messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "recorded:hello"


def test_agent_python_tool_via_agent_prepare_run():
    """Agent._prepare_run pushes Agent.tools onto model_client.tools."""

    @tool
    def echo(value: str) -> str:
        """Echo a value."""
        return value

    client = MockModelClient(["done"])
    agent = Agent(client, tools=[echo])
    agent.run("task")

    assert client.tools == [echo]


def test_python_tool_spec_appears_in_chat_setup_tools():
    """_chat_setup returns the Python tool spec when supports_tools=True and use_tools=True."""

    @tool
    def my_tool(x: str) -> str:
        """Do."""
        return x

    client = MockModelClient(["x"])
    client.tools = [my_tool]
    # MockModelClient's model.supports_tools is True via MagicMock; ensure that's the case
    client.model.supports_tools = True

    _, tools = client._chat_setup("hello")
    spec_names = [t["function"]["name"] for t in tools]
    assert "my_tool" in spec_names


def test_python_tool_takes_precedence_over_mcp():
    """When a name matches both a Python tool and an MCP tool, the Python one wins."""

    @tool
    def shared(value: str) -> str:
        """Python implementation."""
        return f"python:{value}"

    client = MockModelClient([])
    client.tools = [shared]
    # Simulate an MCP tool with the same name; if dispatch hit MCP it would fail (no mcp_client set).
    mcp_spec = {"type": "function", "function": {"name": "shared", "description": "MCP"}}

    client._handle_tool_calls(
        [{"name": "shared", "arguments": {"value": "x"}}],
        tools=[mcp_spec],
    )

    tool_messages = [m for m in client.messages if m["role"] == "tool"]
    assert tool_messages[0]["content"] == "python:x"


def test_python_tool_exception_recorded_as_error_text():
    """If a Python tool raises, the failure is captured as a tool message rather than propagating."""

    @tool
    def broken(x: str) -> str:
        """Always raises."""
        raise RuntimeError("boom")

    client = MockModelClient([])
    client.tools = [broken]

    client._handle_tool_calls(
        [{"name": "broken", "arguments": {"x": "ignored"}}],
        tools=[broken.__tool_spec__],
    )

    tool_messages = [m for m in client.messages if m["role"] == "tool"]
    assert "boom" in tool_messages[0]["content"]


def test_chat_setup_raises_for_undecorated_tool():
    """A function in client.tools without __tool_spec__ raises a clear error."""

    def plain(x: str) -> str:
        return x

    client = MockModelClient(["x"])
    client.tools = [plain]
    client.model.supports_tools = True

    try:
        client._chat_setup("hello")
    except ValueError as exc:
        assert "__tool_spec__" in str(exc)
    else:
        raise AssertionError("expected ValueError for undecorated tool")


# ---------------------------------------------------------------------------
# ModelClient delegation
# ---------------------------------------------------------------------------


def test_model_client_tools_attribute_propagates(monkeypatch):
    """Setting tools on a ModelClient delegates to the inner client."""
    # Build a ModelClient-like wrapper around MockModelClient by patching the factory dispatch.
    from aimu.models.model_client import ModelClient

    inner = MockModelClient(["hi"])

    # Bypass the real factory dispatch
    mc = ModelClient.__new__(ModelClient)
    mc._client = inner
    mc.model = inner.model
    mc.model_kwargs = None
    mc.default_generate_kwargs = {}

    @tool
    def f(x: str) -> str:
        """f"""
        return x

    mc.tools = [f]
    assert inner.tools == [f]
    assert mc.tools == [f]
