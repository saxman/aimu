"""
Tests for the @tool decorator and Python-function tool dispatch through Agent.
"""

import pytest

from aimu.agents import Agent
from aimu.agents._tool_loop import _ToolLoop
from aimu.tools import ToolArgumentError, ToolSignatureError, coerce_tool_arguments, tool
from helpers import MockModelClient


def _stage_tool_calls(client, calls):
    """Append the assistant(tool_calls) message a provider stores, so the engine can dispatch it.

    ``calls`` is a list of ``{"name": ..., "arguments": ...}`` dicts (the old ``_handle_tool_calls``
    input shape).
    """
    client.messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c.get("arguments", {})},
                    "id": f"id{i}",
                }
                for i, c in enumerate(calls)
            ],
        }
    )


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


def test_tool_decorator_full_prose_is_description():
    @tool
    def f(x: str):
        """Short summary.

        Longer details that are now included (guidance the model needs must not be dropped).
        """

    assert f.__tool_spec__["function"]["description"] == (
        "Short summary.\n\nLonger details that are now included (guidance the model needs must not be dropped)."
    )


def test_tool_decorator_description_stops_at_args_section():
    @tool
    def f(word: str, count: int = 1):
        """Repeat a word some number of times.

        Args:
            word: The word to repeat.
            count: How many times to repeat it.
        """

    spec = f.__tool_spec__["function"]
    assert spec["description"] == "Repeat a word some number of times."
    props = spec["parameters"]["properties"]
    assert props["word"]["description"] == "The word to repeat."
    assert props["count"]["description"] == "How many times to repeat it."


def test_tool_decorator_arg_continuation_lines_are_joined():
    @tool
    def f(x: str):
        """Do a thing.

        Args:
            x: A long description that wraps
                onto a second indented line.
        """

    props = f.__tool_spec__["function"]["parameters"]["properties"]
    assert props["x"]["description"] == "A long description that wraps onto a second indented line."


def test_tool_decorator_literal_becomes_enum():
    from typing import Literal, Optional

    @tool
    def f(mode: Literal["once", "daily", "weekly"], flag: Optional[Literal["a", "b"]] = None):
        """Pick a mode."""

    props = f.__tool_spec__["function"]["parameters"]["properties"]
    assert props["mode"]["enum"] == ["once", "daily", "weekly"]
    assert props["mode"]["type"] == "string"
    assert props["flag"]["enum"] == ["a", "b"]  # enum survives Optional unwrapping


def test_tool_decorator_function_still_callable():
    @tool
    def double(n: int) -> int:
        """Double a number."""
        return n * 2

    assert double(5) == 10


# ---------------------------------------------------------------------------
# Decorator: signature validation
# ---------------------------------------------------------------------------


def test_tool_rejects_varargs():
    with pytest.raises(ToolSignatureError, match="variadic"):

        @tool
        def f(*args):
            """Bad."""
            return args


def test_tool_rejects_kwargs():
    with pytest.raises(ToolSignatureError, match="variadic"):

        @tool
        def f(**kwargs):
            """Bad."""
            return kwargs


def test_tool_rejects_param_without_hint_or_default():
    with pytest.raises(ToolSignatureError, match="no type hint"):

        @tool
        def f(x):
            """Bad."""
            return x


def test_tool_accepts_param_with_default_no_hint():
    @tool
    def f(x=1):
        """OK."""
        return x

    spec = f.__tool_spec__
    assert spec["function"]["parameters"]["properties"]["x"]["type"] == "string"
    assert "x" not in spec["function"]["parameters"]["required"]


def test_tool_supports_optional():
    from typing import Optional

    @tool
    def f(name: Optional[str] = None) -> str:
        """OK."""
        return name or ""

    spec = f.__tool_spec__
    assert spec["function"]["parameters"]["properties"]["name"]["type"] == "string"


def test_tool_supports_pipe_none():
    @tool
    def f(value: int | None = None) -> str:
        """OK."""
        return str(value)

    spec = f.__tool_spec__
    assert spec["function"]["parameters"]["properties"]["value"]["type"] == "integer"


def test_tool_maps_list_and_dict_generics():
    @tool
    def f(items: list[str], meta: dict[str, int]) -> str:
        """OK."""
        return ""

    props = f.__tool_spec__["function"]["parameters"]["properties"]
    assert props["items"]["type"] == "array"
    assert props["meta"]["type"] == "object"


# ---------------------------------------------------------------------------
# Dispatch: Python tool wired through the tool-loop engine (_ToolLoop._dispatch)
# ---------------------------------------------------------------------------


def test_agent_python_tool_dispatched():
    """The tool-loop engine invokes a tool by name from the staged assistant tool_calls."""
    calls = []

    @tool
    def record(value: str) -> str:
        """Record a value."""
        calls.append(value)
        return f"recorded:{value}"

    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "record", "arguments": {"value": "hello"}}])
    _ToolLoop(client, [record])._dispatch()

    assert calls == ["hello"]
    tool_messages = [m for m in client.messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "recorded:hello"


def test_agent_python_tool_executed_during_run():
    """Behavioral: the Agent makes its configured tools available to the loop, so a tool the
    model calls is executed during ``run()``. (Replaces the old test asserting the tools were
    pushed onto ``model_client.tools``, which the pure client no longer holds.)"""
    calls = []

    @tool
    def echo() -> str:
        """Echo."""
        calls.append(1)
        return "echoed"

    client = MockModelClient(["tool", "done"])
    agent = Agent(client, tools=[echo])
    assert agent.run("task") == "done"

    assert calls == [1]
    tool_messages = [m for m in client.messages if m["role"] == "tool"]
    assert tool_messages[-1]["content"] == "echoed"


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


def test_duplicate_tool_name_last_in_list_wins():
    """All tools (Python @tool + MCP via as_tools()) live in self.tools as callables, and
    dispatch builds {fn.__name__: fn}. On a name collision the last entry in the list wins,
    so to override an MCP tool with a local Python one, append the Python tool after it."""

    @tool
    def first(value: str) -> str:
        """First implementation."""
        return f"first:{value}"

    @tool
    def second(value: str) -> str:
        """Second implementation."""
        return f"second:{value}"

    # Force a name collision: both callables answer to "shared".
    first.__name__ = "shared"
    second.__name__ = "shared"

    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "shared", "arguments": {"value": "x"}}])
    _ToolLoop(client, [first, second])._dispatch()

    tool_messages = [m for m in client.messages if m["role"] == "tool"]
    assert tool_messages[0]["content"] == "second:x"  # last in list wins


def test_python_tool_exception_recorded_as_error_text():
    """If a Python tool raises, the failure is captured as a tool message rather than propagating."""

    @tool
    def broken(x: str) -> str:
        """Always raises."""
        raise RuntimeError("boom")

    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "broken", "arguments": {"x": "ignored"}}])
    _ToolLoop(client, [broken])._dispatch()

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
# Argument validation / coercion (P0-A)
# ---------------------------------------------------------------------------


def test_tool_decorator_captures_param_adapters_and_allowed_args():
    @tool
    def f(needed: int, opt: str = "x"):
        """Decorator records per-param validators and the allowed-arg set."""

    assert set(f.__tool_param_adapters__) == {"needed", "opt"}
    assert f.__tool_allowed_args__ == {"needed", "opt"}
    assert f.__tool_required__ == ["needed"]


def test_coerce_tool_arguments_lax_coerces_known_types():
    @tool
    def f(count: int, ratio: float, flag: bool) -> str:
        """Coerce stringified scalars to their declared types."""
        return ""

    coerced = coerce_tool_arguments(f, {"count": "5", "ratio": "1.5", "flag": "true"})
    assert coerced == {"count": 5, "ratio": 1.5, "flag": True}
    assert isinstance(coerced["count"], int)


def test_coerce_tool_arguments_rejects_bad_type():
    @tool
    def f(count: int) -> str:
        """Reject an uncoercible value."""
        return ""

    with pytest.raises(ToolArgumentError) as excinfo:
        coerce_tool_arguments(f, {"count": "abc"})
    assert "count" in str(excinfo.value)


def test_coerce_tool_arguments_rejects_missing_required():
    @tool
    def f(needed: int, opt: int = 0) -> str:
        """Reject a call missing a required arg."""
        return ""

    with pytest.raises(ToolArgumentError) as excinfo:
        coerce_tool_arguments(f, {"opt": 1})
    assert "needed" in str(excinfo.value)


def test_coerce_tool_arguments_rejects_unknown_arg():
    @tool
    def f(value: str) -> str:
        """Reject an arg not in the signature."""
        return ""

    with pytest.raises(ToolArgumentError) as excinfo:
        coerce_tool_arguments(f, {"value": "ok", "surprise": 1})
    assert "surprise" in str(excinfo.value)


def test_coerce_tool_arguments_passes_through_without_adapters():
    """MCP-style callables carry __tool_spec__ but are not built via _build_spec, so
    they have no __tool_param_adapters__ and their args pass through unchanged."""

    def mcp_like(**kwargs):
        return ""

    mcp_like.__tool_spec__ = {"type": "function", "function": {"name": "mcp_like"}}
    args = {"anything": "5", "extra": [1, 2]}
    assert coerce_tool_arguments(mcp_like, args) == args


def test_dispatch_coerces_arguments_before_invocation():
    received = {}

    @tool
    def record(count: int) -> str:
        """Record the received value and its type."""
        received["value"] = count
        received["type"] = type(count).__name__
        return "ok"

    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "record", "arguments": {"count": "5"}}])
    _ToolLoop(client, [record])._dispatch()

    assert received == {"value": 5, "type": "int"}


def test_dispatch_bad_argument_returns_validation_message_not_crash_prefix():
    @tool
    def record(count: int) -> str:
        """Should never run with a bad argument."""
        raise AssertionError("tool body must not run on invalid arguments")

    client = MockModelClient([])
    _stage_tool_calls(client, [{"name": "record", "arguments": {"count": "abc"}}])
    _ToolLoop(client, [record])._dispatch()

    tool_messages = [m for m in client.messages if m["role"] == "tool"]
    content = tool_messages[0]["content"]
    assert "count" in content
    assert "raised an error" not in content


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
