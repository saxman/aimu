"""Message-append timestamping (the inert ``timestamp`` key set at append time).

The key is stamped when a message is appended to ``self.messages`` so every consumer (and
persistence) gets an accurate per-message time; it is inert (``INERT_MESSAGE_KEYS``) so it never
reaches a provider. Mock-only; no backend required.
"""

from __future__ import annotations

from datetime import datetime

from aimu.models._internal.chat_state import _ChatStateMixin


class _Stub(_ChatStateMixin):
    def __init__(self):
        self.messages = []
        self._system_message = None
        self.model = None


def test_append_message_stamps_missing_timestamp():
    stub = _Stub()
    stub._append_message({"role": "assistant", "content": "hi"})
    assert "timestamp" in stub.messages[0]
    datetime.fromisoformat(stub.messages[0]["timestamp"])  # parses as ISO-8601


def test_append_message_preserves_existing_timestamp():
    stub = _Stub()
    stub._append_message({"role": "user", "content": "hi", "timestamp": "2020-01-01T00:00:00"})
    assert stub.messages[0]["timestamp"] == "2020-01-01T00:00:00"


def test_append_user_turn_stamps_the_user_message():
    stub = _Stub()
    stub._append_user_turn("hello")
    assert "timestamp" in stub.messages[-1] and stub.messages[-1]["role"] == "user"


def test_append_assistant_tool_calls_stamps_the_message():
    stub = _Stub()
    prepared = [({"name": "t", "arguments": {}}, "id1")]
    stub._append_assistant_tool_calls(prepared, content="calling")
    assert "timestamp" in stub.messages[-1] and stub.messages[-1]["role"] == "assistant"


def test_tool_loop_dispatch_stamps_tool_result():
    """The (non-streamed) tool-loop engine stamps the tool-result message it appends."""
    from aimu.agents._tool_loop import _ToolLoop
    from aimu.tools import tool
    from helpers import MockModelClient

    @tool
    def add(a: int, b: int) -> str:
        """Add two integers."""
        return str(a + b)

    client = MockModelClient([])
    client.messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": "add", "arguments": {"a": 1, "b": 2}}, "id": "id0"}
            ],
        }
    )
    _ToolLoop(client, [add])._dispatch()
    assert client.messages[-1]["role"] == "tool"
    assert "timestamp" in client.messages[-1]
