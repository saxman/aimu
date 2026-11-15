"""get_message: the way back from a masked request, and how every agent with a compaction gets it (mock-only).

A mask leaves a note naming a message by its position; ``get_message`` returns what the note replaced,
read from the live history the compaction never touched. An agent is given it whenever a run has a
compaction, ahead of its tool-group tail, and refused if one of its own tools already has the name.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from aimu.context.compaction import mask, trim
from aimu.tools import ToolResult, tool
from aimu.tools.builtin import HISTORY_TOOL_NAME, make_history_tools
from tests.helpers import MockModelClient
from tests.helpers_aio import MockAsyncModelClient

PNG = b"\x89PNG\r\n\x1a\nfake"
PNG_URL = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")


def _get_message(messages):
    (get_message,) = make_history_tools(SimpleNamespace(messages=messages))
    return get_message


def test_it_returns_a_message_with_who_it_is_from():
    get_message = _get_message(
        [{"role": "user", "content": "task"}, {"role": "tool", "name": "search", "content": "found"}]
    )
    assert get_message(1) == "Message 1 (tool, search)\nfound"


def test_it_shows_the_tool_calls_a_message_made():
    call = {"id": "c1", "type": "function", "function": {"name": "search", "arguments": {"q": "release date"}}}
    get_message = _get_message([{"role": "assistant", "content": "", "tool_calls": [call]}])
    assert get_message(0) == 'Message 0 (assistant)\nCalled search with {"q": "release date"}'


def test_a_long_message_comes_back_in_windows():
    get_message = _get_message([{"role": "tool", "name": "fetch", "content": "x" * 100}])
    first = get_message(0, max_chars=40)
    assert first.startswith("Message 0 (tool, fetch)\n")
    assert "call get_message(0) with offset=41 to continue" in first
    assert not get_message(0, offset=101).startswith("Message")  # the rest, without the header


def test_an_image_comes_back_as_an_image_on_the_first_window():
    message = {
        "role": "user",
        "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": PNG_URL}}],
    }
    result = _get_message([message])(0)
    assert isinstance(result, ToolResult)
    assert result.images == (PNG_URL,)
    assert str(result) == "Message 0 (user)\nlook\n[image, shown below]"


def test_a_bad_index_or_window_is_answered_not_raised():
    get_message = _get_message([{"role": "user", "content": "task"}])
    assert get_message(5) == "There is no message 5: this conversation has messages 0 to 0."
    assert "offset must be 1 or greater" in get_message(0, offset=0)
    assert "past the end of message 0" in get_message(0, offset=999)


def test_it_reads_the_live_history():
    messages = [{"role": "user", "content": "task"}]
    get_message = _get_message(messages)
    messages.append({"role": "assistant", "content": "added after the tool was built"})
    assert get_message(1).endswith("added after the tool was built")


# --------------------------------------------------------------------------- #
# Every agent with a compaction gets it                                       #
# --------------------------------------------------------------------------- #


@tool
def search(q: str) -> str:
    """Search."""
    return "found"


def _tool_names(loop):
    return [fn.__name__ for fn in loop._current_tools()]


def test_a_run_with_a_compaction_carries_get_message():
    from aimu.agents.agent import Agent

    agent = Agent(MockModelClient([]), tools=[search], compaction=trim(1000))
    assert _tool_names(agent._make_tool_loop(None, None, None)) == ["search", HISTORY_TOOL_NAME]


def test_a_run_without_one_does_not():
    from aimu.agents.agent import Agent

    agent = Agent(MockModelClient([]), tools=[search])
    assert _tool_names(agent._make_tool_loop(None, None, None)) == ["search"]
    # A per-run compaction brings it for that run only.
    assert HISTORY_TOOL_NAME in _tool_names(agent._make_tool_loop(None, None, None, compaction=mask(1000)))


def test_it_goes_ahead_of_the_tool_group_tail():
    from aimu.agents.agent import Agent
    from aimu.tools import ToolGroup

    group = ToolGroup(name="extra", description="Extra tools.", build=lambda: [])
    agent = Agent(MockModelClient([]), tools=[search], tool_groups=[group], compaction=trim(1000))
    assert _tool_names(agent._make_tool_loop(None, None, None)) == [
        "search",
        HISTORY_TOOL_NAME,
        "available_tools",
        "load_tools",
    ]


def test_a_tool_already_named_get_message_is_refused_while_there_is_a_compaction():
    from aimu.agents.agent import Agent

    @tool
    def get_message(index: int) -> str:
        """The host's own."""
        return ""

    agent = Agent(MockModelClient([]), tools=[get_message], compaction=trim(1000))
    with pytest.raises(ValueError, match="get_message"):
        agent._make_tool_loop(None, None, None)
    # Without a compaction there is no clash, so nothing to refuse.
    Agent(MockModelClient([]), tools=[get_message])._make_tool_loop(None, None, None)


def test_the_tool_reads_the_agents_own_history():
    from aimu.agents.agent import Agent

    client = MockModelClient([])
    client.messages = [{"role": "user", "content": "task"}, {"role": "tool", "name": "search", "content": "found it"}]
    agent = Agent(client, compaction=trim(1000))
    (history,) = [
        fn for fn in agent._make_tool_loop(None, None, None)._current_tools() if fn.__name__ == HISTORY_TOOL_NAME
    ]
    assert history(1) == "Message 1 (tool, search)\nfound it"


def test_an_async_agent_with_a_compaction_carries_it_too():
    from aimu.aio.agent import Agent

    agent = Agent(MockAsyncModelClient([]), tools=[search], compaction=trim(1000))
    assert _tool_names(agent._make_tool_loop(None, None, None)) == ["search", HISTORY_TOOL_NAME]


def test_a_masked_note_and_the_tool_agree_end_to_end():
    """What a mask sends names a message by position; calling the tool with that position returns the
    original, because the compaction read the same list the tool reads."""
    from aimu.agents.agent import Agent

    big = "word " * 400
    client = MockModelClient(["done"])
    client.model.supports_tools = False
    client.messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": {}}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "search", "content": big},
        {"role": "assistant", "content": "noted"},
    ]
    agent = Agent(client, compaction=mask(80))
    agent.run("and?")

    sent_note = client.sent[0][2]["content"]
    assert sent_note.startswith("[Message 2, the result of search") and "get_message(2)" in sent_note
    (history,) = [
        fn for fn in agent._make_tool_loop(None, None, None)._current_tools() if fn.__name__ == HISTORY_TOOL_NAME
    ]
    assert history(2) == f"Message 2 (tool, search)\n{big}"
