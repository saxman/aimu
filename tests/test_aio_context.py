"""Async mirror of the Agent(compaction=...) tests in tests/test_context.py.

See tests/test_context.py for the sync surface and the full rationale (principle 6: an
applied automatic compaction is never silent). These exercise the same behaviour on
aimu.aio.Agent + the async tool-loop engine.
"""

import logging

from aimu.context import count_tokens, trim_messages
from tests.helpers_aio import MockAsyncModelClient

# _maybe_compact() is inherited from _BaseToolLoop, defined in aimu.agents._tool_loop, so its
# WARNING is logged under that module's name even when the async loop is the one calling it.
_ASYNC_TOOL_LOOP_LOGGER = "aimu.agents._tool_loop"


def _old_conversation() -> list[dict]:
    return [
        {"role": "user", "content": "old question one"},
        {"role": "assistant", "content": "old answer one"},
        {"role": "user", "content": "old question two"},
        {"role": "assistant", "content": "old answer two"},
    ]


async def test_async_compaction_emits_ContextCompacted_carrying_the_dropped_messages():
    """Not a count -- the messages, so a caller who wants them back has them."""
    from aimu.aio.agent import Agent
    from aimu.events import ContextCompacted

    old_messages = _old_conversation()
    client = MockAsyncModelClient(["final answer"])
    client.model.supports_tools = False
    client.messages = list(old_messages)

    seen = []
    budget = count_tokens(old_messages[-2:])
    agent = Agent(
        client,
        events=seen.append,
        compaction=lambda msgs: trim_messages(msgs, max_tokens=budget, keep_last=0),
    )
    result = await agent.run("new question")

    assert result == "final answer"
    compacted = [e for e in seen if isinstance(e, ContextCompacted)]
    assert len(compacted) == 1
    event = compacted[0]
    assert event.dropped == old_messages[:2]
    assert event.dropped[0] is old_messages[0]
    assert event.dropped[1] is old_messages[1]
    assert event.before_tokens == count_tokens(old_messages)
    assert event.after_tokens == count_tokens(old_messages[-2:])


async def test_async_compaction_warns_even_with_no_sink_attached(caplog):
    """A caller who attached no sink still learns their conversation was rewritten."""
    from aimu.aio.agent import Agent

    old_messages = _old_conversation()
    client = MockAsyncModelClient(["final answer"])
    client.model.supports_tools = False
    client.messages = list(old_messages)

    budget = count_tokens(old_messages[-2:])
    agent = Agent(client, compaction=lambda msgs: trim_messages(msgs, max_tokens=budget, keep_last=0))

    with caplog.at_level(logging.WARNING, logger=_ASYNC_TOOL_LOOP_LOGGER):
        result = await agent.run("new question")  # events=None: no sink attached at all

    assert result == "final answer"
    assert "dropped 2 message" in caplog.text


async def test_async_compaction_that_drops_nothing_is_silent(caplog):
    """No spurious warning on a conversation already under the limit."""
    from aimu.aio.agent import Agent
    from aimu.events import ContextCompacted

    client = MockAsyncModelClient(["final answer"])
    client.model.supports_tools = False
    client.messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]

    seen = []
    agent = Agent(
        client,
        events=seen.append,
        compaction=lambda msgs: trim_messages(msgs, max_tokens=10_000),
    )
    with caplog.at_level(logging.WARNING, logger=_ASYNC_TOOL_LOOP_LOGGER):
        result = await agent.run("another question")

    assert result == "final answer"
    assert not any(isinstance(e, ContextCompacted) for e in seen)
    assert "Compacted conversation" not in caplog.text


async def test_async_explicit_trim_does_not_warn(caplog):
    """client.messages = trim_messages(...) needs no announcement: the caller performed
    the drop and holds both lists."""
    from aimu.aio.agent import Agent

    old_messages = _old_conversation()
    client = MockAsyncModelClient(["final answer"])
    client.model.supports_tools = False
    budget = count_tokens(old_messages[-2:])
    client.messages = trim_messages(old_messages, max_tokens=budget, keep_last=0)

    agent = Agent(client)  # compaction=None (default): the automatic hook never runs
    with caplog.at_level(logging.WARNING, logger=_ASYNC_TOOL_LOOP_LOGGER):
        result = await agent.run("new question")

    assert result == "final answer"
    assert "Compacted conversation" not in caplog.text
