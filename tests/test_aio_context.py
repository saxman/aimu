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


async def test_async_compaction_that_rebuilds_without_dropping_is_silent(caplog):
    """Async mirror: a normalize/dedupe callable returning new dict objects for every kept
    message must not be mistaken for a drop by identity comparison."""
    from aimu.aio.agent import Agent
    from aimu.events import ContextCompacted

    old_messages = _old_conversation()
    client = MockAsyncModelClient(["final answer"])
    client.model.supports_tools = False
    client.messages = list(old_messages)

    seen = []
    agent = Agent(client, events=seen.append, compaction=lambda msgs: [dict(m) for m in msgs])
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


async def test_async_compaction_is_a_per_run_override():
    """Mirrors deps / tool_approval / thinking / events: None uses the field, otherwise the
    run()-level override wins."""
    from aimu.aio.agent import Agent
    from aimu.events import ContextCompacted

    old_messages = _old_conversation()
    budget = count_tokens(old_messages[-2:])

    def trimmer(msgs: list[dict]) -> list[dict]:
        return trim_messages(msgs, max_tokens=budget, keep_last=0)

    def no_op(msgs: list[dict]) -> list[dict]:
        return msgs

    # The field is a no-op (would never announce anything on its own); the per-run
    # override is the trimmer. Seeing ContextCompacted proves the override ran, not the field.
    seen = []
    client = MockAsyncModelClient(["final answer"])
    client.model.supports_tools = False
    client.messages = list(old_messages)
    agent = Agent(client, events=seen.append, compaction=no_op)
    await agent.run("q", compaction=trimmer)
    assert any(isinstance(e, ContextCompacted) for e in seen)

    # compaction=None (the default, i.e. omitted) falls back to the field.
    seen2 = []
    client2 = MockAsyncModelClient(["final answer"])
    client2.model.supports_tools = False
    client2.messages = list(old_messages)
    agent2 = Agent(client2, events=seen2.append, compaction=trimmer)
    await agent2.run("q")  # no override -> None -> falls back to self.compaction
    assert any(isinstance(e, ContextCompacted) for e in seen2)
