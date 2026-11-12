"""Async mirror of the Agent(compaction=...) tests in tests/test_context.py.

See tests/test_context.py for the sync surface and the full rationale: a compaction decides what
each request sends, never what the conversation holds, and what it leaves out is announced when
that changes. These exercise the same behaviour on aimu.aio.Agent and the async tool-loop engine,
plus a streamed run, whose requests are built while the engine's generator is suspended.
"""

import logging

import pytest

from aimu.context import count_tokens, trim_messages
from tests.helpers_aio import MockAsyncModelClient

# _compacted_request is inherited from _BaseToolLoop, defined in aimu.agents._tool_loop, so its
# log lines are under that module's name even when the async loop is the one running.
_ASYNC_TOOL_LOOP_LOGGER = "aimu.agents._tool_loop"

NEW = {"role": "user", "content": "new question"}


def _old_conversation() -> list[dict]:
    return [
        {"role": "user", "content": "old question one"},
        {"role": "assistant", "content": "old answer one"},
        {"role": "user", "content": "old question two"},
        {"role": "assistant", "content": "old answer two"},
    ]


def _newest_only():
    budget = count_tokens([NEW])
    return lambda msgs: trim_messages(msgs, max_tokens=budget, keep_first=0, keep_last=1)


def _agent_over(old_messages, compaction, responses=("final answer",), **kwargs):
    from aimu.aio.agent import Agent

    client = MockAsyncModelClient(list(responses))
    client.model.supports_tools = False
    client.messages = list(old_messages)
    return client, Agent(client, compaction=compaction, **kwargs)


async def test_async_compaction_changes_what_is_sent_and_leaves_the_history_whole():
    old_messages = _old_conversation()
    client, agent = _agent_over(old_messages, _newest_only())

    assert await agent.run("new question") == "final answer"

    assert client.sent == [[NEW]]
    assert client.messages[:4] == old_messages
    assert client.messages[4] == NEW


async def test_async_compaction_emits_ContextCompacted_carrying_the_left_out_messages():
    from aimu.events import ContextCompacted

    old_messages = _old_conversation()
    seen = []
    client, agent = _agent_over(old_messages, _newest_only(), events=seen.append)
    await agent.run("new question")

    (event,) = [e for e in seen if isinstance(e, ContextCompacted)]
    assert event.dropped == old_messages
    assert event.restored == []
    assert event.dropped[0] is not client.messages[0]
    client.messages[0]["provenance"] = "mutated afterwards"
    assert "provenance" not in event.dropped[0]
    assert event.before_tokens == count_tokens(old_messages + [NEW])
    assert event.after_tokens == count_tokens([NEW])


async def test_async_compaction_warns_even_with_no_sink_attached(caplog):
    _, agent = _agent_over(_old_conversation(), _newest_only())
    with caplog.at_level(logging.WARNING, logger=_ASYNC_TOOL_LOOP_LOGGER):
        await agent.run("new question")
    assert "4 more message(s) left out" in caplog.text


async def test_async_leaving_out_what_the_previous_request_left_out_is_not_announced_again():
    from aimu.events import ContextCompacted

    old_messages = _old_conversation()
    seen = []
    _, agent = _agent_over(old_messages, lambda msgs: msgs[-2:], responses=["one", "two"], events=seen.append)
    await agent.run("first")
    await agent.run("second")

    first, second = [e for e in seen if isinstance(e, ContextCompacted)]
    assert first.dropped == old_messages[:3]
    assert second.dropped == [{"role": "assistant", "content": "old answer two"}, {"role": "user", "content": "first"}]


async def test_async_a_request_that_sends_again_what_was_left_out_reports_it_restored():
    from aimu.events import ContextCompacted

    old_messages = _old_conversation()
    seen = []
    client, agent = _agent_over(old_messages, _newest_only(), responses=["one", "two"], events=seen.append)
    await agent.run("new question")
    await agent.run("and now?", compaction=lambda msgs: msgs)

    restored = [e for e in seen if isinstance(e, ContextCompacted)][-1]
    assert restored.dropped == []
    assert restored.restored == old_messages
    assert client.sent[-1][:4] == old_messages


async def test_async_compaction_that_drops_nothing_is_silent(caplog):
    from aimu.events import ContextCompacted

    seen = []
    _, agent = _agent_over(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        lambda msgs: trim_messages(msgs, max_tokens=10_000),
        events=seen.append,
    )
    with caplog.at_level(logging.WARNING, logger=_ASYNC_TOOL_LOOP_LOGGER):
        assert await agent.run("another question") == "final answer"
    assert not any(isinstance(e, ContextCompacted) for e in seen)
    assert "Compacted the request" not in caplog.text


async def test_async_compaction_that_rebuilds_without_dropping_is_silent(caplog):
    from aimu.events import ContextCompacted

    seen = []
    _, agent = _agent_over(_old_conversation(), lambda msgs: [dict(m) for m in msgs], events=seen.append)
    with caplog.at_level(logging.WARNING, logger=_ASYNC_TOOL_LOOP_LOGGER):
        assert await agent.run("another question") == "final answer"
    assert not any(isinstance(e, ContextCompacted) for e in seen)
    assert "Compacted the request" not in caplog.text


async def test_async_explicit_trim_does_not_warn(caplog):
    from aimu.aio.agent import Agent

    old_messages = _old_conversation()
    client = MockAsyncModelClient(["final answer"])
    client.model.supports_tools = False
    client.messages = trim_messages(old_messages, max_tokens=count_tokens(old_messages[-2:]), keep_last=0)

    agent = Agent(client)
    with caplog.at_level(logging.WARNING, logger=_ASYNC_TOOL_LOOP_LOGGER):
        assert await agent.run("new question") == "final answer"
    assert "Compacted the request" not in caplog.text


async def test_async_compaction_is_a_per_run_override():
    client, agent = _agent_over(_old_conversation(), lambda msgs: msgs)
    await agent.run("new question", compaction=_newest_only())
    assert client.sent == [[NEW]]

    client2, agent2 = _agent_over(_old_conversation(), _newest_only())
    await agent2.run("new question")
    assert client2.sent == [[NEW]]


async def test_async_streamed_run_compacts_every_request_and_none_after_it():
    """A streamed run builds its requests while the engine's generator is suspended between yields;
    the compaction must reach every one of them, and none once the stream is drained."""
    from aimu.aio.agent import Agent
    from aimu.tools import tool

    @tool
    async def lookup() -> str:
        """Look something up."""
        return "found it"

    client = MockAsyncModelClient(["tool", "final answer", "plain"])
    client.messages = _old_conversation()
    agent = Agent(client, tools=[lookup], compaction=lambda msgs: msgs[-1:])
    stream = await agent.run("new question", stream=True)
    async for _ in stream:
        pass

    assert [len(request) for request in client.sent] == [1, 1]
    assert client.sent[1][0]["role"] == "tool"
    await client.chat("outside any run")
    assert client.sent[-1] == client.messages[:-1]


async def test_async_a_compaction_that_raises_stops_the_run():
    def broken(msgs):
        raise RuntimeError("compaction bug")

    _, agent = _agent_over(_old_conversation(), broken)
    with pytest.raises(RuntimeError, match="compaction bug"):
        await agent.run("new question")


async def test_async_summarize_keeps_the_task_verbatim_by_default():
    """Mirrors test_summarize_keeps_the_task_verbatim_by_default on the async twin."""
    from aimu.aio.context import summarize_messages

    class RecordingSummarizer:
        def __init__(self):
            self.prompts = []

        async def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "a short summary"

    system = {"role": "system", "content": "sys"}
    task = {"role": "user", "content": "Compare the three reports."}
    middle = [{"role": "assistant", "content": "reading report one"}, {"role": "user", "content": "go on"}]
    tail = [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    client = RecordingSummarizer()

    result = await summarize_messages(client, [system, task, *middle, *tail], keep_last=2)

    assert result[0] == system
    assert result[1]["role"] == "system" and "a short summary" in result[1]["content"]
    assert result[2:] == [task, *tail]
    background, summarized = client.prompts[0].split("Summarize this part:")
    assert "do not summarize" in background and "Compare the three reports." in background
    assert "Compare the three reports." not in summarized
