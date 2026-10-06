"""An agent's inbox on the async surface: messages handed to a loop already running."""

from __future__ import annotations

import inspect

import pytest

from aimu.aio import Agent
from aimu.models import StreamingContentType
from tests.helpers_aio import MockAsyncModelClient


class ListInbox:
    """An inbox backed by a plain list, with one cursor per reader."""

    def __init__(self, messages=None):
        self.messages = list(messages or [])

    def reader(self, agent=None):
        seen = 0

        def drain():
            nonlocal seen
            pending = self.messages[seen:]
            seen = len(self.messages)
            return list(pending)

        return drain


def a_tool() -> str:
    """A tool the mock client can be told to call."""
    return "tool result"


async def collect(stream):
    return [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_a_message_pending_at_a_tool_round_becomes_that_rounds_user_message():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])
    inbox = ListInbox(["use the other file"])

    await collect(await agent.run("start", stream=True, inbox=inbox))

    assert {"role": "user", "content": "use the other file"} in client.messages


@pytest.mark.asyncio
async def test_a_round_opened_by_a_message_carries_a_message_chunk_with_the_text():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, inbox=ListInbox(["use the other file"])))

    message_chunks = [c for c in chunks if c.phase == StreamingContentType.INBOX]
    assert [c.content for c in message_chunks] == [{"text": "use the other file"}]


@pytest.mark.asyncio
async def test_an_inbox_message_is_not_tagged_as_a_loop_injection():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    await collect(await agent.run("start", stream=True, inbox=ListInbox(["stop that"])))

    delivered = [m for m in client.messages if m.get("content") == "stop that"]
    assert delivered and all("provenance" not in m for m in delivered)


@pytest.mark.asyncio
async def test_whitespace_only_messages_are_not_delivered():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    await collect(await agent.run("start", stream=True, inbox=ListInbox(["   ", ""])))

    assert [m for m in client.messages if m["role"] == "user"] == [{"role": "user", "content": "start"}]


@pytest.mark.asyncio
async def test_a_raising_inbox_source_does_not_end_the_run():
    class Exploding:
        def reader(self, agent=None):
            def drain():
                raise RuntimeError("the host's mailbox is broken")

            return drain

    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, inbox=Exploding()))

    assert any(c.phase == StreamingContentType.GENERATING and c.content == "done" for c in chunks)


@pytest.mark.asyncio
async def test_a_message_replaces_the_continuation_nudge_after_an_empty_turn():
    client = MockAsyncModelClient(["", "done"])
    agent = Agent(client, tools=[a_tool])

    await collect(await agent.run("start", stream=True, inbox=ListInbox(["try the cache"])))

    user_messages = [m["content"] for m in client.messages if m["role"] == "user"]
    assert user_messages == ["start", "try the cache"]


@pytest.mark.asyncio
async def test_an_empty_turn_still_gets_the_nudge_when_nothing_is_pending():
    client = MockAsyncModelClient(["", "done"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, inbox=ListInbox()))

    assert any(c.phase == StreamingContentType.CONTINUING for c in chunks)
    assert not any(c.phase == StreamingContentType.INBOX for c in chunks)


@pytest.mark.asyncio
async def test_a_message_arriving_before_the_answer_completes_takes_one_more_round():
    client = MockAsyncModelClient(["first answer", "second answer"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, inbox=ListInbox(["also check the log"])))

    generated = [c.content for c in chunks if c.phase == StreamingContentType.GENERATING]
    assert generated == ["first answer", "second answer"]


@pytest.mark.asyncio
async def test_a_healthy_turn_with_nothing_pending_still_ends_the_run():
    client = MockAsyncModelClient(["the answer"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, inbox=ListInbox()))

    generated = [c.content for c in chunks if c.phase == StreamingContentType.GENERATING]
    assert generated == ["the answer"]


@pytest.mark.asyncio
async def test_several_messages_at_one_boundary_are_delivered_as_one_round():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    await collect(await agent.run("start", stream=True, inbox=ListInbox(["first", "second"])))

    user_messages = [m["content"] for m in client.messages if m["role"] == "user"]
    assert user_messages == ["start", "first\n\nsecond"]


@pytest.mark.asyncio
async def test_a_message_delivered_at_the_cap_buys_a_fresh_budget():
    # max_iterations=2 means the bounded loop makes two real calls. Without a reset, the message
    # delivered on the second one would be followed straight by the forced wrap-up.
    client = MockAsyncModelClient(["tool", "tool", "tool", "done"])
    agent = Agent(client, tools=[a_tool], max_iterations=2)
    inbox = ListInbox()

    stream = await agent.run("start", stream=True, inbox=inbox)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        # Hand the message over during the first tool round, so it lands on the second call.
        if chunk.phase == StreamingContentType.TOOL_CALLING and not inbox.messages:
            inbox.messages.append("keep going, use the index")

    assert any(c.phase == StreamingContentType.INBOX for c in chunks)
    # Four calls were made: two on the original budget, then the message-extended one starting a
    # fresh budget of two. A run without the reset stops after three.
    assert client._call_count == 4


@pytest.mark.asyncio
async def test_a_run_with_an_empty_inbox_still_stops_at_its_cap():
    # Only three real calls happen here (two bounded, one forced wrap-up), unlike the fresh-budget
    # test above which needs a fourth: the forced wrap-up asks for a plain answer, so its response
    # must not be "tool" (MockAsyncModelClient's "tool" reply ignores use_tools=False, the same
    # convention the sync MockModelClient uses, so a "tool" reply there would misreport the turn
    # as still pending).
    client = MockAsyncModelClient(["tool", "tool", "done"])
    agent = Agent(client, tools=[a_tool], max_iterations=2)

    await collect(await agent.run("start", stream=True, inbox=ListInbox()))

    # Two bounded calls plus the one forced wrap-up, which is deliberately uncounted.
    assert client._call_count == 3


@pytest.mark.asyncio
async def test_the_non_streamed_driver_delivers_at_a_tool_round():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    result = await agent.run("start", inbox=ListInbox(["use the other file"]))

    assert result == "done"
    assert {"role": "user", "content": "use the other file"} in client.messages


@pytest.mark.asyncio
async def test_the_non_streamed_driver_extends_a_finished_turn():
    client = MockAsyncModelClient(["first answer", "second answer"])
    agent = Agent(client, tools=[a_tool])

    result = await agent.run("start", inbox=ListInbox(["also check the log"]))

    assert result == "second answer"


@pytest.mark.asyncio
async def test_the_non_streamed_driver_replaces_the_nudge():
    client = MockAsyncModelClient(["", "done"])
    agent = Agent(client, tools=[a_tool])

    await agent.run("start", inbox=ListInbox(["try the cache"]))

    user_messages = [m["content"] for m in client.messages if m["role"] == "user"]
    assert user_messages == ["start", "try the cache"]


@pytest.mark.asyncio
async def test_a_structured_run_ignores_an_inbox_message_rather_than_raising():
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    client = MockAsyncModelClient(['{"text": "done"}'])
    # Parse-path: the mock's _chat() takes no response_format, which the supports_structured_output=True
    # branch would add. See the same fix in test_aio_agents.py / test_aio_events.py's schema= tests.
    client.model.supports_structured_output = False
    agent = Agent(client, tools=[a_tool])

    result = await agent.run("start", schema=Answer, inbox=ListInbox(["too late"]))

    assert result.text == "done"


@pytest.mark.asyncio
async def test_a_source_whose_reader_raises_does_not_end_the_run(caplog):
    # The other half of the guarantee the drain's guard makes. `reader()` is called once, at the
    # run's start, and a host that builds its cursor wrong raises there rather than in the drain.
    class ExplodingReader:
        def reader(self, agent=None):
            raise RuntimeError("the host built its mailbox wrong")

    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    with caplog.at_level("WARNING"):
        chunks = await collect(await agent.run("start", stream=True, inbox=ExplodingReader()))

    assert any(c.phase == StreamingContentType.GENERATING and c.content == "done" for c in chunks)
    # The reader's own exception, not a TypeError from the constructor's arity rehearsal: the
    # guarantee this test is named for is about a *conforming* reader whose body raises, and a
    # double the loop refuses outright would pass it without ever running that body.
    assert "the host built its mailbox wrong" in caplog.text


@pytest.mark.asyncio
async def test_a_run_whose_reader_raises_still_reports_that_it_finished():
    # `_open_inbox` runs ahead of the try/finally that emits RunFinished, so a raise there left
    # a sink holding a RunStarted with nothing after it, breaking the one-finish-per-start contract.
    class ExplodingReader:
        def reader(self, agent=None):
            raise RuntimeError("the host built its mailbox wrong")

    client = MockAsyncModelClient(["done"])
    seen = []
    agent = Agent(client, tools=[a_tool], events=seen.append)

    await collect(await agent.run("start", stream=True, inbox=ExplodingReader()))

    assert [type(event).__name__ for event in seen].count("RunFinished") == 1


@pytest.mark.asyncio
async def test_a_drain_returning_a_bare_string_is_refused_rather_than_iterated():
    # A string is iterable, so an unchecked comprehension accepts "stop" and delivers it to the
    # model as four one-character paragraphs. Worse than a raise, because it looks like it worked.
    class StringDrain:
        def reader(self, agent=None):
            return lambda: "stop"

    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    await collect(await agent.run("start", stream=True, inbox=StringDrain()))

    assert [m["content"] for m in client.messages if m["role"] == "user"] == ["start"]


@pytest.mark.asyncio
async def test_a_drain_returning_a_non_list_does_not_end_the_run():
    class NoneDrain:
        def reader(self, agent=None):
            return lambda: None

    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, inbox=NoneDrain()))

    assert any(c.phase == StreamingContentType.GENERATING and c.content == "done" for c in chunks)
    assert [m["content"] for m in client.messages if m["role"] == "user"] == ["start"]


@pytest.mark.asyncio
async def test_a_drain_that_never_advances_cannot_extend_the_budget_forever():
    # A host bug one character wide: returning the whole list instead of the unread slice. Every
    # round then looks like a fresh human message, and `max_iterations` stops bounding the run.
    class NeverAdvancing:
        def reader(self, agent=None):
            return lambda: ["again"]

    # Trailing "done" replies, not "tool": once the budget stops growing the loop reaches its
    # forced wrap-up, which asks for a plain answer and would report a "tool" reply as degenerate.
    client = MockAsyncModelClient(["tool"] * 4 + ["done"] * 50)
    agent = Agent(client, tools=[a_tool], max_iterations=2)

    await collect(await agent.run("start", stream=True, inbox=NeverAdvancing()))

    # Resets are allowed up to `max_iterations`, so the worst case is that many fresh budgets plus
    # the original plus the uncounted wrap-up. The point is that a bound exists at all.
    assert client._call_count <= (2 + 1) * 2 + 1


class RecordingInbox:
    """An inbox that records which agent opened each reader."""

    def __init__(self):
        self.asked: list[str | None] = []

    def reader(self, agent=None):
        self.asked.append(agent)
        return lambda: []


@pytest.mark.asyncio
async def test_the_loop_tells_the_inbox_which_agent_is_opening_a_reader():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool], name="researcher")
    inbox = RecordingInbox()

    await collect(await agent.run("start", stream=True, inbox=inbox))

    assert inbox.asked == ["researcher"]


@pytest.mark.asyncio
async def test_a_loop_built_with_no_agent_name_passes_none_through():
    # Review focus 1, corrected: Agent always names its run (Agent.__post_init__ generates
    # "agent-xxxxxx" when the caller passes none), so agent_name=None is unreachable through
    # Agent and this has to construct the loop directly, as test_aio_agents.py does elsewhere.
    # Still worth pinning: a host implementing Inbox must not be handed a placeholder like ""
    # or "agent" for a nameless run, since that could collide with a real agent's label, and
    # this is the only level where "no name at all" actually occurs. It discriminates only
    # against such a placeholder, though: a recorded None cannot tell "the loop passed None"
    # from "the loop passed nothing and this double's own default supplied it", so the proof
    # that the argument is passed at all is the sibling above,
    # test_the_loop_tells_the_inbox_which_agent_is_opening_a_reader, which records a real label.
    from aimu.aio._tool_loop import _AsyncToolLoop

    client = MockAsyncModelClient(["done"])
    inbox = RecordingInbox()
    loop = _AsyncToolLoop(client, [a_tool], inbox=inbox)

    await loop.run("start")

    assert inbox.asked == [None]


@pytest.mark.asyncio
async def test_two_runs_of_one_agent_each_open_their_own_reader_with_the_same_label():
    # Review focus 3. A host disambiguates concurrent runs of one agent type by counting opens,
    # not by the label, so the label repeating is correct and the count must be per run.
    client = MockAsyncModelClient(["done", "done"])
    agent = Agent(client, tools=[a_tool], name="researcher")
    inbox = RecordingInbox()

    await collect(await agent.run("first", stream=True, inbox=inbox))
    await collect(await agent.run("second", stream=True, inbox=inbox))

    assert inbox.asked == ["researcher", "researcher"]


@pytest.mark.asyncio
async def test_a_structured_run_opens_no_reader_and_does_not_raise():
    # The `schema=` path returns before the tool loop is built (see `Agent.run`), so there is no
    # loop to open a reader and the argument is inert rather than refused. Pinned because "inert"
    # and "raises" are both defensible designs and only one of them is what ships: a host may hand
    # the same inbox to every run it makes, structured ones included.
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    client = MockAsyncModelClient(['{"text": "done"}'])
    # Parse-path, as in test_a_structured_run_ignores_an_inbox_message_rather_than_raising above:
    # the mock's _chat() takes no response_format, which the supports_structured_output=True branch
    # would add.
    client.model.supports_structured_output = False
    agent = Agent(client, tools=[a_tool], name="researcher")
    inbox = RecordingInbox()

    result = await agent.run("start", schema=Answer, inbox=inbox)

    assert result.text == "done"
    assert inbox.asked == []


@pytest.mark.asyncio
async def test_a_reader_that_takes_no_argument_is_refused_before_the_run_starts():
    # The migration trap 0.34.0 introduces, and the check that makes it loud. The `agent=None`
    # default makes the signature in `Inbox` optional *to read*, not optional to accept: the loop
    # passes the label positionally, so a 0.33.0-era `def reader(self)` cannot be called at all.
    # `isinstance` cannot catch that first (a `runtime_checkable` Protocol checks only that the
    # method exists, which is also what `_check_inbox` is limited to), so the loop constructor
    # rehearses the call it is about to make and raises there. It raises from the constructor and
    # not from `_open_inbox` for the reason `_open_inbox`'s docstring gives: raising there would
    # leave a sink holding a `RunStarted` with nothing after it, which is what the empty `seen`
    # below pins.
    from aimu.agents.inbox import Inbox

    class LegacyReader:
        def reader(self):
            return lambda: ["use the other file"]

    inbox = LegacyReader()
    assert isinstance(inbox, Inbox), "the protocol is structural, so this is the gap being pinned"

    client = MockAsyncModelClient(["tool", "done"])
    seen = []
    agent = Agent(client, tools=[a_tool], name="researcher", events=seen.append)

    with pytest.raises(TypeError, match=r"def reader\(self, agent=None\)"):
        await agent.run("start", stream=True, inbox=inbox)

    assert seen == []
    assert [m["content"] for m in client.messages if m["role"] == "user"] == []


@pytest.mark.asyncio
async def test_a_reader_taking_only_keywords_is_refused_too():
    # The check rehearses the real call, `reader(label)`, rather than asking the weaker question
    # "does reader accept an argument". `**kwargs` answers yes to the weaker question and still
    # cannot take a positional, so it has to be refused: the loop's call would fail on it.
    class KeywordOnlyReader:
        def reader(self, **kwargs):
            return lambda: ["use the other file"]

    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool], name="researcher")

    with pytest.raises(TypeError, match="positionally"):
        await agent.run("start", stream=True, inbox=KeywordOnlyReader())


@pytest.mark.asyncio
async def test_a_reader_with_no_default_for_the_label_is_accepted():
    # The other half of rehearsing the real call: a rehearsal cannot refuse an implementation that
    # would have worked. `Inbox` spells the parameter `agent=None`, but the loop always has a label
    # to pass (`Agent` generates one), so a reader that requires the argument works fine.
    class RequiresTheLabel:
        def __init__(self):
            self.asked = []

        def reader(self, agent):
            self.asked.append(agent)
            return lambda: []

    inbox = RequiresTheLabel()
    client = MockAsyncModelClient(["done"])
    agent = Agent(client, tools=[a_tool], name="researcher")

    await collect(await agent.run("start", stream=True, inbox=inbox))

    assert inbox.asked == ["researcher"]


@pytest.mark.asyncio
async def test_an_un_introspectable_reader_is_accepted_unchecked(caplog):
    # The fail-open limit of the rehearsal, stated in the 0.34.0 changelog and pinned here. A
    # `functools.partial` over a *C-implemented* function has no readable signature, so
    # `inspect.signature` raises `ValueError` and there is nothing to rehearse; the inbox is
    # accepted and the call fails where it always did, inside `_open_inbox`'s guard, which logs and
    # lets the run finish. (A partial over a plain Python function is perfectly introspectable and
    # would be checked like any other reader, which is why this double reaches for a builtin.)
    import functools
    import math

    class UnreadableReader:
        reader = functools.partial(math.log)

    with pytest.raises(ValueError):
        inspect.signature(UnreadableReader().reader)

    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool], name="researcher")

    with caplog.at_level("WARNING"):
        chunks = await collect(await agent.run("start", stream=True, inbox=UnreadableReader()))

    assert any(c.phase == StreamingContentType.GENERATING and c.content == "done" for c in chunks)
    assert any("could not open a reader" in record.message for record in caplog.records)
