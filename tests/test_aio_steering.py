"""Mid-run steering on the async surface: messages handed to a loop already running."""

from __future__ import annotations

import pytest

from aimu.aio import Agent
from aimu.models import StreamingContentType
from tests.helpers_aio import MockAsyncModelClient


class ListSteering:
    """A steering source backed by a plain list, with one cursor per reader."""

    def __init__(self, messages=None):
        self.messages = list(messages or [])

    def reader(self):
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
    steering = ListSteering(["use the other file"])

    await collect(await agent.run("start", stream=True, steering=steering))

    assert {"role": "user", "content": "use the other file"} in client.messages


@pytest.mark.asyncio
async def test_a_steered_round_opens_with_a_steering_chunk_carrying_the_text():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, steering=ListSteering(["use the other file"])))

    steering_chunks = [c for c in chunks if c.phase == StreamingContentType.STEERING]
    assert [c.content for c in steering_chunks] == [{"text": "use the other file"}]


@pytest.mark.asyncio
async def test_a_steering_message_is_not_tagged_as_a_loop_injection():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    await collect(await agent.run("start", stream=True, steering=ListSteering(["stop that"])))

    steered = [m for m in client.messages if m.get("content") == "stop that"]
    assert steered and all("provenance" not in m for m in steered)


@pytest.mark.asyncio
async def test_whitespace_only_steering_is_not_delivered():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    await collect(await agent.run("start", stream=True, steering=ListSteering(["   ", ""])))

    assert [m for m in client.messages if m["role"] == "user"] == [{"role": "user", "content": "start"}]


@pytest.mark.asyncio
async def test_a_raising_steering_source_does_not_end_the_run():
    class Exploding:
        def reader(self):
            def drain():
                raise RuntimeError("the host's mailbox is broken")

            return drain

    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, steering=Exploding()))

    assert any(c.phase == StreamingContentType.GENERATING and c.content == "done" for c in chunks)


@pytest.mark.asyncio
async def test_steering_replaces_the_continuation_nudge_after_an_empty_turn():
    client = MockAsyncModelClient(["", "done"])
    agent = Agent(client, tools=[a_tool])

    await collect(await agent.run("start", stream=True, steering=ListSteering(["try the cache"])))

    user_messages = [m["content"] for m in client.messages if m["role"] == "user"]
    assert user_messages == ["start", "try the cache"]


@pytest.mark.asyncio
async def test_an_empty_turn_still_gets_the_nudge_when_nothing_is_pending():
    client = MockAsyncModelClient(["", "done"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, steering=ListSteering()))

    assert any(c.phase == StreamingContentType.CONTINUING for c in chunks)
    assert not any(c.phase == StreamingContentType.STEERING for c in chunks)


@pytest.mark.asyncio
async def test_a_message_arriving_before_the_answer_completes_takes_one_more_round():
    client = MockAsyncModelClient(["first answer", "second answer"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, steering=ListSteering(["also check the log"])))

    generated = [c.content for c in chunks if c.phase == StreamingContentType.GENERATING]
    assert generated == ["first answer", "second answer"]


@pytest.mark.asyncio
async def test_a_healthy_turn_with_nothing_pending_still_ends_the_run():
    client = MockAsyncModelClient(["the answer"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, steering=ListSteering()))

    generated = [c.content for c in chunks if c.phase == StreamingContentType.GENERATING]
    assert generated == ["the answer"]


@pytest.mark.asyncio
async def test_several_messages_at_one_boundary_are_delivered_as_one_round():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    await collect(await agent.run("start", stream=True, steering=ListSteering(["first", "second"])))

    user_messages = [m["content"] for m in client.messages if m["role"] == "user"]
    assert user_messages == ["start", "first\n\nsecond"]


@pytest.mark.asyncio
async def test_a_message_delivered_at_the_cap_buys_a_fresh_budget():
    # max_iterations=2 means the bounded loop makes two real calls. Without a reset, the message
    # delivered on the second one would be followed straight by the forced wrap-up.
    client = MockAsyncModelClient(["tool", "tool", "tool", "done"])
    agent = Agent(client, tools=[a_tool], max_iterations=2)
    steering = ListSteering()

    stream = await agent.run("start", stream=True, steering=steering)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        # Hand the message over during the first tool round, so it lands on the second call.
        if chunk.phase == StreamingContentType.TOOL_CALLING and not steering.messages:
            steering.messages.append("keep going, use the index")

    assert any(c.phase == StreamingContentType.STEERING for c in chunks)
    # Four calls were made: two on the original budget, then the steered one starting a fresh
    # budget of two. A run without the reset stops after three.
    assert client._call_count == 4


@pytest.mark.asyncio
async def test_an_unsteered_run_still_stops_at_its_cap():
    # Only three real calls happen here (two bounded, one forced wrap-up), unlike the fresh-budget
    # test above which needs a fourth: the forced wrap-up asks for a plain answer, so its response
    # must not be "tool" (MockAsyncModelClient's "tool" reply ignores use_tools=False, the same
    # convention the sync MockModelClient uses, so a "tool" reply there would misreport the turn
    # as still pending).
    client = MockAsyncModelClient(["tool", "tool", "done"])
    agent = Agent(client, tools=[a_tool], max_iterations=2)

    await collect(await agent.run("start", stream=True, steering=ListSteering()))

    # Two bounded calls plus the one forced wrap-up, which is deliberately uncounted.
    assert client._call_count == 3


@pytest.mark.asyncio
async def test_the_non_streamed_driver_delivers_at_a_tool_round():
    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    result = await agent.run("start", steering=ListSteering(["use the other file"]))

    assert result == "done"
    assert {"role": "user", "content": "use the other file"} in client.messages


@pytest.mark.asyncio
async def test_the_non_streamed_driver_extends_a_finished_turn():
    client = MockAsyncModelClient(["first answer", "second answer"])
    agent = Agent(client, tools=[a_tool])

    result = await agent.run("start", steering=ListSteering(["also check the log"]))

    assert result == "second answer"


@pytest.mark.asyncio
async def test_the_non_streamed_driver_replaces_the_nudge():
    client = MockAsyncModelClient(["", "done"])
    agent = Agent(client, tools=[a_tool])

    await agent.run("start", steering=ListSteering(["try the cache"]))

    user_messages = [m["content"] for m in client.messages if m["role"] == "user"]
    assert user_messages == ["start", "try the cache"]


@pytest.mark.asyncio
async def test_a_structured_run_ignores_steering_rather_than_raising():
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    client = MockAsyncModelClient(['{"text": "done"}'])
    # Parse-path: the mock's _chat() takes no response_format, which the supports_structured_output=True
    # branch would add. See the same fix in test_aio_agents.py / test_aio_events.py's schema= tests.
    client.model.supports_structured_output = False
    agent = Agent(client, tools=[a_tool])

    result = await agent.run("start", schema=Answer, steering=ListSteering(["too late"]))

    assert result.text == "done"


@pytest.mark.asyncio
async def test_a_source_whose_reader_raises_does_not_end_the_run():
    # The other half of the guarantee the drain's guard makes. `reader()` is called once, at the
    # run's start, and a host that builds its cursor wrong raises there rather than in the drain.
    class ExplodingReader:
        def reader(self):
            raise RuntimeError("the host built its mailbox wrong")

    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, steering=ExplodingReader()))

    assert any(c.phase == StreamingContentType.GENERATING and c.content == "done" for c in chunks)


@pytest.mark.asyncio
async def test_a_run_whose_reader_raises_still_reports_that_it_finished():
    # `_open_steering` runs ahead of the try/finally that emits RunFinished, so a raise there left
    # a sink holding a RunStarted with nothing after it, breaking the one-finish-per-start contract.
    class ExplodingReader:
        def reader(self):
            raise RuntimeError("the host built its mailbox wrong")

    client = MockAsyncModelClient(["done"])
    seen = []
    agent = Agent(client, tools=[a_tool], events=seen.append)

    await collect(await agent.run("start", stream=True, steering=ExplodingReader()))

    assert [type(event).__name__ for event in seen].count("RunFinished") == 1


@pytest.mark.asyncio
async def test_a_drain_returning_a_bare_string_is_refused_rather_than_iterated():
    # A string is iterable, so an unchecked comprehension accepts "stop" and delivers it to the
    # model as four one-character paragraphs. Worse than a raise, because it looks like it worked.
    class StringDrain:
        def reader(self):
            return lambda: "stop"

    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    await collect(await agent.run("start", stream=True, steering=StringDrain()))

    assert [m["content"] for m in client.messages if m["role"] == "user"] == ["start"]


@pytest.mark.asyncio
async def test_a_drain_returning_a_non_list_does_not_end_the_run():
    class NoneDrain:
        def reader(self):
            return lambda: None

    client = MockAsyncModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    chunks = await collect(await agent.run("start", stream=True, steering=NoneDrain()))

    assert any(c.phase == StreamingContentType.GENERATING and c.content == "done" for c in chunks)
    assert [m["content"] for m in client.messages if m["role"] == "user"] == ["start"]


@pytest.mark.asyncio
async def test_a_drain_that_never_advances_cannot_extend_the_budget_forever():
    # A host bug one character wide: returning the whole list instead of the unread slice. Every
    # round then looks like a fresh human message, and `max_iterations` stops bounding the run.
    class NeverAdvancing:
        def reader(self):
            return lambda: ["again"]

    # Trailing "done" replies, not "tool": once the budget stops growing the loop reaches its
    # forced wrap-up, which asks for a plain answer and would report a "tool" reply as degenerate.
    client = MockAsyncModelClient(["tool"] * 4 + ["done"] * 50)
    agent = Agent(client, tools=[a_tool], max_iterations=2)

    await collect(await agent.run("start", stream=True, steering=NeverAdvancing()))

    # Resets are allowed up to `max_iterations`, so the worst case is that many fresh budgets plus
    # the original plus the uncounted wrap-up. The point is that a bound exists at all.
    assert client._call_count <= (2 + 1) * 2 + 1
