"""Mid-run steering on the sync surface. Mirrors tests/test_aio_steering.py."""

from __future__ import annotations

from aimu.agents import Agent
from aimu.models import StreamingContentType
from tests.helpers import MockModelClient


class ListSteering:
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


def test_the_sync_driver_delivers_at_a_tool_round():
    client = MockModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    assert agent.run("start", steering=ListSteering(["use the other file"])) == "done"
    assert {"role": "user", "content": "use the other file"} in client.messages


def test_the_sync_driver_extends_a_finished_turn():
    client = MockModelClient(["first answer", "second answer"])
    agent = Agent(client, tools=[a_tool])

    assert agent.run("start", steering=ListSteering(["also check the log"])) == "second answer"


def test_the_sync_streamed_driver_emits_a_steering_chunk():
    client = MockModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    chunks = list(agent.run("start", stream=True, steering=ListSteering(["stop that"])))

    steering_chunks = [c for c in chunks if c.phase == StreamingContentType.STEERING]
    assert [c.content for c in steering_chunks] == [{"text": "stop that"}]


def test_the_sync_driver_replaces_the_nudge():
    client = MockModelClient(["", "done"])
    agent = Agent(client, tools=[a_tool])

    agent.run("start", steering=ListSteering(["try the cache"]))

    assert [m["content"] for m in client.messages if m["role"] == "user"] == ["start", "try the cache"]


def test_a_source_whose_reader_raises_does_not_end_the_run_on_the_sync_surface():
    class ExplodingReader:
        def reader(self):
            raise RuntimeError("the host built its mailbox wrong")

    client = MockModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    assert agent.run("start", steering=ExplodingReader()) == "done"


def test_a_sync_run_whose_reader_raises_still_reports_that_it_finished():
    class ExplodingReader:
        def reader(self):
            raise RuntimeError("the host built its mailbox wrong")

    client = MockModelClient(["done"])
    seen = []
    agent = Agent(client, tools=[a_tool], events=seen.append)

    agent.run("start", steering=ExplodingReader())

    assert [type(event).__name__ for event in seen].count("RunFinished") == 1


def test_the_sync_surface_refuses_a_bare_string_drain():
    class StringDrain:
        def reader(self):
            return lambda: "stop"

    client = MockModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    agent.run("start", steering=StringDrain())

    assert [m["content"] for m in client.messages if m["role"] == "user"] == ["start"]


def test_the_sync_surface_bounds_a_drain_that_never_advances():
    class NeverAdvancing:
        def reader(self):
            return lambda: ["again"]

    client = MockModelClient(["tool"] * 4 + ["done"] * 50)
    agent = Agent(client, tools=[a_tool], max_iterations=2)

    agent.run("start", steering=NeverAdvancing())

    assert client._call_count <= (2 + 1) * 2 + 1
