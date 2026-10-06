"""An agent's inbox on the sync surface. Mirrors tests/test_aio_inbox.py."""

from __future__ import annotations

from aimu.agents import Agent
from aimu.models import StreamingContentType
from tests.helpers import MockModelClient


class ListInbox:
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


def test_the_sync_driver_delivers_at_a_tool_round():
    client = MockModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    assert agent.run("start", inbox=ListInbox(["use the other file"])) == "done"
    assert {"role": "user", "content": "use the other file"} in client.messages


def test_the_sync_driver_extends_a_finished_turn():
    client = MockModelClient(["first answer", "second answer"])
    agent = Agent(client, tools=[a_tool])

    assert agent.run("start", inbox=ListInbox(["also check the log"])) == "second answer"


def test_the_sync_streamed_driver_emits_a_message_chunk():
    client = MockModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    chunks = list(agent.run("start", stream=True, inbox=ListInbox(["stop that"])))

    message_chunks = [c for c in chunks if c.phase == StreamingContentType.INBOX]
    assert [c.content for c in message_chunks] == [{"text": "stop that"}]


def test_the_sync_driver_replaces_the_nudge():
    client = MockModelClient(["", "done"])
    agent = Agent(client, tools=[a_tool])

    agent.run("start", inbox=ListInbox(["try the cache"]))

    assert [m["content"] for m in client.messages if m["role"] == "user"] == ["start", "try the cache"]


def test_a_source_whose_reader_raises_does_not_end_the_run_on_the_sync_surface():
    class ExplodingReader:
        def reader(self):
            raise RuntimeError("the host built its mailbox wrong")

    client = MockModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    assert agent.run("start", inbox=ExplodingReader()) == "done"


def test_a_sync_run_whose_reader_raises_still_reports_that_it_finished():
    class ExplodingReader:
        def reader(self):
            raise RuntimeError("the host built its mailbox wrong")

    client = MockModelClient(["done"])
    seen = []
    agent = Agent(client, tools=[a_tool], events=seen.append)

    agent.run("start", inbox=ExplodingReader())

    assert [type(event).__name__ for event in seen].count("RunFinished") == 1


def test_the_sync_surface_refuses_a_bare_string_drain():
    class StringDrain:
        def reader(self, agent=None):
            return lambda: "stop"

    client = MockModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool])

    agent.run("start", inbox=StringDrain())

    assert [m["content"] for m in client.messages if m["role"] == "user"] == ["start"]


def test_the_sync_surface_bounds_a_drain_that_never_advances():
    class NeverAdvancing:
        def reader(self, agent=None):
            return lambda: ["again"]

    client = MockModelClient(["tool"] * 4 + ["done"] * 50)
    agent = Agent(client, tools=[a_tool], max_iterations=2)

    agent.run("start", inbox=NeverAdvancing())

    assert client._call_count <= (2 + 1) * 2 + 1


class RecordingInbox:
    """An inbox that records which agent opened each reader. Mirrors tests/test_aio_inbox.py."""

    def __init__(self):
        self.asked: list[str | None] = []

    def reader(self, agent=None):
        self.asked.append(agent)
        return lambda: []


def test_the_sync_loop_tells_the_inbox_which_agent_is_opening_a_reader():
    client = MockModelClient(["tool", "done"])
    agent = Agent(client, tools=[a_tool], name="researcher")
    inbox = RecordingInbox()

    agent.run("start", inbox=inbox)

    assert inbox.asked == ["researcher"]
