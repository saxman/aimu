"""Parity between the sync and async tool-loop drivers' ``max_iterations`` semantics.

``max_iterations`` is the maximum number of model calls **the loop itself** makes (see
``Agent.max_iterations``). The forced wrap-up call -- issued when the loop exhausts that cap
with a tool call still pending -- is one deliberate exception: it runs after the cap and is
never counted against it (both surfaces already implement that exemption; see
``test_run_finished_iteration_matches_the_forced_wrap_up_turn`` in ``test_events.py`` and its
async mirror in ``test_aio_events.py``).

Before the fix this file pins, the two async drivers (``aimu/aio/_tool_loop.py`` ``run`` /
``run_streamed``) let the bounded loop run one round further than ``max_iterations`` permits,
so an async agent made exactly one more real model call than a sync agent configured with the
same ``max_iterations``. Counting is done with an event sink tallying
:class:`~aimu.events.ModelTurnStarted` (the "one sink, one real request" seam from
:mod:`aimu.events`), not by peeking at internal loop state.
"""

from unittest.mock import MagicMock

import pytest

from aimu.aio import Agent as AsyncAgent
from aimu.aio._base import AsyncBaseModelClient
from aimu.agents import Agent as SyncAgent
from aimu.events import ModelTurnStarted, RunFinished
from aimu.models import BaseModelClient, StreamingContentType
from aimu.models.base import StreamChunk
from aimu.tools import tool


@tool
def _noop_tool(x: str) -> str:
    """A no-op tool, present only so the agent has a non-empty tool list."""
    return x


class _AlwaysToolClient(BaseModelClient):
    """Calls a tool on every turn while tools are advertised, and only produces a plain
    answer once tools are disabled (the forced wrap-up turn, tools=[]).

    Never finishes on its own: with this client, the loop always runs every round it is
    permitted, so the bounded loop makes exactly ``max_iterations`` real calls and the
    forced wrap-up always fires exactly one more -- the shape the brief's measured defect
    describes at every ``max_iterations`` value.
    """

    def __init__(self, final_text: str = "DONE"):
        self.model = MagicMock()
        self.model.supports_tools = True
        self.model.supports_thinking = False
        self.model_kwargs = None
        self._system_message = None
        self.default_generate_kwargs = {}
        self.messages = []
        self.tools = []
        self.last_thinking = ""
        self.concurrent_tool_calls = False
        self._streaming_content_type = StreamingContentType.DONE
        self.final_text = final_text
        self.events = None

    def _resolve_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}

    def _chat(self, user_message=None, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
        if stream:
            return self._chat_streamed(user_message, generate_kwargs, use_tools, images=images)
        if user_message is not None:  # None = continuation turn (no new user message)
            self.messages.append({"role": "user", "content": user_message})
        if self.tools:
            self.messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"type": "function", "function": {"name": "t", "arguments": {}}, "id": "x"}],
                }
            )
            self.messages.append({"role": "tool", "name": "t", "content": "result", "tool_call_id": "x"})
            return ""
        self.messages.append({"role": "assistant", "content": self.final_text})  # tools disabled -> wrap up
        return self.final_text

    def _chat_streamed(self, user_message=None, generate_kwargs=None, use_tools=True, images=None):
        text = self._chat(user_message, generate_kwargs, use_tools)
        self._streaming_content_type = StreamingContentType.GENERATING
        yield StreamChunk(StreamingContentType.GENERATING, text)
        self._streaming_content_type = StreamingContentType.DONE

    def _generate(self, prompt, generate_kwargs=None, stream=False, images=None, audio=None):
        return self._chat(prompt, generate_kwargs)


class _AsyncAlwaysToolClient(AsyncBaseModelClient):
    """Async twin of :class:`_AlwaysToolClient`."""

    def __init__(self, final_text: str = "DONE"):
        self.model = MagicMock()
        self.model.supports_tools = True
        self.model.supports_thinking = False
        self.model.supports_vision = False
        self.model.supports_audio = False
        self.model_kwargs = None
        self._system_message = None
        self.default_generate_kwargs = {}
        self.messages = []
        self.tools = []
        self.last_thinking = ""
        self.final_text = final_text
        self.events = None

    def _resolve_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}

    async def _chat(
        self, user_message=None, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None
    ):
        if stream:
            return self._chat_streamed(user_message, generate_kwargs, use_tools, images=images)
        if user_message is not None:
            self.messages.append({"role": "user", "content": user_message})
        if self.tools:
            self.messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"type": "function", "function": {"name": "t", "arguments": {}}, "id": "x"}],
                }
            )
            self.messages.append({"role": "tool", "name": "t", "content": "result", "tool_call_id": "x"})
            return ""
        self.messages.append({"role": "assistant", "content": self.final_text})
        return self.final_text

    async def _chat_streamed(self, user_message=None, generate_kwargs=None, use_tools=True, images=None):
        text = await self._chat(user_message, generate_kwargs, use_tools)
        yield StreamChunk(StreamingContentType.GENERATING, text)

    async def _generate(self, prompt, generate_kwargs=None, stream=False, images=None, audio=None):
        return await self._chat(prompt, generate_kwargs)


def _count(kind, seen):
    return sum(isinstance(e, kind) for e in seen)


# ---------------------------------------------------------------------------
# Step 1-4: exact call-count parity for `run()` and `run(stream=True)`, across
# max_iterations in (1, 2, 3, 4). The client above never finishes on its own, so every
# permitted round runs and the forced wrap-up always fires: total real calls = max_iterations
# (the bounded loop) + 1 (the uncounted wrap-up).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_iterations", [1, 2, 3, 4])
def test_sync_run_call_count_is_max_iterations_plus_forced_wrap_up(max_iterations):
    seen = []
    client = _AlwaysToolClient()
    agent = SyncAgent(client, tools=[_noop_tool], max_iterations=max_iterations, events=seen.append)
    result = agent.run("never-ending task")

    assert result == "DONE"
    assert _count(ModelTurnStarted, seen) == max_iterations + 1


@pytest.mark.parametrize("max_iterations", [1, 2, 3, 4])
async def test_async_run_call_count_is_max_iterations_plus_forced_wrap_up(max_iterations):
    seen = []
    client = _AsyncAlwaysToolClient()
    agent = AsyncAgent(client, tools=[_noop_tool], max_iterations=max_iterations, events=seen.append)
    result = await agent.run("never-ending task")

    assert result == "DONE"
    assert _count(ModelTurnStarted, seen) == max_iterations + 1


@pytest.mark.parametrize("max_iterations", [1, 2, 3, 4])
def test_sync_run_streamed_call_count_is_max_iterations_plus_forced_wrap_up(max_iterations):
    seen = []
    client = _AlwaysToolClient()
    agent = SyncAgent(client, tools=[_noop_tool], max_iterations=max_iterations, events=seen.append)
    list(agent.run("never-ending task", stream=True))

    assert _count(ModelTurnStarted, seen) == max_iterations + 1


@pytest.mark.parametrize("max_iterations", [1, 2, 3, 4])
async def test_async_run_streamed_call_count_is_max_iterations_plus_forced_wrap_up(max_iterations):
    seen = []
    client = _AsyncAlwaysToolClient()
    agent = AsyncAgent(client, tools=[_noop_tool], max_iterations=max_iterations, events=seen.append)
    stream = await agent.run("never-ending task", stream=True)
    async for _ in stream:
        pass

    assert _count(ModelTurnStarted, seen) == max_iterations + 1


@pytest.mark.parametrize("max_iterations", [1, 2, 3, 4])
async def test_sync_and_async_agree_on_call_count_for_the_same_max_iterations(max_iterations):
    sync_seen = []
    sync_agent = SyncAgent(
        _AlwaysToolClient(), tools=[_noop_tool], max_iterations=max_iterations, events=sync_seen.append
    )
    sync_agent.run("never-ending task")

    async_seen = []
    async_agent = AsyncAgent(
        _AsyncAlwaysToolClient(), tools=[_noop_tool], max_iterations=max_iterations, events=async_seen.append
    )
    await async_agent.run("never-ending task")

    assert _count(ModelTurnStarted, sync_seen) == _count(ModelTurnStarted, async_seen)


# ---------------------------------------------------------------------------
# Step 5: the forced-wrap-up path, checked separately with a scripted response sequence and
# `final_answer_prompt` set, mirroring test_events.py's
# `test_run_finished_iteration_matches_the_forced_wrap_up_turn` and pinning that both surfaces
# now share its exact shape (same responses, same max_iterations, same expected iteration).
# v0.21.0 fixed a stale RunFinished.iteration on this path for sync; this guards both surfaces.
# ---------------------------------------------------------------------------


def test_sync_forced_wrap_up_makes_exactly_n_plus_one_calls_with_matching_iteration():
    from tests.helpers import MockModelClient

    seen = []
    client = MockModelClient(["tool", "tool", "final answer"])
    agent = SyncAgent(client, max_iterations=2, final_answer_prompt="Stop and answer.", events=seen.append)
    result = agent.run("go")

    assert result == "final answer"
    from aimu.events import ModelTurnFinished

    turn_finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(turn_finished) == 3  # N=2 bounded-loop calls + 1 forced wrap-up
    finished = next(e for e in seen if isinstance(e, RunFinished))
    assert finished.iteration == turn_finished[-1].iteration == 2


async def test_async_forced_wrap_up_makes_exactly_n_plus_one_calls_with_matching_iteration():
    """Same responses, same max_iterations, same expected outcome as the sync test above --
    now that the two drivers share one convention, the async wrap-up path no longer needs
    different parameters to reach the branch."""
    from tests.helpers_aio import MockAsyncModelClient

    seen = []
    client = MockAsyncModelClient(["tool", "tool", "final answer"])
    agent = AsyncAgent(client, max_iterations=2, final_answer_prompt="Stop and answer.", events=seen.append)
    result = await agent.run("go")

    assert result == "final answer"
    from aimu.events import ModelTurnFinished

    turn_finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(turn_finished) == 3
    finished = next(e for e in seen if isinstance(e, RunFinished))
    assert finished.iteration == turn_finished[-1].iteration == 2


async def test_sync_and_async_announce_the_same_boundaries_for_the_same_run():
    """The drivers agree on the injected rounds, not just on how many calls they make. A seam added to
    one streamed driver and forgotten in the other is the drift this file exists to catch."""

    def kinds(chunks):
        return [c.content["kind"] for c in chunks if c.phase == StreamingContentType.CONTINUING]

    sync_agent = SyncAgent(_AlwaysToolClient(), tools=[_noop_tool], max_iterations=2)
    sync_kinds = kinds(list(sync_agent.run("never-ending task", stream=True)))

    async_agent = AsyncAgent(_AsyncAlwaysToolClient(), tools=[_noop_tool], max_iterations=2)
    stream = await async_agent.run("never-ending task", stream=True)
    async_kinds = kinds([chunk async for chunk in stream])

    # The wire value is spelled out rather than imported on purpose: an assertion against
    # PROVENANCE_FINAL_ANSWER would compare the constant with itself and sail through a silent
    # change to what it holds. This literal and the two in test_models_api.py are the only
    # places the suite pins these two strings.
    assert sync_kinds == async_kinds == ["final_answer"]


# ---------------------------------------------------------------------------
# Inbox: a delivered message moves the round budget's base, which is loop semantics and so
# belongs under this file's parity rule. The two surfaces compute the move differently in surface
# form (the sync drivers compare chat counts, the async ones compare iteration indices) and are
# equal only because `chats == iteration + 1`, so without this the arithmetic could drift between
# them with the rest of this suite green.
# ---------------------------------------------------------------------------


class _OneShotInbox:
    """Delivers one message at the first boundary a run drains, and nothing after it."""

    def reader(self, agent=None):
        sent = False

        def drain():
            nonlocal sent
            if sent:
                return []
            sent = True
            return ["use the index instead"]

        return drain


def _sync_messaged_calls(max_iterations, stream):
    seen = []
    agent = SyncAgent(_AlwaysToolClient(), tools=[_noop_tool], max_iterations=max_iterations, events=seen.append)
    if stream:
        list(agent.run("never-ending task", stream=True, inbox=_OneShotInbox()))
    else:
        agent.run("never-ending task", inbox=_OneShotInbox())
    return _count(ModelTurnStarted, seen)


async def _async_messaged_calls(max_iterations, stream):
    seen = []
    agent = AsyncAgent(_AsyncAlwaysToolClient(), tools=[_noop_tool], max_iterations=max_iterations, events=seen.append)
    if stream:
        stream_obj = await agent.run("never-ending task", stream=True, inbox=_OneShotInbox())
        async for _ in stream_obj:
            pass
    else:
        await agent.run("never-ending task", inbox=_OneShotInbox())
    return _count(ModelTurnStarted, seen)


@pytest.mark.parametrize("max_iterations", [1, 2, 3, 4])
async def test_all_four_drivers_agree_on_the_call_count_of_a_messaged_run(max_iterations):
    counts = {
        "sync run": _sync_messaged_calls(max_iterations, stream=False),
        "sync run_streamed": _sync_messaged_calls(max_iterations, stream=True),
        "async run": await _async_messaged_calls(max_iterations, stream=False),
        "async run_streamed": await _async_messaged_calls(max_iterations, stream=True),
    }

    assert len(set(counts.values())) == 1, counts


@pytest.mark.parametrize("max_iterations", [2, 3, 4])  # 1 offers no boundary to deliver at; see below
async def test_a_delivered_message_buys_a_fresh_budget_on_every_driver(max_iterations):
    # Guards the reset itself, not just agreement about it: four drivers that all ignored the
    # inbox would agree too. The client never finishes on its own, so one delivered message has to
    # buy strictly more real calls than the same run with an empty inbox.
    #
    # max_iterations=1 is excluded because the loop never reaches a round boundary there: the
    # opening call already exhausts the budget, so the while condition is false before the first
    # drain and the run goes straight to its forced wrap-up. Nothing to deliver, by construction.
    # The agreement test above still covers 1, since all four drivers agree on making no drain.
    unmessaged = max_iterations + 1  # the bounded loop plus the uncounted wrap-up
    counts = {
        "sync run": _sync_messaged_calls(max_iterations, stream=False),
        "sync run_streamed": _sync_messaged_calls(max_iterations, stream=True),
        "async run": await _async_messaged_calls(max_iterations, stream=False),
        "async run_streamed": await _async_messaged_calls(max_iterations, stream=True),
    }

    for driver, count in counts.items():
        assert count > unmessaged, f"{driver} made {count} calls, no more than the unmessaged {unmessaged}"


# ---------------------------------------------------------------------------
# Inbox, the label half: which agent a driver says is opening a reader. That label is what lets a
# host route a message to one run rather than broadcast to all of them, so it is loop semantics
# in exactly the sense this file's rule covers, and a driver that forgot to forward it would be
# invisible to every test above (they all drain, none of them look at the argument).
# ---------------------------------------------------------------------------


class _RecordingInbox:
    """Records the label each reader is opened under, and delivers nothing."""

    def __init__(self):
        self.asked: list[str | None] = []

    def reader(self, agent=None):
        self.asked.append(agent)
        return lambda: []


def _sync_opened_labels(stream):
    inbox = _RecordingInbox()
    agent = SyncAgent(_AlwaysToolClient(), tools=[_noop_tool], max_iterations=1, name="researcher")
    if stream:
        list(agent.run("never-ending task", stream=True, inbox=inbox))
    else:
        agent.run("never-ending task", inbox=inbox)
    return tuple(inbox.asked)


async def _async_opened_labels(stream):
    inbox = _RecordingInbox()
    agent = AsyncAgent(_AsyncAlwaysToolClient(), tools=[_noop_tool], max_iterations=1, name="researcher")
    if stream:
        stream_obj = await agent.run("never-ending task", stream=True, inbox=inbox)
        async for _ in stream_obj:
            pass
    else:
        await agent.run("never-ending task", inbox=inbox)
    return tuple(inbox.asked)


async def test_all_four_drivers_name_the_agent_identically():
    """The label a host addresses by is loop semantics, so the four drivers must agree on it.

    Each of the four is driven here, including sync ``run_streamed``, which nothing else in the
    suite checks the label of: ``tests/test_inbox.py`` covers sync ``run``, ``tests/test_aio_inbox.py``
    async ``run_streamed``, and ``tests/test_aio_subagent_tools.py`` async ``run`` by way of a spawn.
    """
    labels = {
        "sync run": _sync_opened_labels(stream=False),
        "sync run_streamed": _sync_opened_labels(stream=True),
        "async run": await _async_opened_labels(stream=False),
        "async run_streamed": await _async_opened_labels(stream=True),
    }

    # The expected value is asserted, not merely agreement: four drivers that each opened no reader
    # would agree on the empty tuple, and four that dropped the argument would agree on (None,).
    assert labels == dict.fromkeys(labels, ("researcher",)), labels
