"""The loop announces the two rounds where it injects a prompt of its own.

A caller watching a streamed run can see tool calls and text, but the two rounds AIMU drives itself
were invisible: a nudge after an empty turn and the forced wrap-up at the round cap. ``chunk.iteration``
said only that the round number moved, which is also what it says after an ordinary tool round, so a
consumer could neither tell the three apart nor quote what the model was told. These tests pin the
chunk that closes that gap, on both drivers, since the two are held to one definition of loop
semantics (see ``test_loop_iteration_parity.py``).
"""

import pytest

from aimu import PROVENANCE_CONTINUATION, PROVENANCE_FINAL_ANSWER
from aimu.agents import Agent as SyncAgent
from aimu.aio import Agent as AsyncAgent
from aimu.models import StreamingContentType
from aimu.tools import tool
from tests.helpers import MockModelClient
from tests.helpers_aio import MockAsyncModelClient


@tool
def _noop_tool(x: str) -> str:
    """A no-op tool, present only so the agent has a non-empty tool list."""
    return x


def _boundaries(chunks):
    """The (kind, prompt) pairs of every CONTINUING chunk, in order."""
    return [(c.content["kind"], c.content["prompt"]) for c in chunks if c.phase == StreamingContentType.CONTINUING]


def _sync_chunks(agent, task="go"):
    return list(agent.run(task, stream=True))


async def _async_chunks(agent, task="go"):
    stream = await agent.run(task, stream=True)
    return [chunk async for chunk in stream]


def test_sync_nudge_announces_itself_with_the_prompt_it_sent():
    client = MockModelClient(["", "recovered answer"])
    agent = SyncAgent(client, name="nudged", continuation_prompt="Keep going.")

    assert _boundaries(_sync_chunks(agent, "do something")) == [(PROVENANCE_CONTINUATION, "Keep going.")]


async def test_async_nudge_announces_itself_with_the_prompt_it_sent():
    client = MockAsyncModelClient(["", "recovered answer"])
    agent = AsyncAgent(client, name="nudged", continuation_prompt="Keep going.")

    assert _boundaries(await _async_chunks(agent, "do something")) == [(PROVENANCE_CONTINUATION, "Keep going.")]


def test_sync_forced_wrap_up_announces_itself_and_quotes_the_configured_prompt():
    """The prompt is read at the call site, so an agent carrying its own ``final_answer_prompt``
    reports that rather than the built-in default. Same responses and cap as
    ``test_loop_iteration_parity``'s wrap-up case: two tool rounds fill the cap, the wrap-up follows."""
    client = MockModelClient(["tool", "tool", "final answer"])
    agent = SyncAgent(client, max_iterations=2, final_answer_prompt="Stop and answer.")

    chunks = _sync_chunks(agent)
    assert _boundaries(chunks) == [(PROVENANCE_FINAL_ANSWER, "Stop and answer.")]

    # iteration is how a renderer decides which round to file the marker under, so it has to be the
    # injected round's own index and not the round that ran before it. A driver that incremented
    # after the yield instead of before would leave every other assertion here green.
    at = next(i for i, c in enumerate(chunks) if c.phase == StreamingContentType.CONTINUING)
    injected_round = chunks[at + 1 :]
    assert injected_round  # the round did produce chunks of its own
    assert {c.iteration for c in injected_round} == {chunks[at].iteration}


async def test_async_forced_wrap_up_announces_itself_and_quotes_the_configured_prompt():
    client = MockAsyncModelClient(["tool", "tool", "final answer"])
    agent = AsyncAgent(client, max_iterations=2, final_answer_prompt="Stop and answer.")

    assert _boundaries(await _async_chunks(agent)) == [(PROVENANCE_FINAL_ANSWER, "Stop and answer.")]


def test_the_default_wrap_up_prompt_is_what_an_unconfigured_agent_reports():
    """The text names the cap and forbids more tools, which is the whole point of surfacing it: a
    consumer showing the nudge's wording here would say the opposite of what the model was told."""
    from aimu.agents._tool_loop import DEFAULT_WRAP_UP_PROMPT

    client = MockModelClient(["tool", "tool", "final answer"])
    agent = SyncAgent(client, max_iterations=2)

    assert _boundaries(_sync_chunks(agent)) == [(PROVENANCE_FINAL_ANSWER, DEFAULT_WRAP_UP_PROMPT)]


@pytest.mark.parametrize("driver", ["sync", "async"])
async def test_an_ordinary_tool_round_announces_nothing(driver):
    """A tool round advances the iteration counter too, and injects nothing. It is the round this
    phase must stay silent for, or the marker means "the counter moved" all over again."""
    if driver == "sync":
        agent = SyncAgent(MockModelClient(["tool", "answer"]), tools=[_noop_tool])
        chunks = _sync_chunks(agent)
    else:
        agent = AsyncAgent(MockAsyncModelClient(["tool", "answer"]), tools=[_noop_tool])
        chunks = await _async_chunks(agent)

    assert _boundaries(chunks) == []
    assert any(c.phase == StreamingContentType.TOOL_CALLING for c in chunks)  # the round did happen


def test_the_boundary_chunk_precedes_the_injected_rounds_own_chunks():
    """Order is the contract a renderer relies on: the marker opens the injected round, so text that
    arrives after it belongs to that round and text before it does not."""
    client = MockModelClient(["", "recovered answer"])
    agent = SyncAgent(client, name="nudged", continuation_prompt="Keep going.")

    chunks = _sync_chunks(agent, "do something")
    phases = [c.phase for c in chunks]
    marker = phases.index(StreamingContentType.CONTINUING)
    generated = [i for i, c in enumerate(chunks) if c.phase == StreamingContentType.GENERATING and c.content]
    assert generated and min(generated) > marker
