"""The forced wrap-up must not strand an un-dispatched tool call.

``chat()`` is single-turn: it records the assistant turn's ``tool_calls`` and leaves execution
to the loop. So when the loop exhausts ``max_iterations`` on a turn that requested tools, the
transcript ends with an assistant message whose tool calls have no results. The wrap-up prompt
is a *user* message, and appending it there produces a transcript that every tool-calling
provider rejects: Anthropic answers 400 ``tool_use ids were found without tool_result blocks
immediately after``, and OpenAI requires an assistant ``tool_calls`` message be followed by
tool messages. The loop therefore settles the pending calls with synthesized results first.

The invariant is asserted on the transcript rather than by mocking a provider error, because
it is the transcript that is wrong: it is what gets persisted, exported, and resumed, so a
provider-level assertion would pin one provider's phrasing instead of the defect. The
Anthropic converter is exercised too, since it is the surface that produced the reported
failure and it is where a valid transcript has to survive translation.

A search-heavy sub-agent is what hits this in practice: it is the shape of run that spends
every round calling tools, so it is the one still holding a pending call when the cap lands.
"""

from unittest.mock import MagicMock

import pytest

from aimu.agents import Agent as SyncAgent
from aimu.aio import Agent as AsyncAgent
from aimu.aio._base import AsyncBaseModelClient
from aimu.models import BaseModelClient, StreamingContentType
from aimu.models.base import StreamChunk
from aimu.models.providers.anthropic import AnthropicClient
from aimu.tools import tool


@tool
def _search(query: str) -> str:
    """Stand-in for a search tool: present so the agent has something to call."""
    return f"results for {query}"


# Two calls per turn, so the assertion covers the parallel case the reported failure had
# (both ids named in one error) and not just a single stranded call.
def _tool_calls(turn: int) -> list[dict]:
    return [
        {"type": "function", "function": {"name": "_search", "arguments": {"query": f"q{turn}a"}}, "id": f"id{turn}a"},
        {"type": "function", "function": {"name": "_search", "arguments": {"query": f"q{turn}b"}}, "id": f"id{turn}b"},
    ]


class _PendingToolClient(BaseModelClient):
    """Records tool calls without results, as the real clients do, and never answers on its own.

    ``_AlwaysToolClient`` in ``test_loop_iteration_parity.py`` appends the tool results itself,
    so the loop never sees a pending turn there. This one leaves them pending, which is what
    makes the cap land on an unsettled transcript.
    """

    def __init__(self, final_text: str = "DONE", tool_turns: int = 99):
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
        self.turn = 0
        # After this many tool-calling turns the client answers instead, so a run that finishes
        # inside its cap is expressible; the default is high enough that it never does.
        self.tool_turns = tool_turns

    def _resolve_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}

    def _chat(self, user_message=None, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
        if stream:
            return self._chat_streamed(user_message, generate_kwargs, use_tools, images=images)
        if user_message is not None:  # None = continuation turn (no new user message)
            self.messages.append({"role": "user", "content": user_message})
        if self.tools and self.turn < self.tool_turns:
            self.turn += 1
            self.messages.append(
                {"role": "assistant", "content": f"searching, round {self.turn}", "tool_calls": _tool_calls(self.turn)}
            )
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


class _AsyncPendingToolClient(AsyncBaseModelClient):
    """Async twin of :class:`_PendingToolClient`."""

    def __init__(self, final_text: str = "DONE", tool_turns: int = 99):
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
        self.turn = 0
        # After this many tool-calling turns the client answers instead, so a run that finishes
        # inside its cap is expressible; the default is high enough that it never does.
        self.tool_turns = tool_turns

    def _resolve_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}

    async def _chat(
        self, user_message=None, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None
    ):
        if stream:
            return self._chat_streamed(user_message, generate_kwargs, use_tools, images=images)
        if user_message is not None:
            self.messages.append({"role": "user", "content": user_message})
        if self.tools and self.turn < self.tool_turns:
            self.turn += 1
            self.messages.append(
                {"role": "assistant", "content": f"searching, round {self.turn}", "tool_calls": _tool_calls(self.turn)}
            )
            return ""
        self.messages.append({"role": "assistant", "content": self.final_text})
        return self.final_text

    async def _chat_streamed(self, user_message=None, generate_kwargs=None, use_tools=True, images=None):
        text = await self._chat(user_message, generate_kwargs, use_tools)
        yield StreamChunk(StreamingContentType.GENERATING, text)

    async def _generate(self, prompt, generate_kwargs=None, stream=False, images=None, audio=None):
        return await self._chat(prompt, generate_kwargs)


class _EmptyAfterToolsClient(_AsyncPendingToolClient):
    """Calls tools on the first turn, then returns degenerate empty turns forever.

    The shape that sends the loop down the empty-turn wrap-up path with an *answered* tool turn
    still behind it in the transcript.
    """

    async def _chat(
        self, user_message=None, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None
    ):
        if stream:
            return self._chat_streamed(user_message, generate_kwargs, use_tools, images=images)
        if user_message is not None:
            self.messages.append({"role": "user", "content": user_message})
        if not self.tools:  # the forced wrap-up turn; answering it is what the loop is here for
            self.messages.append({"role": "assistant", "content": self.final_text})
            return self.final_text
        if self.turn == 0:
            self.turn += 1
            self.messages.append({"role": "assistant", "content": "searching", "tool_calls": _tool_calls(1)})
            return ""
        self.messages.append({"role": "assistant", "content": ""})
        return ""


def assert_every_tool_call_answered(messages: list[dict]) -> None:
    """Every assistant ``tool_calls`` id is answered by tool messages in the run that follows it."""
    for index, msg in enumerate(messages):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        answered = set()
        for following in messages[index + 1 :]:
            if following.get("role") != "tool":
                break
            answered.add(following["tool_call_id"])
        missing = [tc["id"] for tc in msg["tool_calls"] if tc["id"] not in answered]
        assert not missing, (
            f"messages[{index}] requested tools with no result immediately after: {', '.join(missing)}. "
            f"Roles from there: {[m.get('role') for m in messages[index : index + 3]]}"
        )


def assert_anthropic_payload_valid(messages: list[dict]) -> None:
    """The transcript survives translation to Anthropic's format without a dangling ``tool_use``."""
    _, ant = AnthropicClient._openai_messages_to_anthropic(MagicMock(cache_prompt=False), messages)
    for index, msg in enumerate(ant):
        content = msg["content"]
        if not isinstance(content, list):
            continue
        ids = [block["id"] for block in content if block.get("type") == "tool_use"]
        if not ids:
            continue
        following = ant[index + 1] if index + 1 < len(ant) else None
        answered = set()
        if following is not None and isinstance(following.get("content"), list):
            answered = {b.get("tool_use_id") for b in following["content"] if b.get("type") == "tool_result"}
        missing = [i for i in ids if i not in answered]
        assert not missing, (
            f"messages.{index}: `tool_use` ids were found without `tool_result` blocks "
            f"immediately after: {', '.join(missing)}"
        )


@pytest.mark.parametrize("max_iterations", [1, 2, 3])
def test_sync_run_settles_pending_tools_before_the_wrap_up(max_iterations):
    client = _PendingToolClient()
    agent = SyncAgent(client, tools=[_search], max_iterations=max_iterations)
    assert agent.run("research X") == "DONE"
    assert_every_tool_call_answered(client.messages)
    assert_anthropic_payload_valid(client.messages)


@pytest.mark.parametrize("max_iterations", [1, 2, 3])
def test_sync_streamed_run_settles_pending_tools_before_the_wrap_up(max_iterations):
    client = _PendingToolClient()
    agent = SyncAgent(client, tools=[_search], max_iterations=max_iterations)
    list(agent.run("research X", stream=True))
    assert_every_tool_call_answered(client.messages)
    assert_anthropic_payload_valid(client.messages)


@pytest.mark.parametrize("max_iterations", [1, 2, 3])
async def test_async_run_settles_pending_tools_before_the_wrap_up(max_iterations):
    client = _AsyncPendingToolClient()
    agent = AsyncAgent(client, tools=[_search], max_iterations=max_iterations)
    assert await agent.run("research X") == "DONE"
    assert_every_tool_call_answered(client.messages)
    assert_anthropic_payload_valid(client.messages)


@pytest.mark.parametrize("max_iterations", [1, 2, 3])
async def test_async_streamed_run_settles_pending_tools_before_the_wrap_up(max_iterations):
    client = _AsyncPendingToolClient()
    agent = AsyncAgent(client, tools=[_search], max_iterations=max_iterations)
    stream = await agent.run("research X", stream=True)
    async for _ in stream:
        pass
    assert_every_tool_call_answered(client.messages)
    assert_anthropic_payload_valid(client.messages)


async def test_settled_result_says_the_call_was_not_executed():
    """The synthesized result explains itself: the model has to know its search never ran."""
    client = _AsyncPendingToolClient()
    agent = AsyncAgent(client, tools=[_search], max_iterations=1)
    await agent.run("research X")
    settled = [m for m in client.messages if m.get("role") == "tool"]
    assert len(settled) == 2
    assert all("not executed" in m["content"] for m in settled)
    assert all("limit" in m["content"] for m in settled)


async def test_a_run_that_finishes_inside_its_cap_gains_no_synthesized_results():
    """The settle step is inert on a run that dispatched its calls: no phantom tool messages."""
    client = _AsyncPendingToolClient(tool_turns=1)
    agent = AsyncAgent(client, tools=[_search], max_iterations=6)
    assert await agent.run("research X") == "DONE"
    assert [m["content"] for m in client.messages if m.get("role") == "tool"] == [
        "results for q1a",
        "results for q1b",
    ]
    assert_every_tool_call_answered(client.messages)


async def test_an_empty_terminal_turn_after_a_real_tool_round_gains_no_duplicate_result():
    """The empty-turn wrap-up path must not re-answer a call that really did run.

    ``_pending``'s scan walks back past an empty terminal turn to the older tool turn behind it,
    which is why the settle step does its own narrower scan (see ``_settle_pending_tools``).
    """
    client = _EmptyAfterToolsClient()
    agent = AsyncAgent(client, tools=[_search], max_iterations=2)
    await agent.run("research X")
    ids = [m["tool_call_id"] for m in client.messages if m.get("role") == "tool"]
    assert ids == ["id1a", "id1b"], f"a duplicate result was synthesized: {ids}"
    assert not [m for m in client.messages if m.get("role") == "tool" and "not executed" in m["content"]]


def test_anthropic_keeps_assistant_prose_emitted_alongside_tool_calls():
    """Prose the model emits in the same turn as a tool call must reach the next request.

    ``_append_assistant_tool_calls`` stores it deliberately (a turn can carry both), so dropping
    it in translation silently loses the model's stated reason for the call from every later turn.
    """
    messages = [
        {"role": "user", "content": "research X"},
        {"role": "assistant", "content": "I will search for two things.", "tool_calls": _tool_calls(1)},
        {"role": "tool", "name": "_search", "content": "a", "tool_call_id": "id1a"},
        {"role": "tool", "name": "_search", "content": "b", "tool_call_id": "id1b"},
    ]
    _, ant = AnthropicClient._openai_messages_to_anthropic(MagicMock(cache_prompt=False), messages)
    assistant = ant[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "text", "text": "I will search for two things."}
    assert [b["type"] for b in assistant["content"]] == ["text", "tool_use", "tool_use"]


def test_anthropic_omits_an_empty_text_block_when_a_tool_call_carries_no_prose():
    """A tool-only turn must not gain an empty text block, which the API rejects."""
    messages = [
        {"role": "user", "content": "research X"},
        {"role": "assistant", "content": "   ", "tool_calls": _tool_calls(1)},
    ]
    _, ant = AnthropicClient._openai_messages_to_anthropic(MagicMock(cache_prompt=False), messages)
    assert [b["type"] for b in ant[1]["content"]] == ["tool_use", "tool_use"]
