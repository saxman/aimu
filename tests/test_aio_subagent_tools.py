"""Mock-only unit tests for make_async_subagent_tool (async).

No provider/backend/weights: the async ``Agent`` and the fresh-client builder are patched with
recording fakes. Mirrors tests/test_subagent_tools.py.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from aimu.aio.tools import builtin as _aio_builtin
from aimu.aio.tools.builtin import make_async_subagent_tool
from aimu.models import StreamChunk, StreamingContentType

# Capture the real fresh-client builder before the autouse fixture patches the module attribute,
# so the branch tests below can exercise the genuine cloud-vs-in-process logic.
_REAL_FRESH_CLIENT = _aio_builtin._fresh_async_subagent_client


class _FakeAsyncClient:
    instances: list = []

    def __init__(self, model):
        self.model = model
        self.messages: list = []
        _FakeAsyncClient.instances.append(self)


class _RecordingAsyncAgent:
    instances: list = []

    def __init__(
        self,
        model_client,
        system_message=None,
        name=None,
        tools=None,
        max_iterations=10,
        concurrent_tool_calls=False,
        deps=None,
        tool_approval=None,
        thinking=None,
    ):
        self.model_client = model_client
        self.system_message = system_message
        self.name = name
        self.tools = list(tools or [])
        self.max_iterations = max_iterations
        self.concurrent_tool_calls = concurrent_tool_calls
        self.deps = deps
        self.tool_approval = tool_approval
        self.thinking = thinking
        self.enter = None
        self.exit = None
        _RecordingAsyncAgent.instances.append(self)

    # Chunks a streamed run yields, set per test. Empty means "yield nothing".
    chunks: list = []

    async def run(self, task, *args, **kwargs):
        self.enter = time.perf_counter()
        if kwargs.get("stream"):
            return self._streamed(task)
        await asyncio.sleep(0.2)
        self.exit = time.perf_counter()
        return f"[{self.name}] answered: {task}"

    async def _streamed(self, task):
        for chunk in type(self).chunks:
            yield chunk
        self.exit = time.perf_counter()


def _fake_fresh_client(model):
    return _FakeAsyncClient(model)


@pytest.fixture(autouse=True)
def patch_async_agent_and_client(monkeypatch):
    _FakeAsyncClient.instances = []
    _RecordingAsyncAgent.instances = []
    _RecordingAsyncAgent.chunks = []
    monkeypatch.setattr("aimu.aio.agent.Agent", _RecordingAsyncAgent)
    monkeypatch.setattr("aimu.aio.tools.builtin._fresh_async_subagent_client", _fake_fresh_client)
    yield


MODEL = "anthropic:claude-sonnet-4-6"
TYPES = {
    "researcher": {"system_message": "Research the topic thoroughly and cite sources."},
    "writer": {"system_message": "Write clear, concise prose."},
}


def _tool_names(tools):
    return {getattr(t, "__tool_spec__", {}).get("function", {}).get("name") for t in tools}


# ---------------------------------------------------------------------------
# Spec shape
# ---------------------------------------------------------------------------


def test_async_spec_shape():
    spawn = make_async_subagent_tool(MODEL)
    fn = spawn.__tool_spec__["function"]
    assert fn["name"] == "spawn_subagent"
    assert set(fn["parameters"]["properties"]) == {"task"}
    assert spawn.__tool_is_async__ is True
    assert spawn.__tool_is_streaming__ is False


def test_async_typed_spec_menu():
    spawn = make_async_subagent_tool(MODEL, agent_types=TYPES)
    fn = spawn.__tool_spec__["function"]
    assert set(fn["parameters"]["properties"]) == {"agent_type", "task"}
    assert "researcher" in fn["description"] and "writer" in fn["description"]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


async def test_generic_dispatch_awaits_isolated_agent():
    spawn = make_async_subagent_tool(MODEL, system_message="Do the thing.")
    result = await spawn("summarize X")
    assert result == "[subagent] answered: summarize X"
    agent = _RecordingAsyncAgent.instances[-1]
    assert agent.system_message == "Do the thing."
    assert agent.model_client.model == MODEL


async def test_typed_dispatch_uses_type_system_message():
    spawn = make_async_subagent_tool(MODEL, agent_types=TYPES)
    await spawn("writer", "draft an intro")
    agent = _RecordingAsyncAgent.instances[-1]
    assert agent.system_message == "Write clear, concise prose."
    assert agent.name == "subagent-writer"


async def test_unknown_agent_type_returns_message_not_raise():
    spawn = make_async_subagent_tool(MODEL, agent_types=TYPES)
    result = await spawn("nope", "task")
    assert "Unknown agent_type" in result
    assert not _RecordingAsyncAgent.instances


async def test_each_call_creates_an_independent_client():
    spawn = make_async_subagent_tool(MODEL)
    await spawn("t1")
    await spawn("t2")
    assert len(_FakeAsyncClient.instances) == 2
    assert _FakeAsyncClient.instances[0] is not _FakeAsyncClient.instances[1]


# ---------------------------------------------------------------------------
# Depth guard
# ---------------------------------------------------------------------------


async def test_depth_1_child_gets_no_spawn_tool():
    spawn = make_async_subagent_tool(MODEL, max_depth=1)
    await spawn("task")
    agent = _RecordingAsyncAgent.instances[0]
    assert "spawn_subagent" not in _tool_names(agent.tools)


async def test_depth_2_child_gets_a_terminating_spawn_tool():
    spawn = make_async_subagent_tool(MODEL, max_depth=2)
    await spawn("task")
    child = _RecordingAsyncAgent.instances[0]
    child_spawn = next(t for t in child.tools if t.__tool_spec__["function"]["name"] == "spawn_subagent")
    assert child_spawn.__tool_is_async__ is True
    await child_spawn("nested")
    grandchild = _RecordingAsyncAgent.instances[-1]
    assert "spawn_subagent" not in _tool_names(grandchild.tools)


# ---------------------------------------------------------------------------
# Factory-time validation
# ---------------------------------------------------------------------------


def test_max_depth_below_one_raises():
    with pytest.raises(ValueError, match="max_depth"):
        make_async_subagent_tool(MODEL, max_depth=0)


def test_agent_type_missing_system_message_raises():
    with pytest.raises(ValueError, match="system_message"):
        make_async_subagent_tool(MODEL, agent_types={"bad": {}})


def test_agent_type_with_an_unknown_key_raises():
    """Both twins share one validator, so this pins that the async factory calls it too."""
    with pytest.raises(ValueError, match="thinkng"):
        make_async_subagent_tool(MODEL, agent_types={"bad": {"system_message": "S.", "thinkng": "high"}})


# ---------------------------------------------------------------------------
# Parallel overlap under asyncio.TaskGroup
# ---------------------------------------------------------------------------


async def test_concurrent_dispatch_overlaps_spawns():
    from aimu.aio._tool_loop import _AsyncToolLoop

    spawn = make_async_subagent_tool(MODEL)

    class _Client:
        def __init__(self):
            self.messages = [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "spawn_subagent", "arguments": {"task": "A"}},
                            "id": "c1",
                        },
                        {
                            "type": "function",
                            "function": {"name": "spawn_subagent", "arguments": {"task": "B"}},
                            "id": "c2",
                        },
                    ],
                },
            ]

        def _append_message(self, message):
            from aimu.models._internal.chat_state import _ChatStateMixin

            _ChatStateMixin._append_message(self, message)

    client = _Client()
    loop = _AsyncToolLoop(client, [spawn], concurrent_tool_calls=True)
    await loop._dispatch()

    assert len(_RecordingAsyncAgent.instances) == 2
    a, b = _RecordingAsyncAgent.instances
    assert max(a.enter, b.enter) < min(a.exit, b.exit)
    assert len([m for m in client.messages if m.get("role") == "tool"]) == 2


# ---------------------------------------------------------------------------
# _fresh_async_subagent_client: cloud vs in-process wrapping branch
# ---------------------------------------------------------------------------


def test_fresh_client_cloud_constructs_directly(monkeypatch):
    calls = {}
    monkeypatch.setattr(_aio_builtin, "_is_in_process_model", lambda m: False)
    monkeypatch.setattr("aimu.aio._model_client.AsyncModelClient", lambda m: calls.setdefault("arg", m))

    sentinel = object()
    _REAL_FRESH_CLIENT(sentinel)
    assert calls["arg"] is sentinel  # constructed directly from the model


def test_fresh_client_in_process_wraps_sync(monkeypatch):
    import aimu

    sync_sentinel = object()
    seen = {}
    monkeypatch.setattr(_aio_builtin, "_is_in_process_model", lambda m: True)
    monkeypatch.setattr(aimu, "client", lambda m: sync_sentinel)
    monkeypatch.setattr("aimu.aio._model_client.AsyncModelClient", lambda c: seen.setdefault("wrapped", c))

    _REAL_FRESH_CLIENT(object())
    assert seen["wrapped"] is sync_sentinel  # wrapped a fresh sync client


# ---------------------------------------------------------------------------
# tool_approval forwarding
# ---------------------------------------------------------------------------


async def test_tool_approval_forwarded_to_child_agent():
    gate = lambda name, args: False  # noqa: E731

    spawn = make_async_subagent_tool(MODEL, tool_approval=gate)
    await spawn("some task")

    agent = _RecordingAsyncAgent.instances[0]
    assert agent.tool_approval is gate


async def test_tool_approval_forwarded_through_recursive_depth():
    gate = lambda name, args: False  # noqa: E731

    spawn = make_async_subagent_tool(MODEL, max_depth=2, tool_approval=gate)
    await spawn("task")

    child = _RecordingAsyncAgent.instances[0]
    assert child.tool_approval is gate

    child_spawn = next(t for t in child.tools if t.__tool_spec__["function"]["name"] == "spawn_subagent")
    await child_spawn("nested task")
    grandchild = _RecordingAsyncAgent.instances[-1]
    assert grandchild.tool_approval is gate


async def test_no_tool_approval_defaults_to_none():
    spawn = make_async_subagent_tool(MODEL)
    await spawn("task")

    agent = _RecordingAsyncAgent.instances[0]
    assert agent.tool_approval is None


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------


class _RecordingObserver:
    """Records the callback sequence so a test can assert order and payloads."""

    def __init__(self, fail_on: str = ""):
        self.events: list[tuple] = []
        self._fail_on = fail_on

    async def spawned(self, spawn_id, agent_type, task):
        self.events.append(("spawned", spawn_id, agent_type, task))
        if self._fail_on == "spawned":
            raise RuntimeError("observer is broken")

    async def chunk(self, spawn_id, chunk):
        self.events.append(("chunk", spawn_id, chunk.phase, chunk.content))

    async def finished(self, spawn_id, result, error):
        self.events.append(("finished", spawn_id, result, error))


def _text(content, iteration=0):
    return StreamChunk(StreamingContentType.GENERATING, content, iteration=iteration)


async def test_observer_sees_spawn_chunks_and_completion():
    _RecordingAsyncAgent.chunks = [
        StreamChunk(StreamingContentType.THINKING, "let me think"),
        _text("the "),
        _text("answer"),
    ]
    observer = _RecordingObserver()
    spawn = make_async_subagent_tool(MODEL, agent_types=TYPES, observer=observer)
    result = await spawn("researcher", "find X")
    assert result == "the answer"
    kinds = [event[0] for event in observer.events]
    assert kinds == ["spawned", "chunk", "chunk", "chunk", "finished"]
    spawn_id = observer.events[0][1]
    assert spawn_id.startswith("researcher-")
    assert observer.events[0][2:] == ("researcher", "find X")
    assert all(event[1] == spawn_id for event in observer.events)
    assert observer.events[-1][2:] == ("the answer", None)


async def test_generated_text_resets_on_each_loop_iteration():
    """The tool's return value is the FINAL answer, not every intermediate tools-only response."""
    _RecordingAsyncAgent.chunks = [_text("first pass", 0), _text("final answer", 1)]
    observer = _RecordingObserver()
    spawn = make_async_subagent_tool(MODEL, agent_types=TYPES, observer=observer)
    assert await spawn("researcher", "find X") == "final answer"
    assert observer.events[-1][2] == "final answer"


async def test_spawn_ids_are_unique_per_call():
    observer = _RecordingObserver()
    spawn = make_async_subagent_tool(MODEL, agent_types=TYPES, observer=observer)
    await spawn("researcher", "a")
    await spawn("researcher", "b")
    ids = {event[1] for event in observer.events}
    assert len(ids) == 2


async def test_observer_is_told_when_the_child_fails_and_the_error_propagates(monkeypatch):
    class _FailingAgent(_RecordingAsyncAgent):
        async def run(self, task, *args, **kwargs):
            raise ValueError("child exploded")

    monkeypatch.setattr("aimu.aio.agent.Agent", _FailingAgent)
    observer = _RecordingObserver()
    spawn = make_async_subagent_tool(MODEL, agent_types=TYPES, observer=observer)
    with pytest.raises(ValueError, match="child exploded"):
        await spawn("researcher", "find X")
    kind, _spawn_id, result, error = observer.events[-1]
    assert kind == "finished"
    assert result == ""
    assert isinstance(error, ValueError)


async def test_observer_is_told_when_the_spawn_is_cancelled(monkeypatch):
    class _HangingAgent(_RecordingAsyncAgent):
        async def run(self, task, *args, **kwargs):
            async def _gen():
                yield _text("partial")
                await asyncio.sleep(60)

            return _gen()

    monkeypatch.setattr("aimu.aio.agent.Agent", _HangingAgent)
    observer = _RecordingObserver()
    spawn = make_async_subagent_tool(MODEL, agent_types=TYPES, observer=observer)
    task = asyncio.create_task(spawn("researcher", "find X"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    kind, _spawn_id, result, error = observer.events[-1]
    assert kind == "finished"
    assert result == "partial"  # whatever accumulated before the cancellation
    assert isinstance(error, asyncio.CancelledError)


async def test_a_broken_observer_does_not_break_the_spawn():
    _RecordingAsyncAgent.chunks = [_text("answer")]
    observer = _RecordingObserver(fail_on="spawned")
    spawn = make_async_subagent_tool(MODEL, agent_types=TYPES, observer=observer)
    assert await spawn("researcher", "find X") == "answer"


async def test_an_observer_with_a_wrong_signature_does_not_break_the_spawn():
    """Seam drift: an observer written against a different parameter list raises while the call is
    being *built*, not while it is awaited, so the guard has to cover the call itself."""

    class _StaleObserver(_RecordingObserver):
        async def spawned(self, spawn_id):  # the real seam passes agent_type and task too
            raise AssertionError("unreachable: the call itself raises TypeError")  # pragma: no cover

    _RecordingAsyncAgent.chunks = [_text("answer")]
    observer = _StaleObserver()
    spawn = make_async_subagent_tool(MODEL, agent_types=TYPES, observer=observer)

    assert await spawn("researcher", "find X") == "answer"
    # The remaining callbacks still fire: one broken hook must not silence the rest of the report.
    assert [event[0] for event in observer.events] == ["chunk", "finished"]


async def test_a_partial_observer_missing_callbacks_does_not_break_the_spawn():
    """Structural (duck-typed) satisfaction of SubagentObserver means implementing only one callback
    is realistic input, not misuse -- the missing ones must not raise AttributeError."""

    class _ChunkOnlyObserver:
        def __init__(self):
            self.events: list[tuple] = []

        async def chunk(self, spawn_id, chunk):
            self.events.append(("chunk", spawn_id, chunk.content))

    _RecordingAsyncAgent.chunks = [_text("answer")]
    observer = _ChunkOnlyObserver()
    spawn = make_async_subagent_tool(MODEL, agent_types=TYPES, observer=observer)

    assert await spawn("researcher", "find X") == "answer"
    assert [event[0] for event in observer.events] == ["chunk"]


async def test_no_observer_keeps_the_non_streaming_path():
    _RecordingAsyncAgent.chunks = [_text("streamed")]
    spawn = make_async_subagent_tool(MODEL, agent_types=TYPES)
    assert await spawn("researcher", "find X") == "[subagent-researcher] answered: find X"


async def test_nested_spawns_inherit_the_observer():
    observer = _RecordingObserver()
    spawn = make_async_subagent_tool(MODEL, agent_types=TYPES, observer=observer, max_depth=2)
    await spawn("researcher", "find X")
    child = _RecordingAsyncAgent.instances[-1]
    nested = [t for t in child.tools if getattr(t, "__name__", "") == "spawn_subagent"]
    assert nested, "a depth-2 spawn tool must be handed to the child"


async def test_typed_dispatch_honors_per_type_thinking():
    types = {"careful": {"system_message": "Be thorough.", "thinking": "high"}}
    spawn = make_async_subagent_tool(MODEL, agent_types=types)
    await spawn("careful", "hard task")
    assert _RecordingAsyncAgent.instances[-1].thinking == "high"


async def test_typed_dispatch_carries_thinking_false():
    """``False`` is a real request (reasoning off), so the spec read cannot be a truthiness test."""
    types = {"quick": {"system_message": "Be quick.", "thinking": False}}
    spawn = make_async_subagent_tool(MODEL, agent_types=types)
    await spawn("quick", "trivial task")
    assert _RecordingAsyncAgent.instances[-1].thinking is False


async def test_typed_dispatch_leaves_thinking_unset_when_the_spec_omits_it():
    spawn = make_async_subagent_tool(MODEL, agent_types={"plain": {"system_message": "Plain."}})
    await spawn("plain", "task")
    assert _RecordingAsyncAgent.instances[-1].thinking is None
