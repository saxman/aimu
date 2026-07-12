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
    ):
        self.model_client = model_client
        self.system_message = system_message
        self.name = name
        self.tools = list(tools or [])
        self.max_iterations = max_iterations
        self.concurrent_tool_calls = concurrent_tool_calls
        self.deps = deps
        self.enter = None
        self.exit = None
        _RecordingAsyncAgent.instances.append(self)

    async def run(self, task, *args, **kwargs):
        self.enter = time.perf_counter()
        await asyncio.sleep(0.2)
        self.exit = time.perf_counter()
        return f"[{self.name}] answered: {task}"


def _fake_fresh_client(model):
    return _FakeAsyncClient(model)


@pytest.fixture(autouse=True)
def patch_async_agent_and_client(monkeypatch):
    _FakeAsyncClient.instances = []
    _RecordingAsyncAgent.instances = []
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
