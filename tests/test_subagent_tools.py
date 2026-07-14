"""Mock-only unit tests for make_subagent_tool (sync).

No provider/backend/weights: ``ModelClient`` and ``Agent`` are patched with recording fakes so
the tests exercise the factory's wiring (spec shape, isolation, depth guard, dispatch) directly.
"""

from __future__ import annotations

import time

import pytest

from aimu.tools.builtin import make_subagent_tool


class _RecordingModelClient:
    """Fake ModelClient: records the model it was built from; each instance has its own messages."""

    instances: list = []

    def __init__(self, model):
        self.model = model
        self.messages: list = []
        _RecordingModelClient.instances.append(self)


class _RecordingAgent:
    """Fake Agent: records construction kwargs; run() returns a canned, identifiable string."""

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
    ):
        self.model_client = model_client
        self.system_message = system_message
        self.name = name
        self.tools = list(tools or [])
        self.max_iterations = max_iterations
        self.concurrent_tool_calls = concurrent_tool_calls
        self.deps = deps
        self.tool_approval = tool_approval
        self.enter = None
        self.exit = None
        _RecordingAgent.instances.append(self)

    def run(self, task, *args, **kwargs):
        self.enter = time.perf_counter()
        time.sleep(0.2)  # long enough to detect concurrent overlap
        self.exit = time.perf_counter()
        return f"[{self.name}] answered: {task}"


@pytest.fixture(autouse=True)
def patch_agent_and_client(monkeypatch):
    _RecordingModelClient.instances = []
    _RecordingAgent.instances = []
    monkeypatch.setattr("aimu.models.model_client.ModelClient", _RecordingModelClient)
    monkeypatch.setattr("aimu.agents.agent.Agent", _RecordingAgent)
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


def test_generic_spec_shape():
    spawn = make_subagent_tool(MODEL)
    fn = spawn.__tool_spec__["function"]
    assert fn["name"] == "spawn_subagent"
    assert set(fn["parameters"]["properties"]) == {"task"}
    assert fn["parameters"]["required"] == ["task"]
    assert spawn.__tool_is_async__ is False
    assert spawn.__tool_is_streaming__ is False


def test_typed_spec_shape_and_menu_in_description():
    spawn = make_subagent_tool(MODEL, agent_types=TYPES)
    fn = spawn.__tool_spec__["function"]
    assert set(fn["parameters"]["properties"]) == {"agent_type", "task"}
    assert fn["parameters"]["required"] == ["agent_type", "task"]
    # The model must see the available type names — they land in the (first-paragraph) description.
    assert "researcher" in fn["description"]
    assert "writer" in fn["description"]


def test_custom_tool_name():
    spawn = make_subagent_tool(MODEL, tool_name="spawn_researcher")
    assert spawn.__tool_spec__["function"]["name"] == "spawn_researcher"
    assert spawn.__name__ == "spawn_researcher"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_generic_dispatch_builds_isolated_agent():
    spawn = make_subagent_tool(MODEL, system_message="Do the thing.")
    result = spawn("summarize X")
    assert result == "[subagent] answered: summarize X"
    assert len(_RecordingAgent.instances) == 1
    agent = _RecordingAgent.instances[0]
    assert agent.system_message == "Do the thing."
    assert agent.model_client.model == MODEL


def test_typed_dispatch_uses_type_system_message():
    spawn = make_subagent_tool(MODEL, agent_types=TYPES)
    spawn("researcher", "find sources on X")
    agent = _RecordingAgent.instances[-1]
    assert agent.system_message == "Research the topic thoroughly and cite sources."
    assert agent.name == "subagent-researcher"


def test_typed_dispatch_honors_per_type_model_override():
    types = {"fast": {"system_message": "Be quick.", "model": "openai:gpt-4o-mini"}}
    spawn = make_subagent_tool(MODEL, agent_types=types)
    spawn("fast", "quick task")
    agent = _RecordingAgent.instances[-1]
    assert agent.model_client.model == "openai:gpt-4o-mini"


def test_typed_dispatch_uses_factory_tools_when_type_has_none():
    def _dummy():  # a stand-in tool object
        return "x"

    spawn = make_subagent_tool(MODEL, tools=[_dummy], agent_types={"a": {"system_message": "A."}})
    spawn("a", "task")
    agent = _RecordingAgent.instances[-1]
    assert _dummy in agent.tools


def test_each_call_creates_an_independent_client():
    spawn = make_subagent_tool(MODEL)
    spawn("task 1")
    spawn("task 2")
    clients = _RecordingModelClient.instances
    assert len(clients) == 2
    assert clients[0] is not clients[1]
    assert clients[0].messages is not clients[1].messages  # isolated histories


def test_unknown_agent_type_returns_message_not_raise():
    spawn = make_subagent_tool(MODEL, agent_types=TYPES)
    result = spawn("nope", "task")
    assert "Unknown agent_type" in result
    assert "researcher" in result and "writer" in result
    assert not _RecordingAgent.instances  # no agent was built


# ---------------------------------------------------------------------------
# Depth guard
# ---------------------------------------------------------------------------


def test_depth_1_child_gets_no_spawn_tool():
    spawn = make_subagent_tool(MODEL, max_depth=1)
    spawn("task")
    agent = _RecordingAgent.instances[0]
    assert "spawn_subagent" not in _tool_names(agent.tools)


def test_depth_2_child_gets_a_terminating_spawn_tool():
    spawn = make_subagent_tool(MODEL, max_depth=2)
    spawn("task")
    child = _RecordingAgent.instances[0]
    child_spawn = next(t for t in child.tools if t.__tool_spec__["function"]["name"] == "spawn_subagent")
    # Invoking the nested spawn builds a grandchild that gets NO further spawn tool.
    child_spawn("nested task")
    grandchild = _RecordingAgent.instances[-1]
    assert "spawn_subagent" not in _tool_names(grandchild.tools)


# ---------------------------------------------------------------------------
# Factory-time validation (failures apparent)
# ---------------------------------------------------------------------------


def test_max_depth_below_one_raises():
    with pytest.raises(ValueError, match="max_depth"):
        make_subagent_tool(MODEL, max_depth=0)


def test_empty_agent_types_raises():
    with pytest.raises(ValueError, match="non-empty"):
        make_subagent_tool(MODEL, agent_types={})


def test_agent_type_missing_system_message_raises():
    with pytest.raises(ValueError, match="system_message"):
        make_subagent_tool(MODEL, agent_types={"bad": {"tools": []}})


# ---------------------------------------------------------------------------
# Parallel overlap (rides the existing concurrent_tool_calls dispatch)
# ---------------------------------------------------------------------------


def test_concurrent_dispatch_overlaps_spawns():
    from aimu.agents._tool_loop import _ToolLoop

    spawn = make_subagent_tool(MODEL)

    class _Client:
        def __init__(self):
            self.messages = [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
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
    loop = _ToolLoop(client, [spawn], concurrent_tool_calls=True)
    loop._dispatch()

    # Two sub-agents ran; their [0.2s] windows overlapped -> concurrent, not sequential.
    assert len(_RecordingAgent.instances) == 2
    a, b = _RecordingAgent.instances
    assert max(a.enter, b.enter) < min(a.exit, b.exit)
    # Results were appended as tool messages.
    tool_msgs = [m for m in client.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2


# ---------------------------------------------------------------------------
# tool_approval forwarding
# ---------------------------------------------------------------------------


def test_tool_approval_forwarded_to_child_agent():
    gate = lambda name, args: False  # noqa: E731

    spawn = make_subagent_tool(MODEL, tool_approval=gate)
    spawn("some task")

    agent = _RecordingAgent.instances[0]
    assert agent.tool_approval is gate


def test_tool_approval_forwarded_through_recursive_depth():
    gate = lambda name, args: False  # noqa: E731

    spawn = make_subagent_tool(MODEL, max_depth=2, tool_approval=gate)
    spawn("task")

    child = _RecordingAgent.instances[0]
    assert child.tool_approval is gate

    # The nested spawn tool the child received should also carry the gate forward.
    child_spawn = next(t for t in child.tools if t.__tool_spec__["function"]["name"] == "spawn_subagent")
    child_spawn("nested task")
    grandchild = _RecordingAgent.instances[-1]
    assert grandchild.tool_approval is gate


def test_no_tool_approval_defaults_to_none():
    spawn = make_subagent_tool(MODEL)
    spawn("task")

    agent = _RecordingAgent.instances[0]
    assert agent.tool_approval is None
