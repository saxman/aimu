"""Mock-only unit tests for make_subagent_tool (sync).

No provider/backend/weights: ``ModelClient`` and ``Agent`` are patched with recording fakes so
the tests exercise the factory's wiring (spec shape, isolation, depth guard, dispatch) directly.
"""

from __future__ import annotations

import logging
import time

import pytest

# Captured at module load, before the autouse fixture below replaces the attribute with a
# recording fake; importing it fresh inside a test would just re-fetch the fake.
from aimu.agents.agent import Agent as _RealAgent
from aimu.models import ContextOverflowError
from aimu.tools.builtin import make_subagent_tool


class _RecordingModelClient:
    """Fake ModelClient: records the model it was built from; each instance has its own messages."""

    instances: list = []

    def __init__(self, model):
        self.model = model
        self.messages: list = []
        # Mirrors a real client, whose default_generate_kwargs starts empty, so a test can assert that
        # a spec omitting the key leaves it untouched rather than that the attribute is missing.
        self.default_generate_kwargs: dict = {}
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
        thinking=None,
        events=None,
        compaction=None,
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
        self.events = events
        self.compaction = compaction
        self.enter = None
        self.exit = None
        _RecordingAgent.instances.append(self)

    # Set per test to make the child's run fail instead of answering. Reset by the autouse fixture,
    # so one test's failure mode cannot leak into the next.
    raises: BaseException | None = None

    def run(self, task, *args, **kwargs):
        self.enter = time.perf_counter()
        if type(self).raises is not None:
            raise type(self).raises
        time.sleep(0.2)  # long enough to detect concurrent overlap
        self.exit = time.perf_counter()
        return f"[{self.name}] answered: {task}"


@pytest.fixture(autouse=True)
def patch_agent_and_client(monkeypatch):
    _RecordingModelClient.instances = []
    _RecordingAgent.instances = []
    _RecordingAgent.raises = None
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


def test_agent_type_with_an_unknown_key_raises():
    """An ignored key reads as an applied one. A misspelled `"thinking"` would leave the spawned agent at
    its default with nothing raised anywhere, which is the failure this refuses to allow silently."""
    with pytest.raises(ValueError, match="thinkng"):
        make_subagent_tool(MODEL, agent_types={"bad": {"system_message": "S.", "thinkng": "high"}})


def test_the_unknown_key_error_names_the_keys_that_are_accepted():
    with pytest.raises(ValueError, match="generate_kwargs, max_iterations, model, system_message, thinking, tools"):
        make_subagent_tool(MODEL, agent_types={"bad": {"system_message": "S.", "temperture": 0.2}})


def test_the_unknown_key_error_names_the_agent_type_it_came_from():
    with pytest.raises(ValueError, match="'researcher'"):
        make_subagent_tool(
            MODEL,
            agent_types={"writer": {"system_message": "W."}, "researcher": {"system_message": "R.", "nope": 1}},
        )


def test_every_documented_spec_key_is_accepted():
    spec = {
        "system_message": "S.",
        "tools": [],
        "model": MODEL,
        "thinking": "high",
        "generate_kwargs": {"temperature": 0.2},
        "max_iterations": 25,
    }
    make_subagent_tool(MODEL, agent_types={"full": spec})  # must not raise


def test_typed_dispatch_applies_per_type_generate_kwargs():
    types = {"cold": {"system_message": "Be literal.", "generate_kwargs": {"temperature": 0.1}}}
    spawn = make_subagent_tool(MODEL, agent_types=types)
    spawn("cold", "extract the dates")
    assert _RecordingModelClient.instances[-1].default_generate_kwargs == {"temperature": 0.1}


def test_typed_dispatch_leaves_generate_kwargs_empty_when_the_spec_omits_them():
    """Absent must stay absent: this tier sits above the model card, so a filled-in default shadows it."""
    spawn = make_subagent_tool(MODEL, agent_types={"plain": {"system_message": "Plain."}})
    spawn("plain", "task")
    assert _RecordingModelClient.instances[-1].default_generate_kwargs == {}


def test_a_specs_generate_kwargs_dict_is_not_shared_with_the_spawned_client():
    """Two spawns of one agent_type must not accumulate each other's mutations."""
    spec_kwargs = {"temperature": 0.1}
    spawn = make_subagent_tool(MODEL, agent_types={"cold": {"system_message": "S.", "generate_kwargs": spec_kwargs}})
    spawn("cold", "one")
    _RecordingModelClient.instances[-1].default_generate_kwargs["top_p"] = 0.5
    spawn("cold", "two")
    assert spec_kwargs == {"temperature": 0.1}
    assert _RecordingModelClient.instances[-1].default_generate_kwargs == {"temperature": 0.1}


def test_typed_dispatch_applies_per_type_max_iterations():
    types = {"deep": {"system_message": "Dig until you are sure.", "max_iterations": 25}}
    spawn = make_subagent_tool(MODEL, agent_types=types)
    spawn("deep", "research the topic")
    assert _RecordingAgent.instances[-1].max_iterations == 25


def test_typed_dispatch_falls_back_to_the_factory_cap_when_the_spec_omits_it():
    """Unlike "thinking" and "generate_kwargs", a missing cap has a tier to fall back to: this factory's
    own. That is what lets a caller set one default across a roster without writing it into each spec."""
    spawn = make_subagent_tool(MODEL, agent_types={"plain": {"system_message": "Plain."}}, max_iterations=4)
    spawn("plain", "task")
    assert _RecordingAgent.instances[-1].max_iterations == 4


def test_a_spec_cap_overrides_the_factory_cap_rather_than_being_capped_by_it():
    """A spec may ask for more than the factory default, not only less."""
    types = {"deep": {"system_message": "Dig.", "max_iterations": 30}}
    spawn = make_subagent_tool(MODEL, agent_types=types, max_iterations=4)
    spawn("deep", "task")
    assert _RecordingAgent.instances[-1].max_iterations == 30


@pytest.mark.parametrize("bad", [0, -1, True, False, 2.5, "10", None])
def test_a_spec_cap_that_is_not_a_positive_int_raises_at_factory_call_time(bad):
    """`bool` is an `int` subclass, so `True` would otherwise pass as a cap of 1, and `0` is a loop that
    makes no model call at all. Both are programmer errors, so they fail where the roster is written."""
    with pytest.raises(ValueError, match="max_iterations"):
        make_subagent_tool(MODEL, agent_types={"bad": {"system_message": "S.", "max_iterations": bad}})


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

        def _append_message(self, message):
            from aimu.models._internal.chat_state import _ChatStateMixin

            _ChatStateMixin._append_message(self, message)

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


def test_typed_dispatch_honors_per_type_thinking():
    types = {"careful": {"system_message": "Be thorough.", "thinking": "high"}}
    spawn = make_subagent_tool(MODEL, agent_types=types)
    spawn("careful", "hard task")
    assert _RecordingAgent.instances[-1].thinking == "high"


def test_typed_dispatch_carries_thinking_false():
    """``False`` is a real request (reasoning off), so the spec read cannot be a truthiness test."""
    types = {"quick": {"system_message": "Be quick.", "thinking": False}}
    spawn = make_subagent_tool(MODEL, agent_types=types)
    spawn("quick", "trivial task")
    assert _RecordingAgent.instances[-1].thinking is False


def test_typed_dispatch_leaves_thinking_unset_when_the_spec_omits_it():
    spawn = make_subagent_tool(MODEL, agent_types={"plain": {"system_message": "Plain."}})
    spawn("plain", "task")
    assert _RecordingAgent.instances[-1].thinking is None


# ---------------------------------------------------------------------------
# events forwarding
# ---------------------------------------------------------------------------


def test_spawn_forwards_events_to_the_child_agents_sink(monkeypatch):
    """A spawned sub-agent's model turns reach the caller's sink.

    The spawn tool builds its own client, which is deliberately outside the family a scoped
    per-run override reaches, so without an explicit events= a delegated run reports nothing
    and a caller measuring a turn's cost silently under-counts every delegation. The fake Agent
    and ModelClient this module patches in above don't emit turn events at all, so this test
    swaps in the real Agent and a real (mocked) model client to exercise the genuine path.
    """
    from aimu.events import ModelTurnFinished
    from tests.helpers import MockModelClient

    # _RealAgent was imported at module load, before the autouse fixture above replaced the
    # attribute with the fake; importing it fresh here would just re-fetch the fake. The client
    # side doesn't need the same trick: this test replaces ModelClient with a factory of its own
    # rather than restoring the real one.
    monkeypatch.setattr("aimu.agents.agent.Agent", _RealAgent)
    monkeypatch.setattr("aimu.models.model_client.ModelClient", lambda m: MockModelClient(["done"]))

    seen = []
    spawn = make_subagent_tool(
        MODEL,
        agent_types={"worker": {"system_message": "you are a worker", "tools": []}},
        events=seen.append,
    )
    spawn("worker", "do the thing")

    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert finished, "the child's model turn should have reported to the caller's sink"


def test_spawn_without_events_reports_nowhere():
    """The parameter is opt-in: omitting it leaves the child reporting to its own client only."""
    spawn = make_subagent_tool(
        MODEL,
        agent_types={"worker": {"system_message": "you are a worker", "tools": []}},
    )
    spawn("worker", "do the thing")  # must not raise


# ---------------------------------------------------------------------------
# A child that runs out of context
# ---------------------------------------------------------------------------

# A real provider message, kept verbatim: the point of these tests is what the parent model reads,
# and the misattributing half is the "Shorten the conversation" clause this one actually carries.
OVERFLOW_MESSAGE = (
    "The request no longer fits the model's context window: Anthropic rejected the prompt as too "
    "long. Shorten the conversation, advertise fewer tools, or compact history first "
    "(aimu.context.trim_messages / summarize_messages)."
)


def test_a_full_child_becomes_a_tool_result_rather_than_an_exception():
    """Uncaught, this reaches the parent model as ``Tool 'spawn_subagent' raised an error: ...``.

    Every provider composes that message for whoever built the client, so it says to shorten "the
    conversation" and advertise fewer tools. Read by the *parent*, both resolve against the parent's
    own conversation, which is not what filled: the child is built fresh per call and never sees it.
    """
    _RecordingAgent.raises = ContextOverflowError(OVERFLOW_MESSAGE)
    spawn = make_subagent_tool(MODEL)
    result = spawn("summarize X")
    assert "sub-agent" in result
    assert "Your conversation is not the cause" in result


def test_the_overflow_result_names_which_specialist_filled_up():
    _RecordingAgent.raises = ContextOverflowError(OVERFLOW_MESSAGE)
    spawn = make_subagent_tool(MODEL, agent_types=TYPES)
    assert "'researcher'" in spawn("researcher", "dig into X")


def test_the_overflow_result_keeps_the_providers_own_sentence_as_evidence():
    """Dropping it would hide which backend refused and why, so it is disclaimed instead."""
    _RecordingAgent.raises = ContextOverflowError(OVERFLOW_MESSAGE)
    result = make_subagent_tool(MODEL)("summarize X")
    assert OVERFLOW_MESSAGE in result
    assert f'"{OVERFLOW_MESSAGE}"' in result
    assert "That quoted remediation is addressed to the program" in result


def test_a_full_child_is_logged_even_though_the_model_gets_a_string(caplog):
    """The operator's copy must not depend on the model choosing to mention it."""
    _RecordingAgent.raises = ContextOverflowError(OVERFLOW_MESSAGE)
    with caplog.at_level(logging.WARNING, logger="aimu.tools.builtin"):
        make_subagent_tool(MODEL, agent_types=TYPES)("writer", "write it up")
    assert any("ran out of context" in r.getMessage() for r in caplog.records)


def test_other_child_failures_still_propagate():
    """Only the misattributing error is converted; a broken tool is still the caller's to see."""
    _RecordingAgent.raises = RuntimeError("the child broke")
    with pytest.raises(RuntimeError, match="the child broke"):
        make_subagent_tool(MODEL)("summarize X")


# ---------------------------------------------------------------------------
# compaction
# ---------------------------------------------------------------------------


def _noop_compaction(messages):
    return messages


def _other_compaction(messages):
    return messages


def test_the_factory_compaction_reaches_every_spawned_agent():
    spawn = make_subagent_tool(MODEL, compaction=_noop_compaction)
    spawn("task")
    assert _RecordingAgent.instances[-1].compaction is _noop_compaction


def test_a_spec_compaction_overrides_the_factory_one():
    types = {"heavy": {"system_message": "Read a lot.", "compaction": _other_compaction}}
    spawn = make_subagent_tool(MODEL, agent_types=types, compaction=_noop_compaction)
    spawn("heavy", "task")
    assert _RecordingAgent.instances[-1].compaction is _other_compaction


def test_a_spec_omitting_compaction_inherits_the_factory_one():
    spawn = make_subagent_tool(MODEL, agent_types=TYPES, compaction=_noop_compaction)
    spawn("writer", "task")
    assert _RecordingAgent.instances[-1].compaction is _noop_compaction


def test_a_spec_can_turn_the_factory_compaction_off():
    """``"compaction": None`` is a decision, and ``.get()`` could not tell it from an absent key."""
    types = {"short": {"system_message": "Answer briefly.", "compaction": None}}
    spawn = make_subagent_tool(MODEL, agent_types=types, compaction=_noop_compaction)
    spawn("short", "task")
    assert _RecordingAgent.instances[-1].compaction is None


def test_no_compaction_by_default():
    make_subagent_tool(MODEL)("task")
    assert _RecordingAgent.instances[-1].compaction is None


def test_a_nested_spawn_tool_carries_the_factory_compaction():
    """The factory tier, like the cap: a nested tool serves the whole roster again."""
    spawn = make_subagent_tool(MODEL, max_depth=2, compaction=_noop_compaction)
    spawn("task")
    nested = [t for t in _RecordingAgent.instances[-1].tools if getattr(t, "__name__", "") == "spawn_subagent"]
    assert nested, "depth 2 should have injected a nested spawn tool"
    nested[0]("deeper task")
    assert _RecordingAgent.instances[-1].compaction is _noop_compaction


def test_a_non_callable_compaction_raises_at_factory_call_time():
    with pytest.raises(ValueError, match="compaction must be a callable"):
        make_subagent_tool(MODEL, compaction="trim")


def test_a_non_callable_spec_compaction_raises_at_factory_call_time():
    types = {"bad": {"system_message": "x", "compaction": 5}}
    with pytest.raises(ValueError, match=r"agent_types\['bad'\]\['compaction'\]"):
        make_subagent_tool(MODEL, agent_types=types)
