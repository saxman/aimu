"""Behavioural tests for the API redesign:

* Top-level ``aimu.chat()`` / ``aimu.client()`` and model-string parser.
* ``ModelSpec`` migration preserves capability flags + ``.value`` semantics.
* ``StreamChunk`` is the only chunk type (back-compat aliases included).
* ``system_message`` is immutable after the first chat; ``reset()`` unlocks it.
* ``@tool`` validates signatures; supports ``Optional[T]`` / ``T | None``.
* ``include=`` stream filter selects phases.
* Workflow ``.of()`` factories build runnable workflows.
* ``OrchestratorAgent.assemble()`` works without subclassing.
"""

from __future__ import annotations

import pytest

from aimu.agents import Chain, Parallel, Router
from aimu.agents.agent import Agent
from aimu.models import ModelSpec, StreamChunk, StreamingContentType, resolve_model_string
from aimu.tools import ToolSignatureError, builtin, tool
from helpers import MockModelClient


# ---------------------------------------------------------------------------
# Model strings + ModelSpec
# ---------------------------------------------------------------------------


def test_resolve_model_string_known_id():
    """resolve_model_string finds an enum member by provider:id."""
    from aimu.models import HAS_ANTHROPIC

    if not HAS_ANTHROPIC:
        pytest.skip("anthropic not installed")
    m = resolve_model_string("anthropic:claude-sonnet-4-6")
    assert m.value == "claude-sonnet-4-6"
    assert m.supports_tools


def test_resolve_model_string_unknown_provider_lists_options():
    with pytest.raises(ValueError, match="Unknown provider"):
        resolve_model_string("nonexistent:foo")


def test_resolve_model_string_requires_colon():
    with pytest.raises(ValueError, match="provider:model_id"):
        resolve_model_string("just-an-id")


def test_modelspec_holds_capabilities():
    spec = ModelSpec("x", tools=True, thinking=True, vision=False)
    assert spec.id == "x"
    assert spec.tools and spec.thinking
    assert not spec.vision


def test_modelspec_equality_by_id_only():
    """ModelSpec equality uses ``id`` so it stays hashable even with a dict field."""
    assert ModelSpec("a", tools=True) == ModelSpec("a")
    assert ModelSpec("a") != ModelSpec("b")


def test_model_enum_value_is_string():
    """``Model.X.value`` returns the id string, preserving back-compat."""
    from aimu.models import AnthropicModel

    assert AnthropicModel.CLAUDE_SONNET_4_6.value == "claude-sonnet-4-6"
    assert isinstance(AnthropicModel.CLAUDE_SONNET_4_6.value, str)


def test_model_enum_capability_attrs_accessible():
    """Each member exposes supports_tools / supports_thinking / supports_vision."""
    from aimu.models import AnthropicModel, OllamaModel

    assert AnthropicModel.CLAUDE_SONNET_4_6.supports_tools is True
    assert AnthropicModel.CLAUDE_SONNET_4_6.supports_vision is True
    assert OllamaModel.QWEN_3_8B.supports_thinking is True


# ---------------------------------------------------------------------------
# StreamChunk unification + helpers
# ---------------------------------------------------------------------------


def test_streamchunk_has_agent_and_iteration_defaults():
    c = StreamChunk(StreamingContentType.GENERATING, "hi")
    assert c.agent is None
    assert c.iteration == 0


def test_streamchunk_is_text_helpers():
    text = StreamChunk(StreamingContentType.GENERATING, "hi")
    thinking = StreamChunk(StreamingContentType.THINKING, "hmm")
    tool_call = StreamChunk(StreamingContentType.TOOL_CALLING, {"name": "x", "response": "y"})
    assert text.is_text() and not text.is_tool_call()
    assert thinking.is_text()
    assert tool_call.is_tool_call() and not tool_call.is_text()


def test_agentchunk_is_alias_for_streamchunk():
    """AgentChunk is back-compat alias for StreamChunk."""
    from aimu.agents import AgentChunk

    assert AgentChunk is StreamChunk


def test_chainchunk_is_alias_for_streamchunk():
    from aimu.agents import ChainChunk

    assert ChainChunk is StreamChunk


# ---------------------------------------------------------------------------
# system_message immutability + reset()
# ---------------------------------------------------------------------------


def test_system_message_locks_after_first_chat():
    client = MockModelClient(["hello"])
    client.system_message = "v1"
    client.chat("hi")
    with pytest.raises(RuntimeError, match="immutable"):
        client.system_message = "v2"


def test_reset_unlocks_system_message():
    client = MockModelClient(["hello", "world"])
    client.system_message = "v1"
    client.chat("hi")
    client.reset()
    # Now mutable again. After chat resets messages, reset() restores system_message in messages on next chat.
    client.system_message = "v2"
    assert client.system_message == "v2"


def test_reset_preserves_system_message_by_default():
    client = MockModelClient(["hello"])
    client.system_message = "preserved"
    client.chat("hi")
    client.reset()
    assert client.system_message == "preserved"


def test_reset_can_clear_system_message():
    client = MockModelClient(["hello"])
    client.system_message = "v1"
    client.chat("hi")
    client.reset(system_message=None)
    assert client.system_message is None


# ---------------------------------------------------------------------------
# @tool validation
# ---------------------------------------------------------------------------


def test_tool_rejects_varargs():
    with pytest.raises(ToolSignatureError, match="variadic"):

        @tool
        def f(*args):
            """Bad."""
            return args


def test_tool_rejects_kwargs():
    with pytest.raises(ToolSignatureError, match="variadic"):

        @tool
        def f(**kwargs):
            """Bad."""
            return kwargs


def test_tool_rejects_param_without_hint_or_default():
    with pytest.raises(ToolSignatureError, match="no type hint"):

        @tool
        def f(x):
            """Bad."""
            return x


def test_tool_accepts_param_with_default_no_hint():
    @tool
    def f(x=1):
        """OK."""
        return x

    spec = f.__tool_spec__
    assert spec["function"]["parameters"]["properties"]["x"]["type"] == "string"
    assert "x" not in spec["function"]["parameters"]["required"]


def test_tool_supports_optional():
    from typing import Optional

    @tool
    def f(name: Optional[str] = None) -> str:
        """OK."""
        return name or ""

    spec = f.__tool_spec__
    assert spec["function"]["parameters"]["properties"]["name"]["type"] == "string"


def test_tool_supports_pipe_none():
    @tool
    def f(value: int | None = None) -> str:
        """OK."""
        return str(value)

    spec = f.__tool_spec__
    assert spec["function"]["parameters"]["properties"]["value"]["type"] == "integer"


def test_tool_maps_list_and_dict_generics():
    @tool
    def f(items: list[str], meta: dict[str, int]) -> str:
        """OK."""
        return ""

    props = f.__tool_spec__["function"]["parameters"]["properties"]
    assert props["items"]["type"] == "array"
    assert props["meta"]["type"] == "object"


# ---------------------------------------------------------------------------
# include= stream filter
# ---------------------------------------------------------------------------


def test_chat_stream_include_filters_phases():
    """include=['generating'] drops THINKING and TOOL_CALLING chunks."""

    class _MultiPhaseClient(MockModelClient):
        def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None):
            if not stream:
                return super()._chat(user_message, generate_kwargs, use_tools, stream, images)

            def _gen():
                yield StreamChunk(StreamingContentType.THINKING, "thinking")
                yield StreamChunk(StreamingContentType.GENERATING, "text")
                self.messages.append({"role": "user", "content": user_message})
                self.messages.append({"role": "assistant", "content": "text"})

            return _gen()

    client = _MultiPhaseClient(["text"])
    chunks = list(client.chat("hi", stream=True, include=["generating"]))
    assert all(c.phase == StreamingContentType.GENERATING for c in chunks)


def test_chat_stream_include_thinking_only():
    class _MultiPhaseClient(MockModelClient):
        def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None):
            if not stream:
                return super()._chat(user_message, generate_kwargs, use_tools, stream, images)

            def _gen():
                yield StreamChunk(StreamingContentType.THINKING, "thinking")
                yield StreamChunk(StreamingContentType.GENERATING, "text")
                self.messages.append({"role": "user", "content": user_message})

            return _gen()

    client = _MultiPhaseClient(["x"])
    chunks = list(client.chat("hi", stream=True, include=["thinking"]))
    assert all(c.phase == StreamingContentType.THINKING for c in chunks)


# ---------------------------------------------------------------------------
# builtin grouping
# ---------------------------------------------------------------------------


def test_builtin_web_group_contains_expected_tools():
    names = {t.__name__ for t in builtin.web}
    assert names == {"get_weather", "get_webpage", "search", "wikipedia"}


def test_builtin_fs_group_contains_expected_tools():
    names = {t.__name__ for t in builtin.fs}
    assert names == {"list_directory", "read_file"}


def test_builtin_all_tools_still_exposed():
    assert builtin.calculate in builtin.ALL_TOOLS


# ---------------------------------------------------------------------------
# Workflow .of() factories
# ---------------------------------------------------------------------------


def test_chain_of_builds_runnable_chain():
    client = MockModelClient(["first", "second"])
    chain = Chain.of(client, ["Step 1 prompt.", "Step 2 prompt."])
    assert len(chain.agents) == 2
    result = chain.run("task")
    assert result == "second"


def test_router_of_dispatches_to_handler():
    classifier = MockModelClient(["code"])
    coder = MockModelClient(["wrote code"])
    handler = Agent(coder, name="coder")

    router = Router.of(classifier, "Reply with 'code'.", handlers={"code": handler})
    result = router.run("task")
    assert result == "wrote code"


def test_parallel_of_without_aggregator_joins_workers():
    worker_a = MockModelClient(["A output"])
    worker_b = MockModelClient(["B output"])

    # Build a Parallel by hand to use the per-worker mock clients.
    parallel = Parallel(workers=[Agent(worker_a, name="a"), Agent(worker_b, name="b")])
    result = parallel.run("topic")
    assert "A output" in result
    assert "B output" in result


# ---------------------------------------------------------------------------
# OrchestratorAgent.assemble()
# ---------------------------------------------------------------------------


def test_orchestrator_assemble_registers_one_tool_per_worker():
    from aimu.agents import OrchestratorAgent

    client = MockModelClient(["orchestrator response"])
    workers = [
        Agent(MockModelClient(["w1"]), "Worker one role.", name="worker_one"),
        Agent(MockModelClient(["w2"]), "Worker two role.", name="worker_two"),
    ]
    orch = OrchestratorAgent.assemble(client, "Use workers.", workers=workers)

    assert {t.__name__ for t in client.tools} == {"worker_one", "worker_two"}
    # Each generated tool has __tool_spec__ from the @tool decorator.
    for fn in client.tools:
        assert hasattr(fn, "__tool_spec__")
        assert fn.__tool_spec__["function"]["name"] in {"worker_one", "worker_two"}

    assert orch.run("anything") == "orchestrator response"
