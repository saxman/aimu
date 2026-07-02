"""Tests for message provenance tagging and inert-key request hygiene.

Covers:
  - ``strip_inert_keys`` removes inert metadata keys, preserves the rest, leaves clean dicts by identity.
  - The sync ``Agent`` loop tags injected continuation / final-answer turns; user turns stay untagged.
  - OpenAI-compat and Ollama request paths exclude inert keys from the outgoing ``messages=`` payload.

Mock-only; no backend required. The async ``Agent`` mirror lives in ``tests/test_aio_agents.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aimu.agents import Agent
from aimu.models import BaseModelClient
from aimu.models._internal.message_meta import (
    INERT_MESSAGE_KEYS,
    PROVENANCE_CONTINUATION,
    PROVENANCE_FINAL_ANSWER,
    PROVENANCE_KEY,
    PROVENANCE_PROACTIVE,
    strip_inert_keys,
)


# --------------------------------------------------------------------------------------
# strip_inert_keys
# --------------------------------------------------------------------------------------


def test_strip_removes_inert_keys_and_preserves_the_rest():
    messages = [
        {"role": "user", "content": "hi", PROVENANCE_KEY: PROVENANCE_CONTINUATION, "timestamp": "t"},
        {"role": "assistant", "content": "yo", "thinking": "hmm", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "name": "t", "content": "r", "tool_call_id": "1"},
    ]
    out = strip_inert_keys(messages)

    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1] == {"role": "assistant", "content": "yo", "tool_calls": [{"id": "1"}]}
    # A dict without inert keys passes through by identity (no needless copy on the hot path).
    assert out[2] is messages[2]


def test_strip_does_not_mutate_the_input():
    messages = [{"role": "user", "content": "hi", PROVENANCE_KEY: PROVENANCE_PROACTIVE}]
    strip_inert_keys(messages)
    assert messages[0][PROVENANCE_KEY] == PROVENANCE_PROACTIVE


def test_inert_key_set_contents():
    assert INERT_MESSAGE_KEYS == frozenset({"thinking", PROVENANCE_KEY, "timestamp"})


# --------------------------------------------------------------------------------------
# Agent loop tagging
# --------------------------------------------------------------------------------------


class _LoopClient(BaseModelClient):
    """Fake client with precise control over when a turn "calls tools".

    ``tool_turns`` is how many leading ``chat()`` calls end in a pending tool result (so the
    agent loops); after that a turn ends cleanly. A final-answer turn (``tools == []``) always
    ends cleanly.
    """

    def __init__(self, tool_turns: int):
        self.model = MagicMock()
        self.model.supports_tools = True
        self.model.supports_thinking = False
        self.model_kwargs = None
        self._system_message = None
        self.default_generate_kwargs = {}
        self.messages: list[dict] = []
        self.tools: list = []
        self.last_thinking = ""
        self.last_usage = None
        self.tool_context_deps = None
        self.tool_approval = None
        self._tool_turns = tool_turns
        self._turn = 0

    def _update_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}

    def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None):
        self.messages.append({"role": "user", "content": user_message})
        self._turn += 1
        pending_tool = use_tools and self._turn <= self._tool_turns
        if pending_tool:
            self.messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"type": "function", "function": {"name": "t", "arguments": {}}, "id": "x"}],
                }
            )
            self.messages.append({"role": "tool", "name": "t", "content": "r", "tool_call_id": "x"})
            return ""
        self.messages.append({"role": "assistant", "content": "done"})
        return "done"

    def _generate(self, prompt, generate_kwargs=None, stream=False, images=None, audio=None):
        return self._chat(prompt, generate_kwargs)


def test_continuation_turn_tagged_user_turn_untagged():
    client = _LoopClient(tool_turns=1)  # first turn calls tools, then a continuation finishes
    agent = Agent(client, tools=[])
    agent.run("real question")

    roles = [(m.get("role"), m.get(PROVENANCE_KEY)) for m in client.messages]
    # index 0: genuine user turn -> untagged
    assert client.messages[0]["content"] == "real question"
    assert client.messages[0].get(PROVENANCE_KEY) is None
    # a continuation user turn exists and is tagged
    continuation_turns = [m for m in client.messages if m.get(PROVENANCE_KEY) == PROVENANCE_CONTINUATION]
    assert len(continuation_turns) == 1, roles


def test_final_answer_turn_tagged():
    # Always calls tools + a low cap forces the final-answer wrap-up.
    client = _LoopClient(tool_turns=99)
    agent = Agent(client, tools=[], max_iterations=2, final_answer_prompt="wrap up now")
    agent.run("real question")

    tags = [m.get(PROVENANCE_KEY) for m in client.messages]
    assert PROVENANCE_CONTINUATION in tags
    assert tags.count(PROVENANCE_FINAL_ANSWER) == 1, tags


def test_no_continuation_no_tags():
    client = _LoopClient(tool_turns=0)  # first turn finishes cleanly, no loop
    agent = Agent(client, tools=[])
    agent.run("real question")

    assert all(m.get(PROVENANCE_KEY) is None for m in client.messages)


# --------------------------------------------------------------------------------------
# Request-time inert-key stripping (provider adapters)
# --------------------------------------------------------------------------------------


def test_openai_compat_excludes_inert_keys_from_request():
    pytest.importorskip("openai")
    from aimu.models.providers.openai_compat import LMStudioOpenAIClient, LMStudioOpenAIModel

    client = LMStudioOpenAIClient(list(LMStudioOpenAIModel)[0])
    client.messages = [
        {"role": "user", "content": "hi", PROVENANCE_KEY: PROVENANCE_CONTINUATION, "thinking": "z", "timestamp": "t"},
    ]

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        msg = MagicMock()
        msg.tool_calls = None
        msg.content = "ok"
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        return resp

    client._client = MagicMock()
    client._client.chat.completions.create = fake_create

    client._chat("next", use_tools=False)

    sent = captured["messages"]
    for message in sent:
        assert INERT_MESSAGE_KEYS.isdisjoint(message), message


def test_ollama_adapter_strips_inert_keys():
    from aimu.models._internal.image_input import _adapt_messages_for_ollama

    messages = [
        {"role": "user", "content": "hi", PROVENANCE_KEY: PROVENANCE_PROACTIVE, "timestamp": "t"},
        {"role": "assistant", "content": "yo", "thinking": "hmm"},
    ]
    out = _adapt_messages_for_ollama(messages)
    for message in out:
        assert INERT_MESSAGE_KEYS.isdisjoint(message), message
    assert out[0]["content"] == "hi"
    assert out[1]["content"] == "yo"
