"""Mock-only tests for FallbackClient (P0-B): ordered failover across BaseModelClients.

Uses lightweight stub clients (duck-typed to the BaseModelClient surface FallbackClient
touches) so no backend is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aimu.models import BaseModelClient
from aimu.models.fallback import FallbackClient, FallbackExhaustedError


def _fake_model():
    model = MagicMock()
    model.supports_tools = True
    model.supports_thinking = False
    model.supports_vision = False
    model.supports_audio = False
    model.supports_structured_output = False
    return model


class StubClient:
    """Minimal client: succeeds with canned replies (mutating its own history like a real
    client) or raises ``error`` on every call."""

    def __init__(self, *, replies=None, error=None):
        self.model = _fake_model()
        self.messages: list[dict] = []
        self.tools: list = []
        self._system_message = None
        self.tool_context_deps = None
        self.last_thinking = ""
        self.last_usage = None
        self.concurrent_tool_calls = False
        self._replies = list(replies or [])
        self._error = error
        self.chat_calls = 0
        self.generate_calls = 0
        self.seen_kwargs: dict = {}

    @property
    def system_message(self):
        return self._system_message

    @system_message.setter
    def system_message(self, value):
        self._system_message = value

    def reset(self, system_message="__keep__"):
        self.messages = []
        if system_message != "__keep__":
            self._system_message = system_message
        self.last_thinking = ""
        self.last_usage = None

    def _reply(self):
        return self._replies.pop(0) if self._replies else "ok"

    def chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, **kwargs):
        self.chat_calls += 1
        self.seen_kwargs = kwargs
        if stream:
            return self._chat_stream(user_message)
        if self._error is not None:
            raise self._error
        if not self.messages and self._system_message:
            self.messages.append({"role": "system", "content": self._system_message})
        self.messages.append({"role": "user", "content": user_message})
        reply = self._reply()
        self.messages.append({"role": "assistant", "content": reply})
        self.last_usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return reply

    def _chat_stream(self, user_message):
        if self._error is not None:
            raise self._error
        if not self.messages and self._system_message:
            self.messages.append({"role": "system", "content": self._system_message})
        self.messages.append({"role": "user", "content": user_message})
        reply = self._reply()
        for token in reply.split():
            yield token
        self.messages.append({"role": "assistant", "content": reply})

    def generate(self, prompt, generate_kwargs=None, stream=False, **kwargs):
        self.generate_calls += 1
        if stream:
            return self._generate_stream()
        if self._error is not None:
            raise self._error
        return self._reply()

    def _generate_stream(self):
        if self._error is not None:
            raise self._error
        for token in self._reply().split():
            yield token


# ---------------------------------------------------------------------------
# Construction / contract
# ---------------------------------------------------------------------------


def test_fallback_is_base_model_client():
    fb = FallbackClient([StubClient(), StubClient()])
    assert isinstance(fb, BaseModelClient)


def test_empty_client_list_raises():
    with pytest.raises(ValueError):
        FallbackClient([])


# ---------------------------------------------------------------------------
# chat() failover
# ---------------------------------------------------------------------------


def test_first_healthy_client_answers_and_others_untouched():
    a = StubClient(replies=["A"])
    b = StubClient(replies=["B"])
    fb = FallbackClient([a, b])

    assert fb.chat("hi") == "A"
    assert a.chat_calls == 1
    assert b.chat_calls == 0


def test_falls_back_to_next_on_error():
    a = StubClient(error=ConnectionError("down"))
    b = StubClient(replies=["B"])
    fb = FallbackClient([a, b])

    assert fb.chat("hi") == "B"
    assert a.chat_calls == 1
    assert b.chat_calls == 1


def test_all_clients_fail_raises_exhausted_with_chained_cause():
    last = TimeoutError("second timed out")
    fb = FallbackClient([StubClient(error=ConnectionError("first")), StubClient(error=last)])

    with pytest.raises(FallbackExhaustedError) as excinfo:
        fb.chat("hi")
    assert excinfo.value.__cause__ is last


def test_retry_on_narrows_which_errors_trigger_fallback():
    a = StubClient(error=ValueError("permanent"))
    b = StubClient(replies=["B"])
    fb = FallbackClient([a, b], retry_on=(TimeoutError,))

    # ValueError is not in retry_on -> propagates, second client never tried.
    with pytest.raises(ValueError, match="permanent"):
        fb.chat("hi")
    assert b.chat_calls == 0


def test_history_preserved_across_failover():
    a = StubClient(replies=["A1"])
    b = StubClient(replies=["B2"])
    fb = FallbackClient([a, b])

    assert fb.chat("turn one") == "A1"  # client A answers, history grows
    a._error = ConnectionError("A is now down")
    assert fb.chat("turn two") == "B2"  # fails over to B

    # B saw the full prior conversation, not just turn two.
    user_msgs = [m["content"] for m in b.messages if m["role"] == "user"]
    assert user_msgs == ["turn one", "turn two"]
    # Canonical history reflects the winning client.
    fb_user_msgs = [m["content"] for m in fb.messages if m["role"] == "user"]
    assert fb_user_msgs == ["turn one", "turn two"]


def test_system_message_propagates_to_clients():
    a = StubClient(replies=["A"])
    fb = FallbackClient([a], system_message="You are terse.")
    fb.chat("hi")
    assert a.messages[0] == {"role": "system", "content": "You are terse."}


def test_kwargs_forwarded_to_inner_chat():
    a = StubClient(replies=["A"])
    fb = FallbackClient([a])
    sentinel = object()
    fb.chat("hi", tools=[], schema=None, audio=sentinel if False else None)
    assert "tools" in a.seen_kwargs


def test_last_usage_adopted_from_winning_client():
    a = StubClient(error=ConnectionError("x"))
    b = StubClient(replies=["B"])
    fb = FallbackClient([a, b])
    fb.chat("hi")
    assert fb.last_usage == {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


# ---------------------------------------------------------------------------
# generate() failover
# ---------------------------------------------------------------------------


def test_generate_falls_back():
    a = StubClient(error=ConnectionError("down"))
    b = StubClient(replies=["G"])
    fb = FallbackClient([a, b])
    assert fb.generate("prompt") == "G"
    assert a.generate_calls == 1
    assert b.generate_calls == 1


def test_generate_all_fail_raises():
    fb = FallbackClient([StubClient(error=ConnectionError("a")), StubClient(error=ConnectionError("b"))])
    with pytest.raises(FallbackExhaustedError):
        fb.generate("prompt")


# ---------------------------------------------------------------------------
# streaming failover (only before the first chunk is yielded)
# ---------------------------------------------------------------------------


def test_streaming_falls_back_before_first_chunk():
    a = StubClient(error=ConnectionError("down"))
    b = StubClient(replies=["hello world"])
    fb = FallbackClient([a, b])

    chunks = list(fb.chat("hi", stream=True))
    assert chunks == ["hello", "world"]
    assert a.chat_calls == 1
    assert b.chat_calls == 1


def test_streaming_all_fail_raises():
    fb = FallbackClient([StubClient(error=ConnectionError("a")), StubClient(error=ConnectionError("b"))])
    with pytest.raises(FallbackExhaustedError):
        list(fb.chat("hi", stream=True))


# ---------------------------------------------------------------------------
# Composition: a FallbackClient is a BaseModelClient, so it drops into an Agent
# ---------------------------------------------------------------------------


def test_composes_as_agent_model_client():
    from helpers import MockModelClient

    from aimu.agents import Agent

    primary = StubClient(error=ConnectionError("primary down"))
    backup = MockModelClient(["the answer"])
    agent = Agent(FallbackClient([primary, backup]), "You are helpful.", max_iterations=2)

    assert agent.run("question") == "the answer"
    assert primary.chat_calls == 1  # primary was tried and failed over
