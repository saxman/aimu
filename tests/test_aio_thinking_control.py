"""Async mirror of tests/test_thinking_control.py. Resolution logic is shared through
_ChatStateMixin, so this file covers only the async surface and async provider paths."""

from __future__ import annotations

import pytest

from aimu.models._internal.thinking import THINKING_KWARG, ResolvedThinking


def _fake_async_client(model, recorder):
    from aimu.aio._base import AsyncBaseModelClient

    class _Fake(AsyncBaseModelClient):
        MODELS = None

        def __init__(self):
            self.model = model
            self.model_kwargs = None
            self._system_message = None
            self.default_generate_kwargs = {}
            self.messages = []
            self.tools = []
            self.last_thinking = ""
            self.last_usage = None
            self.last_output_truncated = False
            self.last_structured = None

        def _update_generate_kwargs(self, generate_kwargs=None):
            return dict(generate_kwargs or {})

        async def _chat(self, user_message=None, generate_kwargs=None, **kw):
            recorder.append(generate_kwargs)
            return "ok"

        async def _generate(self, prompt, generate_kwargs=None, **kw):
            recorder.append(generate_kwargs)
            return "ok"

    return _Fake()


class _Model:
    def __init__(self, value="m", thinking=True, levels=True, optional=True):
        self.value = value
        self.supports_thinking = thinking
        self.thinking_levels = levels
        self.thinking_optional = optional
        self.generation_kwargs = {"temperature": 1.0}
        self.nonthinking_generation_kwargs = {}


async def test_async_chat_forwards_a_resolved_request():
    seen: list = []
    client = _fake_async_client(_Model(), seen)

    await client.chat("hi", thinking="medium")

    assert seen[0][THINKING_KWARG] == ResolvedThinking(enabled=True, level="medium")


async def test_async_generate_forwards_a_resolved_request():
    seen: list = []
    client = _fake_async_client(_Model(), seen)

    await client.generate("hi", thinking=False)

    assert seen[0][THINKING_KWARG] == ResolvedThinking(enabled=False, level=None)


async def test_async_omitting_thinking_injects_nothing():
    seen: list = []
    client = _fake_async_client(_Model(), seen)

    await client.chat("hi")

    assert THINKING_KWARG not in (seen[0] or {})


async def test_async_invalid_value_raises():
    seen: list = []
    client = _fake_async_client(_Model(), seen)

    with pytest.raises(ValueError):
        await client.chat("hi", thinking="xhigh")

    assert seen == []
