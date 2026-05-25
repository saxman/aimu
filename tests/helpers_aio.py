"""Async test helpers — mock async model client.

Mirrors :class:`helpers.MockModelClient` but with async ``_chat``/``_generate``.
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import MagicMock

from aimu.aio._base import AsyncBaseModelClient
from aimu.models import StreamChunk, StreamingContentType


class MockAsyncModelClient(AsyncBaseModelClient):
    """An async ModelClient stub whose chat() responses come from a fixed queue.

    ``responses`` entries follow the same convention as :class:`helpers.MockModelClient`:
    a plain ``str`` is a direct response; ``"tool"`` simulates one tool-call round
    by appending the relevant messages and consuming a follow-up response.
    """

    def __init__(self, responses: list):
        self.model = MagicMock()
        self.model.supports_tools = True
        self.model.supports_thinking = False
        self.model.supports_vision = False
        self.model_kwargs = None
        self._system_message = None
        self._system_message_locked = False
        self.default_generate_kwargs = {}
        self.messages = []
        self.mcp_client = None
        self.tools = []
        self.last_thinking = ""
        self.concurrent_tool_calls = False
        self._responses = list(responses)
        self._call_count = 0

    def _update_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}

    async def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None):
        if stream:
            return self._chat_streamed(user_message, generate_kwargs, use_tools, images=images)
        if images:
            from aimu.models._images import _build_user_content_blocks

            self.messages.append({"role": "user", "content": _build_user_content_blocks(user_message, images)})
        else:
            self.messages.append({"role": "user", "content": user_message})
        self._system_message_locked = True
        response = self._responses[self._call_count]
        self._call_count += 1

        if response == "tool":
            self.messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"type": "function", "function": {"name": "mock_tool", "arguments": {}}, "id": "x"}],
                }
            )
            self.messages.append({"role": "tool", "name": "mock_tool", "content": "tool result", "tool_call_id": "x"})
            text = self._responses[self._call_count]
            self._call_count += 1
            self.messages.append({"role": "assistant", "content": text})
            return text
        self.messages.append({"role": "assistant", "content": response})
        return response

    async def _chat_streamed(self, user_message, generate_kwargs=None, use_tools=True, images=None) -> AsyncIterator[StreamChunk]:
        response = await self._chat(user_message, generate_kwargs, use_tools, images=images)
        yield StreamChunk(StreamingContentType.GENERATING, response)

    async def _generate(self, prompt, generate_kwargs=None, stream=False):
        if stream:
            return self._generate_streamed(prompt, generate_kwargs)
        return await self._chat(prompt, generate_kwargs)

    async def _generate_streamed(self, prompt, generate_kwargs=None) -> AsyncIterator[StreamChunk]:
        text = await self._generate(prompt, generate_kwargs)
        yield StreamChunk(StreamingContentType.GENERATING, text)
