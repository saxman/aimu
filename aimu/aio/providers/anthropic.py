"""Async Anthropic client.

Mirrors ``aimu.models.providers.anthropic.AnthropicClient`` using
``anthropic.AsyncAnthropic``. Reuses the sync client's pure format adapters
(message/tool/id conversion) via composition — they don't touch I/O.
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from typing import Any, AsyncIterator, Optional, Union

import anthropic
from dotenv import load_dotenv

from aimu.models.providers.anthropic import AnthropicClient as _SyncAnthropicClient
from aimu.models.providers.anthropic import AnthropicModel
from aimu.models.base import Model, StreamChunk, StreamingContentType, classproperty

from .._base import AsyncBaseModelClient

logger = logging.getLogger(__name__)

_DEFAULT_THINKING_BUDGET = 8000


class AsyncAnthropicClient(AsyncBaseModelClient):
    """Async client for Anthropic Claude models using ``anthropic.AsyncAnthropic``.

    ``self.messages`` is stored in OpenAI format; conversion to Anthropic format
    happens at request time via the same pure helpers as the sync client.
    """

    MODELS = AnthropicModel

    DEFAULT_GENERATE_KWARGS = {
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    _UNSUPPORTED_KWARGS = frozenset({"max_new_tokens", "do_sample", "num_return_sequences"})

    def __init__(
        self,
        model: AnthropicModel,
        model_kwargs: Optional[dict] = None,
        system_message: Optional[str] = None,
    ):
        super().__init__(model, model_kwargs, system_message)
        self.default_generate_kwargs = self.DEFAULT_GENERATE_KWARGS.copy()
        load_dotenv()
        self._client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_thinking]

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_tools]

    @classproperty
    def VISION_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_vision]

    @classproperty
    def AUDIO_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_audio]

    # Pure helpers reused from sync client — no I/O involved.
    _update_generate_kwargs = _SyncAnthropicClient._update_generate_kwargs
    _thinking_kwargs = _SyncAnthropicClient._thinking_kwargs
    _openai_messages_to_anthropic = _SyncAnthropicClient._openai_messages_to_anthropic
    _openai_tools_to_anthropic = _SyncAnthropicClient._openai_tools_to_anthropic
    _patch_tool_ids = _SyncAnthropicClient._patch_tool_ids

    async def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)
        generate_kwargs = self._thinking_kwargs(generate_kwargs)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs, images=images, audio=audio)

        response = await self._client.messages.create(
            model=self.model.value,
            messages=[{"role": "user", "content": _SyncAnthropicClient._generate_content(prompt, images, audio=audio)}],
            **generate_kwargs,
        )

        self.last_thinking = ""
        content = ""
        for block in response.content:
            if block.type == "thinking":
                self.last_thinking = block.thinking
            elif block.type == "text":
                content = block.text
        return content

    async def _generate_streamed(
        self,
        prompt: str,
        generate_kwargs: dict[str, Any],
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> AsyncIterator[StreamChunk]:
        self.last_thinking = ""

        async with self._client.messages.stream(
            model=self.model.value,
            messages=[{"role": "user", "content": _SyncAnthropicClient._generate_content(prompt, images, audio=audio)}],
            **generate_kwargs,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "thinking_delta":
                        self.last_thinking += delta.thinking
                        yield StreamChunk(StreamingContentType.THINKING, delta.thinking)
                    elif delta.type == "text_delta":
                        yield StreamChunk(StreamingContentType.GENERATING, delta.text)

    async def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        generate_kwargs, tools = await self._chat_setup(user_message, generate_kwargs, use_tools, images=images, audio=audio)
        generate_kwargs = self._thinking_kwargs(generate_kwargs)

        if stream:
            return self._chat_streamed(generate_kwargs, tools)

        system_str, ant_messages = self._openai_messages_to_anthropic(self.messages)
        ant_tools = self._openai_tools_to_anthropic(tools) if tools else anthropic.NOT_GIVEN

        response = await self._client.messages.create(
            model=self.model.value,
            system=system_str if system_str else anthropic.NOT_GIVEN,
            messages=ant_messages,
            tools=ant_tools,
            **generate_kwargs,
        )

        self.last_thinking = ""
        tool_use_blocks = []
        text_content = ""

        for block in response.content:
            if block.type == "thinking":
                self.last_thinking = block.thinking
            elif block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        if tool_use_blocks:
            tool_calls = [{"name": b.name, "arguments": b.input} for b in tool_use_blocks]
            msgs_before = len(self.messages)
            await self._handle_tool_calls(tool_calls)
            self._patch_tool_ids(msgs_before, tool_use_blocks)

            if self.last_thinking:
                self.messages[msgs_before]["thinking"] = self.last_thinking

            first_call_thinking = self.last_thinking

            system_str, ant_messages = self._openai_messages_to_anthropic(self.messages)
            response = await self._client.messages.create(
                model=self.model.value,
                system=system_str if system_str else anthropic.NOT_GIVEN,
                messages=ant_messages,
                tools=ant_tools,
                **generate_kwargs,
            )

            self.last_thinking = ""
            text_content = ""
            for block in response.content:
                if block.type == "thinking":
                    self.last_thinking = block.thinking
                elif block.type == "text":
                    text_content = block.text

            if not self.last_thinking:
                self.last_thinking = first_call_thinking

        assistant_msg: dict = {"role": "assistant", "content": text_content}
        if self.last_thinking:
            assistant_msg["thinking"] = self.last_thinking
        self.messages.append(assistant_msg)
        return text_content

    async def _chat_streamed(self, generate_kwargs: dict[str, Any], tools: list) -> AsyncIterator[StreamChunk]:
        system_str, ant_messages = self._openai_messages_to_anthropic(self.messages)
        ant_tools = self._openai_tools_to_anthropic(tools) if tools else anthropic.NOT_GIVEN

        tool_use_acc: list[dict] = []
        first_pass_chunks: list[StreamChunk] = []
        self.last_thinking = ""

        async with self._client.messages.stream(
            model=self.model.value,
            system=system_str if system_str else anthropic.NOT_GIVEN,
            messages=ant_messages,
            tools=ant_tools,
            **generate_kwargs,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        tool_use_acc.append({"id": block.id, "name": block.name, "input_json": ""})
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "thinking_delta":
                        self.last_thinking += delta.thinking
                        first_pass_chunks.append(StreamChunk(StreamingContentType.THINKING, delta.thinking))
                    elif delta.type == "text_delta":
                        first_pass_chunks.append(StreamChunk(StreamingContentType.GENERATING, delta.text))
                    elif delta.type == "input_json_delta" and tool_use_acc:
                        tool_use_acc[-1]["input_json"] += delta.partial_json

        if not tool_use_acc:
            full_content = ""
            for sc in first_pass_chunks:
                if sc.phase == StreamingContentType.GENERATING:
                    full_content += sc.content
                yield sc
            self.messages.append({"role": "assistant", "content": full_content})
            return

        parsed_blocks = [
            SimpleNamespace(
                id=tub["id"],
                name=tub["name"],
                input=json.loads(tub["input_json"]) if tub["input_json"] else {},
            )
            for tub in tool_use_acc
        ]

        tool_calls = [{"name": b.name, "arguments": b.input} for b in parsed_blocks]
        msgs_before = len(self.messages)
        async for chunk in self._handle_tool_calls_streamed(tool_calls):
            yield chunk
        self._patch_tool_ids(msgs_before, parsed_blocks)

        if self.last_thinking:
            self.messages[msgs_before]["thinking"] = self.last_thinking

        system_str, ant_messages = self._openai_messages_to_anthropic(self.messages)
        full_content = ""
        first_pass_thinking = self.last_thinking
        self.last_thinking = ""

        async with self._client.messages.stream(
            model=self.model.value,
            system=system_str if system_str else anthropic.NOT_GIVEN,
            messages=ant_messages,
            tools=ant_tools,
            **generate_kwargs,
        ) as stream2:
            async for event in stream2:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "thinking_delta":
                        self.last_thinking += delta.thinking
                        yield StreamChunk(StreamingContentType.THINKING, delta.thinking)
                    elif delta.type == "text_delta":
                        full_content += delta.text
                        yield StreamChunk(StreamingContentType.GENERATING, delta.text)

        if not self.last_thinking:
            self.last_thinking = first_pass_thinking

        assistant_msg: dict = {"role": "assistant", "content": full_content}
        if self.last_thinking:
            assistant_msg["thinking"] = self.last_thinking
        self.messages.append(assistant_msg)
