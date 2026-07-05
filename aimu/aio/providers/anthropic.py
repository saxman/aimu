"""Async Anthropic client.

Mirrors ``aimu.models.providers.anthropic.AnthropicClient`` using
``anthropic.AsyncAnthropic``. Reuses the sync client's pure format adapters
(message/tool/id conversion) via composition; they don't touch I/O.
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
from aimu.models._internal.sdk_config import sdk_client_kwargs
from aimu.models._internal.usage import usage_from_anthropic
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
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        cache_prompt: bool = False,
    ):
        super().__init__(model, model_kwargs, system_message)
        self.default_generate_kwargs = self.DEFAULT_GENERATE_KWARGS.copy()
        # Opt-in prompt caching; the cache-marking format adapters are inherited from the
        # sync client via composition, so only the flag needs setting here.
        self.cache_prompt = cache_prompt
        load_dotenv()
        self._client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"), **sdk_client_kwargs(timeout, max_retries)
        )

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

    @classproperty
    def STRUCTURED_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_structured_output]

    async def _structured_call(
        self, system_str, ant_messages: list, generate_kwargs: dict, response_format: dict
    ) -> str:
        """Async forced-tool structured output (see sync ``AnthropicClient._structured_call``)."""
        import re

        name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(response_format.get("title", "Response")))[:64] or "Response"
        tool = {"name": name, "description": f"Emit the answer as a {name} object.", "input_schema": response_format}
        response = await self._client.messages.create(
            model=self.model.value,
            system=system_str,
            messages=ant_messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": name},
            **generate_kwargs,
        )
        self.last_usage = usage_from_anthropic(response)
        for block in response.content:
            if block.type == "tool_use":
                return json.dumps(block.input)
        return "{}"

    async def _structured_call_streamed(
        self,
        system_str,
        ant_messages: list,
        generate_kwargs: dict,
        response_format: dict,
        *,
        append_message: bool,
    ) -> AsyncIterator[StreamChunk]:
        """Async streamed forced-tool structured output (see sync ``_structured_call_streamed``).

        Streams the tool-input JSON (``GENERATING`` from ``input_json_delta``); no ``THINKING``.
        """
        import re

        self.last_thinking = ""
        self.last_usage = None
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(response_format.get("title", "Response")))[:64] or "Response"
        tool = {"name": name, "description": f"Emit the answer as a {name} object.", "input_schema": response_format}
        async with self._client.messages.stream(
            model=self.model.value,
            system=system_str,
            messages=ant_messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": name},
            **generate_kwargs,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "input_json_delta":
                    yield StreamChunk(StreamingContentType.GENERATING, event.delta.partial_json)
            final = await stream.get_final_message()
        self.last_usage = usage_from_anthropic(final)
        text = "{}"
        for block in final.content:
            if block.type == "tool_use":
                text = json.dumps(block.input)
                break
        if append_message:
            self.messages.append({"role": "assistant", "content": text})

    # Pure helpers reused from sync client; no I/O involved.
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
        response_format: Optional[dict] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        if response_format is not None:
            content = _SyncAnthropicClient._generate_content(prompt, images, audio=audio)
            messages = [{"role": "user", "content": content}]
            if stream:
                return self._structured_call_streamed(
                    anthropic.NOT_GIVEN, messages, generate_kwargs, response_format, append_message=False
                )
            return await self._structured_call(anthropic.NOT_GIVEN, messages, generate_kwargs, response_format)

        generate_kwargs = self._thinking_kwargs(generate_kwargs)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs, images=images, audio=audio)

        response = await self._client.messages.create(
            model=self.model.value,
            messages=[{"role": "user", "content": _SyncAnthropicClient._generate_content(prompt, images, audio=audio)}],
            **generate_kwargs,
        )
        self.last_usage = usage_from_anthropic(response)

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
        self.last_usage = None

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
            self.last_usage = usage_from_anthropic(await stream.get_final_message())

    async def _chat(
        self,
        user_message: Optional[str] = None,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        if response_format is not None and use_tools and self.tools:
            raise ValueError(
                "Anthropic structured output uses a forced tool, which is incompatible with active "
                "action tools. Drop tools (or use_tools=False), or use a provider whose response_format "
                "composes with tools (e.g. OpenAI)."
            )

        generate_kwargs, tools = await self._chat_setup(
            user_message, generate_kwargs, use_tools, images=images, audio=audio
        )

        if response_format is not None:
            system_str, ant_messages = self._openai_messages_to_anthropic(self.messages)
            system = system_str if system_str else anthropic.NOT_GIVEN
            if stream:
                return self._structured_call_streamed(
                    system, ant_messages, generate_kwargs, response_format, append_message=True
                )
            text = await self._structured_call(system, ant_messages, generate_kwargs, response_format)
            self.messages.append({"role": "assistant", "content": text})
            return text

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

        self.last_usage = usage_from_anthropic(response)

        # Single turn: if the model called tools, execute them and return. The model's response
        # to the tool results comes on the next chat() call (the loop lives in Agent).
        if tool_use_blocks:
            tool_calls = [{"name": b.name, "arguments": b.input} for b in tool_use_blocks]
            msgs_before = len(self.messages)
            await self._handle_tool_calls(tool_calls, content=text_content)
            self._patch_tool_ids(msgs_before, tool_use_blocks)
            if self.last_thinking:
                self.messages[msgs_before]["thinking"] = self.last_thinking
            return text_content

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
        self.last_usage = None

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

        self.last_usage = usage_from_anthropic(await stream.get_final_message())

        if not tool_use_acc:
            full_content = ""
            for sc in first_pass_chunks:
                if sc.phase == StreamingContentType.GENERATING:
                    full_content += sc.content
                yield sc
            assistant_msg: dict = {"role": "assistant", "content": full_content}
            if self.last_thinking:
                assistant_msg["thinking"] = self.last_thinking
            self.messages.append(assistant_msg)
            return

        # Single turn: parse accumulated JSON, dispatch, yield TOOL_CALLING chunks, and return.
        # The model's response to the tool results comes on the next chat() call (loop in Agent).
        parsed_blocks = [
            SimpleNamespace(
                id=tub["id"],
                name=tub["name"],
                input=json.loads(tub["input_json"]) if tub["input_json"] else {},
            )
            for tub in tool_use_acc
        ]

        tool_calls = [{"name": b.name, "arguments": b.input} for b in parsed_blocks]
        # Yield any prose/thinking emitted alongside the tool call and store the prose as content.
        full_content = ""
        for sc in first_pass_chunks:
            if sc.phase == StreamingContentType.GENERATING:
                full_content += sc.content
            yield sc
        msgs_before = len(self.messages)
        async for chunk in self._handle_tool_calls_streamed(tool_calls, content=full_content):
            yield chunk
        self._patch_tool_ids(msgs_before, parsed_blocks)
        if self.last_thinking:
            self.messages[msgs_before]["thinking"] = self.last_thinking
