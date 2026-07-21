"""Async OpenAI-compatible clients: the base plus local-inference-server subclasses.

One base ``AsyncOpenAICompatClient`` that uses ``openai.AsyncOpenAI``, plus thin
subclasses for each *local* server (LM Studio, Ollama-OpenAI, vLLM, HF-Serve,
llama-server, SGLang) that supply the right base URL. Mirrors
``aimu.models.providers.openai_compat``. The cloud-brand clients live in their own
subpackages: ``aimu.aio.providers.openai`` and ``aimu.aio.providers.gemini``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional, Union

import openai

from aimu.models._internal.audio_input import _build_audio_content_blocks
from aimu.models._internal.image_input import _build_user_content_blocks
from aimu.models._internal.message_meta import strip_inert_keys
from aimu.models._internal.sdk_config import sdk_client_kwargs
from aimu.models._internal.usage import usage_from_openai
from aimu.models.providers._thinking import _ThinkingParser, _split_thinking
from aimu.models.base import Model, ModelConnectionError, StreamChunk, StreamingContentType, classproperty
from aimu.models.providers.openai_compat import (
    HFOpenAIModel,
    LlamaServerOpenAIModel,
    LMStudioOpenAIModel,
    OllamaOpenAIModel,
    OpenAICompatClient as _SyncOpenAICompatClient,
    SGLangOpenAIModel,
    VLLMOpenAIModel,
)

from .._base import AsyncBaseModelClient

logger = logging.getLogger(__name__)


async def _guarded_create(sdk_client, **kwargs):
    """Call the chat-completions endpoint, translating a server-unreachable failure into
    ``ModelConnectionError``. The SDK's ``APIConnectionError`` message is generic ("Connection
    error."); the specific cause (e.g. "Connection refused") is preserved on ``__cause__``."""
    try:
        return await sdk_client.chat.completions.create(**kwargs)
    except openai.APIConnectionError as exc:
        raise ModelConnectionError(str(exc)) from exc


async def _guard_stream(stream):
    """Yield from a streaming response, translating a mid-stream connection drop into
    ``ModelConnectionError`` (the connection can fail while consuming, not only on create)."""
    try:
        async for chunk in stream:
            yield chunk
    except openai.APIConnectionError as exc:
        raise ModelConnectionError(str(exc)) from exc


class AsyncOpenAICompatClient(AsyncBaseModelClient):
    """Async base for any OpenAI-compatible endpoint."""

    MODELS = Model

    DEFAULT_GENERATE_KWARGS = {
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    def __init__(
        self,
        model: Model,
        base_url: str,
        api_key: str = "not-needed",
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        super().__init__(model, model_kwargs, system_message)
        self.default_generate_kwargs = self.DEFAULT_GENERATE_KWARGS.copy()
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, **sdk_client_kwargs(timeout, max_retries))

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

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        if not generate_kwargs:
            return self.default_generate_kwargs.copy()
        return {**self.default_generate_kwargs, **generate_kwargs}

    async def _iter_stream(self, stream) -> AsyncIterator[StreamChunk]:
        self.last_thinking = ""
        self.last_usage = None
        parser = _ThinkingParser() if self.is_thinking_model else None

        async for chunk in _guard_stream(stream):
            if getattr(chunk, "usage", None):
                self.last_usage = usage_from_openai(chunk)
            if not chunk.choices:  # terminal usage chunk (empty choices) or keep-alive
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                self.last_thinking += reasoning
                yield StreamChunk(StreamingContentType.THINKING, reasoning)
            if delta.content is None:
                continue
            if parser:
                for phase, text in parser.feed(delta.content):
                    if phase == StreamingContentType.THINKING:
                        self.last_thinking += text
                        yield StreamChunk(StreamingContentType.THINKING, text)
                    else:
                        yield StreamChunk(StreamingContentType.GENERATING, text)
            else:
                yield StreamChunk(StreamingContentType.GENERATING, delta.content)

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
        generate_kwargs = _SyncOpenAICompatClient._with_response_format(generate_kwargs, response_format)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs, images=images, audio=audio)

        if audio:
            content_in = _build_audio_content_blocks(prompt, audio)
        elif images:
            content_in = _build_user_content_blocks(prompt, images)
        else:
            content_in = prompt
        response = await _guarded_create(
            self._client,
            model=self.model.value,
            messages=[{"role": "user", "content": content_in}],
            **generate_kwargs,
        )
        msg = response.choices[0].message
        self.last_usage = usage_from_openai(response)
        content = msg.content or ""

        self.last_thinking = ""
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            self.last_thinking = reasoning
        elif self.is_thinking_model:
            self.last_thinking, content = _split_thinking(content)

        return content

    async def _generate_streamed(
        self,
        prompt: str,
        generate_kwargs: dict[str, Any],
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> AsyncIterator[StreamChunk]:
        if audio:
            content_in = _build_audio_content_blocks(prompt, audio)
        elif images:
            content_in = _build_user_content_blocks(prompt, images)
        else:
            content_in = prompt
        stream = await _guarded_create(
            self._client,
            model=self.model.value,
            messages=[{"role": "user", "content": content_in}],
            stream=True,
            stream_options={"include_usage": True},
            **generate_kwargs,
        )
        async for sc in self._iter_stream(stream):
            yield sc

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
        generate_kwargs, tools = await self._chat_setup(
            user_message, generate_kwargs, use_tools, images=images, audio=audio
        )
        generate_kwargs = _SyncOpenAICompatClient._with_response_format(generate_kwargs, response_format)

        if stream:
            return self._chat_streamed(generate_kwargs, tools)

        response = await _guarded_create(
            self._client,
            model=self.model.value,
            messages=strip_inert_keys(self.messages),
            tools=tools if tools else openai.NOT_GIVEN,
            **generate_kwargs,
        )
        msg = response.choices[0].message
        self.last_usage = usage_from_openai(response)
        self.last_thinking = ""
        # Servers that strip <think> tags server-side (llama-server, vLLM/SGLang reasoning parsers)
        # return reasoning in a separate reasoning_content field; prefer it when present, else fall
        # back to parsing inline tags out of content.
        reasoning = getattr(msg, "reasoning_content", None)

        # Single turn: if the model called tools, execute them and return. The model's response
        # to the tool results comes on the next chat() call (the loop lives in Agent).
        if msg.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": json.loads(tc.function.arguments)} for tc in msg.tool_calls
            ]
            text = msg.content or ""
            if reasoning:
                self.last_thinking = reasoning
            elif self.is_thinking_model:
                self.last_thinking, text = _split_thinking(text)
            msgs_before = len(self.messages)
            self._record_tool_calls(tool_calls, content=text)
            if self.last_thinking:
                self.messages[msgs_before]["thinking"] = self.last_thinking
            return text

        content = msg.content or ""
        if reasoning:
            self.last_thinking = reasoning
        elif self.is_thinking_model:
            self.last_thinking, content = _split_thinking(content)

        self.messages.append({"role": "assistant", "content": content})
        if self.last_thinking:
            self.messages[-1]["thinking"] = self.last_thinking
        return content

    async def _chat_streamed(self, generate_kwargs: dict[str, Any], tools: list) -> AsyncIterator[StreamChunk]:
        stream = await _guarded_create(
            self._client,
            model=self.model.value,
            messages=strip_inert_keys(self.messages),
            stream=True,
            stream_options={"include_usage": True},
            tools=tools if tools else openai.NOT_GIVEN,
            **generate_kwargs,
        )

        # Yield content/thinking chunks as they arrive (incremental streaming) while accumulating
        # any tool-call deltas separately; content and tool_call deltas don't require buffering.
        tool_calls_acc: dict[int, dict] = {}
        full_content = ""
        parser = _ThinkingParser() if self.is_thinking_model else None
        self.last_thinking = ""
        self.last_usage = None

        async for chunk in _guard_stream(stream):
            if getattr(chunk, "usage", None):
                self.last_usage = usage_from_openai(chunk)
            if not chunk.choices:  # terminal usage chunk (empty choices) or keep-alive
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                self.last_thinking += reasoning
                yield StreamChunk(StreamingContentType.THINKING, reasoning)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    acc = tool_calls_acc.setdefault(tc_delta.index, {"name": "", "arguments": ""})
                    if tc_delta.function and tc_delta.function.name:
                        acc["name"] += tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        acc["arguments"] += tc_delta.function.arguments
            elif delta.content is not None:
                if parser:
                    for phase, text in parser.feed(delta.content):
                        if phase == StreamingContentType.THINKING:
                            self.last_thinking += text
                        else:
                            full_content += text
                        yield StreamChunk(phase, text)
                else:
                    full_content += delta.content
                    yield StreamChunk(StreamingContentType.GENERATING, delta.content)

        if not tool_calls_acc:
            self.messages.append({"role": "assistant", "content": full_content})
            if self.last_thinking:
                self.messages[-1]["thinking"] = self.last_thinking
            return

        # Tool call path (single turn): prose/thinking already streamed above; now dispatch and
        # return. The model's response to the tool results comes on the next chat() call (the loop
        # lives in Agent). No second stream here.
        tool_calls = [{"name": tc["name"], "arguments": json.loads(tc["arguments"])} for tc in tool_calls_acc.values()]
        tool_turn_thinking = self.last_thinking
        msgs_before = len(self.messages)
        self._record_tool_calls(tool_calls, content=full_content)
        if tool_turn_thinking:
            self.messages[msgs_before]["thinking"] = tool_turn_thinking


# --- Local-server subclasses (cloud OpenAI / Gemini live in their own subpackages) ---


class AsyncLMStudioOpenAIClient(AsyncOpenAICompatClient):
    MODELS = LMStudioOpenAIModel

    def __init__(
        self,
        model: LMStudioOpenAIModel,
        base_url: str = "http://localhost:1234/v1",
        **kwargs,
    ):
        super().__init__(model, base_url=base_url, **kwargs)


class AsyncOllamaOpenAIClient(AsyncOpenAICompatClient):
    MODELS = OllamaOpenAIModel

    def __init__(
        self,
        model: OllamaOpenAIModel,
        base_url: str = "http://localhost:11434/v1",
        **kwargs,
    ):
        super().__init__(model, base_url=base_url, **kwargs)


class AsyncHFOpenAIClient(AsyncOpenAICompatClient):
    MODELS = HFOpenAIModel

    def __init__(
        self,
        model: HFOpenAIModel,
        base_url: str = "http://localhost:8000/v1",
        **kwargs,
    ):
        super().__init__(model, base_url=base_url, **kwargs)


class AsyncVLLMOpenAIClient(AsyncOpenAICompatClient):
    MODELS = VLLMOpenAIModel

    def __init__(
        self,
        model: VLLMOpenAIModel,
        base_url: str = "http://localhost:8000/v1",
        **kwargs,
    ):
        super().__init__(model, base_url=base_url, **kwargs)


class AsyncLlamaServerOpenAIClient(AsyncOpenAICompatClient):
    MODELS = LlamaServerOpenAIModel

    def __init__(
        self,
        model: LlamaServerOpenAIModel,
        base_url: str = "http://localhost:8080/v1",
        **kwargs,
    ):
        super().__init__(model, base_url=base_url, **kwargs)


class AsyncSGLangOpenAIClient(AsyncOpenAICompatClient):
    MODELS = SGLangOpenAIModel

    def __init__(
        self,
        model: SGLangOpenAIModel,
        base_url: str = "http://localhost:30000/v1",
        **kwargs,
    ):
        super().__init__(model, base_url=base_url, **kwargs)
