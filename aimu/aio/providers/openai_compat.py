"""Async OpenAI-compatible clients.

One base ``AsyncOpenAICompatClient`` that uses ``openai.AsyncOpenAI``, plus thin
subclasses for each service (OpenAI cloud, Gemini, LM Studio, Ollama-OpenAI, etc.)
that supply the right base URL / API key. Mirrors ``aimu.models.openai_compat``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator, Optional, Union

import openai

from aimu.models._thinking import _ThinkingParser, _split_thinking
from aimu.models.base import Model, StreamChunk, StreamingContentType, classproperty
from aimu.models.openai_compat.gemini_client import GeminiModel
from aimu.models.openai_compat.hf_openai_client import HFOpenAIModel
from aimu.models.openai_compat.llamaserver_openai_client import LlamaServerOpenAIModel
from aimu.models.openai_compat.lmstudio_openai_client import LMStudioOpenAIModel
from aimu.models.openai_compat.ollama_openai_client import OllamaOpenAIModel
from aimu.models.openai_compat.openai_client import OpenAIModel
from aimu.models.openai_compat.sglang_openai_client import SGLangOpenAIModel
from aimu.models.openai_compat.vllm_openai_client import VLLMOpenAIModel

from .._base import AsyncBaseModelClient

logger = logging.getLogger(__name__)


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
    ):
        super().__init__(model, model_kwargs, system_message)
        self.default_generate_kwargs = self.DEFAULT_GENERATE_KWARGS.copy()
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_thinking]

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_tools]

    @classproperty
    def VISION_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_vision]

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        if not generate_kwargs:
            return self.default_generate_kwargs.copy()
        return {**self.default_generate_kwargs, **generate_kwargs}

    async def _iter_stream(self, stream) -> AsyncIterator[StreamChunk]:
        self.last_thinking = ""
        parser = _ThinkingParser() if self.is_thinking_model else None

        async for chunk in stream:
            delta = chunk.choices[0].delta
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
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs)

        response = await self._client.chat.completions.create(
            model=self.model.value,
            messages=[{"role": "user", "content": prompt}],
            **generate_kwargs,
        )
        content = response.choices[0].message.content or ""

        self.last_thinking = ""
        if self.is_thinking_model:
            self.last_thinking, content = _split_thinking(content)

        return content

    async def _generate_streamed(
        self,
        prompt: str,
        generate_kwargs: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        stream = await self._client.chat.completions.create(
            model=self.model.value,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **generate_kwargs,
        )
        async for sc in self._iter_stream(stream):
            yield sc

    async def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        generate_kwargs, tools = await self._chat_setup(user_message, generate_kwargs, use_tools, images=images)

        if stream:
            return self._chat_streamed(generate_kwargs, tools)

        response = await self._client.chat.completions.create(
            model=self.model.value,
            messages=self.messages,
            tools=tools if tools else openai.NOT_GIVEN,
            **generate_kwargs,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": json.loads(tc.function.arguments)} for tc in msg.tool_calls
            ]
            await self._handle_tool_calls(tool_calls, tools)

            response = await self._client.chat.completions.create(
                model=self.model.value,
                messages=self.messages,
                tools=tools if tools else openai.NOT_GIVEN,
                **generate_kwargs,
            )
            msg = response.choices[0].message

        content = msg.content or ""
        self.last_thinking = ""
        if self.is_thinking_model:
            self.last_thinking, content = _split_thinking(content)

        self.messages.append({"role": "assistant", "content": content})
        return content

    async def _chat_streamed(
        self, generate_kwargs: dict[str, Any], tools: list
    ) -> AsyncIterator[StreamChunk]:
        stream = await self._client.chat.completions.create(
            model=self.model.value,
            messages=self.messages,
            stream=True,
            tools=tools if tools else openai.NOT_GIVEN,
            **generate_kwargs,
        )

        tool_calls_acc: dict[int, dict] = {}
        first_pass_chunks: list[StreamChunk] = []
        parser = _ThinkingParser() if self.is_thinking_model else None
        self.last_thinking = ""

        async for chunk in stream:
            delta = chunk.choices[0].delta
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
                        first_pass_chunks.append(StreamChunk(phase, text))
                else:
                    first_pass_chunks.append(StreamChunk(StreamingContentType.GENERATING, delta.content))

        if not tool_calls_acc:
            full_content = ""
            for sc in first_pass_chunks:
                if sc.phase == StreamingContentType.GENERATING:
                    full_content += sc.content
                yield sc
            self.messages.append({"role": "assistant", "content": full_content})
            return

        tool_calls = [{"name": tc["name"], "arguments": json.loads(tc["arguments"])} for tc in tool_calls_acc.values()]
        msgs_before = len(self.messages)
        await self._handle_tool_calls(tool_calls, tools)

        for i, tc in enumerate(self.messages[msgs_before]["tool_calls"]):
            yield StreamChunk(
                StreamingContentType.TOOL_CALLING,
                {"name": tc["function"]["name"], "response": self.messages[msgs_before + 1 + i]["content"]},
            )

        stream2 = await self._client.chat.completions.create(
            model=self.model.value,
            messages=self.messages,
            stream=True,
            tools=tools if tools else openai.NOT_GIVEN,
            **generate_kwargs,
        )

        full_content = ""
        async for sc in self._iter_stream(stream2):
            if sc.phase == StreamingContentType.GENERATING:
                full_content += sc.content
            yield sc
        self.messages.append({"role": "assistant", "content": full_content})


# --- Service-specific subclasses ---


class AsyncOpenAIClient(AsyncOpenAICompatClient):
    """OpenAI cloud API."""

    MODELS = OpenAIModel

    def __init__(
        self,
        model: OpenAIModel,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        from dotenv import load_dotenv

        load_dotenv()
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
            system_message=system_message,
            model_kwargs=model_kwargs,
        )

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        kwargs = super()._update_generate_kwargs(generate_kwargs)
        # o-series models: rename max_tokens, force temperature=1.
        model_id = self.model.value
        if any(model_id.startswith(prefix) for prefix in ("o1", "o3", "o4")):
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            kwargs["temperature"] = 1
        return kwargs


class AsyncGeminiClient(AsyncOpenAICompatClient):
    """Google Gemini via OpenAI-compatible endpoint."""

    MODELS = GeminiModel

    def __init__(
        self,
        model: GeminiModel,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        from dotenv import load_dotenv

        load_dotenv()
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or os.environ.get("GOOGLE_API_KEY", "not-needed"),
            system_message=system_message,
            model_kwargs=model_kwargs,
        )


class AsyncLMStudioOpenAIClient(AsyncOpenAICompatClient):
    MODELS = LMStudioOpenAIModel

    def __init__(
        self,
        model: LMStudioOpenAIModel,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "not-needed",
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__(model, base_url, api_key, system_message, model_kwargs)


class AsyncOllamaOpenAIClient(AsyncOpenAICompatClient):
    MODELS = OllamaOpenAIModel

    def __init__(
        self,
        model: OllamaOpenAIModel,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "not-needed",
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__(model, base_url, api_key, system_message, model_kwargs)


class AsyncHFOpenAIClient(AsyncOpenAICompatClient):
    MODELS = HFOpenAIModel

    def __init__(
        self,
        model: HFOpenAIModel,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__(model, base_url, api_key, system_message, model_kwargs)


class AsyncVLLMOpenAIClient(AsyncOpenAICompatClient):
    MODELS = VLLMOpenAIModel

    def __init__(
        self,
        model: VLLMOpenAIModel,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__(model, base_url, api_key, system_message, model_kwargs)


class AsyncLlamaServerOpenAIClient(AsyncOpenAICompatClient):
    MODELS = LlamaServerOpenAIModel

    def __init__(
        self,
        model: LlamaServerOpenAIModel,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "not-needed",
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__(model, base_url, api_key, system_message, model_kwargs)


class AsyncSGLangOpenAIClient(AsyncOpenAICompatClient):
    MODELS = SGLangOpenAIModel

    def __init__(
        self,
        model: SGLangOpenAIModel,
        base_url: str = "http://localhost:30000/v1",
        api_key: str = "not-needed",
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__(model, base_url, api_key, system_message, model_kwargs)
