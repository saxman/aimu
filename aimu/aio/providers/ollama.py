"""Async Ollama client using ``ollama.AsyncClient``."""

from __future__ import annotations

import logging
from typing import AsyncIterator, Optional, Union

import ollama

from aimu.models._internal.image_input import _adapt_messages_for_ollama
from aimu.models._internal.usage import usage_from_ollama
from aimu.models.base import Model, StreamChunk, StreamingContentType, classproperty
from aimu.models.providers.ollama import OllamaClient, OllamaModel

from .._base import AsyncBaseModelClient

logger = logging.getLogger(__name__)


class AsyncOllamaClient(AsyncBaseModelClient):
    """Async Ollama client. Mirrors :class:`OllamaClient` using ``ollama.AsyncClient``."""

    MODELS = OllamaModel

    def __init__(
        self,
        model: OllamaModel,
        system_message: Optional[str] = None,
        model_keep_alive_seconds: int = 60,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        super().__init__(model, None, system_message)

        if max_retries is not None:
            raise ValueError(
                "Ollama's native client has no request-retry support. Omit max_retries, "
                "or use the 'ollama-openai' provider, which routes through the OpenAI SDK and supports it."
            )

        self.model_keep_alive_seconds = model_keep_alive_seconds
        self.default_generate_kwargs = dict(model.generation_kwargs)

        # Model pull happens lazily on first request; `ollama.AsyncClient` doesn't
        # expose a separate pull. Users who want eager pull can call
        # `ollama.pull(model.value)` themselves.
        self._client = ollama.AsyncClient(**({"timeout": timeout} if timeout is not None else {}))

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

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict] = None) -> dict:
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs
        if "max_tokens" in generate_kwargs:
            generate_kwargs["num_predict"] = generate_kwargs.pop("max_tokens")
        return generate_kwargs

    async def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)
        gen_images = OllamaClient._extract_ollama_images(images)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs, images=gen_images)

        response = await self._client.generate(
            model=self.model.value,
            prompt=prompt,
            images=gen_images,
            options=generate_kwargs,
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
            format=response_format,
        )
        self.last_usage = usage_from_ollama(response)

        self.last_thinking = ""
        if not self.is_thinking_model:
            return response["response"]
        self.last_thinking = response.thinking
        return response.response

    async def _generate_streamed(
        self,
        prompt: str,
        generate_kwargs: dict,
        images: Optional[list] = None,
    ) -> AsyncIterator[StreamChunk]:
        response = await self._client.generate(
            model=self.model.value,
            prompt=prompt,
            images=images,
            options=generate_kwargs,
            stream=True,
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
        )

        self.last_thinking = ""
        self.last_usage = None
        response_part = None
        async for response_part in response:
            if response_part.thinking:
                self.last_thinking += response_part.thinking
                yield StreamChunk(StreamingContentType.THINKING, response_part.thinking)
            yield StreamChunk(StreamingContentType.GENERATING, response_part["response"])

        # The final part (done=true) carries the eval counts.
        if response_part is not None:
            self.last_usage = usage_from_ollama(response_part)

    async def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        generate_kwargs, tools = await self._chat_setup(
            user_message, generate_kwargs, use_tools, images=images, audio=audio
        )

        if stream:
            return self._chat_streamed(generate_kwargs, tools)

        response = await self._client.chat(
            model=self.model.value,
            messages=_adapt_messages_for_ollama(self.messages),
            options=generate_kwargs,
            tools=tools,
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
            format=response_format,
        )

        if response["message"].tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": tc.function.arguments} for tc in response["message"].tool_calls
            ]
            await self._handle_tool_calls(tool_calls)

            if response["message"].thinking:
                self.messages[-1 - len(tool_calls)]["thinking"] = response["message"].thinking

            response = await self._client.chat(
                model=self.model.value,
                messages=_adapt_messages_for_ollama(self.messages),
                options=generate_kwargs,
                tools=tools,
                think=self.is_thinking_model,
                keep_alive=self.model_keep_alive_seconds,
                format=response_format,
            )

        self.last_usage = usage_from_ollama(response)
        self.messages.append({"role": response["message"].role, "content": response["message"].content})
        if response["message"].thinking:
            self.messages[-1]["thinking"] = response["message"].thinking
        return response["message"].content

    async def _chat_streamed(self, generate_kwargs: dict, tools: list) -> AsyncIterator[StreamChunk]:
        self.last_usage = None
        response = await self._client.chat(
            model=self.model.value,
            messages=_adapt_messages_for_ollama(self.messages),
            options=generate_kwargs,
            tools=tools,
            stream=True,
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
        )

        response_iter = response.__aiter__()
        response_part = await response_iter.__anext__()

        thinking = ""
        if response_part["message"].thinking:
            thinking = response_part["message"].thinking
            yield StreamChunk(StreamingContentType.THINKING, thinking)
            while True:
                response_part = await response_iter.__anext__()
                if response_part["message"].thinking:
                    thinking += response_part["message"].thinking
                    yield StreamChunk(StreamingContentType.THINKING, response_part["message"].thinking)
                else:
                    break

        if response_part["message"].tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in response_part["message"].tool_calls
            ]

            msgs_before = len(self.messages)
            async for chunk in self._handle_tool_calls_streamed(tool_calls):
                yield chunk

            if thinking:
                self.messages[msgs_before]["thinking"] = thinking
                thinking = ""

            response = await self._client.chat(
                model=self.model.value,
                messages=_adapt_messages_for_ollama(self.messages),
                options=generate_kwargs,
                tools=tools,
                stream=True,
                think=self.is_thinking_model,
                keep_alive=self.model_keep_alive_seconds,
            )
            response_iter = response.__aiter__()
            response_part = await response_iter.__anext__()

            thinking = ""
            if response_part["message"].thinking:
                thinking = response_part["message"].thinking
                yield StreamChunk(StreamingContentType.THINKING, thinking)
                while True:
                    response_part = await response_iter.__anext__()
                    if response_part["message"].thinking:
                        thinking += response_part["message"].thinking
                        yield StreamChunk(StreamingContentType.THINKING, response_part["message"].thinking)
                    else:
                        break

        content = response_part["message"].content
        yield StreamChunk(StreamingContentType.GENERATING, content)

        async for response_part in response_iter:
            content += response_part["message"].content
            yield StreamChunk(StreamingContentType.GENERATING, response_part["message"].content)

        message = {"role": response_part["message"].role, "content": content}
        if thinking:
            message["thinking"] = thinking
        self.messages.append(message)

        # The final part (done=true) of the last stream carries the eval counts.
        self.last_usage = usage_from_ollama(response_part)
