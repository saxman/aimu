"""Async Ollama client using ``ollama.AsyncClient``."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional, Union

import ollama

from aimu.models._images import _adapt_messages_for_ollama
from aimu.models.base import Model, StreamChunk, StreamingContentType, classproperty
from aimu.models.ollama.ollama_client import OllamaModel

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
    ):
        super().__init__(model, None, system_message)
        self.model_keep_alive_seconds = model_keep_alive_seconds
        self.default_generate_kwargs = dict(model.generation_kwargs)

        # Model pull happens lazily on first request — `ollama.AsyncClient` doesn't
        # expose a separate pull. Users who want eager pull can call
        # `ollama.pull(model.value)` themselves.
        self._client = ollama.AsyncClient()

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_thinking]

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_tools]

    @classproperty
    def VISION_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_vision]

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
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs)

        response = await self._client.generate(
            model=self.model.value,
            prompt=prompt,
            options=generate_kwargs,
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
        )

        self.last_thinking = ""
        if not self.is_thinking_model:
            return response["response"]
        self.last_thinking = response.thinking
        return response.response

    async def _generate_streamed(
        self,
        prompt: str,
        generate_kwargs: dict,
    ) -> AsyncIterator[StreamChunk]:
        response = await self._client.generate(
            model=self.model.value,
            prompt=prompt,
            options=generate_kwargs,
            stream=True,
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
        )

        self.last_thinking = ""
        async for response_part in response:
            if response_part.thinking:
                self.last_thinking += response_part.thinking
                yield StreamChunk(StreamingContentType.THINKING, response_part.thinking)
            yield StreamChunk(StreamingContentType.GENERATING, response_part["response"])

    async def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        generate_kwargs, tools = await self._chat_setup(user_message, generate_kwargs, use_tools, images=images)

        if stream:
            return self._chat_streamed(generate_kwargs, tools)

        response = await self._client.chat(
            model=self.model.value,
            messages=_adapt_messages_for_ollama(self.messages),
            options=generate_kwargs,
            tools=tools,
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
        )

        if response["message"].tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in response["message"].tool_calls
            ]
            await self._handle_tool_calls(tool_calls, tools)

            if response["message"].thinking:
                self.messages[-1 - len(tool_calls)]["thinking"] = response["message"].thinking

            response = await self._client.chat(
                model=self.model.value,
                messages=_adapt_messages_for_ollama(self.messages),
                options=generate_kwargs,
                tools=tools,
                think=self.is_thinking_model,
                keep_alive=self.model_keep_alive_seconds,
            )

        self.messages.append({"role": response["message"].role, "content": response["message"].content})
        if response["message"].thinking:
            self.messages[-1]["thinking"] = response["message"].thinking
        return response["message"].content

    async def _chat_streamed(
        self, generate_kwargs: dict, tools: list
    ) -> AsyncIterator[StreamChunk]:
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
            async for chunk in self._handle_tool_calls_streamed(tool_calls, tools):
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
