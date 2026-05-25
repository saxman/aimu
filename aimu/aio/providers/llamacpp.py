"""Async LlamaCpp client — wraps an existing sync :class:`LlamaCppClient`.

Same pattern as :mod:`aimu.aio.providers.hf` — see Decision 7. GGUF weights load
once via the sync client; this class adds an awaitable interface on top.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional, Union

from aimu.models.base import Model, StreamChunk, classproperty
from aimu.models.llamacpp.llamacpp_client import LlamaCppClient, LlamaCppModel

from .._base import AsyncBaseModelClient


class AsyncLlamaCppClient(AsyncBaseModelClient):
    """Async wrapper around a sync :class:`LlamaCppClient` — no weight reload."""

    MODELS = LlamaCppModel

    def __init__(self, sync_client: LlamaCppClient):
        if not isinstance(sync_client, LlamaCppClient):
            raise TypeError(
                f"AsyncLlamaCppClient requires an existing sync LlamaCppClient. "
                f"Got {type(sync_client).__name__}."
            )
        self._sync = sync_client
        self.model = sync_client.model
        self.model_kwargs = sync_client.model_kwargs
        self.default_generate_kwargs = sync_client.default_generate_kwargs

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_thinking]

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_tools]

    @classproperty
    def VISION_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_vision]

    @property
    def messages(self) -> list[dict]:
        return self._sync.messages

    @messages.setter
    def messages(self, value: list[dict]) -> None:
        self._sync.messages = value

    @property
    def mcp_client(self) -> Any:
        return self._sync.mcp_client

    @mcp_client.setter
    def mcp_client(self, value: Any) -> None:
        self._sync.mcp_client = value

    @property
    def tools(self) -> list:
        return self._sync.tools

    @tools.setter
    def tools(self, value: list) -> None:
        self._sync.tools = value

    @property
    def system_message(self) -> Optional[str]:
        return self._sync.system_message

    @system_message.setter
    def system_message(self, message: Optional[str]) -> None:
        self._sync.system_message = message

    @property
    def last_thinking(self) -> str:
        return self._sync.last_thinking

    @last_thinking.setter
    def last_thinking(self, value: str) -> None:
        self._sync.last_thinking = value

    @property
    def concurrent_tool_calls(self) -> bool:
        return self._sync.concurrent_tool_calls

    @concurrent_tool_calls.setter
    def concurrent_tool_calls(self, value: bool) -> None:
        self._sync.concurrent_tool_calls = value

    def reset(self, system_message: Optional[str] = "__keep__") -> None:
        self._sync.reset(system_message)

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        return self._sync._update_generate_kwargs(generate_kwargs)

    async def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        if stream:
            return self._stream_via_thread(self._sync._generate(prompt, generate_kwargs, stream=True))
        return await asyncio.to_thread(self._sync._generate, prompt, generate_kwargs, False)

    async def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        if stream:
            return self._stream_via_thread(
                self._sync._chat(user_message, generate_kwargs, use_tools=use_tools, stream=True, images=images)
            )
        return await asyncio.to_thread(
            self._sync._chat, user_message, generate_kwargs, use_tools, False, images
        )

    async def _stream_via_thread(self, sync_iterator) -> AsyncIterator[StreamChunk]:
        sentinel = object()
        loop_iter = iter(sync_iterator)
        while True:
            chunk = await asyncio.to_thread(next, loop_iter, sentinel)
            if chunk is sentinel:
                break
            yield chunk
