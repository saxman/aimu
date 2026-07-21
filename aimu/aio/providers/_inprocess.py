"""Shared base for async wrappers over in-process sync clients.

Per Decision 7 in the async design, the in-process providers (HuggingFace,
LlamaCpp) do *not* load model weights independently on the async surface. Each
async wrapper takes an existing sync client and routes work through
``asyncio.to_thread``: the async surface exists for event-loop integration, not
coroutine concurrency (these backends are sync-only and GIL/CUDA-bound). State
(``messages``, ``system_message``, ``tools``) is shared with the wrapped sync
client -- there's conceptually one client, and the async version just adds an
awaitable interface.

Concrete subclasses set two class attributes: ``MODELS`` (the provider's model
enum) and ``_SYNC_CLASS`` (the sync client type the wrapper accepts).
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional, Union

from aimu.models.base import Model, StreamChunk, classproperty

from .._base import AsyncBaseModelClient


class _AsyncInProcessClient(AsyncBaseModelClient):
    """Async wrapper around a sync in-process client (no weight reload)."""

    #: The sync client type this wrapper accepts. Set by subclasses.
    _SYNC_CLASS: type

    def __init__(self, sync_client):
        if not isinstance(sync_client, self._SYNC_CLASS):
            raise TypeError(
                f"{type(self).__name__} requires an existing sync {self._SYNC_CLASS.__name__}. "
                f"Got {type(sync_client).__name__}."
            )
        self._sync = sync_client
        # super().__init__ would clobber the sync client's state; mirror attributes instead.
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

    @classproperty
    def AUDIO_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_audio]

    # --- Share state with the wrapped sync client ---

    @property
    def messages(self) -> list[dict]:
        return self._sync.messages

    @messages.setter
    def messages(self, value: list[dict]) -> None:
        self._sync.messages = value

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

    # --- Async surface: route sync work through to_thread so the event loop doesn't block ---

    async def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        if stream:
            return self._stream_via_thread(
                self._sync._generate(prompt, generate_kwargs, stream=True, images=images, audio=audio)
            )
        return await asyncio.to_thread(self._sync._generate, prompt, generate_kwargs, False, images, audio)

    async def _chat(
        self,
        user_message: Optional[str] = None,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        # The sync _chat is a single model turn: it parses and *stores* any tool calls but does
        # not execute them (the async tool-loop engine dispatches them, awaiting async tools).
        # So the sync loop only needs to advertise the tools (via their __tool_spec__, which async
        # tools carry) — no async->sync tool bridging is required.
        if stream:
            sync_iter = self._sync._chat(
                user_message, generate_kwargs, use_tools=use_tools, stream=True, images=images, audio=audio
            )
            return self._stream_via_thread(sync_iter)
        return await asyncio.to_thread(self._sync._chat, user_message, generate_kwargs, use_tools, False, images, audio)

    async def _stream_via_thread(self, sync_iterator) -> AsyncIterator[StreamChunk]:
        """Yield from a sync iterator by hopping each ``next()`` to a worker thread.

        Crude but correct: the sync streaming iterator blocks until the next token is
        generated. ``asyncio.to_thread`` keeps the event loop free between chunks.
        """
        sentinel = object()
        loop_iter = iter(sync_iterator)
        while True:
            chunk = await asyncio.to_thread(next, loop_iter, sentinel)
            if chunk is sentinel:
                break
            yield chunk
