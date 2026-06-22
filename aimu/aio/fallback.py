"""Async failover across model clients (P0-B): the ``aimu.aio`` twin of ``FallbackClient``.

Same semantics as the sync :class:`aimu.models.fallback.FallbackClient` (ordered failover,
shared conversation state, ``FallbackExhaustedError`` when all clients fail), but the
``chat`` / ``generate`` loops are async and streaming returns an ``AsyncIterator``. Reuses
the sync module's state plumbing and error type.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterable, Optional, Union

from aimu.models.base import StreamChunk, StreamingContentType
from aimu.models.fallback import _FallbackStateMixin, _label

from ._base import AsyncBaseModelClient

logger = logging.getLogger(__name__)


class AsyncFallbackClient(_FallbackStateMixin, AsyncBaseModelClient):
    """An ``AsyncBaseModelClient`` that delegates to an ordered list of async clients with failover.

    See :class:`aimu.models.fallback.FallbackClient` for the full semantics; this is the
    async mirror (``await`` on ``chat`` / ``generate``; streaming yields via ``async for``).
    """

    async def chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
        tools: Optional[list] = None,
        audio: Optional[list] = None,
        schema: Optional[type] = None,
    ) -> Union[str, Any, AsyncIterator[StreamChunk]]:
        if stream:
            return self._chat_streamed(user_message, generate_kwargs, use_tools, images, include, tools, audio)
        errors: list[tuple[Any, BaseException]] = []
        for client in self.clients:
            self._load_state(client)
            try:
                result = await client.chat(
                    user_message,
                    generate_kwargs,
                    use_tools=use_tools,
                    stream=False,
                    images=images,
                    include=include,
                    tools=tools,
                    audio=audio,
                    schema=schema,
                )
            except self.retry_on as exc:
                logger.warning("Fallback: client %r failed (%s); trying next.", _label(client), exc)
                errors.append((client, exc))
                continue
            self._adopt_state(client)
            return result
        raise self._exhausted(errors)

    async def _chat_streamed(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]],
        use_tools: bool,
        images: Optional[list],
        include: Optional[Iterable[Union[str, StreamingContentType]]],
        tools: Optional[list],
        audio: Optional[list],
    ) -> AsyncIterator[StreamChunk]:
        errors: list[tuple[Any, BaseException]] = []
        for client in self.clients:
            self._load_state(client)
            stream = await client.chat(
                user_message,
                generate_kwargs,
                use_tools=use_tools,
                stream=True,
                images=images,
                include=include,
                tools=tools,
                audio=audio,
            )
            iterator = stream.__aiter__()
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration:
                self._adopt_state(client)  # empty but successful stream
                return
            except self.retry_on as exc:
                logger.warning("Fallback: client %r failed before first chunk (%s); trying next.", _label(client), exc)
                errors.append((client, exc))
                continue
            yield first
            async for chunk in iterator:
                yield chunk
            self._adopt_state(client)
            return
        raise self._exhausted(errors)

    async def generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
        audio: Optional[list] = None,
        schema: Optional[type] = None,
    ) -> Union[str, Any, AsyncIterator[StreamChunk]]:
        if stream:
            return self._generate_streamed(prompt, generate_kwargs, images, include, audio)
        errors: list[tuple[Any, BaseException]] = []
        for client in self.clients:
            client.last_thinking = ""
            client.last_usage = None
            try:
                result = await client.generate(
                    prompt, generate_kwargs, stream=False, images=images, include=include, audio=audio, schema=schema
                )
            except self.retry_on as exc:
                logger.warning("Fallback: client %r failed on generate (%s); trying next.", _label(client), exc)
                errors.append((client, exc))
                continue
            self.last_thinking = getattr(client, "last_thinking", "")
            self.last_usage = getattr(client, "last_usage", None)
            return result
        raise self._exhausted(errors)

    async def _generate_streamed(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]],
        images: Optional[list],
        include: Optional[Iterable[Union[str, StreamingContentType]]],
        audio: Optional[list],
    ) -> AsyncIterator[StreamChunk]:
        errors: list[tuple[Any, BaseException]] = []
        for client in self.clients:
            stream = await client.generate(
                prompt, generate_kwargs, stream=True, images=images, include=include, audio=audio
            )
            iterator = stream.__aiter__()
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration:
                return
            except self.retry_on as exc:
                errors.append((client, exc))
                continue
            yield first
            async for chunk in iterator:
                yield chunk
            return
        raise self._exhausted(errors)

    # Abstract-method stubs: the public methods above are overridden.
    async def _chat(self, *args, **kwargs):  # pragma: no cover - not reachable
        raise NotImplementedError("AsyncFallbackClient delegates via chat(); _chat is unused.")

    async def _generate(self, *args, **kwargs):  # pragma: no cover - not reachable
        raise NotImplementedError("AsyncFallbackClient delegates via generate(); _generate is unused.")

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict] = None) -> dict:  # pragma: no cover
        return generate_kwargs or {}
