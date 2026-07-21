"""Async base class for provider model clients.

Mirrors ``aimu.models.base.BaseModelClient`` but with async ``_chat`` / ``_generate``
abstract methods. State-management mechanics (system_message lifecycle, reset,
message-history append, image-block normalization) are inherited from the shared
``_ChatStateMixin``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterable, Optional, Union

from aimu.models._internal.chat_state import _ChatStateMixin
from aimu.models._internal.streaming import afilter_chunks, resolve_include
from aimu.models.base import Model, StreamChunk, StreamingContentType, classproperty

logger = logging.getLogger(__name__)


class AsyncBaseModelClient(_ChatStateMixin, ABC):
    """Abstract async base for all provider clients.

    Subclasses implement :meth:`_chat`, :meth:`_generate`, and
    :meth:`_update_generate_kwargs`. Tool calling, message history, vision input,
    and streaming filters are handled here once for every provider.
    """

    MODELS = Model

    model: Model
    model_kwargs: Optional[dict]
    _system_message: Optional[str]
    default_generate_kwargs: dict
    messages: list[dict]
    last_thinking: str | None
    last_usage: dict | None
    last_structured: Any | None

    @abstractmethod
    def __init__(self, model: Model, model_kwargs: Optional[dict] = None, system_message: Optional[str] = None):
        self.model = model
        self.model_kwargs = model_kwargs
        self._system_message = system_message
        self.default_generate_kwargs = {}
        self.messages = []
        # Transient set advertised to the model for the current call (swapped by the ``tools=``
        # per-call override); NOT a persistent registry, and the client never executes tools.
        # Tool execution lives in the Agent's tool-loop engine (``aimu.aio._tool_loop``).
        self.tools: list = []
        self.last_thinking = ""
        self.last_usage = None
        self.last_structured = None

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    @classproperty
    def VISION_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    @classproperty
    def AUDIO_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    @classproperty
    def STRUCTURED_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    @abstractmethod
    async def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        """Provider-specific async generate implementation. Use :meth:`generate`.

        ``response_format`` (a JSON Schema dict) is passed only on the native structured-output
        path and only to providers with ``supports_structured_output=True``.
        """
        pass

    @abstractmethod
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
        """Provider-specific single-turn async chat implementation. Use :meth:`chat`.

        One model request. ``user_message=None`` runs a turn on the current messages
        (continuation). If the model requests tools, execute them and return — no follow-up.
        """
        pass

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
        """Single-turn, stateless async generation. See sync :meth:`BaseModelClient.generate`.

        ``images`` / ``audio`` accept the same forms as :meth:`chat` and raise ``ValueError``
        for models that don't support the respective modality. The call stays stateless
        (does not touch ``self.messages``). Passing both raises ``ValueError``. ``schema``
        mirrors the sync surface; see :meth:`chat`.
        """
        if images and audio:
            raise ValueError("Pass either images= or audio= per call, not both.")
        if images:
            self._require_vision()
        if audio:
            self._require_audio()
        if schema is not None:
            if stream:
                return self._generate_structured_streamed(prompt, generate_kwargs, images, audio, schema, include)
            return await self._generate_structured(prompt, generate_kwargs, images, audio, schema)
        result = await self._generate(prompt, generate_kwargs, stream=stream, images=images, audio=audio)
        if stream and include is not None:
            return afilter_chunks(result, resolve_include(include))
        return result

    async def chat(
        self,
        user_message: Optional[str] = None,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
        tools: Optional[list] = None,
        audio: Optional[list] = None,
        schema: Optional[type] = None,
    ) -> Union[str, Any, AsyncIterator[StreamChunk]]:
        """One async model turn against the persistent message history.

        A single call issues one model request; if the model requests tools they are executed
        and appended, and the call returns (the response to the tool results comes on the next
        call — the multi-turn loop lives in :class:`~aimu.aio.Agent`). ``user_message=None``
        runs a turn on the current messages without appending a user turn (continuation).

        Args mirror the sync :meth:`BaseModelClient.chat`, including the per-call
        ``tools`` override and ``schema`` (structured output: native when
        ``supports_structured_output`` else prompt-and-parse; Anthropic native + active
        tools raises). Streaming returns an ``AsyncIterator[StreamChunk]``, consumed with
        ``async for``. With ``schema`` + ``stream=True`` the stream ends in a terminal
        ``DONE`` chunk carrying ``{"result": <object>}`` and sets ``self.last_structured``
        (Anthropic streams the JSON but emits no thinking; see the sync surface).
        """
        if schema is not None:
            if stream:
                return self._chat_structured_streamed(
                    user_message, generate_kwargs, use_tools, images, audio, tools, schema, include
                )
            return await self._chat_structured(user_message, generate_kwargs, use_tools, images, audio, tools, schema)

        if tools is None:
            result = await self._chat(
                user_message, generate_kwargs, use_tools=use_tools, stream=stream, images=images, audio=audio
            )
            if stream and include is not None:
                return afilter_chunks(result, resolve_include(include))
            return result

        if stream:
            return self._chat_with_tools_streamed(
                user_message, generate_kwargs, use_tools, images, include, tools, audio=audio
            )
        with self._tools_override(tools):
            return await self._chat(
                user_message, generate_kwargs, use_tools=use_tools, stream=False, images=images, audio=audio
            )

    async def _chat_structured(self, user_message, generate_kwargs, use_tools, images, audio, tools, schema):
        from aimu.models._internal.json import parse_json_response

        response_format, suffix = self._structured_request(schema)
        message = user_message + suffix
        extra = {"response_format": response_format} if response_format is not None else {}
        if tools is None:
            text = await self._chat(
                message, generate_kwargs, use_tools=use_tools, stream=False, images=images, audio=audio, **extra
            )
        else:
            with self._tools_override(tools):
                text = await self._chat(
                    message, generate_kwargs, use_tools=use_tools, stream=False, images=images, audio=audio, **extra
                )
        return parse_json_response(text, schema)

    async def _generate_structured(self, prompt, generate_kwargs, images, audio, schema):
        from aimu.models._internal.json import parse_json_response

        response_format, suffix = self._structured_request(schema)
        extra = {"response_format": response_format} if response_format is not None else {}
        text = await self._generate(prompt + suffix, generate_kwargs, stream=False, images=images, audio=audio, **extra)
        return parse_json_response(text, schema)

    async def _chat_structured_streamed(
        self, user_message, generate_kwargs, use_tools, images, audio, tools, schema, include
    ) -> AsyncIterator[StreamChunk]:
        """Async streamed structured output for :meth:`chat`.

        Yields live THINKING / GENERATING / TOOL_CALLING chunks (subject to ``include=``),
        accumulates the GENERATING text, then parses it into ``schema`` and yields a terminal
        ``StreamChunk(DONE, {"result": obj})`` (always emitted, regardless of ``include=``).
        Sets ``self.last_structured`` just before the terminal chunk.
        """
        from aimu.models._internal.json import parse_json_response

        response_format, suffix = self._structured_request(schema)
        message = user_message + suffix
        extra = {"response_format": response_format} if response_format is not None else {}
        selected = resolve_include(include) if include is not None else None

        async def _run() -> AsyncIterator[StreamChunk]:
            buffer: list[str] = []
            result = await self._chat(
                message, generate_kwargs, use_tools=use_tools, stream=True, images=images, audio=audio, **extra
            )
            async for chunk in result:
                if chunk.phase == StreamingContentType.GENERATING and isinstance(chunk.content, str):
                    buffer.append(chunk.content)
                if selected is None or chunk.phase in selected:
                    yield chunk
            obj = parse_json_response("".join(buffer), schema)
            self.last_structured = obj
            yield StreamChunk(StreamingContentType.DONE, {"result": obj})

        if tools is None:
            async for chunk in _run():
                yield chunk
        else:
            with self._tools_override(tools):
                async for chunk in _run():
                    yield chunk

    async def _generate_structured_streamed(
        self, prompt, generate_kwargs, images, audio, schema, include
    ) -> AsyncIterator[StreamChunk]:
        """Async streamed structured output for :meth:`generate`. See :meth:`_chat_structured_streamed`."""
        from aimu.models._internal.json import parse_json_response

        response_format, suffix = self._structured_request(schema)
        extra = {"response_format": response_format} if response_format is not None else {}
        selected = resolve_include(include) if include is not None else None
        buffer: list[str] = []
        result = await self._generate(
            prompt + suffix, generate_kwargs, stream=True, images=images, audio=audio, **extra
        )
        async for chunk in result:
            if chunk.phase == StreamingContentType.GENERATING and isinstance(chunk.content, str):
                buffer.append(chunk.content)
            if selected is None or chunk.phase in selected:
                yield chunk
        obj = parse_json_response("".join(buffer), schema)
        self.last_structured = obj
        yield StreamChunk(StreamingContentType.DONE, {"result": obj})

    async def _chat_with_tools_streamed(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]],
        use_tools: bool,
        images: Optional[list],
        include: Optional[Iterable[Union[str, StreamingContentType]]],
        tools: list,
        audio: Optional[list] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Async streaming chat with a per-call ``tools`` override.

        The override must stay active while the stream is consumed (tool dispatch reads
        ``self.tools``), so the swap wraps the ``async for`` rather than just ``_chat``.
        """
        with self._tools_override(tools):
            result = await self._chat(
                user_message, generate_kwargs, use_tools=use_tools, stream=True, images=images, audio=audio
            )
            if include is not None:
                result = afilter_chunks(result, resolve_include(include))
            async for chunk in result:
                yield chunk

    @abstractmethod
    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        pass

    async def _chat_setup(
        self,
        user_message: Optional[str] = None,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Async equivalent of the sync ``_chat_setup``."""
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        # user_message=None → continuation: run a turn on current messages, append nothing.
        if user_message is not None:
            self._append_user_turn(user_message, images, audio=audio)

        tools: list[dict] = []
        if self.model.supports_tools and use_tools:
            tools.extend(self._collect_python_tool_specs())

        return generate_kwargs, tools
