"""Text-modality base types: ``ModelSpec``, the ``Model`` enum, and ``BaseModelClient``.

``BaseModelClient`` is the abstract base for every text/chat provider. Tool calling,
message history, vision input, and streaming filters live here once for all providers;
concrete clients implement ``_chat`` / ``_generate`` / ``_update_generate_kwargs``.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Optional, Union

from .._internal.chat_state import _ChatStateMixin
from .._internal.streaming import filter_chunks as _filter_chunks_fn
from .._internal.streaming import resolve_include as _resolve_include_fn
from .shared import StreamChunk, StreamingContentType, classproperty

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Capability descriptor for a single model.

    Holds the provider-side model id plus universal capability flags. Provider-specific
    extras (e.g. HuggingFace tool-call format) live on the provider's ``Model`` subclass,
    not here.

    Equality and hash are by ``id`` only, so a ``ModelSpec`` can be used directly as an
    enum value even when ``generation_kwargs`` is a dict.
    """

    id: str
    tools: bool = False
    thinking: bool = False
    vision: bool = False
    audio: bool = False
    structured_output: bool = False
    generation_kwargs: Optional[dict] = field(default=None)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ModelSpec):
            return self.id == other.id
        return NotImplemented


class Model(Enum):
    """Base enum for provider model catalogs.

    Each member's value is a ``ModelSpec``; capability flags are mirrored as plain
    attributes (``supports_tools``, ``supports_thinking``, ``supports_vision``,
    ``generation_kwargs``) for direct read access. ``.value`` returns the provider id
    string so code can call e.g. ``ollama.pull(model.value)``.
    """

    def __init__(self, spec: ModelSpec):
        self._value_ = spec.id
        self.spec = spec
        self.supports_tools = spec.tools
        self.supports_thinking = spec.thinking
        self.supports_vision = spec.vision
        self.supports_audio = spec.audio
        self.supports_structured_output = spec.structured_output
        self.generation_kwargs = dict(spec.generation_kwargs or {})


class AdHocModel:
    """A ``Model``-like object for a model id not present in any provider enum.

    Built from a parsed model string's id plus capability flags, it mirrors the read
    surface of a :class:`Model` enum member so provider clients treat it exactly like an
    enum member. It is not an ``Enum`` member: the client factories route it by provider
    prefix rather than by ``isinstance``.
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.value = spec.id
        self.name = spec.id
        self.supports_tools = spec.tools
        self.supports_thinking = spec.thinking
        self.supports_vision = spec.vision
        self.supports_audio = spec.audio
        self.supports_structured_output = spec.structured_output
        self.generation_kwargs = dict(spec.generation_kwargs or {})


class BaseModelClient(_ChatStateMixin, ABC):
    """Abstract base for all provider clients.

    Subclasses implement :meth:`generate`, :meth:`chat`, and :meth:`_update_generate_kwargs`.
    Tool calling, message history, vision input, and streaming filters are handled here
    once for every provider.
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
        # ``tools`` is the transient set advertised to the model for the current call (swapped
        # by the ``tools=`` per-call override via ``_tools_override``); it is NOT a persistent
        # tool registry and the client never executes tools. Tool execution (dispatch, approval,
        # deps, concurrency) lives in the Agent's tool-loop engine (``aimu.agents._tool_loop``).
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
    def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Provider-specific generate implementation. Use :meth:`generate`.

        ``response_format`` (a JSON Schema dict) is passed only by the structured-output
        path and only to providers with ``supports_structured_output=True``; others never
        receive it and need not accept it.
        """
        pass

    @abstractmethod
    def _chat(
        self,
        user_message: Optional[str] = None,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Provider-specific single-turn chat implementation. Use :meth:`chat`.

        Issues exactly one model request. ``user_message=None`` runs a turn on the current
        messages without appending a user turn (continuation). If the model requests tools,
        execute them and return; do NOT make a follow-up generation. ``response_format``
        semantics match :meth:`_generate`.
        """
        pass

    def generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
        audio: Optional[list] = None,
        schema: Optional[type] = None,
    ) -> Union[str, Any, Iterator[StreamChunk]]:
        """Single-turn, stateless generation. See :meth:`chat` for the ``include`` filter semantics.

        Args:
            prompt: The text to generate from.
            generate_kwargs: Provider-specific generation parameters.
            stream: If True, return an iterator of :class:`StreamChunk` instead of a string.
            images: Optional list of images for vision-capable models, same accepted forms as
                :meth:`chat` (file path, ``pathlib.Path``, ``bytes``, http(s) URL, or data URL).
                Raises ``ValueError`` if the model does not support vision. Unlike :meth:`chat`,
                this does not touch ``self.messages``; the call stays single-turn and stateless.
            include: Optional iterable of stream phases to yield. Has no effect when ``stream=False``.
            audio: Optional list of audio clips for audio-capable models. Each entry may be a
                file path (str or ``pathlib.Path``), raw ``bytes``, a ``data:audio/...;base64,...``
                URL, or an http(s) URL (fetched eagerly). Raises ``ValueError`` if the model does
                not support audio input. Like ``images``, this does not touch ``self.messages``.
                ``images`` and ``audio`` are mutually exclusive.
            schema: Optional dataclass type or Pydantic v2 model. When set, returns a validated
                instance of that type instead of a string. See :meth:`chat` for the structured-output
                semantics (native enforcement when ``supports_structured_output`` is True, otherwise
                prompt-and-parse). With ``stream=True`` returns an iterator of :class:`StreamChunk`
                ending in a terminal ``DONE`` chunk whose ``content`` is ``{"result": <object>}``; the
                validated object is also stored on ``self.last_structured`` after the stream is consumed.
        """
        if images and audio:
            raise ValueError("images= and audio= are mutually exclusive. Pass one or the other, not both.")
        if images:
            self._require_vision()
        if audio:
            self._require_audio()
        if schema is not None:
            if stream:
                return self._generate_structured_streamed(prompt, generate_kwargs, images, audio, schema, include)
            return self._generate_structured(prompt, generate_kwargs, images, audio, schema)
        result = self._generate(prompt, generate_kwargs, stream=stream, images=images, audio=audio)
        if stream and include is not None:
            return self._filter_chunks(result, self._resolve_include(include))
        return result

    def chat(
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
    ) -> Union[str, Any, Iterator[StreamChunk]]:
        """One model turn against the persistent message history.

        A single call issues exactly one model request. If the model requests tools, they are
        executed and their results appended, and the call returns (the model's *response* to the
        tool results comes on the next :meth:`chat` call — the multi-turn tool loop lives in
        :class:`~aimu.agents.Agent`, which wraps this method). Message history persists across
        calls on the same client.

        Args:
            user_message: The text the user is sending this turn. Pass ``None`` (the default) to
                run a turn on the *current* messages without appending a new user turn — the
                continuation primitive the agent loop uses after a tool turn.
            generate_kwargs: Provider-specific generation parameters. Unknown keys are
                dropped per-provider; see each client for accepted names.
            use_tools: If False, suppress tool calls even when the model supports tools.
            stream: If True, return an iterator of :class:`StreamChunk` instead of a string.
            images: Optional list of images for vision-capable models. Each entry may be a
                file path (str or ``pathlib.Path``), raw ``bytes``, an ``http(s)://`` URL,
                or a ``data:image/...;base64,...`` URL. Raises ``ValueError`` if the model
                does not support vision. Only used on the initial user turn.
            include: Optional iterable of stream phases to yield. Defaults to all phases
                (THINKING, TOOL_CALLING, GENERATING, DONE). Has no effect when ``stream=False``.
                Values may be :class:`StreamingContentType` members or their string equivalents
                (``"thinking"``, ``"tool_calling"``, ``"generating"``, ``"done"``).
            tools: Optional per-call override of the Python ``@tool`` callables. ``None``
                (default) uses the client's configured ``self.tools``; any other value
                (including ``[]`` to disable Python tools for this call) replaces them for
                this call only and is restored afterwards (MCP tools, being callables in
                ``self.tools`` via ``MCPClient.as_tools()``, are included in the swap).
                Ignored when ``use_tools=False``.
            audio: Optional list of audio clips for audio-capable models. Same accepted forms
                as :meth:`generate`. Raises ``ValueError`` if the model does not support audio
                input. Audio blocks persist in ``self.messages`` for multi-turn context.
                ``images`` and ``audio`` are mutually exclusive per turn.
            schema: Optional dataclass type or Pydantic v2 model. When set, returns a validated
                instance of that type instead of a string. If the model has
                ``supports_structured_output=True`` the provider enforces the schema natively
                (OpenAI ``response_format``, Ollama ``format=``, Anthropic forced-tool); otherwise
                the schema is appended to the prompt and the response is parsed. Raises
                ``ValueError`` on parse failure. With ``stream=True`` returns an iterator of
                :class:`StreamChunk`: thinking/generation chunks stream live, then a terminal
                ``DONE`` chunk carries ``{"result": <object>}`` and ``self.last_structured`` is set
                once the stream is consumed. (Anthropic streams the JSON as it is built but emits no
                thinking, since its forced-tool structured mode is incompatible with extended thinking.)
                On Anthropic (native, forced-tool) combining ``schema`` with active ``tools`` raises.
        """
        if schema is not None:
            if stream:
                return self._chat_structured_streamed(
                    user_message, generate_kwargs, use_tools, images, audio, tools, schema, include
                )
            return self._chat_structured(user_message, generate_kwargs, use_tools, images, audio, tools, schema)

        if tools is None:
            result = self._chat(
                user_message, generate_kwargs, use_tools=use_tools, stream=stream, images=images, audio=audio
            )
            if stream and include is not None:
                return self._filter_chunks(result, self._resolve_include(include))
            return result

        if stream:
            return self._chat_with_tools_streamed(
                user_message, generate_kwargs, use_tools, images, include, tools, audio=audio
            )
        with self._tools_override(tools):
            return self._chat(
                user_message, generate_kwargs, use_tools=use_tools, stream=False, images=images, audio=audio
            )

    def _chat_structured(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]],
        use_tools: bool,
        images: Optional[list],
        audio: Optional[list],
        tools: Optional[list],
        schema: type,
    ) -> Any:
        from .._internal.json import parse_json_response

        response_format, suffix = self._structured_request(schema)
        message = user_message + suffix
        extra = {"response_format": response_format} if response_format is not None else {}
        if tools is None:
            text = self._chat(
                message, generate_kwargs, use_tools=use_tools, stream=False, images=images, audio=audio, **extra
            )
        else:
            with self._tools_override(tools):
                text = self._chat(
                    message, generate_kwargs, use_tools=use_tools, stream=False, images=images, audio=audio, **extra
                )
        return parse_json_response(text, schema)

    def _generate_structured(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]],
        images: Optional[list],
        audio: Optional[list],
        schema: type,
    ) -> Any:
        from .._internal.json import parse_json_response

        response_format, suffix = self._structured_request(schema)
        extra = {"response_format": response_format} if response_format is not None else {}
        text = self._generate(prompt + suffix, generate_kwargs, stream=False, images=images, audio=audio, **extra)
        return parse_json_response(text, schema)

    def _chat_structured_streamed(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]],
        use_tools: bool,
        images: Optional[list],
        audio: Optional[list],
        tools: Optional[list],
        schema: type,
        include: Optional[Iterable[Union[str, StreamingContentType]]],
    ) -> Iterator[StreamChunk]:
        """Streamed structured output for :meth:`chat`.

        Yields the provider's live THINKING / GENERATING / TOOL_CALLING chunks (subject to
        ``include=``), accumulates the GENERATING text, then parses it into ``schema`` and
        yields a terminal ``StreamChunk(DONE, {"result": obj})`` (always, regardless of
        ``include=``, so the payload is never filtered away). ``self.last_structured`` is set
        just before the terminal chunk.
        """
        from .._internal.json import parse_json_response

        response_format, suffix = self._structured_request(schema)
        message = user_message + suffix
        extra = {"response_format": response_format} if response_format is not None else {}
        selected = self._resolve_include(include) if include is not None else None

        def _run() -> Iterator[StreamChunk]:
            buffer: list[str] = []
            result = self._chat(
                message, generate_kwargs, use_tools=use_tools, stream=True, images=images, audio=audio, **extra
            )
            for chunk in result:
                if chunk.phase == StreamingContentType.GENERATING and isinstance(chunk.content, str):
                    buffer.append(chunk.content)
                if selected is None or chunk.phase in selected:
                    yield chunk
            obj = parse_json_response("".join(buffer), schema)
            self.last_structured = obj
            yield StreamChunk(StreamingContentType.DONE, {"result": obj})

        if tools is None:
            yield from _run()
        else:
            with self._tools_override(tools):
                yield from _run()

    def _generate_structured_streamed(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]],
        images: Optional[list],
        audio: Optional[list],
        schema: type,
        include: Optional[Iterable[Union[str, StreamingContentType]]],
    ) -> Iterator[StreamChunk]:
        """Streamed structured output for :meth:`generate`. See :meth:`_chat_structured_streamed`."""
        from .._internal.json import parse_json_response

        response_format, suffix = self._structured_request(schema)
        extra = {"response_format": response_format} if response_format is not None else {}
        selected = self._resolve_include(include) if include is not None else None
        buffer: list[str] = []
        result = self._generate(prompt + suffix, generate_kwargs, stream=True, images=images, audio=audio, **extra)
        for chunk in result:
            if chunk.phase == StreamingContentType.GENERATING and isinstance(chunk.content, str):
                buffer.append(chunk.content)
            if selected is None or chunk.phase in selected:
                yield chunk
        obj = parse_json_response("".join(buffer), schema)
        self.last_structured = obj
        yield StreamChunk(StreamingContentType.DONE, {"result": obj})

    def _chat_with_tools_streamed(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]],
        use_tools: bool,
        images: Optional[list],
        include: Optional[Iterable[Union[str, StreamingContentType]]],
        tools: list,
        audio: Optional[list] = None,
    ) -> Iterator[StreamChunk]:
        """Streaming chat with a per-call ``tools`` override.

        The override must stay active while the stream is consumed (the provider only
        executes the generator on iteration, and tool dispatch reads ``self.tools``), so
        the swap wraps the ``yield from`` rather than just the ``_chat`` call.
        """
        with self._tools_override(tools):
            result = self._chat(
                user_message, generate_kwargs, use_tools=use_tools, stream=True, images=images, audio=audio
            )
            if include is not None:
                result = self._filter_chunks(result, self._resolve_include(include))
            yield from result

    @abstractmethod
    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        pass

    @staticmethod
    def _resolve_include(
        include: Iterable[Union[str, StreamingContentType]],
    ) -> set[StreamingContentType]:
        """Normalise an ``include=`` argument to a set of :class:`StreamingContentType`."""
        return _resolve_include_fn(include)

    @staticmethod
    def _filter_chunks(
        chunks: Iterator[StreamChunk],
        include: set[StreamingContentType],
    ) -> Iterator[StreamChunk]:
        """Drop chunks whose phase isn't in the include set."""
        return _filter_chunks_fn(chunks, include)

    def _chat_setup(
        self,
        user_message: Optional[str] = None,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        # user_message=None is the continuation primitive: run a turn on the current messages
        # without appending a new user turn (the agent loop uses it after a tool turn).
        if user_message is not None:
            self._append_user_turn(user_message, images, audio)

        tools: list[dict] = []
        if self.model.supports_tools and use_tools:
            tools.extend(self._collect_python_tool_specs())

        return generate_kwargs, tools
