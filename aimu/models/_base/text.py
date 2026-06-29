"""Text-modality base types: ``ModelSpec``, the ``Model`` enum, and ``BaseModelClient``.

``BaseModelClient`` is the abstract base for every text/chat provider. Tool calling,
message history, vision input, and streaming filters live here once for all providers;
concrete clients implement ``_chat`` / ``_generate`` / ``_update_generate_kwargs``.
"""

import logging
import random
import string
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

    @abstractmethod
    def __init__(self, model: Model, model_kwargs: Optional[dict] = None, system_message: Optional[str] = None):
        self.model = model
        self.model_kwargs = model_kwargs
        self._system_message = system_message
        self.default_generate_kwargs = {}
        self.messages = []
        self.tools: list = []
        self.last_thinking = ""
        self.last_usage = None
        self.concurrent_tool_calls = False
        # Value injected as ``ctx.deps`` into tools that declare a ToolContext parameter.
        # Set by Agent from its deps= field / run(deps=) override; None for bare chat().
        self.tool_context_deps = None
        # Optional gate run before each tool call: (tool_name, arguments) -> bool. Default
        # approves everything (no behavior change). Set directly for bare chat(), or via
        # Agent(tool_approval=...) / run(tool_approval=...). Sync surface: must be a plain function.
        from aimu.tools.approval import approve_all

        self.tool_approval = approve_all

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    def _tool_call_approved(self, name: str, arguments: dict) -> bool:
        """Run the approval policy for one tool call (sync). Default approves everything.

        A coroutine-returning policy can't be awaited here; raise a clear error pointing at the
        async surface rather than silently approving.
        """
        import inspect

        from aimu.tools.approval import approve_all

        policy = getattr(self, "tool_approval", None) or approve_all
        result = policy(name, arguments)
        if inspect.isawaitable(result):
            result.close()  # avoid "coroutine was never awaited" warning
            raise ValueError(
                "tool_approval returned a coroutine on the sync client. Use a synchronous policy "
                "(plain function), or run on the aimu.aio surface for async approval."
            )
        return bool(result)

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
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Provider-specific chat implementation. Use :meth:`chat`.

        ``response_format`` semantics match :meth:`_generate`.
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
                prompt-and-parse). Mutually exclusive with ``stream=True``.
        """
        if images and audio:
            raise ValueError("images= and audio= are mutually exclusive. Pass one or the other, not both.")
        if images:
            self._require_vision()
        if audio:
            self._require_audio()
        if schema is not None:
            if stream:
                raise ValueError("schema= and stream=True are mutually exclusive (a typed object can't be streamed).")
            return self._generate_structured(prompt, generate_kwargs, images, audio, schema)
        result = self._generate(prompt, generate_kwargs, stream=stream, images=images, audio=audio)
        if stream and include is not None:
            return self._filter_chunks(result, self._resolve_include(include))
        return result

    def chat(
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
    ) -> Union[str, Any, Iterator[StreamChunk]]:
        """Multi-turn chat with persistent message history.

        Args:
            user_message: The text the user is sending this turn.
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
                ``ValueError`` on parse failure. Mutually exclusive with ``stream=True``. On
                Anthropic (native, forced-tool) combining ``schema`` with active ``tools`` raises.
        """
        if schema is not None:
            if stream:
                raise ValueError("schema= and stream=True are mutually exclusive (a typed object can't be streamed).")
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

    def _structured_request(self, schema: type) -> tuple[Optional[dict], str]:
        """Resolve a schema to ``(response_format, prompt_suffix)`` for the active model.

        Native models get the JSON Schema dict as ``response_format`` and no prompt suffix;
        parse-path models get ``None`` and a suffix instructing JSON output. The provider
        only ever receives ``response_format`` when it's non-None (native).
        """
        from .._internal.structured import json_schema_instruction, schema_to_json_schema

        json_schema = schema_to_json_schema(schema)
        if self.supports_structured_output:
            return json_schema, ""
        return None, "\n\n" + json_schema_instruction(json_schema)

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
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        self._append_user_turn(user_message, images, audio)

        tools: list[dict] = []
        if self.model.supports_tools and use_tools:
            tools.extend(self._collect_python_tool_specs())

        return generate_kwargs, tools

    def _prepare_tool_calls(self, tool_calls: list[dict]) -> list[tuple[dict, str]]:
        """Normalize ``arguments``/``parameters`` and assign tool_call_ids upfront.

        Concurrent execution can use pre-assigned IDs and still append results in
        original order.
        """
        prepared = []
        for tc in tool_calls:
            # llama 3.1 uses 'parameters' instead of 'arguments'
            if "arguments" not in tc and "parameters" in tc:
                tc["arguments"] = tc.pop("parameters")
            tc_id = "".join(random.choices(string.ascii_letters + string.digits, k=9))
            prepared.append((tc, tc_id))
        return prepared

    def _append_assistant_tool_calls(self, prepared: list[tuple[dict, str]]) -> None:
        """Append the assistant message that records the tool calls being made."""
        self.messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}, "id": tc_id}
                    for tc, tc_id in prepared
                ],
            }
        )

    def _call_plain_tool(self, tc: dict, tc_id: str) -> dict:
        """Dispatch a single non-streaming tool call. Returns the tool message dict.

        Every tool (in-process ``@tool`` functions and MCP tools alike) lives in
        ``self.tools`` as a callable (MCP tools are wrapped by ``MCPClient.as_tools()``),
        so dispatch is a single by-name lookup.
        """
        from aimu.tools.decorator import ToolArgumentError

        python_tools_by_name = {fn.__name__: fn for fn in self.tools}
        fn = python_tools_by_name.get(tc["name"])
        if fn is not None:
            if getattr(fn, "__tool_is_async__", False):
                raise ValueError(
                    f"Tool '{tc['name']}' is an async function (`async def`). The sync "
                    "BaseModelClient cannot dispatch async tools. Use the aimu.aio surface, "
                    "or convert the tool to a regular `def`."
                )
            if getattr(fn, "__tool_is_streaming__", False):
                raise ValueError(
                    f"Tool '{tc['name']}' is a generator (streaming) tool. Streaming tools "
                    "require the streaming dispatch path: call chat() / agent.run() with "
                    "stream=True. For non-streaming use, convert the tool to a plain function."
                )
            if not self._tool_call_approved(tc["name"], tc["arguments"]):
                return self._tool_not_approved_message(tc, tc_id)
            try:
                response = fn(**self._tool_call_kwargs(fn, tc["arguments"]))
                content = str(response)
            except ToolArgumentError as exc:
                content = str(exc)
            except Exception as exc:
                content = f"Tool '{tc['name']}' raised an error: {exc}"
                logger.warning("Tool call '%s' failed: %s", tc["name"], exc)
            return {"role": "tool", "name": tc["name"], "content": content, "tool_call_id": tc_id}

        return {
            "role": "tool",
            "name": tc["name"],
            "content": f"Tool '{tc['name']}' not found.",
            "tool_call_id": tc_id,
        }

    def _handle_tool_calls(self, tool_calls: list[dict]) -> None:
        """Non-streaming tool dispatch, used by non-streaming ``_chat`` paths.

        Streaming (generator) tools are rejected here; call ``chat(stream=True)``
        to dispatch them via :meth:`_handle_tool_calls_streamed`.
        """
        prepared = self._prepare_tool_calls(tool_calls)
        self._append_assistant_tool_calls(prepared)

        if self.concurrent_tool_calls and len(prepared) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._call_plain_tool, tc, tc_id) for tc, tc_id in prepared]
                results = [f.result() for f in futures]
        else:
            results = [self._call_plain_tool(tc, tc_id) for tc, tc_id in prepared]

        self.messages.extend(results)

    def _handle_tool_calls_streamed(self, tool_calls: list[dict]) -> Iterator[StreamChunk]:
        """Streaming tool dispatch: generator version used by streaming ``_chat``.

        For each tool call, yields zero-or-more in-flight :class:`StreamChunk` objects
        (when the tool is a generator function decorated with ``@tool``) followed by a
        final ``TOOL_CALLING`` chunk with the tool's canonical response.

        **Result extraction for streaming tools**, in priority order:

        1. The generator's ``return`` value (captured via ``StopIteration.value``).
        2. The last yielded chunk's ``content["result"]`` if it's a dict with that key
           (matches the convention used by ``IMAGE_GENERATING`` final chunks).
        3. The last yielded chunk's content (stringified).

        **Concurrency**: when ``concurrent_tool_calls=True`` and *no* tools in the
        batch are streaming, the existing ``ThreadPoolExecutor`` path is reused for
        speed. When any tool is streaming, dispatch is sequential (chunks from
        concurrent generators would interleave).
        """
        from aimu.tools.decorator import ToolArgumentError

        prepared = self._prepare_tool_calls(tool_calls)
        self._append_assistant_tool_calls(prepared)

        python_tools_by_name = {fn.__name__: fn for fn in self.tools}
        has_streaming_tool = any(
            getattr(python_tools_by_name.get(tc["name"]), "__tool_is_streaming__", False) for tc, _ in prepared
        )

        if self.concurrent_tool_calls and len(prepared) > 1 and not has_streaming_tool:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._call_plain_tool, tc, tc_id) for tc, tc_id in prepared]
                results = [f.result() for f in futures]
            for (tc, tc_id), result_msg in zip(prepared, results):
                self.messages.append(result_msg)
                yield StreamChunk(
                    StreamingContentType.TOOL_CALLING,
                    {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "response": result_msg["content"],
                    },
                )
            return

        for tc, tc_id in prepared:
            fn = python_tools_by_name.get(tc["name"])
            if fn is not None and getattr(fn, "__tool_is_streaming__", False):
                if getattr(fn, "__tool_is_async__", False):
                    raise ValueError(
                        f"Tool '{tc['name']}' is an async streaming tool. Use the aimu.aio surface to dispatch it."
                    )
                if not self._tool_call_approved(tc["name"], tc["arguments"]):
                    result_msg = self._tool_not_approved_message(tc, tc_id)
                    self.messages.append(result_msg)
                    yield StreamChunk(
                        StreamingContentType.TOOL_CALLING,
                        {"name": tc["name"], "arguments": tc["arguments"], "response": result_msg["content"]},
                    )
                    continue
                try:
                    gen = fn(**self._tool_call_kwargs(fn, tc["arguments"]))
                    return_value = None
                    last_content: Any = None
                    while True:
                        try:
                            chunk = next(gen)
                        except StopIteration as stop:
                            return_value = stop.value
                            break
                        yield chunk
                        last_content = chunk.content
                    if return_value is not None:
                        response = return_value
                    elif isinstance(last_content, dict) and "result" in last_content:
                        response = last_content["result"]
                    else:
                        response = last_content if last_content is not None else "(no response)"
                    content = str(response)
                except ToolArgumentError as exc:
                    content = str(exc)
                except Exception as exc:
                    content = f"Tool '{tc['name']}' raised an error: {exc}"
                    logger.warning("Tool call '%s' failed: %s", tc["name"], exc)
                result_msg = {"role": "tool", "name": tc["name"], "content": content, "tool_call_id": tc_id}
            else:
                result_msg = self._call_plain_tool(tc, tc_id)

            self.messages.append(result_msg)
            yield StreamChunk(
                StreamingContentType.TOOL_CALLING,
                {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "response": result_msg["content"],
                },
            )
