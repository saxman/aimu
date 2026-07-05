"""Async base class for provider model clients.

Mirrors ``aimu.models.base.BaseModelClient`` but with async ``_chat`` / ``_generate``
abstract methods. State-management mechanics (system_message lifecycle, reset,
message-history append, image-block normalization) are inherited from the shared
``_ChatStateMixin``.
"""

from __future__ import annotations

import asyncio
import logging
import random
import string
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
        self.tools: list = []
        self.last_thinking = ""
        self.last_usage = None
        self.last_structured = None
        self.concurrent_tool_calls = False
        # Value injected as ``ctx.deps`` into tools that declare a ToolContext parameter.
        # Set by Agent from its deps= field / run(deps=) override; None for bare chat().
        self.tool_context_deps = None
        # Optional gate run before each tool call: (tool_name, arguments) -> bool (a coroutine is
        # awaited). Default approves everything (no behavior change). Set directly for bare chat(),
        # or via Agent(tool_approval=...) / run(tool_approval=...).
        from aimu.tools.approval import approve_all

        self.tool_approval = approve_all

    async def _tool_call_approved(self, name: str, arguments: dict) -> bool:
        """Run the approval policy for one tool call (async); awaits a coroutine result."""
        import inspect

        from aimu.tools.approval import approve_all

        policy = getattr(self, "tool_approval", None) or approve_all
        result = policy(name, arguments)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

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

    def _structured_request(self, schema: type) -> tuple[Optional[dict], str]:
        """Resolve a schema to ``(response_format, prompt_suffix)`` for the active model.

        Native models get the JSON Schema dict; parse-path models get ``None`` and an
        instruction suffix. Mirrors :meth:`BaseModelClient._structured_request`.
        """
        from aimu.models._internal.structured import json_schema_instruction, schema_to_json_schema

        json_schema = schema_to_json_schema(schema)
        if self.supports_structured_output:
            return json_schema, ""
        return None, "\n\n" + json_schema_instruction(json_schema)

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

    def _prepare_tool_calls(self, tool_calls: list[dict]) -> list[tuple[dict, str]]:
        """Normalize ``arguments``/``parameters`` and assign tool_call_ids upfront."""
        prepared = []
        for tc in tool_calls:
            if "arguments" not in tc and "parameters" in tc:
                tc["arguments"] = tc.pop("parameters")
            tc_id = "".join(random.choices(string.ascii_letters + string.digits, k=9))
            prepared.append((tc, tc_id))
        return prepared

    def _append_assistant_tool_calls(self, prepared: list[tuple[dict, str]], content: str = "") -> None:
        """Append the assistant message that records the tool calls being made.

        ``content`` is any text the model emitted alongside the tool call in the same turn; it
        is stored on the message when non-empty (see the sync surface for rationale).
        """
        message: dict = {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}, "id": tc_id}
                for tc, tc_id in prepared
            ],
        }
        if content:
            message["content"] = content
        self.messages.append(message)

    async def _call_plain_tool(self, tc: dict, tc_id: str) -> dict:
        """Dispatch a single non-streaming tool call. Returns the tool message dict.

        Awaits async tools directly; routes sync tools through ``asyncio.to_thread``.
        Every tool (in-process ``@tool`` functions and MCP tools alike) lives in
        ``self.tools`` as a callable (MCP tools are wrapped by ``aio.MCPClient.as_tools()``),
        so dispatch is a single by-name lookup. Generator (streaming) tools are rejected here.
        """
        from aimu.tools.decorator import ToolArgumentError

        python_tools_by_name = {fn.__name__: fn for fn in self.tools}
        fn = python_tools_by_name.get(tc["name"])
        if fn is not None:
            if getattr(fn, "__tool_is_streaming__", False):
                raise ValueError(
                    f"Tool '{tc['name']}' is a generator (streaming) tool. Streaming tools "
                    "require the streaming dispatch path: call chat() / agent.run() with "
                    "stream=True. For non-streaming use, convert the tool to a plain function."
                )
            if not await self._tool_call_approved(tc["name"], tc["arguments"]):
                return self._tool_not_approved_message(tc, tc_id)
            try:
                kwargs = self._tool_call_kwargs(fn, tc["arguments"])
                if getattr(fn, "__tool_is_async__", False):
                    response = await fn(**kwargs)
                else:
                    response = await asyncio.to_thread(lambda: fn(**kwargs))
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

    async def _handle_tool_calls(self, tool_calls: list[dict], content: str = "") -> None:
        """Non-streaming async tool dispatch, used by non-streaming ``_chat`` paths.

        ``content`` is any text the model emitted alongside the tool call this turn; it is
        stored on the assistant message. Streaming (generator) tools are rejected; call
        ``chat(stream=True)`` to dispatch them via :meth:`_handle_tool_calls_streamed`.
        """
        prepared = self._prepare_tool_calls(tool_calls)
        self._append_assistant_tool_calls(prepared, content)

        if self.concurrent_tool_calls and len(prepared) > 1:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._call_plain_tool(tc, tc_id)) for tc, tc_id in prepared]
            results = [t.result() for t in tasks]
        else:
            results = [await self._call_plain_tool(tc, tc_id) for tc, tc_id in prepared]

        self.messages.extend(results)

    async def _handle_tool_calls_streamed(
        self, tool_calls: list[dict], content: str = ""
    ) -> AsyncIterator[StreamChunk]:
        """Async streaming tool dispatch, used by streaming ``_chat`` paths.

        ``content`` is any text the model emitted alongside the tool call this turn; it is
        stored on the assistant message. Mirrors :meth:`BaseModelClient._handle_tool_calls_streamed`. Async generator tools (``async def fn(): yield ...``) are drained
        via ``async for``; sync generator tools are dispatched through
        ``asyncio.to_thread`` (one step at a time, the thread re-enters per
        ``next()`` so the event loop stays free between yields).

        Result extraction priority for streaming tools (matching the sync version):

        1. Sync generators: ``StopIteration.value``.
        2. The last yielded chunk's ``content["result"]`` if dict-with-key.
        3. ``str(last_chunk.content)``.

        Async generators have no ``return``-value channel (PEP 525) so (1) does
        not apply; (2)/(3) cover them.

        **Concurrency**: when ``concurrent_tool_calls=True`` and no tools in the
        batch are streaming, the existing ``asyncio.TaskGroup`` fast path is used.
        With any streaming tool, dispatch is sequential.
        """
        from aimu.tools.decorator import ToolArgumentError

        prepared = self._prepare_tool_calls(tool_calls)
        self._append_assistant_tool_calls(prepared, content)

        python_tools_by_name = {fn.__name__: fn for fn in self.tools}
        has_streaming_tool = any(
            getattr(python_tools_by_name.get(tc["name"]), "__tool_is_streaming__", False) for tc, _ in prepared
        )

        if self.concurrent_tool_calls and len(prepared) > 1 and not has_streaming_tool:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._call_plain_tool(tc, tc_id)) for tc, tc_id in prepared]
            results = [t.result() for t in tasks]
            for (tc, _tc_id), result_msg in zip(prepared, results):
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
                if not await self._tool_call_approved(tc["name"], tc["arguments"]):
                    result_msg = self._tool_not_approved_message(tc, tc_id)
                    self.messages.append(result_msg)
                    yield StreamChunk(
                        StreamingContentType.TOOL_CALLING,
                        {"name": tc["name"], "arguments": tc["arguments"], "response": result_msg["content"]},
                    )
                    continue
                try:
                    return_value: Any = None
                    last_content: Any = None
                    kwargs = self._tool_call_kwargs(fn, tc["arguments"])
                    if getattr(fn, "__tool_is_async__", False):
                        # Async generator tool, drained with `async for`.
                        agen = fn(**kwargs)
                        async for chunk in agen:
                            yield chunk
                            last_content = chunk.content
                        # Async generators have no return-value channel.
                    else:
                        # Sync generator dispatched in the async surface: pull steps
                        # via to_thread so the event loop stays free between yields.
                        gen = fn(**kwargs)
                        _SENTINEL = object()

                        def _next_chunk(_gen=gen):
                            try:
                                return next(_gen), None
                            except StopIteration as stop:
                                return _SENTINEL, stop.value

                        while True:
                            chunk, ret_val = await asyncio.to_thread(_next_chunk)
                            if chunk is _SENTINEL:
                                return_value = ret_val
                                break
                            yield chunk
                            last_content = chunk.content
                    if return_value is not None:
                        response: Any = return_value
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
                result_msg = await self._call_plain_tool(tc, tc_id)

            self.messages.append(result_msg)
            yield StreamChunk(
                StreamingContentType.TOOL_CALLING,
                {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "response": result_msg["content"],
                },
            )
