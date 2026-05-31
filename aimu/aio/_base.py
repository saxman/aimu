"""Async base class for provider model clients.

Mirrors ``aimu.models.base.BaseModelClient`` but with async ``_chat`` / ``_generate``
abstract methods. State-management mechanics (system_message lifecycle, reset,
message-history append, image-block normalization) are inherited from the shared
``_ChatStateMixin``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import string
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterable, Optional, Union

from aimu.models._chat_state import _ChatStateMixin
from aimu.models._streaming import afilter_chunks, resolve_include
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
    _system_message_locked: bool
    default_generate_kwargs: dict
    messages: list[dict]
    mcp_client: Optional[Any]
    last_thinking: str | None

    @abstractmethod
    def __init__(self, model: Model, model_kwargs: Optional[dict] = None, system_message: Optional[str] = None):
        self.model = model
        self.model_kwargs = model_kwargs
        self._system_message = system_message
        self._system_message_locked = False
        self.default_generate_kwargs = {}
        self.messages = []
        self.mcp_client = None
        self.tools: list = []
        self.last_thinking = ""
        self.concurrent_tool_calls = False

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    @classproperty
    def VISION_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    @abstractmethod
    async def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        """Provider-specific async generate implementation. Use :meth:`generate`."""
        pass

    @abstractmethod
    async def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        """Provider-specific async chat implementation. Use :meth:`chat`."""
        pass

    async def generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        """Single-turn, stateless async generation. See sync :meth:`BaseModelClient.generate`.

        ``images`` accepts the same forms as :meth:`chat` and raises ``ValueError`` for
        non-vision models; the call stays stateless (does not touch ``self.messages``).
        """
        if images:
            self._require_vision()
        result = await self._generate(prompt, generate_kwargs, stream=stream, images=images)
        if stream and include is not None:
            return afilter_chunks(result, resolve_include(include))
        return result

    async def chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        """Multi-turn async chat with persistent message history.

        Args mirror the sync :meth:`BaseModelClient.chat`. Streaming returns an
        ``AsyncIterator[StreamChunk]`` — consume with ``async for``.
        """
        result = await self._chat(user_message, generate_kwargs, use_tools=use_tools, stream=stream, images=images)
        if stream and include is not None:
            return afilter_chunks(result, resolve_include(include))
        return result

    @abstractmethod
    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        pass

    async def _chat_setup(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        images: Optional[list] = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Async equivalent of the sync ``_chat_setup``. The only async point is the
        MCP ``get_tools()`` call when an async ``mcp_client`` is attached."""
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        self._append_user_turn(user_message, images)

        tools: list[dict] = []
        if self.model.supports_tools and use_tools:
            if self.mcp_client:
                mcp_tools = self.mcp_client.get_tools()
                if inspect.iscoroutine(mcp_tools):
                    mcp_tools = await mcp_tools
                tools.extend(mcp_tools)
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

    async def _call_plain_tool(self, tc: dict, tc_id: str, tools: list) -> dict:
        """Dispatch a single non-streaming tool call. Returns the tool message dict.

        Awaits async tools directly; routes sync tools through ``asyncio.to_thread``.
        Generator (streaming) tools are rejected here.
        """
        python_tools_by_name = {fn.__name__: fn for fn in self.tools}
        fn = python_tools_by_name.get(tc["name"])
        if fn is not None:
            if getattr(fn, "__tool_is_streaming__", False):
                raise ValueError(
                    f"Tool '{tc['name']}' is a generator (streaming) tool. Streaming tools "
                    "require the streaming dispatch path — call chat() / agent.run() with "
                    "stream=True. For non-streaming use, convert the tool to a plain function."
                )
            try:
                if getattr(fn, "__tool_is_async__", False):
                    response = await fn(**tc["arguments"])
                else:
                    response = await asyncio.to_thread(lambda: fn(**tc["arguments"]))
                content = str(response)
            except Exception as exc:
                content = f"Tool '{tc['name']}' raised an error: {exc}"
                logger.warning("Tool call '%s' failed: %s", tc["name"], exc)
            return {"role": "tool", "name": tc["name"], "content": content, "tool_call_id": tc_id}

        for tool in tools:
            if tool["type"] == "function" and tool["function"]["name"] == tc["name"]:
                if self.mcp_client is None:
                    raise ValueError(
                        "MCP client not initialized. Please initialize and assign an MCP client before using MCP tools."
                    )
                try:
                    tool_response = self.mcp_client.call_tool(tool["function"]["name"], tc["arguments"])
                    if inspect.iscoroutine(tool_response):
                        tool_response = await tool_response
                    response_content = tool_response if isinstance(tool_response, list) else tool_response.content
                    content = ""
                    for part in response_content:
                        if part.type == "text":
                            content += part.text
                        else:
                            logger.debug("Skipping unsupported tool response part type: %s", part.type)
                except Exception as exc:
                    content = f"Tool '{tc['name']}' raised an error: {exc}"
                    logger.warning("Tool call '%s' failed: %s", tc["name"], exc)
                return {"role": "tool", "name": tool["function"]["name"], "content": content, "tool_call_id": tc_id}

        return {
            "role": "tool",
            "name": tc["name"],
            "content": f"Tool '{tc['name']}' not found.",
            "tool_call_id": tc_id,
        }

    async def _handle_tool_calls(self, tool_calls: list[dict], tools: list) -> None:
        """Non-streaming async tool dispatch — used by non-streaming ``_chat`` paths.

        Streaming (generator) tools are rejected; call ``chat(stream=True)`` to
        dispatch them via :meth:`_handle_tool_calls_streamed`.
        """
        prepared = self._prepare_tool_calls(tool_calls)
        self._append_assistant_tool_calls(prepared)

        if self.concurrent_tool_calls and len(prepared) > 1:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._call_plain_tool(tc, tc_id, tools)) for tc, tc_id in prepared]
            results = [t.result() for t in tasks]
        else:
            results = [await self._call_plain_tool(tc, tc_id, tools) for tc, tc_id in prepared]

        self.messages.extend(results)

    async def _handle_tool_calls_streamed(self, tool_calls: list[dict], tools: list) -> AsyncIterator[StreamChunk]:
        """Async streaming tool dispatch — used by streaming ``_chat`` paths.

        Mirrors :meth:`BaseModelClient._handle_tool_calls_streamed` on the sync
        surface. Async generator tools (``async def fn(): yield ...``) are drained
        via ``async for``; sync generator tools are dispatched through
        ``asyncio.to_thread`` (one step at a time — the thread re-enters per
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
        prepared = self._prepare_tool_calls(tool_calls)
        self._append_assistant_tool_calls(prepared)

        python_tools_by_name = {fn.__name__: fn for fn in self.tools}
        has_streaming_tool = any(
            getattr(python_tools_by_name.get(tc["name"]), "__tool_is_streaming__", False) for tc, _ in prepared
        )

        if self.concurrent_tool_calls and len(prepared) > 1 and not has_streaming_tool:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._call_plain_tool(tc, tc_id, tools)) for tc, tc_id in prepared]
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
                try:
                    return_value: Any = None
                    last_content: Any = None
                    if getattr(fn, "__tool_is_async__", False):
                        # Async generator tool — drain with `async for`.
                        agen = fn(**tc["arguments"])
                        async for chunk in agen:
                            yield chunk
                            last_content = chunk.content
                        # Async generators have no return-value channel.
                    else:
                        # Sync generator dispatched in the async surface — pull steps
                        # via to_thread so the event loop stays free between yields.
                        gen = fn(**tc["arguments"])
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
                except Exception as exc:
                    content = f"Tool '{tc['name']}' raised an error: {exc}"
                    logger.warning("Tool call '%s' failed: %s", tc["name"], exc)
                result_msg = {"role": "tool", "name": tc["name"], "content": content, "tool_call_id": tc_id}
            else:
                result_msg = await self._call_plain_tool(tc, tc_id, tools)

            self.messages.append(result_msg)
            yield StreamChunk(
                StreamingContentType.TOOL_CALLING,
                {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "response": result_msg["content"],
                },
            )
