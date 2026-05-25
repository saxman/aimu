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
        include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        """Single-turn async generation. See :meth:`chat` for ``include`` semantics."""
        result = await self._generate(prompt, generate_kwargs, stream=stream)
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

    async def _handle_tool_calls(self, tool_calls: list[dict], tools: list) -> None:
        """Async tool dispatch.

        Mirrors the sync version's structure but awaits async tools and dispatches
        sync ones via ``asyncio.to_thread``. When ``concurrent_tool_calls=True``,
        uses ``asyncio.TaskGroup`` for structured concurrency: a failing tool
        cancels in-flight siblings and surfaces all errors as ``ExceptionGroup``.
        """
        prepared = []
        for tc in tool_calls:
            if "arguments" not in tc and "parameters" in tc:
                tc["arguments"] = tc.pop("parameters")
            tc_id = "".join(random.choices(string.ascii_letters + string.digits, k=9))
            prepared.append((tc, tc_id))

        self.messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}, "id": tc_id}
                    for tc, tc_id in prepared
                ],
            }
        )

        python_tools_by_name = {fn.__name__: fn for fn in self.tools}

        async def _call_one(tc: dict, tc_id: str) -> dict:
            fn = python_tools_by_name.get(tc["name"])
            if fn is not None:
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
                        response_content = (
                            tool_response if isinstance(tool_response, list) else tool_response.content
                        )
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

        if self.concurrent_tool_calls and len(prepared) > 1:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(_call_one(tc, tc_id)) for tc, tc_id in prepared]
            results = [t.result() for t in tasks]
        else:
            results = [await _call_one(tc, tc_id) for tc, tc_id in prepared]

        self.messages.extend(results)
