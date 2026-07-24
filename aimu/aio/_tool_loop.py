"""The iterative tool-calling engine (async twin of ``aimu.agents._tool_loop._ToolLoop``).

Same responsibility — run the model<->tools loop over a pure async model client — with
``await``/``asyncio.TaskGroup`` in place of threads. Internal; composed by ``aimu.aio.Agent``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional

from aimu.agents._tool_loop import (
    TERMINAL_EMPTY,
    TERMINAL_HEALTHY,
    TERMINAL_PENDING_TOOLS,
    DegenerateTurnError,
    _BaseToolLoop,
    classify_terminal_turn,
)
from aimu.models._internal.message_meta import PROVENANCE_CONTINUATION, PROVENANCE_FINAL_ANSWER
from aimu.models.base import StreamChunk, StreamingContentType

logger = logging.getLogger(__name__)


class _AsyncToolLoop(_BaseToolLoop):
    """Runs the async model<->tools loop over a pure async model client.

    Construction and the sync-safe helpers (``_current_tools``, ``_pending``,
    ``_tag_injected``, ``_wrap_up_prompt``, ``_tool_call_kwargs``, ``_not_approved``)
    are inherited from :class:`aimu.agents._tool_loop._BaseToolLoop`.
    """

    # ------------------------------------------------------------------ #
    # The loop                                                            #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        user_message: Optional[str] = None,
        *,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> str:
        response = await self._client.chat(
            user_message, generate_kwargs=generate_kwargs, images=images, tools=self._current_tools()
        )
        rounds = 0
        while rounds < self._max_rounds:
            state = classify_terminal_turn(self._client.messages)
            if state == TERMINAL_PENDING_TOOLS:
                await self._dispatch()
                response = await self._client.chat(generate_kwargs=generate_kwargs, tools=self._current_tools())
            elif state == TERMINAL_EMPTY:
                # A degenerate empty turn: nudge with tools still enabled so the model can resume
                # a multi-step plan (not just answer from nothing).
                injected_at = len(self._client.messages)
                response = await self._client.chat(
                    self._continuation_prompt, generate_kwargs=generate_kwargs, tools=self._current_tools()
                )
                self._tag_injected(injected_at, PROVENANCE_CONTINUATION)
            else:  # TERMINAL_HEALTHY
                return response
            rounds += 1

        return await self._forced_wrap_up(response, generate_kwargs)

    async def run_streamed(
        self,
        user_message: Optional[str] = None,
        *,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> AsyncIterator[StreamChunk]:
        iteration = 0
        stream = await self._client.chat(
            user_message, generate_kwargs=generate_kwargs, stream=True, images=images, tools=self._current_tools()
        )
        async for chunk in stream:
            yield StreamChunk(chunk.phase, chunk.content, agent=chunk.agent, iteration=iteration)
        while iteration < self._max_rounds:
            state = classify_terminal_turn(self._client.messages)
            if state == TERMINAL_PENDING_TOOLS:
                async for chunk in self._dispatch_streamed(iteration):
                    yield chunk
                iteration += 1
                stream = await self._client.chat(
                    generate_kwargs=generate_kwargs, stream=True, tools=self._current_tools()
                )
                async for chunk in stream:
                    yield StreamChunk(chunk.phase, chunk.content, agent=chunk.agent, iteration=iteration)
            elif state == TERMINAL_EMPTY:
                iteration += 1
                injected_at = len(self._client.messages)
                stream = await self._client.chat(
                    self._continuation_prompt, generate_kwargs=generate_kwargs, stream=True, tools=self._current_tools()
                )
                async for chunk in stream:
                    yield StreamChunk(chunk.phase, chunk.content, agent=chunk.agent, iteration=iteration)
                self._tag_injected(injected_at, PROVENANCE_CONTINUATION)
            else:  # TERMINAL_HEALTHY
                return

        if classify_terminal_turn(self._client.messages) != TERMINAL_HEALTHY:
            injected_at = len(self._client.messages)
            iteration += 1
            stream = await self._client.chat(
                self._wrap_up_prompt(), generate_kwargs=generate_kwargs, stream=True, use_tools=False, tools=[]
            )
            async for chunk in stream:
                yield StreamChunk(chunk.phase, chunk.content, agent=chunk.agent, iteration=iteration)
            self._tag_injected(injected_at, PROVENANCE_FINAL_ANSWER)
            if classify_terminal_turn(self._client.messages) != TERMINAL_HEALTHY:
                raise DegenerateTurnError(
                    "The model produced no answer (empty or tools-only turn) even after a forced wrap-up."
                )

    async def _forced_wrap_up(self, response: str, generate_kwargs: Optional[dict[str, Any]]) -> str:
        """At the round cap with a degenerate terminal turn, force one tools-disabled answer.

        Async twin of :meth:`aimu.agents._tool_loop._ToolLoop._forced_wrap_up`. Raises
        :class:`DegenerateTurnError` if the wrap-up is still degenerate rather than returning
        silent empty output.
        """
        if classify_terminal_turn(self._client.messages) == TERMINAL_HEALTHY:
            return response
        injected_at = len(self._client.messages)
        response = await self._client.chat(
            self._wrap_up_prompt(), generate_kwargs=generate_kwargs, use_tools=False, tools=[]
        )
        self._tag_injected(injected_at, PROVENANCE_FINAL_ANSWER)
        if classify_terminal_turn(self._client.messages) != TERMINAL_HEALTHY:
            raise DegenerateTurnError(
                "The model produced no answer (empty or tools-only turn) even after a forced wrap-up."
            )
        return response

    # ------------------------------------------------------------------ #
    # Dispatch                                                            #
    # ------------------------------------------------------------------ #

    async def _dispatch(self) -> None:
        prepared = self._pending()
        if self._concurrent and len(prepared) > 1:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._call_plain_tool(tc, tc_id)) for tc, tc_id in prepared]
            results = [t.result() for t in tasks]
        else:
            results = [await self._call_plain_tool(tc, tc_id) for tc, tc_id in prepared]
        for result_msg in results:
            self._client._append_message(result_msg)

    async def _dispatch_streamed(self, iteration: int) -> AsyncIterator[StreamChunk]:
        from aimu.tools.decorator import ToolArgumentError

        prepared = self._pending()
        by_name = {fn.__name__: fn for fn in self._current_tools()}
        has_streaming_tool = any(getattr(by_name.get(tc["name"]), "__tool_is_streaming__", False) for tc, _ in prepared)

        def _tool_chunk(tc: dict, response: str) -> StreamChunk:
            return StreamChunk(
                StreamingContentType.TOOL_CALLING,
                {"name": tc["name"], "arguments": tc["arguments"], "response": response},
                iteration=iteration,
            )

        if self._concurrent and len(prepared) > 1 and not has_streaming_tool:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._call_plain_tool(tc, tc_id)) for tc, tc_id in prepared]
            results = [t.result() for t in tasks]
            for (tc, _tc_id), result_msg in zip(prepared, results):
                self._client._append_message(result_msg)
                yield _tool_chunk(tc, result_msg["content"])
            return

        for tc, tc_id in prepared:
            fn = by_name.get(tc["name"])
            if fn is not None and getattr(fn, "__tool_is_streaming__", False):
                if not await self._tool_call_approved(tc["name"], tc["arguments"]):
                    result_msg = self._not_approved(tc, tc_id)
                    self._client._append_message(result_msg)
                    yield _tool_chunk(tc, result_msg["content"])
                    continue
                try:
                    return_value: Any = None
                    last_content: Any = None
                    kwargs = self._tool_call_kwargs(fn, tc["arguments"])
                    if getattr(fn, "__tool_is_async__", False):
                        agen = fn(**kwargs)
                        async for chunk in agen:
                            yield chunk
                            last_content = chunk.content
                    else:
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

            self._client._append_message(result_msg)
            yield _tool_chunk(tc, result_msg["content"])

    async def _call_plain_tool(self, tc: dict, tc_id: str) -> dict:
        from aimu.tools.decorator import ToolArgumentError

        fn = {f.__name__: f for f in self._current_tools()}.get(tc["name"])
        if fn is None:
            return {
                "role": "tool",
                "name": tc["name"],
                "content": f"Tool '{tc['name']}' not found.",
                "tool_call_id": tc_id,
            }
        if getattr(fn, "__tool_is_streaming__", False):
            raise ValueError(
                f"Tool '{tc['name']}' is a generator (streaming) tool. Run the agent with stream=True "
                "to dispatch it, or convert the tool to a plain function."
            )
        if not await self._tool_call_approved(tc["name"], tc["arguments"]):
            return self._not_approved(tc, tc_id)
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

    async def _tool_call_approved(self, name: str, arguments: dict) -> bool:
        import inspect

        from aimu.tools.approval import approve_all

        policy = self._tool_approval or approve_all
        result = policy(name, arguments)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
