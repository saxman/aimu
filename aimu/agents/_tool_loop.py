"""The iterative tool-calling engine (sync).

This is the middle layer between a pure model client and an autonomous ``Agent``. A model
client's ``chat()`` is a single turn: it advertises the tools it is given, parses any tool
calls out of the response, stores them on the assistant message, and returns — it never runs
a tool. ``_ToolLoop`` owns the *iterative tool-calling logic*: call the client, and while the
model's turn requested tools, execute them (dispatch, approval, deps injection, concurrency),
append the results, and call the client again — until a turn makes no tool calls (bounded by
``max_rounds``). It holds the tool callables and resolves tool names against them.

It is internal: the public ladder is ``chat()`` (one turn) -> ``Agent`` (autonomy + composition)
-> workflows. ``Agent`` composes a ``_ToolLoop`` per run. The async twin is
``aimu.aio._tool_loop._AsyncToolLoop``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterator, Optional, Union

from aimu.events import EventSink, RunFinished, RunStarted, ToolCalled, ToolDenied, emit
from aimu.models._internal.message_meta import PROVENANCE_CONTINUATION, PROVENANCE_FINAL_ANSWER, PROVENANCE_KEY
from aimu.models.base import StreamChunk, StreamingContentType

logger = logging.getLogger(__name__)

# Forced wrap-up prompt used when the loop hits the round cap with tools still pending and the
# agent configured no ``final_answer_prompt``. Tools are disabled for this turn, so it asks the
# model to answer from the context it has already gathered.
DEFAULT_WRAP_UP_PROMPT = (
    "You have reached the tool-use limit for this task. Do not call any more tools. "
    "Provide your best final answer now using the information you have already gathered."
)

# Terminal-turn classifications (shared by the sync and async loops).
TERMINAL_PENDING_TOOLS = "pending_tools"  # last turn requested tools (or a tool result trails)
TERMINAL_EMPTY = "empty"  # last assistant turn has no tool calls and no usable content
TERMINAL_HEALTHY = "healthy"  # last assistant turn is a real answer


class DegenerateTurnError(RuntimeError):
    """The loop could not obtain a real answer from the model.

    Raised when, even after a forced tools-disabled wrap-up, the model's terminal turn is still
    degenerate (empty, or still only requesting tools). Small local models occasionally emit such
    turns; raising surfaces the failure to the caller instead of returning silent empty output.
    """


class TruncatedTurnError(DegenerateTurnError):
    """The model's turn was cut off at an output limit before it produced anything usable.

    A subclass because the loop's outcome is the same -- no answer -- but the cause is not the model
    failing to answer, it is the model never getting the room to. The loop raises this *instead of*
    injecting its continuation nudge, because the nudge adds tokens to a request that already had
    none to spare: retrying makes each turn shorter than the last rather than better.
    """


def classify_terminal_turn(messages: list[dict]) -> str:
    """Classify the transcript's most recent turn as pending-tools, empty, or healthy.

    ``chat()`` is single-turn: it stores parsed tool_calls but does not execute, so the transcript
    ends in an ``assistant`` message carrying ``tool_calls`` (wants tools), a trailing ``tool``
    result (mid dispatch), an ``assistant`` message with usable content (a real answer), or an
    ``assistant`` message with neither (a degenerate empty turn). Shared by the sync and async loops.
    """
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "tool":
            return TERMINAL_PENDING_TOOLS
        if role == "assistant":
            if msg.get("tool_calls"):
                return TERMINAL_PENDING_TOOLS
            content = msg.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                return TERMINAL_EMPTY
            return TERMINAL_HEALTHY
    return TERMINAL_HEALTHY


def last_turn_called_tools(messages: list[dict]) -> bool:
    """True if the model's most recent turn ended by requesting tools.

    A trailing ``tool`` result (mid dispatch) also counts as "still working". Shared by the sync
    and async loops. Thin wrapper over :func:`classify_terminal_turn` kept for existing callers.
    """
    return classify_terminal_turn(messages) == TERMINAL_PENDING_TOOLS


class _BaseToolLoop:
    """State and sync-safe helpers shared by the sync and async tool-loop engines.

    Holds only members free of ``async``/``await``: construction, tool-list resolution,
    pending-call extraction, provenance tagging, wrap-up prompt selection, argument
    coercion + ``deps`` injection, and the not-approved message. The loop drivers
    (``run`` / ``run_streamed``) and dispatch (which differ by threads vs
    ``asyncio.TaskGroup`` and sync vs ``await``) live on the concrete subclasses.
    """

    def __init__(
        self,
        model_client: Any,
        tools,
        *,
        deps: Optional[Any] = None,
        tool_approval: Optional[Callable] = None,
        concurrent_tool_calls: bool = False,
        max_rounds: int = 10,
        final_answer_prompt: Optional[str] = None,
        continuation_prompt: Optional[str] = None,
        thinking: Optional[Union[bool, str]] = None,
        events: Optional[EventSink] = None,
        agent_name: Optional[str] = None,
    ):
        # ``tools`` is either the tool-callable list, or a zero-arg callable returning it
        # (re-read each round so tools added mid-run — e.g. SkillAgent.reload_skills authoring a
        # skill callable in the same turn — are advertised and dispatchable on the next round).
        self._client = model_client
        self._tools = tools
        self._deps = deps
        self._tool_approval = tool_approval
        self._concurrent = concurrent_tool_calls
        self._max_rounds = max_rounds
        self._final_answer_prompt = final_answer_prompt
        self._continuation_prompt = continuation_prompt or DEFAULT_WRAP_UP_PROMPT
        # The public thinking= argument, not a resolved request: the client's own chat() validates
        # it against its model and warns once, so re-passing it every round is safe and silent.
        self._thinking = thinking
        # The event sink and the agent's name -- the loop is the one caller that knows both a
        # run started and which tool a policy refused, so it emits RunStarted/RunFinished and
        # ToolCalled/ToolDenied directly, stamped with the agent's name and the current round.
        self._events = events
        self._agent_name = agent_name

    def _current_tools(self) -> list[Callable]:
        return list(self._tools() if callable(self._tools) else self._tools)

    def _pending(self) -> list[tuple[dict, str]]:
        """Extract ``[( {"name","arguments"}, tool_call_id ), ...]`` from the last assistant turn.

        The provider already parsed the response and stored the assistant message with
        ``tool_calls`` (each ``{"type":"function","function":{"name","arguments"},"id"}``); the
        engine only executes them.
        """
        for msg in reversed(self._client.messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                return [
                    ({"name": t["function"]["name"], "arguments": t["function"]["arguments"]}, t["id"])
                    for t in msg["tool_calls"]
                ]
            if msg.get("role") == "user":
                break
        return []

    def _raise_if_truncated(self) -> None:
        """Turn a provider's "output ran out of room" report into a typed error, with the numbers.

        Only called where the turn came back with nothing usable. A truncated turn that still carries
        an answer is a caller's own ``max_tokens`` doing its job, and is left alone.
        """
        if not getattr(self._client, "last_output_truncated", False):
            return
        usage = getattr(self._client, "last_usage", None) or {}
        spent = usage.get("input_tokens")
        detail = f" The prompt used {spent} input tokens." if spent else ""
        raise TruncatedTurnError(
            "The model's output was cut off before it produced an answer, so there is nothing to "
            f"continue from.{detail} Shorten the conversation or advertise fewer tools, or raise the "
            "model's context window (generate_kwargs={'context_length': N} where the backend "
            "accepts it; see docs/how-to/set-context-length.md)."
        )

    def _tag_injected(self, index: int, provenance: str) -> None:
        messages = self._client.messages
        if 0 <= index < len(messages) and messages[index].get("role") == "user":
            messages[index][PROVENANCE_KEY] = provenance

    def _wrap_up_prompt(self) -> str:
        """The forced tools-disabled wrap-up prompt: the configured one, else the built-in default."""
        return self._final_answer_prompt or DEFAULT_WRAP_UP_PROMPT

    def _tool_call_kwargs(self, fn: Callable, arguments: dict) -> dict:
        """Coerce model-supplied args to the tool's hints and inject ``ToolContext(deps)``."""
        from aimu.tools.decorator import coerce_tool_arguments

        kwargs = coerce_tool_arguments(fn, arguments)
        injected = getattr(fn, "__tool_injected__", None)
        if injected:
            from aimu.tools.context import ToolContext

            ctx = ToolContext(deps=self._deps)
            for name in injected:
                kwargs[name] = ctx
        return kwargs

    def _not_approved(self, tc: dict, tc_id: str, iteration: int = 0) -> dict:
        emit(
            self._events,
            ToolDenied(agent=self._agent_name, iteration=iteration, name=tc["name"], arguments=tc["arguments"]),
        )
        return {
            "role": "tool",
            "name": tc["name"],
            "content": f"Tool '{tc['name']}' was not approved.",
            "tool_call_id": tc_id,
        }


class _ToolLoop(_BaseToolLoop):
    """Runs the model<->tools loop over a pure model client. See module docstring."""

    # ------------------------------------------------------------------ #
    # The loop                                                            #
    # ------------------------------------------------------------------ #

    def run(
        self,
        user_message: Optional[str] = None,
        *,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> str:
        emit(self._events, RunStarted(agent=self._agent_name, iteration=0, task=user_message or ""))
        result: Optional[str] = None
        error: Optional[BaseException] = None
        last_iteration = 0
        try:
            with self._client._events_override(self._events):
                response = self._client.chat(
                    user_message,
                    generate_kwargs=generate_kwargs,
                    images=images,
                    tools=self._current_tools(),
                    thinking=self._thinking,
                )
                chats = 1  # ``max_rounds`` caps the total number of model turns in the loop.
                while chats < self._max_rounds:
                    last_iteration = chats - 1
                    state = classify_terminal_turn(self._client.messages)
                    if state == TERMINAL_PENDING_TOOLS:
                        self._dispatch(last_iteration)
                        response = self._client.chat(
                            generate_kwargs=generate_kwargs, tools=self._current_tools(), thinking=self._thinking
                        )
                    elif state == TERMINAL_EMPTY:
                        # A degenerate empty turn: nudge with tools still enabled so the model can
                        # resume a multi-step plan (not just answer from nothing). Unless the turn
                        # was empty because it was cut off, in which case there is nothing to resume
                        # and nudging only shrinks the next one.
                        self._raise_if_truncated()
                        injected_at = len(self._client.messages)
                        response = self._client.chat(
                            self._continuation_prompt,
                            generate_kwargs=generate_kwargs,
                            tools=self._current_tools(),
                            thinking=self._thinking,
                        )
                        self._tag_injected(injected_at, PROVENANCE_CONTINUATION)
                    else:  # TERMINAL_HEALTHY
                        result = response
                        return result
                    chats += 1

                last_iteration = chats - 1
                result = self._forced_wrap_up(response, generate_kwargs)
                return result
        except BaseException as exc:
            error = exc
            raise
        finally:
            emit(
                self._events,
                RunFinished(agent=self._agent_name, iteration=last_iteration, result=result, error=error),
            )

    def run_streamed(
        self,
        user_message: Optional[str] = None,
        *,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> Iterator[StreamChunk]:
        emit(self._events, RunStarted(agent=self._agent_name, iteration=0, task=user_message or ""))
        error: Optional[BaseException] = None
        iteration = 0
        try:
            with self._client._events_override(self._events):
                yield from self._retag(
                    self._client.chat(
                        user_message,
                        generate_kwargs=generate_kwargs,
                        stream=True,
                        images=images,
                        tools=self._current_tools(),
                        thinking=self._thinking,
                    ),
                    iteration,
                )
                while iteration + 1 < self._max_rounds:
                    state = classify_terminal_turn(self._client.messages)
                    if state == TERMINAL_PENDING_TOOLS:
                        yield from self._dispatch_streamed(iteration)
                        iteration += 1
                        yield from self._retag(
                            self._client.chat(
                                generate_kwargs=generate_kwargs,
                                stream=True,
                                tools=self._current_tools(),
                                thinking=self._thinking,
                            ),
                            iteration,
                        )
                    elif state == TERMINAL_EMPTY:
                        self._raise_if_truncated()  # cut off, not degenerate: a nudge cannot recover it
                        iteration += 1
                        injected_at = len(self._client.messages)
                        yield from self._retag(
                            self._client.chat(
                                self._continuation_prompt,
                                generate_kwargs=generate_kwargs,
                                stream=True,
                                tools=self._current_tools(),
                                thinking=self._thinking,
                            ),
                            iteration,
                        )
                        self._tag_injected(injected_at, PROVENANCE_CONTINUATION)
                    else:  # TERMINAL_HEALTHY
                        return

                if classify_terminal_turn(self._client.messages) != TERMINAL_HEALTHY:
                    injected_at = len(self._client.messages)
                    iteration += 1
                    yield from self._retag(
                        self._client.chat(
                            self._wrap_up_prompt(),
                            generate_kwargs=generate_kwargs,
                            stream=True,
                            use_tools=False,
                            tools=[],
                            thinking=self._thinking,
                        ),
                        iteration,
                    )
                    self._tag_injected(injected_at, PROVENANCE_FINAL_ANSWER)
                    if classify_terminal_turn(self._client.messages) != TERMINAL_HEALTHY:
                        self._raise_if_truncated()  # says which of the two failures this was
                        raise DegenerateTurnError(
                            "The model produced no answer (empty or tools-only turn) even after a forced wrap-up."
                        )
        except BaseException as exc:
            error = exc
            raise
        finally:
            emit(self._events, RunFinished(agent=self._agent_name, iteration=iteration, result=None, error=error))

    def _forced_wrap_up(self, response: str, generate_kwargs: Optional[dict[str, Any]]) -> str:
        """At the round cap with a degenerate terminal turn, force one tools-disabled answer.

        Runs when the loop exhausted ``max_rounds`` while the last turn was still pending tools or
        empty. Disables tools so the model must synthesize an answer from gathered context. Raises
        :class:`DegenerateTurnError` if the wrap-up is *still* degenerate, rather than returning
        silent empty output.
        """
        if classify_terminal_turn(self._client.messages) == TERMINAL_HEALTHY:
            return response
        injected_at = len(self._client.messages)
        response = self._client.chat(
            self._wrap_up_prompt(),
            generate_kwargs=generate_kwargs,
            use_tools=False,
            tools=[],
            thinking=self._thinking,
        )
        self._tag_injected(injected_at, PROVENANCE_FINAL_ANSWER)
        if classify_terminal_turn(self._client.messages) != TERMINAL_HEALTHY:
            self._raise_if_truncated()  # says which of the two failures this was
            raise DegenerateTurnError(
                "The model produced no answer (empty or tools-only turn) even after a forced wrap-up."
            )
        return response

    @staticmethod
    def _retag(chunks: Iterator[StreamChunk], iteration: int) -> Iterator[StreamChunk]:
        for chunk in chunks:
            yield StreamChunk(chunk.phase, chunk.content, agent=chunk.agent, iteration=iteration)

    # ------------------------------------------------------------------ #
    # Dispatch (execute the pending tool calls stored on the last turn)   #
    # ------------------------------------------------------------------ #

    def _dispatch(self, iteration: int = 0) -> None:
        prepared = self._pending()
        if self._concurrent and len(prepared) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._call_plain_tool, tc, tc_id, iteration) for tc, tc_id in prepared]
                results = [f.result() for f in futures]
        else:
            results = [self._call_plain_tool(tc, tc_id, iteration) for tc, tc_id in prepared]
        for result_msg in results:
            self._client._append_message(result_msg)

    def _dispatch_streamed(self, iteration: int) -> Iterator[StreamChunk]:
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
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._call_plain_tool, tc, tc_id, iteration) for tc, tc_id in prepared]
                results = [f.result() for f in futures]
            for (tc, _tc_id), result_msg in zip(prepared, results):
                self._client._append_message(result_msg)
                yield _tool_chunk(tc, result_msg["content"])
            return

        for tc, tc_id in prepared:
            fn = by_name.get(tc["name"])
            if fn is not None and getattr(fn, "__tool_is_streaming__", False):
                if getattr(fn, "__tool_is_async__", False):
                    raise ValueError(
                        f"Tool '{tc['name']}' is an async streaming tool. Use the aimu.aio surface to dispatch it."
                    )
                if not self._tool_call_approved(tc["name"], tc["arguments"]):
                    result_msg = self._not_approved(tc, tc_id, iteration)
                    self._client._append_message(result_msg)
                    yield _tool_chunk(tc, result_msg["content"])
                    continue
                started = time.monotonic()
                error_str: Optional[str] = None
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
                    error_str = content
                except Exception as exc:
                    content = f"Tool '{tc['name']}' raised an error: {exc}"
                    error_str = str(exc)
                    logger.warning("Tool call '%s' failed: %s", tc["name"], exc)
                emit(
                    self._events,
                    ToolCalled(
                        agent=self._agent_name,
                        iteration=iteration,
                        name=tc["name"],
                        arguments=tc["arguments"],
                        result=content,
                        error=error_str,
                        duration_s=time.monotonic() - started,
                    ),
                )
                result_msg = {"role": "tool", "name": tc["name"], "content": content, "tool_call_id": tc_id}
            else:
                result_msg = self._call_plain_tool(tc, tc_id, iteration)

            self._client._append_message(result_msg)
            yield _tool_chunk(tc, result_msg["content"])

    def _call_plain_tool(self, tc: dict, tc_id: str, iteration: int = 0) -> dict:
        """Dispatch one non-streaming tool call. Returns the ``role:"tool"`` message dict."""
        from aimu.tools.decorator import ToolArgumentError

        fn = {f.__name__: f for f in self._current_tools()}.get(tc["name"])
        if fn is None:
            return {
                "role": "tool",
                "name": tc["name"],
                "content": f"Tool '{tc['name']}' not found.",
                "tool_call_id": tc_id,
            }
        if getattr(fn, "__tool_is_async__", False):
            raise ValueError(
                f"Tool '{tc['name']}' is an async function (`async def`). The sync Agent cannot "
                "dispatch async tools. Use the aimu.aio surface, or convert the tool to a regular `def`."
            )
        if getattr(fn, "__tool_is_streaming__", False):
            raise ValueError(
                f"Tool '{tc['name']}' is a generator (streaming) tool. Run the agent with stream=True "
                "to dispatch it, or convert the tool to a plain function."
            )
        if not self._tool_call_approved(tc["name"], tc["arguments"]):
            return self._not_approved(tc, tc_id, iteration)
        started = time.monotonic()
        error_str: Optional[str] = None
        try:
            response = fn(**self._tool_call_kwargs(fn, tc["arguments"]))
            content = str(response)
        except ToolArgumentError as exc:
            content = str(exc)
            error_str = content
        except Exception as exc:
            content = f"Tool '{tc['name']}' raised an error: {exc}"
            error_str = str(exc)
            logger.warning("Tool call '%s' failed: %s", tc["name"], exc)
        emit(
            self._events,
            ToolCalled(
                agent=self._agent_name,
                iteration=iteration,
                name=tc["name"],
                arguments=tc["arguments"],
                result=content,
                error=error_str,
                duration_s=time.monotonic() - started,
            ),
        )
        return {"role": "tool", "name": tc["name"], "content": content, "tool_call_id": tc_id}

    def _tool_call_approved(self, name: str, arguments: dict) -> bool:
        """Run the approval policy (default approves everything). Rejects a coroutine policy."""
        import inspect

        from aimu.tools.approval import approve_all

        policy = self._tool_approval or approve_all
        result = policy(name, arguments)
        if inspect.isawaitable(result):
            result.close()
            raise ValueError(
                "tool_approval returned a coroutine on the sync Agent. Use a synchronous policy, "
                "or run on the aimu.aio surface for async approval."
            )
        return bool(result)
