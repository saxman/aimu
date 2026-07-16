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
from typing import Any, Callable, Iterator, Optional

from aimu.models._internal.message_meta import PROVENANCE_CONTINUATION, PROVENANCE_FINAL_ANSWER, PROVENANCE_KEY
from aimu.models.base import BaseModelClient, StreamChunk, StreamingContentType

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


class _ToolLoop:
    """Runs the model<->tools loop over a pure model client. See module docstring."""

    def __init__(
        self,
        model_client: BaseModelClient,
        tools,
        *,
        deps: Optional[Any] = None,
        tool_approval: Optional[Callable] = None,
        concurrent_tool_calls: bool = False,
        max_rounds: int = 10,
        final_answer_prompt: Optional[str] = None,
        continuation_prompt: Optional[str] = None,
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

    def _current_tools(self) -> list[Callable]:
        return list(self._tools() if callable(self._tools) else self._tools)

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
        response = self._client.chat(
            user_message, generate_kwargs=generate_kwargs, images=images, tools=self._current_tools()
        )
        chats = 1  # ``max_rounds`` caps the total number of model turns in the loop.
        while chats < self._max_rounds:
            state = classify_terminal_turn(self._client.messages)
            if state == TERMINAL_PENDING_TOOLS:
                self._dispatch()
                response = self._client.chat(generate_kwargs=generate_kwargs, tools=self._current_tools())
            elif state == TERMINAL_EMPTY:
                # A degenerate empty turn: nudge with tools still enabled so the model can resume
                # a multi-step plan (not just answer from nothing).
                injected_at = len(self._client.messages)
                response = self._client.chat(
                    self._continuation_prompt, generate_kwargs=generate_kwargs, tools=self._current_tools()
                )
                self._tag_injected(injected_at, PROVENANCE_CONTINUATION)
            else:  # TERMINAL_HEALTHY
                return response
            chats += 1

        return self._forced_wrap_up(response, generate_kwargs)

    def run_streamed(
        self,
        user_message: Optional[str] = None,
        *,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> Iterator[StreamChunk]:
        iteration = 0
        yield from self._retag(
            self._client.chat(
                user_message, generate_kwargs=generate_kwargs, stream=True, images=images, tools=self._current_tools()
            ),
            iteration,
        )
        while iteration + 1 < self._max_rounds:
            state = classify_terminal_turn(self._client.messages)
            if state == TERMINAL_PENDING_TOOLS:
                yield from self._dispatch_streamed(iteration)
                iteration += 1
                yield from self._retag(
                    self._client.chat(generate_kwargs=generate_kwargs, stream=True, tools=self._current_tools()),
                    iteration,
                )
            elif state == TERMINAL_EMPTY:
                iteration += 1
                injected_at = len(self._client.messages)
                yield from self._retag(
                    self._client.chat(
                        self._continuation_prompt,
                        generate_kwargs=generate_kwargs,
                        stream=True,
                        tools=self._current_tools(),
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
                    self._wrap_up_prompt(), generate_kwargs=generate_kwargs, stream=True, use_tools=False, tools=[]
                ),
                iteration,
            )
            self._tag_injected(injected_at, PROVENANCE_FINAL_ANSWER)
            if classify_terminal_turn(self._client.messages) != TERMINAL_HEALTHY:
                raise DegenerateTurnError(
                    "The model produced no answer (empty or tools-only turn) even after a forced wrap-up."
                )

    def _wrap_up_prompt(self) -> str:
        """The forced tools-disabled wrap-up prompt: the configured one, else the built-in default."""
        return self._final_answer_prompt or DEFAULT_WRAP_UP_PROMPT

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
        response = self._client.chat(self._wrap_up_prompt(), generate_kwargs=generate_kwargs, use_tools=False, tools=[])
        self._tag_injected(injected_at, PROVENANCE_FINAL_ANSWER)
        if classify_terminal_turn(self._client.messages) != TERMINAL_HEALTHY:
            raise DegenerateTurnError(
                "The model produced no answer (empty or tools-only turn) even after a forced wrap-up."
            )
        return response

    @staticmethod
    def _retag(chunks: Iterator[StreamChunk], iteration: int) -> Iterator[StreamChunk]:
        for chunk in chunks:
            yield StreamChunk(chunk.phase, chunk.content, agent=chunk.agent, iteration=iteration)

    def _tag_injected(self, index: int, provenance: str) -> None:
        messages = self._client.messages
        if 0 <= index < len(messages) and messages[index].get("role") == "user":
            messages[index][PROVENANCE_KEY] = provenance

    # ------------------------------------------------------------------ #
    # Dispatch (execute the pending tool calls stored on the last turn)   #
    # ------------------------------------------------------------------ #

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

    def _dispatch(self) -> None:
        prepared = self._pending()
        if self._concurrent and len(prepared) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._call_plain_tool, tc, tc_id) for tc, tc_id in prepared]
                results = [f.result() for f in futures]
        else:
            results = [self._call_plain_tool(tc, tc_id) for tc, tc_id in prepared]
        self._client.messages.extend(results)

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
                futures = [executor.submit(self._call_plain_tool, tc, tc_id) for tc, tc_id in prepared]
                results = [f.result() for f in futures]
            for (tc, _tc_id), result_msg in zip(prepared, results):
                self._client.messages.append(result_msg)
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
                    result_msg = self._not_approved(tc, tc_id)
                    self._client.messages.append(result_msg)
                    yield _tool_chunk(tc, result_msg["content"])
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

            self._client.messages.append(result_msg)
            yield _tool_chunk(tc, result_msg["content"])

    def _call_plain_tool(self, tc: dict, tc_id: str) -> dict:
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
            return self._not_approved(tc, tc_id)
        try:
            response = fn(**self._tool_call_kwargs(fn, tc["arguments"]))
            content = str(response)
        except ToolArgumentError as exc:
            content = str(exc)
        except Exception as exc:
            content = f"Tool '{tc['name']}' raised an error: {exc}"
            logger.warning("Tool call '%s' failed: %s", tc["name"], exc)
        return {"role": "tool", "name": tc["name"], "content": content, "tool_call_id": tc_id}

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

    @staticmethod
    def _not_approved(tc: dict, tc_id: str) -> dict:
        return {
            "role": "tool",
            "name": tc["name"],
            "content": f"Tool '{tc['name']}' was not approved.",
            "tool_call_id": tc_id,
        }
