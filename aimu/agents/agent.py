from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Union

from aimu.agents._loop import _AgentLoopMixin
from aimu.agents.base import MessageHistory, Runner
from aimu.models._internal.message_meta import PROVENANCE_FINAL_ANSWER
from aimu.models.base import BaseModelClient, StreamChunk

logger = logging.getLogger(__name__)

# Deprecated: the agent no longer injects a continuation prompt. It continues a tool-using run
# by calling chat() with no user message (a turn on the current messages). The ``continuation_prompt``
# field and this constant are kept only so existing constructors / from_config configs don't break.
DEFAULT_CONTINUATION_PROMPT = (
    "Continue working on the task using available tools as needed. If you have the answer "
    "and don't need to use any more tools, just provide the final response."
)


@dataclass
class Agent(_AgentLoopMixin, Runner):
    """A model client wrapped in an agentic loop.

    ``model_client.chat()`` is a single turn: it issues one model request and, if the model
    requests tools, executes them and returns. This agent is the loop over that: it calls
    ``chat(task)`` and then ``chat()`` (no user message — a continuation turn on the current
    messages) until a turn makes no tool calls, or ``max_iterations`` turns are reached. No
    synthetic "continue" prompt is injected into the transcript.

    Tools are plain callables in ``tools=``: functions decorated with
    ``@aimu.tools.tool`` for in-process tools, and/or ``MCPClient(...).as_tools()`` for
    cross-process FastMCP tools (each MCP tool becomes a callable). Mix them freely in
    one list, ``tools=builtin.web + mcp.as_tools()``.

    When ``system_message`` is set or ``reset_messages_on_run`` is True, the agent
    clears ``model_client.messages`` and re-applies ``system_message`` before every
    run. This isolates state when a client is shared (e.g. inside a :class:`Chain`).

    ``final_answer_prompt`` (opt-in, default ``None``) guarantees a final answer when the
    loop exhausts ``max_iterations`` while the model is still calling tools. Instead of
    returning whatever the last (possibly tool-only) turn produced, the agent sends this
    prompt once with tools disabled, forcing the model to synthesize an answer from the
    context it has gathered. This wrap-up turn is *not* counted against ``max_iterations``
    (the cap bounds tool-using turns; this is the guaranteed finish). It fires only on the
    cap-with-pending-tools path; a natural finish (a turn with no tool calls) is unaffected.

    Quick start::

        from aimu.tools import tool
        from aimu.agents import Agent
        import aimu

        @tool
        def letter_counter(word: str, letter: str) -> int:
            \"\"\"Count occurrences of a letter in a word.\"\"\"
            return word.lower().count(letter.lower())

        client = aimu.client("ollama:qwen3.5:9b")
        agent = Agent(client, "You are a helpful assistant.", tools=[letter_counter])
        print(agent.run("How many r's in strawberry?"))
    """

    model_client: BaseModelClient
    system_message: Optional[str] = None
    name: Optional[str] = None
    tools: list[Callable] = field(default_factory=list)
    max_iterations: int = 10
    continuation_prompt: str = field(default=DEFAULT_CONTINUATION_PROMPT)
    reset_messages_on_run: bool = False
    final_answer_prompt: Optional[str] = None
    deps: Optional[Any] = None
    tool_approval: Optional[Callable] = None
    _last_messages: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name is None:
            # Stable but unique-per-instance default. Users who need readable names
            # in messages histories should pass ``name=`` explicitly.
            self.name = f"agent-{id(self) & 0xFFFFFF:06x}"

    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        tools: Optional[list[Callable]] = None,
        deps: Optional[Any] = None,
        tool_approval: Optional[Callable] = None,
        schema: Optional[type] = None,
    ) -> Union[str, Any, Iterator[StreamChunk]]:
        """Run the agentic loop. ``images`` attach only to the initial turn.

        ``tools`` is a per-run override of the agent's configured ``self.tools``: ``None``
        (default) uses them, any other value (including ``[]`` to disable Python tools for
        this run) replaces them for every ``chat()`` call in the loop and is restored
        afterward.

        ``deps`` is a per-run override of the agent's ``self.deps`` field, the value injected
        as ``ctx.deps`` into tools that declare a :class:`~aimu.tools.ToolContext` parameter.

        ``tool_approval`` is a per-run override of the agent's ``self.tool_approval`` field, the
        gate run before each tool call (``(name, arguments) -> bool``; deny appends a refusal tool
        message). Defaults to approving everything.

        ``schema`` (a dataclass or Pydantic v2 model) makes the run a single structured-output
        turn that returns a validated instance instead of looping with tools. Use it for an
        agent whose job is to return a typed object (e.g. a critic's verdict). It is mutually
        exclusive with the tool-calling loop. With ``stream=True`` the run yields
        :class:`StreamChunk`s (thinking/generation) ending in a terminal ``DONE`` chunk carrying
        ``{"result": <object>}``; the object is also on ``model_client.last_structured``.
        """
        if schema is not None:
            if stream:
                return self._run_structured_streamed(task, generate_kwargs, images, deps, tool_approval, schema)
            self._prepare_run(deps, tool_approval)
            result = self.model_client.chat(task, generate_kwargs=generate_kwargs, images=images, schema=schema)
            self._last_messages = list(self.model_client.messages)
            return result
        if stream:
            return self._run_streamed(
                task, generate_kwargs, images=images, tools=tools, deps=deps, tool_approval=tool_approval
            )
        self._prepare_run(deps, tool_approval)
        # Turn 1 executes any requested tools and returns. Continue (chat() with no user
        # message → a turn on the current messages) until a turn makes no tool calls.
        response = self.model_client.chat(task, generate_kwargs=generate_kwargs, images=images, tools=tools)
        iteration = 1
        while self._last_turn_called_tools() and iteration < self.max_iterations:
            logger.debug("Agent '%s' continuing (iteration %d).", self.name, iteration)
            response = self.model_client.chat(generate_kwargs=generate_kwargs, tools=tools)
            iteration += 1

        if self.final_answer_prompt is not None and self._last_turn_called_tools():
            logger.debug("Agent '%s' hit max_iterations with tools pending; forcing final answer.", self.name)
            injected_at = len(self.model_client.messages)
            response = self.model_client.chat(self.final_answer_prompt, generate_kwargs=generate_kwargs, tools=[])
            self._tag_injected_turn(injected_at, PROVENANCE_FINAL_ANSWER)

        self._last_messages = list(self.model_client.messages)
        return response

    def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
        tools: Optional[list[Callable]] = None,
        deps: Optional[Any] = None,
        tool_approval: Optional[Callable] = None,
    ) -> Iterator[StreamChunk]:
        self._prepare_run(deps, tool_approval)
        iteration = 0
        for chunk in self.model_client.chat(
            task, generate_kwargs=generate_kwargs, stream=True, images=images, tools=tools
        ):
            yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=iteration)

        iteration += 1
        while self._last_turn_called_tools() and iteration < self.max_iterations:
            logger.debug("Agent '%s' continuing (iteration %d).", self.name, iteration)
            # chat() with no user message → a turn on the current messages (tool results).
            for chunk in self.model_client.chat(generate_kwargs=generate_kwargs, stream=True, tools=tools):
                yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=iteration)
            iteration += 1

        if self.final_answer_prompt is not None and self._last_turn_called_tools():
            logger.debug("Agent '%s' hit max_iterations with tools pending; forcing final answer.", self.name)
            injected_at = len(self.model_client.messages)
            for chunk in self.model_client.chat(
                self.final_answer_prompt, generate_kwargs=generate_kwargs, stream=True, tools=[]
            ):
                yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=iteration)
            self._tag_injected_turn(injected_at, PROVENANCE_FINAL_ANSWER)
            iteration += 1

        self._last_messages = list(self.model_client.messages)

    def _run_structured_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]],
        images: Optional[list],
        deps: Optional[Any],
        tool_approval: Optional[Callable],
        schema: type,
    ) -> Iterator[StreamChunk]:
        """Single structured-output turn, streamed: forward the client's chunks (thinking /
        generation / terminal DONE) tagged with this agent's name. Snapshots ``_last_messages``
        in a ``finally`` so a cancelled/partial run still records its turn."""
        self._prepare_run(deps, tool_approval)
        try:
            for chunk in self.model_client.chat(
                task, generate_kwargs=generate_kwargs, stream=True, images=images, schema=schema
            ):
                yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=0)
        finally:
            self._last_messages = list(self.model_client.messages)

    @property
    def messages(self) -> MessageHistory:
        return {self.name: self._last_messages}

    def as_model_client(self) -> BaseModelClient:
        """Return a :class:`BaseModelClient` view of this agent.

        Each ``chat()`` call on the returned object runs the full agent loop, looping
        until the model stops calling tools. Use this only where an API expects a
        ``BaseModelClient``. For direct use, call :meth:`run` instead.
        """
        from aimu.agents.agentic_client import _AgenticView

        return _AgenticView(self)

    @classmethod
    def from_config(cls, config: dict[str, Any], model_client: BaseModelClient) -> Agent:
        """Create an Agent from a plain dict config.

        Recognised keys: ``name``, ``system_message``, ``max_iterations``,
        ``continuation_prompt``, ``final_answer_prompt``.
        """
        sm = config.get("system_message")
        return cls(
            model_client=model_client,
            system_message=sm,
            name=config.get("name"),
            max_iterations=config.get("max_iterations", 10),
            continuation_prompt=config.get("continuation_prompt", DEFAULT_CONTINUATION_PROMPT),
            final_answer_prompt=config.get("final_answer_prompt"),
        )
