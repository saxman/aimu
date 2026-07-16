from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Union

from aimu.agents._loop import _AgentLoopMixin
from aimu.agents._tool_loop import _ToolLoop
from aimu.agents.base import MessageHistory, Runner
from aimu.models.base import BaseModelClient, StreamChunk

logger = logging.getLogger(__name__)

# Between successful tool rounds the agent continues by calling chat() with no user message. The
# ``continuation_prompt`` is injected only to recover a degenerate empty turn (the model returned no
# content and no tool calls): it nudges the model to continue, with tools still enabled.
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
    synthetic "continue" prompt is injected between successful tool rounds.

    Tools are plain callables in ``tools=``: functions decorated with
    ``@aimu.tools.tool`` for in-process tools, and/or ``MCPClient(...).as_tools()`` for
    cross-process FastMCP tools (each MCP tool becomes a callable). Mix them freely in
    one list, ``tools=builtin.web + mcp.as_tools()``.

    When ``system_message`` is set or ``reset_messages_on_run`` is True, the agent
    clears ``model_client.messages`` and re-applies ``system_message`` before every
    run. This isolates state when a client is shared (e.g. inside a :class:`Chain`).

    The loop guards against degenerate turns so a run never ends silently:

    - **Empty turn.** If a turn comes back with no content and no tool calls, the agent injects
      ``continuation_prompt`` (tools still enabled, so the model can resume a multi-step plan) and
      continues. These nudges are bounded by ``max_iterations`` and tagged so a UI can hide them.
    - **Cap with tools pending.** On exhausting ``max_iterations`` with a tool call still pending,
      the agent sends one forced wrap-up turn with tools disabled, so the model must synthesize an
      answer from the context it has gathered. ``final_answer_prompt`` customizes this wrap-up
      prompt; when unset a built-in default is used. This turn is *not* counted against
      ``max_iterations``. A natural finish (a real answer) is unaffected.
    - **Still degenerate after wrap-up.** If even the wrap-up yields no answer, the agent raises
      :class:`~aimu.agents.DegenerateTurnError` rather than returning empty output.

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
    concurrent_tool_calls: bool = False
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
        loop = self._make_tool_loop(tools, deps, tool_approval)
        try:
            return loop.run(task, generate_kwargs=generate_kwargs, images=images)
        finally:
            self._last_messages = list(self.model_client.messages)

    def _effective_tools(self, tools: Optional[list[Callable]]) -> list[Callable]:
        """The tool callables for this run: the ``tools=`` override, else the agent's ``self.tools``.
        (``SkillAgent`` overrides this to add its discovered skill tools.)"""
        return list(tools) if tools is not None else list(self.tools)

    def _make_tool_loop(
        self, tools: Optional[list[Callable]], deps: Optional[Any], tool_approval: Optional[Callable]
    ) -> _ToolLoop:
        """Build the iterative tool-calling engine with this run's effective tools + policy."""
        from aimu.tools.approval import approve_all

        return _ToolLoop(
            self.model_client,
            lambda: self._effective_tools(tools),  # re-read each round (SkillAgent may add skill tools mid-run)
            deps=deps if deps is not None else self.deps,
            tool_approval=tool_approval or self.tool_approval or approve_all,
            concurrent_tool_calls=self.concurrent_tool_calls,
            max_rounds=self.max_iterations,
            final_answer_prompt=self.final_answer_prompt,
            continuation_prompt=self.continuation_prompt,
        )

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
        loop = self._make_tool_loop(tools, deps, tool_approval)
        try:
            for chunk in loop.run_streamed(task, generate_kwargs=generate_kwargs, images=images):
                yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=chunk.iteration)
        finally:
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
