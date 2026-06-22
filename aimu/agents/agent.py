from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Union

from aimu.agents.base import MessageHistory, Runner
from aimu.models.base import BaseModelClient, StreamChunk

logger = logging.getLogger(__name__)

DEFAULT_CONTINUATION_PROMPT = (
    "Continue working on the task using available tools as needed. If you have the answer "
    "and don't need to use any more tools, just provide the final response."
)


@dataclass
class Agent(Runner):
    """A model client wrapped in an agentic loop.

    Calls ``model_client.chat()`` repeatedly until the model produces a turn without
    invoking tools, or ``max_iterations`` is reached. The stop condition scans
    ``model_client.messages`` in reverse for a ``"tool"`` role message after the last
    ``"user"`` role. If found, the agent sends ``continuation_prompt`` and loops.

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
    _last_messages: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name is None:
            # Stable but unique-per-instance default. Users who need readable names
            # in messages histories should pass ``name=`` explicitly.
            self.name = f"agent-{id(self) & 0xFFFFFF:06x}"

    def _prepare_run(self, deps: Any = None) -> None:
        """Reset client state and re-apply system_message before a run, when configured.

        ``deps`` (a per-run override) takes precedence over the agent's ``self.deps`` field;
        the effective value is published to the model client for ``ToolContext`` injection.
        """
        if self.reset_messages_on_run or self.system_message is not None:
            self.model_client.reset(system_message=self.system_message)
        if self.tools:
            self.model_client.tools = list(self.tools)
        self.model_client.tool_context_deps = deps if deps is not None else self.deps

    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        tools: Optional[list[Callable]] = None,
        deps: Optional[Any] = None,
        schema: Optional[type] = None,
    ) -> Union[str, Any, Iterator[StreamChunk]]:
        """Run the agentic loop. ``images`` attach only to the initial turn.

        ``tools`` is a per-run override of the agent's configured ``self.tools``: ``None``
        (default) uses them, any other value (including ``[]`` to disable Python tools for
        this run) replaces them for every ``chat()`` call in the loop and is restored
        afterward.

        ``deps`` is a per-run override of the agent's ``self.deps`` field, the value injected
        as ``ctx.deps`` into tools that declare a :class:`~aimu.tools.ToolContext` parameter.

        ``schema`` (a dataclass or Pydantic v2 model) makes the run a single structured-output
        turn that returns a validated instance instead of looping with tools. Use it for an
        agent whose job is to return a typed object (e.g. a critic's verdict). It is mutually
        exclusive with ``stream=True`` and with the tool-calling loop.
        """
        if schema is not None:
            if stream:
                raise ValueError("schema= and stream=True are mutually exclusive (a typed object can't be streamed).")
            self._prepare_run(deps)
            result = self.model_client.chat(task, generate_kwargs=generate_kwargs, images=images, schema=schema)
            self._last_messages = list(self.model_client.messages)
            return result
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images, tools=tools, deps=deps)
        self._prepare_run(deps)
        response = self.model_client.chat(task, generate_kwargs=generate_kwargs, images=images, tools=tools)

        for _ in range(self.max_iterations - 1):
            if not self._last_turn_called_tools():
                break
            logger.debug("Agent '%s' continuing, tools were used in last turn.", self.name)
            response = self.model_client.chat(self.continuation_prompt, generate_kwargs=generate_kwargs, tools=tools)

        if self.final_answer_prompt is not None and self._last_turn_called_tools():
            logger.debug("Agent '%s' hit max_iterations with tools pending; forcing final answer.", self.name)
            response = self.model_client.chat(self.final_answer_prompt, generate_kwargs=generate_kwargs, tools=[])

        self._last_messages = list(self.model_client.messages)
        return response

    def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
        tools: Optional[list[Callable]] = None,
        deps: Optional[Any] = None,
    ) -> Iterator[StreamChunk]:
        self._prepare_run(deps)
        iteration = 0
        for chunk in self.model_client.chat(
            task, generate_kwargs=generate_kwargs, stream=True, images=images, tools=tools
        ):
            yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=iteration)

        iteration += 1
        while self._last_turn_called_tools() and iteration < self.max_iterations:
            logger.debug("Agent '%s' continuing (iteration %d).", self.name, iteration)
            for chunk in self.model_client.chat(
                self.continuation_prompt, generate_kwargs=generate_kwargs, stream=True, tools=tools
            ):
                yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=iteration)
            iteration += 1

        if self.final_answer_prompt is not None and self._last_turn_called_tools():
            logger.debug("Agent '%s' hit max_iterations with tools pending; forcing final answer.", self.name)
            for chunk in self.model_client.chat(
                self.final_answer_prompt, generate_kwargs=generate_kwargs, stream=True, tools=[]
            ):
                yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=iteration)
            iteration += 1

        self._last_messages = list(self.model_client.messages)

    @property
    def messages(self) -> MessageHistory:
        return {self.name: self._last_messages}

    def restore(self, messages: list[dict]) -> None:
        """Restore agent state from a saved message list for resuming after failure.

        Saves the partial state from a failed run via ``agent.model_client.messages``
        (the live list, not the post-run snapshot from ``agent.messages``), then call
        this method before the next ``run()`` to resume from that point.

        Handles the system-message duplication hazard: ``model_client.reset()``
        clears history and preserves the ``system_message`` attribute, and this method
        strips the leading system message from *messages* (if present) so it is not
        prepended twice on the next ``chat()`` call.

        Example::

            try:
                result = agent.run(task)
            except Exception:
                import json
                with open("checkpoint.json", "w") as f:
                    json.dump(agent.model_client.messages, f)
                raise

            # On retry:
            with open("checkpoint.json") as f:
                saved = json.load(f)
            agent.restore(saved)
            result = agent.run(continuation_prompt)
        """
        self.model_client.reset()  # clears messages, keeps system_message value
        stripped = [m for m in messages if m.get("role") != "system"]
        self.model_client.messages = stripped

    def _last_turn_called_tools(self) -> bool:
        for msg in reversed(self.model_client.messages):
            if msg.get("role") == "user":
                return False
            if msg.get("role") == "tool":
                return True
        return False

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
