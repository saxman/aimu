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
    ``"user"`` role — if found, the agent sends ``continuation_prompt`` and loops.

    Tools may be supplied two ways and combined:

    * ``tools=[fn1, fn2]`` — Python functions decorated with ``@aimu.tools.tool``.
    * ``model_client.mcp_client = MCPClient(...)`` — a FastMCP-backed tool server.

    When ``system_message`` is set or ``reset_messages_on_run`` is True, the agent
    clears ``model_client.messages`` and re-applies ``system_message`` before every
    run. This isolates state when a client is shared (e.g. inside a :class:`Chain`).

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
    _last_messages: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name is None:
            # Stable but unique-per-instance default. Users who need readable names
            # in messages histories should pass ``name=`` explicitly.
            self.name = f"agent-{id(self) & 0xFFFFFF:06x}"

    def _prepare_run(self) -> None:
        """Reset client state and re-apply system_message before a run, when configured."""
        if self.reset_messages_on_run or self.system_message is not None:
            self.model_client.reset(system_message=self.system_message)
        if self.tools:
            self.model_client.tools = list(self.tools)

    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Run the agentic loop. ``images`` attach only to the initial turn."""
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images)
        self._prepare_run()
        response = self.model_client.chat(task, generate_kwargs=generate_kwargs, images=images)

        for _ in range(self.max_iterations - 1):
            if not self._last_turn_called_tools():
                break
            logger.debug("Agent '%s' continuing, tools were used in last turn.", self.name)
            response = self.model_client.chat(self.continuation_prompt, generate_kwargs=generate_kwargs)

        self._last_messages = list(self.model_client.messages)
        return response

    def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> Iterator[StreamChunk]:
        self._prepare_run()
        iteration = 0
        for chunk in self.model_client.chat(task, generate_kwargs=generate_kwargs, stream=True, images=images):
            yield StreamChunk(chunk.phase, chunk.content, agent=self.name, iteration=iteration)

        iteration += 1
        while self._last_turn_called_tools() and iteration < self.max_iterations:
            logger.debug("Agent '%s' continuing (iteration %d).", self.name, iteration)
            for chunk in self.model_client.chat(self.continuation_prompt, generate_kwargs=generate_kwargs, stream=True):
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
        unlocks and preserves the ``system_message`` attribute, and this method strips
        the leading system message from *messages* (if present) so it is not
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
        self.model_client.reset()  # clears messages, unlocks system_message, keeps its value
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
        ``BaseModelClient`` — for direct use, call :meth:`run` instead.
        """
        from aimu.agents.agentic_client import _AgenticView

        return _AgenticView(self)

    @classmethod
    def from_config(cls, config: dict[str, Any], model_client: BaseModelClient) -> Agent:
        """Create an Agent from a plain dict config.

        Recognised keys: ``name``, ``system_message``, ``max_iterations``,
        ``continuation_prompt``.
        """
        sm = config.get("system_message")
        return cls(
            model_client=model_client,
            system_message=sm,
            name=config.get("name"),
            max_iterations=config.get("max_iterations", 10),
            continuation_prompt=config.get("continuation_prompt", DEFAULT_CONTINUATION_PROMPT),
        )
