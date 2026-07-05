from __future__ import annotations

from typing import Any, Iterator, Optional, Union

from aimu.agents._agentic_view import _AgenticViewMixin
from aimu.agents.agent import Agent
from aimu.models.base import BaseModelClient, StreamChunk


class _AgenticView(_AgenticViewMixin, BaseModelClient):
    """Internal: a :class:`BaseModelClient` view backed by an :class:`Agent`.

    Each ``chat()`` runs the full agent loop. Returned by ``Agent.as_model_client()``;
    not part of the public API. Use ``Agent.run()`` directly unless you specifically
    need to drop an agent into an API that expects a ``BaseModelClient``.
    """

    def __init__(self, agent: Agent):
        if not isinstance(agent, Agent):
            raise TypeError(
                f"_AgenticView only accepts Agent, got {type(agent).__name__}. "
                "For workflows (Chain, Router, ...) call their run() / run(stream=True) directly."
            )
        self._bind_agent(agent)

    # --- Agentic overrides ---

    def _chat(
        self,
        user_message: Optional[str] = None,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        if stream:
            return self._agent.run(user_message, generate_kwargs=generate_kwargs, stream=True, images=images)
        return self._agent.run(user_message, generate_kwargs=generate_kwargs, images=images)

    def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        # generate() bypasses the agent loop (no message history, no tools).
        return self._inner_client._generate(prompt, generate_kwargs, stream=stream, images=images, audio=audio)
