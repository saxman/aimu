"""Shared state delegation for the agentic-view wrappers (sync + aio).

``_AgenticView`` / ``_AsyncAgenticView`` wrap an ``Agent`` as a ``BaseModelClient`` /
``AsyncBaseModelClient`` so the agent loop can stand in wherever a model client is
expected. The mutable-state delegation to the wrapped agent's inner client is
identical on both surfaces; only ``_chat`` / ``_generate`` (the ``await`` points) and
the constructor's type-check message differ, so those stay on each concrete view.
"""

from __future__ import annotations

from typing import Any, Optional


class _AgenticViewMixin:
    """Delegates model-client state to the wrapped agent's inner client."""

    _inner_client: Any

    def _bind_agent(self, agent: Any) -> None:
        """Wire the view to *agent* and mirror the inner client's static attributes.

        ``super().__init__()`` is intentionally not called; it would reset inner-client state.
        """
        self._agent = agent
        self._inner_client = agent.model_client
        self.model = self._inner_client.model
        self.model_kwargs = self._inner_client.model_kwargs
        self.default_generate_kwargs = self._inner_client.default_generate_kwargs

    # --- Delegate mutable state to inner_client so both stay in sync ---

    @property
    def messages(self) -> list[dict]:
        return self._inner_client.messages

    @messages.setter
    def messages(self, value: list[dict]) -> None:
        self._inner_client.messages = value

    @property
    def tools(self) -> list:
        return self._inner_client.tools

    @tools.setter
    def tools(self, value: list) -> None:
        self._inner_client.tools = value

    @property
    def system_message(self) -> Optional[str]:
        return self._inner_client.system_message

    @system_message.setter
    def system_message(self, message: Optional[str]) -> None:
        self._inner_client.system_message = message

    @property
    def last_thinking(self) -> str:
        return self._inner_client.last_thinking

    @last_thinking.setter
    def last_thinking(self, value: str) -> None:
        self._inner_client.last_thinking = value

    @property
    def last_structured(self):
        return self._inner_client.last_structured

    @last_structured.setter
    def last_structured(self, value) -> None:
        self._inner_client.last_structured = value

    def reset(self, system_message: Optional[str] = "__keep__") -> None:
        self._inner_client.reset(system_message)

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        return self._inner_client._update_generate_kwargs(generate_kwargs)
