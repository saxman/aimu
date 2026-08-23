"""Shared state delegation for the agentic-view wrappers (sync + aio).

``_AgenticView`` / ``_AsyncAgenticView`` wrap an ``Agent`` as a ``BaseModelClient`` /
``AsyncBaseModelClient`` so the agent loop can stand in wherever a model client is
expected. The mutable-state delegation to the wrapped agent's inner client is
identical on both surfaces; only ``_chat`` / ``_generate`` (the ``await`` points) and
the constructor's type-check message differ, so those stay on each concrete view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aimu.events import EventSink


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

    # --- Turn-event suppression: the view is never itself one real model request ---
    #
    # A ModelTurnStarted/ModelTurnFinished pair means one real request to a provider. This
    # view's chat() drives a whole Agent loop -- each real request inside that loop already
    # emits via the inner client's own chat() calls -- and its generate() delegates straight
    # to the inner client's real generate() (see _generate on each concrete view), which
    # emits for that one real request on its own. Either way, this view calling the
    # inherited chat()/generate() turn-tracking would be a phantom event with no
    # corresponding request, so it is unconditionally a no-op here. `events` (below) still
    # delegates to the inner client so the real requests keep reaching the sink.
    def _emit_turn_started(self, pending_user_message: Optional[str] = None) -> tuple[None, str]:
        return None, ""

    def _emit_turn_finished(self, model_id: str, started: Optional[float], result: Any) -> None:
        return None

    def _emit_when_drained(self, chunks: Any, started: Optional[float], model_id: str) -> Any:
        return chunks

    # --- Delegate mutable state to inner_client so both stay in sync ---

    @property
    def events(self) -> Optional["EventSink"]:
        return getattr(self._inner_client, "events", None)

    @events.setter
    def events(self, value: Optional["EventSink"]) -> None:
        self._inner_client.events = value

    @property
    def default_generate_kwargs(self) -> dict:
        return self._inner_client.default_generate_kwargs

    @default_generate_kwargs.setter
    def default_generate_kwargs(self, value: dict) -> None:
        self._inner_client.default_generate_kwargs = value

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

    def _resolve_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        return self._inner_client._resolve_generate_kwargs(generate_kwargs)
