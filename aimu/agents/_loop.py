"""Pure agent-loop helpers shared by the sync and async ``Agent``.

These three methods are byte-for-byte identical across :class:`aimu.agents.Agent`
and :class:`aimu.aio.Agent`: they only read or assign model-client state and never
``await``, so they live here and are mixed into both. The surface-specific
``run()`` / ``_run_streamed()`` bodies stay on each class (their return types and
``await`` points differ).

The mixin reads attributes (``model_client``, ``reset_messages_on_run``,
``system_message``, ``tools``, ``deps``) that the concrete ``Agent`` dataclasses
define as fields. It is intentionally **not** a dataclass, so its annotations are
not collected as fields by the ``@dataclass`` decorator on the concrete classes.
"""

from __future__ import annotations

from typing import Any


class _AgentLoopMixin:
    """Shared, surface-agnostic helpers for the tool-calling loop."""

    # Provided by the concrete Agent dataclasses (annotations only; not dataclass fields here).
    model_client: Any
    reset_messages_on_run: bool
    system_message: Any
    tools: list
    deps: Any
    tool_approval: Any

    def _prepare_run(self, deps: Any = None, tool_approval: Any = None) -> None:
        """Reset client state and re-apply system_message before a run, when configured.

        Tool callables, ``deps``, and the approval policy are the Agent's own state now and are
        passed to the tool-loop engine per run (see :meth:`Agent._make_tool_loop`); the model
        client no longer holds them. ``deps`` / ``tool_approval`` are accepted for call-site
        compatibility but are not pushed onto the client here.
        """
        if self.reset_messages_on_run or self.system_message is not None:
            self.model_client.reset(system_message=self.system_message)

    def restore(self, messages: list[dict]) -> None:
        """Restore agent state from a saved message list for resuming after failure.

        Saves the partial state from a failed run via ``agent.model_client.messages``
        (the live list, not the post-run snapshot from ``agent.messages``), then call
        this method before the next ``run()`` to resume from that point.

        Handles the system-message duplication hazard: ``model_client.reset()`` clears
        history and preserves the ``system_message`` attribute, and this method strips
        the leading system message from *messages* (if present) so it is not prepended
        twice on the next ``chat()`` call.

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
