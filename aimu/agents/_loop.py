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

        ``deps`` and ``tool_approval`` (per-run overrides) take precedence over the agent's
        ``self.deps`` / ``self.tool_approval`` fields; the effective values are published to the
        model client (``ToolContext`` injection and the tool-call approval gate, respectively).
        """
        from aimu.tools.approval import approve_all

        if self.reset_messages_on_run or self.system_message is not None:
            self.model_client.reset(system_message=self.system_message)
        if self.tools:
            self.model_client.tools = list(self.tools)
        self.model_client.tool_context_deps = deps if deps is not None else self.deps
        self.model_client.tool_approval = tool_approval or self.tool_approval or approve_all

    def _last_turn_called_tools(self) -> bool:
        """True if a ``"tool"`` message appears after the most recent ``"user"`` message."""
        for msg in reversed(self.model_client.messages):
            if msg.get("role") == "user":
                return False
            if msg.get("role") == "tool":
                return True
        return False

    def _tag_injected_turn(self, index: int, provenance: str) -> None:
        """Mark the framework-injected user turn at ``index`` with a provenance value.

        The continuation / final-answer prompts are appended to ``model_client.messages`` at
        the recorded index by ``_append_user_turn``. Tagging is done here rather than at append
        time so the public ``chat()`` signature stays free of an internal concept, mirroring how
        providers attach the inert ``"thinking"`` key to a message by index. Call after the
        ``chat()`` turn completes (streamed: after the stream is fully consumed). No-op if the
        index no longer names a user turn, so a caller error can't corrupt the transcript.
        """
        from aimu.models._internal.message_meta import PROVENANCE_KEY

        messages = self.model_client.messages
        if 0 <= index < len(messages) and messages[index].get("role") == "user":
            messages[index][PROVENANCE_KEY] = provenance

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
