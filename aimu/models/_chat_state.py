"""Pure state mechanics for chat model clients.

Shared by sync ``BaseModelClient`` and async ``AsyncBaseModelClient``. Contains no
I/O — only the bits that mutate ``self.messages`` and ``self._system_message``.
Subclasses provide the underlying attributes via their own ``__init__``.
"""

from __future__ import annotations

from typing import Optional


class _ChatStateMixin:
    """Mixin providing system-message lifecycle, reset, and user-turn append.

    Subclasses must provide attributes:
      - ``model``: a :class:`Model` enum member (for capability flags)
      - ``messages``: list of OpenAI-format message dicts
      - ``_system_message``: ``str | None``
      - ``_system_message_locked``: ``bool``
      - ``last_thinking``: ``str | None``
      - ``tools``: list of ``@tool``-decorated callables
    """

    @property
    def system_message(self) -> Optional[str]:
        return self._system_message

    @system_message.setter
    def system_message(self, message: Optional[str]) -> None:
        if self._system_message_locked:
            raise RuntimeError(
                "system_message is immutable after the conversation starts. "
                "Call client.reset() to clear messages, then assign a new system_message."
            )
        self._system_message = message
        if self.messages:
            if self.messages[0]["role"] == "system":
                if message is None:
                    self.messages.pop(0)
                else:
                    self.messages[0]["content"] = message
            elif message is not None:
                self.messages.insert(0, {"role": "system", "content": message})

    def reset(self, system_message: Optional[str] = "__keep__") -> None:
        """Clear the conversation history and unlock ``system_message``.

        Default keeps the existing ``system_message``. Pass ``None`` to clear it or a
        new string to replace it.
        """
        self.messages = []
        self._system_message_locked = False
        if system_message != "__keep__":
            self._system_message = system_message
        self.last_thinking = ""

    def __deepcopy__(self, memo):
        # Stateful conversation history and non-copyable backend resources.
        memo[id(self)] = self
        return self

    @property
    def is_thinking_model(self) -> bool:
        return self.model.supports_thinking

    @property
    def is_tool_using_model(self) -> bool:
        return self.model.supports_tools

    @property
    def is_vision_model(self) -> bool:
        return self.model.supports_vision

    def _append_user_turn(self, user_message: str, images: Optional[list] = None) -> None:
        """Append the system message (if first turn) and the user turn to ``self.messages``.

        Normalises images into OpenAI-format content blocks for vision-capable models.
        Locks ``system_message`` against further mutation.
        """
        if len(self.messages) == 0 and self._system_message:
            self.messages.append({"role": "system", "content": self._system_message})

        if images:
            if not self.model.supports_vision:
                raise ValueError(
                    f"Model {self.model.name} does not support vision input. Use a model with supports_vision=True."
                )
            from ._images import _build_user_content_blocks

            self.messages.append({"role": "user", "content": _build_user_content_blocks(user_message, images)})
        else:
            self.messages.append({"role": "user", "content": user_message})

        self._system_message_locked = True

    def _collect_python_tool_specs(self) -> list[dict]:
        """Collect ``__tool_spec__`` dicts from every registered Python tool callable.

        Raises ``ValueError`` if a callable lacks ``__tool_spec__`` (i.e. wasn't
        decorated with ``@aimu.tools.tool``).
        """
        specs = []
        for fn in self.tools:
            spec = getattr(fn, "__tool_spec__", None)
            if spec is None:
                raise ValueError(
                    f"Tool '{getattr(fn, '__name__', fn)}' is missing __tool_spec__. Decorate it with @aimu.tools.tool."
                )
            specs.append(spec)
        return specs
