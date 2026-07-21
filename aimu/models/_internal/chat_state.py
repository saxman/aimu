"""I/O-free helpers shared by both base chat clients.

Shared by sync ``BaseModelClient`` and async ``AsyncBaseModelClient``. Contains no
I/O: state mechanics (the bits that mutate ``self.messages`` / ``self._system_message``),
tool-call recording (parse + store a requested tool turn; execution is the Agent's job),
and structured-request resolution (schema -> ``response_format`` / prompt suffix).
Subclasses provide the underlying attributes via their own ``__init__``.
"""

from __future__ import annotations

import random
import string
from contextlib import contextmanager
from typing import Iterator, Optional


class _ChatStateMixin:
    """Mixin providing system-message lifecycle, reset, and user-turn append.

    Subclasses must provide attributes:
      - ``model``: a :class:`Model` enum member (for capability flags)
      - ``messages``: list of OpenAI-format message dicts
      - ``_system_message``: ``str | None``
      - ``last_thinking``: ``str | None``
      - ``last_usage``: ``dict | None``
      - ``last_structured``: ``Any | None`` (validated object from the most recent ``schema=`` call)
      - ``tools``: list of ``@tool``-decorated callables
    """

    @property
    def system_message(self) -> Optional[str]:
        return self._system_message

    @system_message.setter
    def system_message(self, message: Optional[str]) -> None:
        """Set the active system prompt.

        Assigning ``system_message`` mid-conversation rewrites the system entry in
        ``self.messages`` in place (or inserts/removes it), so the change takes effect on
        the next request while the conversation history is preserved. The model is
        re-conditioned on the new prompt for every subsequent turn; prior assistant turns
        remain in the transcript even though they predate the new prompt. Before the first
        chat (``messages`` empty) this just seeds the value, which is injected on the first
        ``chat()`` call.
        """
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
        """Clear the conversation history.

        Default keeps the existing ``system_message``. Pass ``None`` to clear it or a
        new string to replace it.
        """
        self.messages = []
        if system_message != "__keep__":
            self._system_message = system_message
        self.last_thinking = ""
        self.last_usage = None
        self.last_structured = None

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

    @property
    def is_audio_model(self) -> bool:
        return self.model.supports_audio

    @property
    def supports_structured_output(self) -> bool:
        return self.model.supports_structured_output

    def _require_vision(self) -> None:
        """Raise ``ValueError`` if the model lacks vision support.

        Shared by the stateful ``chat(images=...)`` path (via ``_append_user_turn``) and the
        stateless ``generate(images=...)`` path, so both reject images with one message.
        """
        if not self.model.supports_vision:
            raise ValueError(
                f"Model {self.model.name} does not support vision input. Use a model with supports_vision=True."
            )

    def _require_audio(self) -> None:
        """Raise ``ValueError`` if the model lacks audio input support."""
        if not self.model.supports_audio:
            raise ValueError(
                f"Model {self.model.name} does not support audio input. Use a model with supports_audio=True."
            )

    def _append_user_turn(self, user_message: str, images: Optional[list] = None, audio: Optional[list] = None) -> None:
        """Append the system message (if first turn) and the user turn to ``self.messages``.

        Normalises images or audio into OpenAI-format content blocks. ``images`` and
        ``audio`` are mutually exclusive per turn.
        """
        if len(self.messages) == 0 and self._system_message:
            self.messages.append({"role": "system", "content": self._system_message})

        if images and audio:
            raise ValueError("images= and audio= are mutually exclusive per turn. Pass one or the other, not both.")

        if images:
            self._require_vision()
            from .image_input import _build_user_content_blocks

            self.messages.append({"role": "user", "content": _build_user_content_blocks(user_message, images)})
        elif audio:
            self._require_audio()
            from .audio_input import _build_audio_content_blocks

            self.messages.append({"role": "user", "content": _build_audio_content_blocks(user_message, audio)})
        else:
            self.messages.append({"role": "user", "content": user_message})

    @contextmanager
    def _tools_override(self, tools: Optional[list]) -> Iterator[None]:
        """Temporarily replace ``self.tools`` for the span of a single ``chat()`` call.

        ``tools=None`` is a no-op; the client's configured ``self.tools`` are used.
        Any other value (including ``[]`` to disable tools for one call) replaces the
        registered tool callables for the duration of the call and is restored afterwards.
        Since MCP tools also live in ``self.tools`` (via ``MCPClient.as_tools()``), the
        override covers them too. The swap covers both request-spec building
        (``_collect_python_tool_specs``) and dispatch (``_call_plain_tool``), since both
        read ``self.tools``.

        Not safe across concurrent ``chat()`` calls on a shared client, but neither is
        ``self.messages``, so this matches the existing single-conversation contract.
        """
        if tools is None:
            yield
            return
        saved = self.tools
        self.tools = list(tools)
        try:
            yield
        finally:
            self.tools = saved

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

    # ------------------------------------------------------------------ #
    # Tool-call recording (parse + store; execution is the Agent's job)   #
    # ------------------------------------------------------------------ #

    def _prepare_tool_calls(self, tool_calls: list[dict]) -> list[tuple[dict, str]]:
        """Normalize ``arguments``/``parameters`` and assign tool_call_ids upfront.

        Concurrent execution can use pre-assigned IDs and still append results in
        original order.
        """
        prepared = []
        for tc in tool_calls:
            # llama 3.1 uses 'parameters' instead of 'arguments'
            if "arguments" not in tc and "parameters" in tc:
                tc["arguments"] = tc.pop("parameters")
            tc_id = "".join(random.choices(string.ascii_letters + string.digits, k=9))
            prepared.append((tc, tc_id))
        return prepared

    def _append_assistant_tool_calls(self, prepared: list[tuple[dict, str]], content: str = "") -> None:
        """Append the assistant message that records the tool calls being made.

        ``content`` is the natural-language text the model emitted alongside the tool call in
        the same turn (models can generate both). It is stored on the message when non-empty so
        the transcript preserves it, matching how HuggingFace / OpenAI / Anthropic represent a
        single generation that carries both ``content`` and ``tool_calls``.
        """
        message: dict = {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}, "id": tc_id}
                for tc, tc_id in prepared
            ],
        }
        if content:
            message["content"] = content
        self.messages.append(message)

    def _record_tool_calls(self, tool_calls: list[dict], content: str = "") -> None:
        """Store the assistant turn that requested tools — parse + record, **no execution**.

        The model client's job ends at parsing: it records the requested ``tool_calls`` (with ids
        and any prose ``content``) on the assistant message. Executing the tools and appending
        their results is the Agent's job — see the tool-loop engine ``aimu.agents._tool_loop``.
        """
        self._append_assistant_tool_calls(self._prepare_tool_calls(tool_calls), content)

    # ------------------------------------------------------------------ #
    # Structured-request resolution (schema -> response_format / suffix)  #
    # ------------------------------------------------------------------ #

    def _structured_request(self, schema: type) -> tuple[Optional[dict], str]:
        """Resolve a schema to ``(response_format, prompt_suffix)`` for the active model.

        Native models get the JSON Schema dict as ``response_format`` and no prompt suffix;
        parse-path models get ``None`` and a suffix instructing JSON output. The provider
        only ever receives ``response_format`` when it's non-None (native).
        """
        from .structured import json_schema_instruction, schema_to_json_schema

        json_schema = schema_to_json_schema(schema)
        if self.supports_structured_output:
            return json_schema, ""
        return None, "\n\n" + json_schema_instruction(json_schema)
