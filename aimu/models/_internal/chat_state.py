"""I/O-free helpers shared by both base chat clients.

Shared by sync ``BaseModelClient`` and async ``AsyncBaseModelClient``. Contains no
I/O: state mechanics (the bits that mutate ``self.messages`` / ``self._system_message``),
tool-call recording (parse + store a requested tool turn; execution is the Agent's job),
and structured-request resolution (schema -> ``response_format`` / prompt suffix).
Subclasses provide the underlying attributes via their own ``__init__``.
"""

from __future__ import annotations

import logging
import random
import string
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Iterator, Optional, Union

from .thinking import THINKING_KWARG, ResolvedThinking, resolve_thinking

logger = logging.getLogger(__name__)

# The scoped event-sink override, held per execution context rather than on the client.
#
# A client can be shared by agents running concurrently -- Parallel.from_client builds every
# worker Agent over one -- and a plain attribute swap (the original implementation) is visible
# to all of them: two workers' overlapping swap/restore sequences interleave and clobber each
# other's sink (dropped and misattributed events; see tests/test_workflow_parallel.py's
# concurrent-workers test, which replaced a test that pinned this as a known gap).
#
# A ContextVar isolates by *execution context*, not by client instance: each OS thread gets its
# own independent Context (nothing to copy at submit time -- see the isolation note on
# _events_override below), and each asyncio Task gets its own copy of the context it was created
# in, by construction. `self.events` on the client remains the durable, always-shared setting;
# only the temporary per-run override moves off the client.
_ACTIVE_EVENT_SINK: ContextVar[Optional[Any]] = ContextVar("aimu_active_event_sink", default=None)


def _effective_sink(client: Any) -> Optional[Any]:
    """The event sink for the current execution context, else the client's own.

    Every emit site on both the sync and async client bases calls this instead of reading
    ``client.events`` directly, so a scoped override (see ``_ChatStateMixin._events_override``)
    reaches only the execution context that installed it.

    A tool called under ``concurrent_tool_calls=True`` behaves *differently* on the two
    surfaces here, verified rather than assumed -- see ``tests/test_events.py``'s
    concurrent-tool-dispatch tests:

    * **Sync**: the tool-loop engine dispatches via ``concurrent.futures.ThreadPoolExecutor``,
      a plain ``.submit()`` with no ``contextvars.copy_context()``. Each OS thread starts with
      its own independent default ``Context``, so a client called from inside such a tool does
      **not** see the run's active override there; it falls back to its own ``self.events``.
    * **Async**: the tool-loop engine dispatches via ``asyncio.TaskGroup.create_task()``, which
      *always* copies the current ``Context`` into the new task (standard ``asyncio`` behaviour,
      not something this module opts into). Since dispatch happens while the run's
      ``_events_override`` scope is still open, a client called from inside a concurrently
      dispatched async tool **does** inherit that override -- and, because the attribution
      wrapper (``_attributing`` in ``aimu.agents._tool_loop`` / ``aimu.aio._tool_loop``) stamps
      *any* event lacking its own ``agent``, that inner client's turn events land in the outer
      run's sink attributed to the *calling* agent, not to whatever produced them.

    Either way, pass ``events=`` explicitly to a client constructed inside a tool if it needs
    reliable, correctly-attributed delivery to a particular sink.
    """
    return _ACTIVE_EVENT_SINK.get() or getattr(client, "events", None)


class _ChatStateMixin:
    """Mixin providing system-message lifecycle, reset, and user-turn append.

    Subclasses must provide attributes:
      - ``model``: a :class:`Model` enum member (for capability flags)
      - ``messages``: list of OpenAI-format message dicts
      - ``_system_message``: ``str | None``
      - ``last_thinking``: ``str | None``
      - ``last_usage``: ``dict | None``
      - ``last_structured``: ``Any | None`` (validated object from the most recent ``schema=`` call)
      - ``last_request``: ``Any | None`` (the payload of the most recent request, post-adaptation)
      - ``tools``: list of ``@tool``-decorated callables

    Generation-kwarg resolution is a separate concern (it serves ``generate()`` as much as
    ``chat()``) and lives in :class:`~aimu.models._internal.generate_kwargs._GenerateKwargsMixin`.
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
        self.last_request = None

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

    def _warn_once(self, message: str) -> None:
        """Log ``message`` at WARNING the first time only.

        Thinking-control warnings are raised per call, so an agent loop would repeat one
        message every round without this.
        """
        # Initialised lazily so no provider __init__ (nor FallbackClient, which mirrors
        # BaseModelClient.__init__ by hand) has to be touched.
        seen = getattr(self, "_warned_messages", None)
        if seen is None:
            seen = self._warned_messages = set()
        if message in seen:
            return
        seen.add(message)
        logger.warning(message)

    def _apply_thinking(
        self,
        generate_kwargs: Optional[dict],
        thinking: Optional[Union[bool, str]],
    ) -> Optional[dict]:
        """Resolve ``thinking=`` and carry the result in the generate_kwargs dict.

        Returns ``generate_kwargs`` unchanged when there is nothing for a provider to do, so
        ``thinking=None`` leaves every request path byte-for-byte as it was.
        """
        # A `ResolvedThinking` value already there is this module's own re-threading of an
        # already-resolved request through a nested chat() call (e.g. Agent -> tool-loop ->
        # inner client), not caller input, so it passes through untouched below. Anything
        # else under this key is a caller mistake: reject it here, at the layer where it is
        # actionable, rather than let it reach a provider that assumes a ResolvedThinking and
        # raises an opaque AttributeError.
        if generate_kwargs and THINKING_KWARG in generate_kwargs:
            existing = generate_kwargs[THINKING_KWARG]
            if not isinstance(existing, ResolvedThinking):
                raise ValueError(
                    f"generate_kwargs contains the reserved key {THINKING_KWARG!r}, which AIMU uses "
                    "internally to carry a resolved thinking request to the provider. Use the "
                    "thinking= parameter instead of setting this key directly."
                )
        resolved = resolve_thinking(self.model, thinking, warn=self._warn_once)
        if resolved is None:
            return generate_kwargs
        return {**(generate_kwargs or {}), THINKING_KWARG: resolved}

    def _append_message(self, message: dict) -> None:
        """Append ``message`` to ``self.messages``, stamping it with an append-time ``timestamp``.

        ``setdefault`` so a caller that already set one (e.g. a restore replay) wins. The key is inert
        (see ``INERT_MESSAGE_KEYS``) and stripped before any provider request, so stamping here never
        changes the payload sent to a model; it exists for UIs and persistence.
        """
        message.setdefault("timestamp", datetime.now().isoformat())
        self.messages.append(message)

    def _pending_message_count(self, pending_user_message: Optional[str]) -> int:
        """How many messages the request about to be issued will carry.

        Mirrors ``_append_user_turn``: the system message is seeded only on a turn that
        appends a user message to an empty history.
        """
        if pending_user_message is None:
            return len(self.messages)
        seeds_system = 1 if (not self.messages and self.system_message) else 0
        return len(self.messages) + 1 + seeds_system

    def _append_user_turn(self, user_message: str, images: Optional[list] = None, audio: Optional[list] = None) -> None:
        """Append the system message (if first turn) and the user turn to ``self.messages``.

        Normalises images or audio into OpenAI-format content blocks. ``images`` and
        ``audio`` are mutually exclusive per turn.
        """
        if len(self.messages) == 0 and self._system_message:
            self._append_message({"role": "system", "content": self._system_message})

        if images and audio:
            raise ValueError("images= and audio= are mutually exclusive per turn. Pass one or the other, not both.")

        if images:
            self._require_vision()
            from .image_input import _build_user_content_blocks

            self._append_message({"role": "user", "content": _build_user_content_blocks(user_message, images)})
        elif audio:
            self._require_audio()
            from .audio_input import _build_audio_content_blocks

            self._append_message({"role": "user", "content": _build_audio_content_blocks(user_message, audio)})
        else:
            self._append_message({"role": "user", "content": user_message})

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

    @contextmanager
    def _events_override(self, events: Optional[object]) -> Iterator[None]:
        """Temporarily install ``events`` as the active sink for this execution context.

        ``events=None`` is a no-op: whatever the current effective sink is (the
        ``_ACTIVE_EVENT_SINK`` ContextVar if one is already active, else ``self.events``)
        is left alone. An ``Agent`` uses this to make its per-run event sink reach the
        client's own turn events (``ModelTurnStarted`` / ``ModelTurnFinished`` /
        ``RequestPrepared``, all read via ``_effective_sink``) for the duration of the run.

        Unlike ``_tools_override`` (a genuine mutation of ``self.tools``, since request-spec
        building has to read a plain attribute), this does **not** touch ``self.events`` --
        it sets a module-level ``ContextVar`` and resets its token in a ``finally``, so the
        override is visible only within the execution context (OS thread / asyncio Task)
        that entered it. That is what makes it safe for two concurrently-running agents to
        share one ``model_client`` (e.g. every worker ``Agent`` in a ``Parallel`` built via
        ``Parallel.from_client``): each worker's thread gets its own independent copy of the
        ContextVar, so one worker's override can never clobber another's. See
        ``_ACTIVE_EVENT_SINK`` / ``_effective_sink`` above for the mechanism, and
        ``tests/test_workflow_parallel.py``'s concurrent-workers test for the proof.

        ``self.events`` set directly on a shared client (rather than through this scoped
        override) is still genuinely shared -- that assignment has no execution-context
        boundary to isolate by, and is documented as such on ``events`` itself.
        """
        if events is None:
            yield
            return
        token = _ACTIVE_EVENT_SINK.set(events)
        try:
            yield
        finally:
            _ACTIVE_EVENT_SINK.reset(token)

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
        self._append_message(message)

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
