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
#
# An entry carries a `family`, not a bare sink: a module-level ContextVar is necessarily
# client-agnostic, so a bare sink would reach *any* client called while the scope is open --
# including a wholly unrelated client a tool happens to call (e.g. make_subagent_tool's fresh
# per-call ModelClient), stamping that client's turns with the calling agent's name and
# swallowing the callee's own RunStarted/RunFinished. `family` is the set of client objects this
# override is actually *for* -- see `_client_family` -- and `_effective_sink` only returns an
# entry's sink when the client asking is a member of it.
#
# The payload is a *stack* of mutable `_EventScope` objects rather than one `(sink, family)`
# tuple, and teardown flips `scope.active` rather than resetting a `Token`. A Token is the
# obvious way to restore a ContextVar, and it is the wrong one here, for two measured reasons:
#
#   1. A Token can only be reset in the Context that created it. `_events_override` brackets a
#      `yield` inside async generators (`_AsyncToolLoop.run_streamed` consumed by
#      `aio.Agent._run_loop_streamed`), and a consumer that stops iterating early -- a UI's stop
#      button on `run(stream=True, events=sink)` -- leaves those generators to the event loop's
#      asyncgen finalizer, which closes them in a *separate Task*. `reset(token)` there raises
#      `ValueError: <Token ...> was created in a different Context` and the caller's context
#      keeps the entry forever, so a later, wholly un-instrumented `chat()` on that client still
#      reports to the abandoned run's sink.
#   2. Resetting out of LIFO order restores `token.old_value`, which silently *drops* any scope
#      opened after it. Two streamed runs open on one execution context and drained
#      out of order (legal: they are independent generators) end with the var stuck holding the
#      first run's payload.
#
# Mutating a plain object, by contrast, is context-independent: the finalizer Task and the
# caller share the same `_EventScope` instance, so `active = False` is always effective and
# `_effective_sink` skips the entry no matter which context observes it. The stack is rebuilt
# with `set()` (which never raises) on both push and pop, dropping inactive entries each time,
# so ordinary nesting behaves exactly as a Token would while neither failure above can occur.
_ACTIVE_EVENT_SINK: ContextVar[tuple] = ContextVar("aimu_active_event_sink", default=())


class _EventScope:
    """One open ``_events_override`` scope: which sink, for which clients, still live?

    ``active`` is deliberately mutable and deliberately *not* a ContextVar: flipping it is the
    one teardown operation that works from any execution context, including the asyncgen
    finalizer Task that closes an abandoned streamed run. See ``_ACTIVE_EVENT_SINK`` above.

    That reach is the point, and it cuts both ways: teardown is global, so a task still holding
    a *copy* of a context in which this scope was open stops seeing the sink once the scope
    closes, where a ``Token`` reset (visible only in the resetting context) would have left it
    reporting into a finished run. Nothing in AIMU detaches such a task -- ``TaskGroup`` awaits
    its children and ``RunHandle`` wraps the whole run -- and ending with the run is the
    intended reading of a per-run sink either way.
    """

    __slots__ = ("sink", "family", "active")

    def __init__(self, sink: Any, family: list):
        self.sink = sink
        self.family = family
        self.active = True


# Attribute names used across this codebase's client wrappers to hold the inner client(s) they
# delegate mutable state to (self.events, self.messages, ...). Duck-typed on attribute name
# rather than isinstance, so this low-level module -- inherited by every base client -- never
# has to import the higher-level wrapper classes that define them: ModelClient / AsyncModelClient
# (`_client`), _AsyncInProcessClient and its HuggingFace/llamacpp subclasses (`_sync`), and
# _AgenticView / _AsyncAgenticView (`_inner_client`).
# The stop reasons that mean "output ran out of room" rather than "the model finished". Each
# backend spells it its own way -- OpenAI-compatible servers and Ollama say "length", Anthropic
# says "max_tokens" -- and one set beats a per-provider comparison for the same reason the
# generate_kwargs merge lives on the base: it is a rule that has to hold everywhere.
TRUNCATED_STOP_REASONS = frozenset({"length", "max_tokens"})

# Anthropic's safety classifiers decline with this rather than an HTTP error.
REFUSAL_STOP_REASON = "refusal"

_DELEGATE_ATTRS = ("_client", "_sync", "_inner_client")
# Same idea for a wrapper delegating to *several* clients: FallbackClient / AsyncFallbackClient
# (`clients`), which tries each in turn by calling each inner client's own public chat()/generate().
_DELEGATE_LIST_ATTRS = ("clients",)


def _client_family(client: Any) -> list:
    """``client`` plus every inner client it (transitively) delegates state to.

    A scoped override is installed on whatever object the caller holds as ``model_client`` --
    but the object whose own code actually runs a given emit site can be a *different* one.
    ``BaseModelClient.chat()`` (inherited by a wrapper that doesn't override it) calls
    ``self._emit_turn_started()`` / ``self._emit_turn_finished()`` with ``self`` bound to
    whatever object ``chat()`` was called on; a delegating wrapper's own ``_chat()`` then hands
    off to an inner client's *raw* ``_chat()`` (not its public ``chat()``), so
    ``self._record_request(...)`` inside that call runs with ``self`` bound to the *inner*
    client instead. Both must resolve to the same override, so ``_events_override`` records the
    whole family (this function) rather than just the object it was called on.

    Traversal is iterative with an explicit stack (not recursion) and a ``seen`` set keyed on
    ``id()`` purely to avoid revisiting a node in a cyclic or diamond delegation graph within
    this one call -- every object visited is held live in ``family`` for the call's duration,
    so there is no id-reuse hazard here (contrast the *membership test* in ``_effective_sink``,
    which must compare by identity against long-lived references for the same reason).
    """
    family = [client]
    seen = {id(client)}
    stack = [client]
    while stack:
        current = stack.pop()
        for attr in _DELEGATE_ATTRS:
            inner = getattr(current, attr, None)
            if inner is not None and id(inner) not in seen:
                family.append(inner)
                seen.add(id(inner))
                stack.append(inner)
        for attr in _DELEGATE_LIST_ATTRS:
            for inner in getattr(current, attr, None) or ():
                if id(inner) not in seen:
                    family.append(inner)
                    seen.add(id(inner))
                    stack.append(inner)
    return family


def _effective_sink(client: Any) -> Optional[Any]:
    """The event sink for the current execution context, else the client's own.

    Every emit site on both the sync and async client bases calls this instead of reading
    ``client.events`` directly, so a scoped override (see ``_ChatStateMixin._events_override``)
    reaches only ``client`` and the family it was installed for (see ``_client_family``) -- not
    every client that happens to run any code while the scope is open. Membership is checked by
    identity against the family's held object references (``is``, never a bare ``id()``
    comparison or a set keyed by ``__eq__``/``__hash__``), which is what makes this safe even if
    some other object were later allocated at a reused address: the family list itself keeps
    every one of its members alive for the scope's duration, so no id it holds can be reused
    while the scope is open.

    A tool that calls a *different* client than the one the run's override was installed for
    (e.g. ``make_subagent_tool``'s fresh per-call ``ModelClient``, or any client built inside a
    tool) never matches this family, on either surface and regardless of
    ``concurrent_tool_calls`` -- it falls back to that client's own ``self.events``, so passing
    ``events=`` explicitly to such a client is both necessary and sufficient for it to report
    reliably to a particular sink.

    The one case that still depends on the surface is a tool that calls the *same* client the
    run's override was installed for (the ordinary, intended case -- e.g. every worker `Agent`
    in a `Parallel` sharing one `model_client`, or a tool that reuses `ctx.deps` holding that
    client): sync dispatches a concurrent tool via a plain ``ThreadPoolExecutor.submit()`` with
    no ``contextvars.copy_context()``, so that thread's empty ``Context`` means the override is
    invisible there even though the client matches the family; async's
    ``asyncio.TaskGroup.create_task()`` always copies the current context, so it *is* visible
    there. See ``tests/test_events.py`` / ``tests/test_aio_events.py``'s
    concurrent-tool-dispatch tests for both the same-client and different-client cases,
    verified rather than assumed.

    The stack is scanned from the top down (innermost open scope first) and inactive entries are
    skipped, so an abandoned run's entry -- which its finalizer marked inert but could not remove
    from this context -- is passed over, and an outer scope stays reachable to the clients *it*
    covers even while an inner, narrower one is open. See ``_ACTIVE_EVENT_SINK``.
    """
    for scope in reversed(_ACTIVE_EVENT_SINK.get()):
        if scope.active and any(client is member for member in scope.family):
            return scope.sink
    return getattr(client, "events", None)


class _ChatStateMixin:
    """Mixin providing system-message lifecycle, reset, and user-turn append.

    Subclasses must provide attributes:
      - ``model``: a :class:`Model` enum member (for capability flags)
      - ``messages``: list of OpenAI-format message dicts
      - ``_system_message``: ``str | None``
      - ``last_thinking``: ``str | None``
      - ``last_usage``: ``dict | None``
      - ``last_stop_reason``: ``str | None`` (the provider's own word for how the turn ended)
      - ``last_output_truncated``: ``bool``
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
        self.last_stop_reason = None
        self.last_output_truncated = False
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
        """Temporarily install ``events`` as the active sink for ``self``'s client family.

        ``events=None`` is a no-op: whatever the current effective sink is (the
        ``_ACTIVE_EVENT_SINK`` ContextVar if one is already active *and ``self`` is a member
        of the family it was installed for*, else ``self.events``) is left alone. An ``Agent``
        uses this to make its per-run event sink reach the client's own turn events
        (``ModelTurnStarted`` / ``ModelTurnFinished`` / ``RequestPrepared``, all read via
        ``_effective_sink``) for the duration of the run.

        Unlike ``_tools_override`` (a genuine mutation of ``self.tools``, since request-spec
        building has to read a plain attribute), this does **not** touch ``self.events`` --
        it pushes a ``_EventScope`` (the sink plus ``_client_family(self)``) onto a stack held
        in a module-level ``ContextVar``, so the override is visible only within the execution
        context (OS thread / asyncio Task) that entered it, and only to ``self`` and whatever
        it delegates state to or from (see ``_client_family``) -- not to an unrelated client a
        tool happens to call while the scope is open. That combination is what makes it safe
        for two concurrently-running agents to share one ``model_client`` (e.g. every worker
        ``Agent`` in a ``Parallel`` built via ``Parallel.from_client``): each worker's thread
        gets its own independent copy of the ContextVar, so one worker's override can never
        clobber another's, *and* a worker's override can never leak onto a client outside its
        own family. See ``_ACTIVE_EVENT_SINK`` / ``_client_family`` / ``_effective_sink`` above
        for the mechanism, and ``tests/test_workflow_parallel.py``'s concurrent-workers test
        for the proof.

        ``self.events`` set directly on a shared client (rather than through this scoped
        override) is still genuinely shared -- that assignment has no execution-context
        boundary to isolate by, and is documented as such on ``events`` itself.

        Teardown flips the scope's ``active`` flag and rebuilds the stack without it. It does
        **not** reset a ``Token``, because this context manager brackets a ``yield`` inside async
        generators: a consumer that abandons a streamed run leaves them to the event loop's
        asyncgen finalizer, which runs in a different ``Context`` where a ``Token`` reset raises
        and the entry would leak permanently. ``_ACTIVE_EVENT_SINK`` documents that measurement
        and the out-of-LIFO-order case in full. Nothing here can raise, so nothing is swallowed:
        the flag lives on an object both contexts share, and ``set()`` never fails.
        """
        if events is None:
            yield
            return
        scope = _EventScope(events, _client_family(self))
        # Drop any entry a cross-context teardown could only mark inert, so the stack stays
        # bounded by the number of genuinely open scopes rather than growing per abandoned run.
        _ACTIVE_EVENT_SINK.set(tuple(s for s in _ACTIVE_EVENT_SINK.get() if s.active) + (scope,))
        try:
            yield
        finally:
            scope.active = False
            _ACTIVE_EVENT_SINK.set(tuple(s for s in _ACTIVE_EVENT_SINK.get() if s.active))

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

    def _record_stop_reason(self, stop_reason: Optional[str]) -> None:
        """Record how the provider says this turn ended, and derive ``last_output_truncated``.

        The single seam for it, mirroring ``_record_request``: ``last_output_truncated`` is
        consumed by the agent loop (``_ToolLoop._raise_if_truncated`` turns it into a
        ``TruncatedTurnError`` naming the remedy), and its default of ``False`` means "nobody
        looked", not "not truncated". A provider that skips this therefore does not merely lose a
        field -- it makes an actionable error silently stop firing. Only Ollama set it before
        v0.27.0, so on every other backend a turn cut off mid-answer surfaced as a bare empty
        string. A guard test enforces the rule rather than a convention.

        ``None`` is recorded as-is: it means the provider said nothing, which is different from
        saying the turn finished normally.
        """
        self.last_stop_reason = stop_reason
        self.last_output_truncated = stop_reason in TRUNCATED_STOP_REASONS

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
