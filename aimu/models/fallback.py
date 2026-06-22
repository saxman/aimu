"""Composable failover across model clients (P0-B).

``FallbackClient`` wraps an ordered list of :class:`BaseModelClient` instances and tries
them in turn: the first client that answers wins; if a client raises (by default any
``Exception``), the next is tried with the same conversation state. When every client
fails, :class:`FallbackExhaustedError` is raised with the last error chained as its cause.

Because ``FallbackClient`` *is* a ``BaseModelClient``, it drops into anywhere a plain
client is accepted (``Agent``, workflows, ``Benchmark``, ``agent.as_model_client()``),
with no failover-specific wiring. It is the principle-aligned answer to provider
failover: one composable class, not a gateway.

This is a thin policy layer, not a retry/backoff engine: it does not sleep, jitter, or
bound attempts beyond the client list. Pair it with each client's own ``max_retries`` /
``timeout`` (see the provider constructors) for in-SDK retry plus cross-provider failover.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Iterable, Iterator, Optional, Union

from .base import BaseModelClient, StreamChunk, StreamingContentType

logger = logging.getLogger(__name__)


class FallbackExhaustedError(RuntimeError):
    """Raised when every client in a :class:`FallbackClient` failed.

    The most recent client's exception is chained as ``__cause__``; the full list of
    ``(client, exception)`` pairs is available on ``.errors`` for inspection.
    """

    def __init__(self, message: str, errors: list[tuple[Any, BaseException]]):
        super().__init__(message)
        self.errors = errors


class _FallbackStateMixin:
    """Shared construction + state plumbing for the sync and async fallback clients.

    Holds the canonical conversation (``messages`` / ``system_message`` / ``tools``) on the
    fallback client and loads it into whichever inner client runs, so history survives a
    failover. The sync/async subclasses add the (a)sync ``chat`` / ``generate`` loops.
    """

    def __init__(
        self,
        clients: list,
        *,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
        system_message: Optional[str] = None,
        name: Optional[str] = None,
    ):
        if not clients:
            raise ValueError("FallbackClient requires at least one client.")
        self.clients = list(clients)
        self.retry_on = retry_on
        self.name = name

        # Own conversation/state, mirroring BaseModelClient.__init__ (which is abstract and
        # provider-specific, so it isn't called here).
        self.model = self.clients[0].model
        self.model_kwargs = None
        self._system_message = system_message
        self.default_generate_kwargs = {}
        self.messages: list[dict] = []
        self.tools: list = []
        self.last_thinking = ""
        self.last_usage: Optional[dict] = None
        self.concurrent_tool_calls = False
        self.tool_context_deps = None

    def _load_state(self, client) -> None:
        """Load the shared conversation state into ``client`` before delegating to it."""
        client.reset(self.system_message)  # clears history, syncs system_message (delegates on ModelClient)
        client.messages = copy.deepcopy(self.messages)
        client.tools = list(self.tools)
        try:
            client.tool_context_deps = self.tool_context_deps
        except AttributeError:
            pass
        client.last_thinking = ""
        client.last_usage = None

    def _adopt_state(self, client: BaseModelClient) -> None:
        """Adopt the winning client's resulting state as the canonical conversation."""
        self.messages = client.messages
        self.last_thinking = getattr(client, "last_thinking", "")
        self.last_usage = getattr(client, "last_usage", None)

    def _exhausted(self, errors: list[tuple[Any, BaseException]]) -> "FallbackExhaustedError":
        last = errors[-1][1] if errors else None
        message = f"All {len(self.clients)} fallback client(s) failed"
        if last is not None:
            message += f"; last error: {type(last).__name__}: {last}"
        exc = FallbackExhaustedError(message, errors)
        exc.__cause__ = last
        return exc


class FallbackClient(_FallbackStateMixin, BaseModelClient):
    """A ``BaseModelClient`` that delegates to an ordered list of clients with failover.

    Args:
        clients: Ordered client list; index 0 is preferred, later entries are fallbacks.
        retry_on: Exception types that trigger failover to the next client. Defaults to
            ``(Exception,)`` (fail over on any error). Narrow it (e.g. to timeout/connection
            errors) to let permanent errors surface immediately instead of being masked.
        system_message: Optional system prompt for the shared conversation; synced into
            whichever client runs.
        name: Optional label (currently informational only).

    Conversation state (``messages``, ``system_message``, ``tools``) lives on the
    ``FallbackClient`` and is loaded into whichever client runs, so history is preserved
    across a failover. Capability flags (``is_thinking_model`` etc.) and ``model`` reflect
    the **first** client; use capability-compatible clients in one fallback set.

    Streaming caveat: failover only happens *before the first chunk is emitted*. If a
    client fails mid-stream (after yielding output), the error propagates rather than
    silently replaying from another client.
    """

    # ------------------------------------------------------------------ #
    # Public surface (override; we delegate to inner clients' public API)  #
    # ------------------------------------------------------------------ #

    def chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
        tools: Optional[list] = None,
        audio: Optional[list] = None,
        schema: Optional[type] = None,
    ) -> Union[str, Any, Iterator[StreamChunk]]:
        if stream:
            return self._chat_streamed(user_message, generate_kwargs, use_tools, images, include, tools, audio)
        errors: list[tuple[BaseModelClient, BaseException]] = []
        for client in self.clients:
            self._load_state(client)
            try:
                result = client.chat(
                    user_message,
                    generate_kwargs,
                    use_tools=use_tools,
                    stream=False,
                    images=images,
                    include=include,
                    tools=tools,
                    audio=audio,
                    schema=schema,
                )
            except self.retry_on as exc:
                logger.warning("Fallback: client %r failed (%s); trying next.", _label(client), exc)
                errors.append((client, exc))
                continue
            self._adopt_state(client)
            return result
        raise self._exhausted(errors)

    def _chat_streamed(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]],
        use_tools: bool,
        images: Optional[list],
        include: Optional[Iterable[Union[str, StreamingContentType]]],
        tools: Optional[list],
        audio: Optional[list],
    ) -> Iterator[StreamChunk]:
        errors: list[tuple[BaseModelClient, BaseException]] = []
        for client in self.clients:
            self._load_state(client)
            stream = client.chat(
                user_message,
                generate_kwargs,
                use_tools=use_tools,
                stream=True,
                images=images,
                include=include,
                tools=tools,
                audio=audio,
            )
            iterator = iter(stream)
            try:
                first = next(iterator)
            except StopIteration:
                self._adopt_state(client)  # empty but successful stream
                return
            except self.retry_on as exc:
                logger.warning("Fallback: client %r failed before first chunk (%s); trying next.", _label(client), exc)
                errors.append((client, exc))
                continue
            # Committed to this client; a later error propagates (can't replay emitted output).
            yield first
            yield from iterator
            self._adopt_state(client)
            return
        raise self._exhausted(errors)

    def generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
        audio: Optional[list] = None,
        schema: Optional[type] = None,
    ) -> Union[str, Any, Iterator[StreamChunk]]:
        if stream:
            return self._generate_streamed(prompt, generate_kwargs, images, include, audio)
        errors: list[tuple[BaseModelClient, BaseException]] = []
        for client in self.clients:
            client.last_thinking = ""
            client.last_usage = None
            try:
                result = client.generate(
                    prompt, generate_kwargs, stream=False, images=images, include=include, audio=audio, schema=schema
                )
            except self.retry_on as exc:
                logger.warning("Fallback: client %r failed on generate (%s); trying next.", _label(client), exc)
                errors.append((client, exc))
                continue
            self.last_thinking = getattr(client, "last_thinking", "")
            self.last_usage = getattr(client, "last_usage", None)
            return result
        raise self._exhausted(errors)

    def _generate_streamed(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]],
        images: Optional[list],
        include: Optional[Iterable[Union[str, StreamingContentType]]],
        audio: Optional[list],
    ) -> Iterator[StreamChunk]:
        errors: list[tuple[BaseModelClient, BaseException]] = []
        for client in self.clients:
            stream = client.generate(prompt, generate_kwargs, stream=True, images=images, include=include, audio=audio)
            iterator = iter(stream)
            try:
                first = next(iterator)
            except StopIteration:
                return
            except self.retry_on as exc:
                errors.append((client, exc))
                continue
            yield first
            yield from iterator
            return
        raise self._exhausted(errors)

    # ------------------------------------------------------------------ #
    # Abstract-method stubs: the public methods above are overridden, so   #
    # the base never routes through these.                                 #
    # ------------------------------------------------------------------ #

    def _chat(self, *args, **kwargs):  # pragma: no cover - not reachable
        raise NotImplementedError("FallbackClient delegates via chat(); _chat is unused.")

    def _generate(self, *args, **kwargs):  # pragma: no cover - not reachable
        raise NotImplementedError("FallbackClient delegates via generate(); _generate is unused.")

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict] = None) -> dict:  # pragma: no cover
        return generate_kwargs or {}


def _label(client: object) -> str:
    """A short, safe label for logging which client failed."""
    model = getattr(client, "model", None)
    name = getattr(model, "name", None) or getattr(model, "value", None)
    return name or type(client).__name__
