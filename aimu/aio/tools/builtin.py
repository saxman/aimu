"""Async equivalents of :mod:`aimu.tools.builtin`.

Re-exports every sync tool from the sync module (the async client dispatches them
through :func:`asyncio.to_thread`) and provides an async-native ``generate_image``
streaming tool that yields :class:`~aimu.models.StreamChunk` progress chunks during
generation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Protocol
from uuid import uuid4

from aimu.models import StreamChunk, StreamingContentType
from aimu.tools.builtin import (  # noqa: F401 (re-exports)
    DEFAULT_SUBAGENT_SYSTEM_MESSAGE,
    calculate,
    compute,
    convert_time,
    echo,
    fs,
    get_current_date_and_time,
    get_weather,
    get_webpage,
    get_webpage_html,
    list_directory,
    make_document_tools,
    make_memory_tools,
    make_retrieval_tool,
    make_web_tools,
    misc,
    read_file,
    time,
    web_search,
    web,
    wikipedia,
)
from aimu.tools.builtin import _subagent_docstring, _validate_subagent_config
from aimu.tools.decorator import tool

logger = logging.getLogger(__name__)


class SubagentObserver(Protocol):
    """Display hook for one sub-agent spawn, so a front end can show its work as it happens.

    Passing an observer to :func:`make_async_subagent_tool` switches the spawn to a streamed child
    run: every chunk is forwarded here while the tool's return value stays the child's final answer.
    The spawn tool itself remains a plain (non-streaming) tool, so concurrent spawns still overlap
    under the parent's ``concurrent_tool_calls``. Callbacks are display-only; an exception raised by
    one is logged and swallowed rather than failing the spawn.

    Attaching an observer is therefore not purely additive: an observed spawn issues its model calls
    through the provider's *streaming* request path, where an unobserved one uses the non-streaming
    path. The answer is the same either way, but any behavior that differs between a provider's two
    request paths applies to observed spawns.
    """

    async def spawned(self, spawn_id: str, agent_type: Optional[str], task: str) -> None:
        """A sub-agent has been built for ``task``. ``agent_type`` is None in generic (untyped) mode."""

    async def chunk(self, spawn_id: str, chunk: StreamChunk) -> None:
        """One chunk from the child's streamed run."""

    async def finished(self, spawn_id: str, result: str, error: Optional[BaseException]) -> None:
        """The spawn ended. ``result`` is the final (or partial, on failure) generated text, and
        ``error`` is the exception that ended it, including a ``CancelledError``."""


async def _notify_observer(observer: SubagentObserver, name: str, *args) -> None:
    """Call one named callback on a display-only observer, logging rather than propagating its failure.

    The observer and the method name are passed separately, rather than an already-resolved callable,
    so that both failure modes stay inside this guard instead of at the call site, where nothing would
    catch them and a display hook would break the spawn: a partial observer (the Protocol is satisfied
    structurally, so implementing only some callbacks is a realistic input) missing ``name`` entirely,
    and a callback with the wrong signature (the seam-drift case: an observer written against an older
    parameter list).
    """
    callback = getattr(observer, name, None)
    if callback is None:
        logger.warning("A sub-agent observer has no %r callback; skipping.", name)
        return
    try:
        await callback(*args)
    except Exception:
        logger.warning("A sub-agent observer callback failed; continuing the spawn.", exc_info=True)


async def _run_observed(agent, agent_type: Optional[str], task: str, observer: SubagentObserver) -> str:
    """Run one spawn streamed, reporting it to ``observer``, and return the child's final answer.

    ``parts`` is cleared whenever the child's loop advances an iteration, so the return value is the
    final answer rather than every intermediate tools-only response concatenated -- which is what
    keeps this path's result identical to the non-streamed one.
    """
    spawn_id = f"{agent_type or 'subagent'}-{uuid4().hex[:8]}"
    await _notify_observer(observer, "spawned", spawn_id, agent_type, task)
    parts: list[str] = []
    iteration = 0
    error: Optional[BaseException] = None
    try:
        async for chunk in await agent.run(task, stream=True):
            if chunk.iteration > iteration:
                iteration = chunk.iteration
                parts.clear()
            if chunk.phase == StreamingContentType.GENERATING and isinstance(chunk.content, str):
                parts.append(chunk.content)
            await _notify_observer(observer, "chunk", spawn_id, chunk)
        return "".join(parts)
    except BaseException as exc:  # including CancelledError: the observer must hear about it
        error = exc
        raise
    finally:
        await _notify_observer(observer, "finished", spawn_id, "".join(parts), error)


_async_image_client = None


def _get_async_image_client():
    """Lazy singleton :class:`AsyncImageClient` for the async built-in tool.

    Reads ``AIMU_IMAGE_MODEL`` from the environment. Raises ``ValueError`` if it is
    unset (no model is downloaded implicitly). Accepts any model string supported by
    :func:`aimu.aio.image_client`: ``"hf:..."`` / ``"gemini:..."``.
    """
    global _async_image_client
    if _async_image_client is None:
        import aimu
        from aimu.aio import image_client as _aio_image_client

        _async_image_client = _aio_image_client(aimu.image_client())  # resolves AIMU_IMAGE_MODEL or raises
    return _async_image_client


@tool
async def generate_image(prompt: str):
    """Generate an image from a text prompt and return the saved file path.

    **Streaming async tool**: async generator yielding
    :attr:`~aimu.models.StreamingContentType.IMAGE_GENERATING` chunks during
    denoising. When dispatched by ``aio.Agent.run(stream=True)``, chunks flow
    through the agent's own stream and into the UI live.

    Uses an :class:`aimu.aio.AsyncImageClient`. The model is controlled by
    ``AIMU_IMAGE_MODEL`` (required; the tool raises if it is unset). Use
    :func:`make_async_image_tool` to override the client or opt into
    ``preview_every=N`` intermediate previews.

    Args:
        prompt: A description of the desired image.
    """
    client = _get_async_image_client()
    final_result: Optional[str] = None
    async for chunk in await client.generate(prompt, format="path", stream=True):
        yield chunk
        content = chunk.content
        if isinstance(content, dict) and content.get("final"):
            final_result = content.get("result")
    # Final chunk's content["result"] is picked up by the tool-loop engine's
    # _dispatch_streamed as the canonical tool response; no return-value needed
    # (PEP 525 async generators don't carry return values anyway).
    del final_result


def make_async_image_tool(client, *, preview_every: Optional[int] = None):
    """Build an async streaming ``generate_image`` tool bound to a specific client.

    Pass a sync :class:`aimu.BaseImageClient` (e.g.
    :class:`HuggingFaceImageClient`, :class:`GeminiImageClient`), which will be
    wrapped automatically, or an existing :class:`aimu.aio.AsyncImageClient`.
    ``preview_every=N`` opts into intermediate denoised-image previews (HF only;
    Gemini ignores it).
    """
    from aimu.aio import image_client as _aio_image_client
    from aimu.aio.image import AsyncImageClient
    from aimu.models.base import BaseImageClient

    if isinstance(client, BaseImageClient):
        client = _aio_image_client(client)
    elif not isinstance(client, AsyncImageClient):
        # Permit the per-provider async classes (AsyncHuggingFaceImageClient,
        # AsyncGeminiImageClient) directly; they expose .generate() too.
        pass

    @tool
    async def generate_image(prompt: str):
        """Generate an image from a text prompt and return the saved file path.

        Async streaming tool: yields progress chunks during generation.

        Args:
            prompt: A description of the desired image.
        """
        async for chunk in await client.generate(prompt, format="path", stream=True, preview_every=preview_every):
            yield chunk

    return generate_image


image = [generate_image]


def _is_in_process_model(model) -> bool:
    """True for HuggingFace / LlamaCpp enum members (which the aio surface must wrap, not construct)."""
    try:
        from aimu.models.providers.hf.text import HuggingFaceModel

        if isinstance(model, HuggingFaceModel):
            return True
    except ImportError:
        pass
    try:
        from aimu.models.providers.llamacpp import LlamaCppModel

        if isinstance(model, LlamaCppModel):
            return True
    except ImportError:
        pass
    return False


def _fresh_async_subagent_client(model):
    """Build a fresh isolated :class:`AsyncModelClient` for one spawn.

    Cloud/Ollama models are constructed directly. In-process providers (HuggingFace, LlamaCpp) can't
    be built from an enum on the aio surface, so a *fresh* sync client is wrapped per spawn (fresh
    preserves message isolation, and the process weight cache prevents a reload).

    A string model reaches :class:`AsyncModelClient` unresolved, and is resolved here only to answer
    the in-process question. That split is the point: an extended model string carries an
    ``@base_url`` and ``;flags`` that no resolved enum can hold, so resolving *for* the constructor
    would drop the endpoint and run the sub-agent against the provider default while its parent
    talked to the override. The sync twin hands its string to ``ModelClient`` for the same reason.
    """
    from aimu.aio._model_client import AsyncModelClient
    from aimu.models.model_client import resolve_model

    resolved = resolve_model(model).model if isinstance(model, str) else model
    if _is_in_process_model(resolved):
        import aimu

        return AsyncModelClient(aimu.client(resolved))
    return AsyncModelClient(model if isinstance(model, str) else resolved)


def make_async_subagent_tool(
    model,
    *,
    system_message: str = DEFAULT_SUBAGENT_SYSTEM_MESSAGE,
    tools: Optional[list[Callable]] = None,
    agent_types: Optional[dict[str, dict]] = None,
    max_depth: int = 1,
    max_iterations: int = 10,
    concurrent_tool_calls: bool = True,
    deps: Any = None,
    tool_approval: Optional[Callable] = None,
    tool_name: str = "spawn_subagent",
    observer: Optional[SubagentObserver] = None,
) -> Callable:
    """Async twin of :func:`aimu.tools.builtin.make_subagent_tool`.

    Produces an ``async def spawn_subagent`` tool (``__tool_is_async__=True``) that builds a fresh,
    isolated :class:`aimu.aio.Agent` per call and awaits its ``run``. Parallelism is free: give the
    parent :class:`aimu.aio.Agent` ``concurrent_tool_calls=True`` and multiple spawn calls in one turn
    overlap under an ``asyncio.TaskGroup``. See the sync docstring for the full contract (generic vs
    typed mode, the per-spec ``"model"`` / ``"thinking"`` / ``"generate_kwargs"`` keys, ``max_depth``
    recursion guard, unknown-``agent_type`` handling, and the ``tool_approval`` gate forwarded to every
    spawned sub-agent).

    In-process providers (HuggingFace, LlamaCpp) are wrapped per spawn via a fresh sync client (the aio
    surface can't construct them from an enum); the process weight cache prevents reloading weights.

    Passing ``observer`` (a :class:`SubagentObserver`) switches each spawn to a streamed child run and
    reports it as it happens, without making this a streaming tool (which would disable the parent's
    concurrent dispatch). Nested spawns inherit it.
    """
    from aimu.models.base import BaseModelClient

    _validate_subagent_config(max_depth, agent_types)
    default_model = model.model if isinstance(model, BaseModelClient) else model

    def _build_agent(
        sys_msg: str,
        agent_tools: Optional[list[Callable]],
        name: str,
        model_override=None,
        thinking=None,
        generate_kwargs=None,
    ):
        from aimu.aio.agent import Agent

        m = model_override if model_override is not None else default_model
        child_tools = list(agent_tools or [])
        if max_depth > 1:
            child_tools.append(
                make_async_subagent_tool(
                    m,
                    system_message=system_message,
                    tools=tools,
                    agent_types=agent_types,
                    max_depth=max_depth - 1,
                    max_iterations=max_iterations,
                    concurrent_tool_calls=concurrent_tool_calls,
                    deps=deps,
                    tool_approval=tool_approval,
                    tool_name=tool_name,
                    observer=observer,
                )
            )
        client = _fresh_async_subagent_client(m)
        if generate_kwargs:
            # Copied, not aliased: see the sync twin.
            client.default_generate_kwargs = dict(generate_kwargs)
        return Agent(
            client,
            system_message=sys_msg,
            name=name,
            tools=child_tools,
            max_iterations=max_iterations,
            concurrent_tool_calls=concurrent_tool_calls,
            deps=deps,
            tool_approval=tool_approval,
            thinking=thinking,
        )

    if agent_types is None:

        async def spawn_subagent(task: str) -> str:
            agent = _build_agent(system_message, tools, name="subagent")
            if observer is None:
                return await agent.run(task)
            return await _run_observed(agent, None, task, observer)

    else:

        async def spawn_subagent(agent_type: str, task: str) -> str:
            spec = agent_types.get(agent_type)
            if spec is None:
                return (
                    f"Unknown agent_type {agent_type!r}. Available agent_type values: {', '.join(sorted(agent_types))}."
                )
            agent = _build_agent(
                spec["system_message"],
                spec.get("tools", tools),
                name=f"subagent-{agent_type}",
                model_override=spec.get("model"),
                thinking=spec.get("thinking"),
                generate_kwargs=spec.get("generate_kwargs"),
            )
            if observer is None:
                return await agent.run(task)
            return await _run_observed(agent, agent_type, task, observer)

    spawn_subagent.__name__ = tool_name
    spawn_subagent.__qualname__ = tool_name
    spawn_subagent.__doc__ = _subagent_docstring(agent_types)
    return tool(spawn_subagent)
