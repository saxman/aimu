"""Async equivalents of :mod:`aimu.tools.builtin`.

Re-exports every sync tool from the sync module (the async client dispatches them
through :func:`asyncio.to_thread`) and provides an async-native ``generate_image``
streaming tool that yields :class:`~aimu.models.StreamChunk` progress chunks during
generation.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from aimu.tools.builtin import (  # noqa: F401 (re-exports)
    DEFAULT_SUBAGENT_SYSTEM_MESSAGE,
    calculate,
    compute,
    echo,
    fs,
    get_current_date_and_time,
    get_weather,
    get_webpage,
    list_directory,
    make_document_tools,
    make_memory_tools,
    make_retrieval_tool,
    misc,
    read_file,
    web_search,
    web,
    wikipedia,
)
from aimu.tools.builtin import _subagent_docstring, _validate_subagent_config
from aimu.tools.decorator import tool

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
    be built from an enum on the aio surface, so a *fresh* sync client is wrapped per spawn — fresh
    preserves message isolation, and the process weight cache prevents a reload.
    """
    from aimu.aio._model_client import AsyncModelClient
    from aimu.models.model_client import resolve_model_string

    resolved = resolve_model_string(model) if isinstance(model, str) else model
    if _is_in_process_model(resolved):
        import aimu

        return AsyncModelClient(aimu.client(resolved))
    return AsyncModelClient(resolved)


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
) -> Callable:
    """Async twin of :func:`aimu.tools.builtin.make_subagent_tool`.

    Produces an ``async def spawn_subagent`` tool (``__tool_is_async__=True``) that builds a fresh,
    isolated :class:`aimu.aio.Agent` per call and awaits its ``run``. Parallelism is free: give the
    parent :class:`aimu.aio.Agent` ``concurrent_tool_calls=True`` and multiple spawn calls in one turn
    overlap under an ``asyncio.TaskGroup``. See the sync docstring for the full contract (generic vs
    typed mode, ``max_depth`` recursion guard, unknown-``agent_type`` handling).

    In-process providers (HuggingFace, LlamaCpp) are wrapped per spawn via a fresh sync client (the aio
    surface can't construct them from an enum); the process weight cache prevents reloading weights.
    """
    from aimu.models.base import BaseModelClient

    _validate_subagent_config(max_depth, agent_types)
    default_model = model.model if isinstance(model, BaseModelClient) else model

    def _build_agent(sys_msg: str, agent_tools: Optional[list[Callable]], name: str, model_override=None):
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
                )
            )
        return Agent(
            _fresh_async_subagent_client(m),
            system_message=sys_msg,
            name=name,
            tools=child_tools,
            max_iterations=max_iterations,
            concurrent_tool_calls=concurrent_tool_calls,
            deps=deps,
            tool_approval=tool_approval,
        )

    if agent_types is None:

        async def spawn_subagent(task: str) -> str:
            return await _build_agent(system_message, tools, name="subagent").run(task)

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
            )
            return await agent.run(task)

    spawn_subagent.__name__ = tool_name
    spawn_subagent.__qualname__ = tool_name
    spawn_subagent.__doc__ = _subagent_docstring(agent_types)
    return tool(spawn_subagent)
