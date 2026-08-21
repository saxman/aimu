"""Async surface for AIMU.

Mirrors the sync public surface one-for-one. The sync ladder (``aimu.chat()`` →
``aimu.client()`` → ``Agent`` → workflows) stays untouched; this submodule is opt-in.

Quick start::

    import asyncio
    from aimu import aio

    async def main():
        client = aio.client("anthropic:claude-sonnet-4-6")
        agent = aio.Agent(client, tools=[my_async_tool])
        reply = await agent.run("Hello")
        async for chunk in agent.run("Stream this", stream=True):
            print(chunk.content, end="")

    asyncio.run(main())

**Streaming types differ between surfaces.** ``aimu.chat(stream=True)`` returns
``Iterator[StreamChunk]``; ``aio.chat(stream=True)`` returns
``AsyncIterator[StreamChunk]``. They cannot unify without a hidden event loop;
this asymmetry is by design.

**Per-call timeouts** use ``asyncio.timeout()``:

    async with asyncio.timeout(30):
        result = await agent.run("Long task")

**In-process providers (HuggingFace, LlamaCpp).** These load model weights into
memory; constructing both a sync and async client for the same model would load
weights twice. Instead, build a sync client first and pass it to ``aio.client()``::

    sync_client = aimu.client(HuggingFaceModel.LLAMA_70B)   # loads weights
    async_client = aio.client(sync_client)                  # wraps; shares weights

**Structured concurrency.** ``aio.Parallel`` and ``concurrent_tool_calls=True``
use ``asyncio.TaskGroup``: if one worker raises, siblings are cancelled and an
``ExceptionGroup`` surfaces with all failures.
"""

from ._model_client import AsyncModelClient, client, chat
from aimu.models import ContextOverflowError, ModelConnectionError
from .fallback import AsyncFallbackClient
from aimu.agents import DegenerateTurnError, TruncatedTurnError
from .agent import Agent, AsyncRunner
from .channels import CLIChannel, Channel, ChannelMessage, WebChannel
from .run_handle import RunHandle
from .scheduler import Scheduler
from .audio import AsyncAudioClient, audio_client, generate_audio
from .embedding import AsyncEmbeddingClient, embed, embedding_client
from .image import AsyncImageClient, generate_image, image_client
from .speech import AsyncSpeechClient, speech_client, generate_speech
from .transcription import AsyncTranscriptionClient, transcription_client, transcribe
from .skill_agent import SkillAgent
from .orchestrator_agent import OrchestratorAgent
from .workflows.chain import Chain
from .workflows.router import Router
from .workflows.parallel import Parallel
from .workflows.evaluator import EvaluatorOptimizer
from .workflows.plan_execute_evaluator import PlanExecuteEvaluator

from importlib import import_module as _import_module

from aimu.models._internal.factory import installed as _installed

# symbol name -> (module, third-party dependency probed for availability)
_LAZY_ASYNC_SYMBOLS: dict[str, tuple[str, str]] = {
    "AsyncHuggingFaceAudioClient": ("aimu.aio.providers.hf.audio", "soundfile"),
    "AsyncHuggingFaceImageClient": ("aimu.aio.providers.hf.image", "diffusers"),
    "AsyncGeminiImageClient": ("aimu.aio.providers.gemini.image", "google.genai"),
    "AsyncHuggingFaceSpeechClient": ("aimu.aio.providers.hf.speech", "soundfile"),
    "AsyncOpenAISpeechClient": ("aimu.aio.providers.openai.speech", "openai"),
    "AsyncHuggingFaceTranscriptionClient": ("aimu.aio.providers.hf.transcription", "transformers"),
    "AsyncOpenAITranscriptionClient": ("aimu.aio.providers.openai_transcription", "openai"),
}

# MCPClient lives in ._mcp_client, which imports fastmcp (and its own dependency tree --
# mcp, jsonschema, ...) at module level. fastmcp is a *required* (not optional) dependency,
# so this isn't the installed()-gated pattern above -- it's the same plain lazy import
# aimu.tools.__init__ already uses for the sync MCPClient, deferring fastmcp's real cost
# until a caller actually touches MCP.
_LAZY_REQUIRED_SYMBOLS = frozenset({"MCPClient"})


def __getattr__(name: str):
    """Resolve MCPClient and async provider clients on first access (PEP 562).

    Absent dependency yields None, matching the contract the deleted except-ImportError
    blocks established; an installed-but-broken dependency raises with the cause chained.
    """
    if name in _LAZY_REQUIRED_SYMBOLS:
        return getattr(_import_module("._mcp_client", __name__), name)
    entry = _LAZY_ASYNC_SYMBOLS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, requires = entry
    if not _installed(requires):
        return None
    try:
        return getattr(_import_module(module_name), name)
    except ImportError as exc:
        raise ImportError(
            f"{name} could not be loaded from {module_name!r} ({requires!r} is installed): {exc}"
        ) from exc


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_ASYNC_SYMBOLS, *_LAZY_REQUIRED_SYMBOLS})


__all__ = [
    "Agent",
    "AsyncAudioClient",
    "AsyncEmbeddingClient",
    "AsyncFallbackClient",
    "AsyncGeminiImageClient",
    "AsyncHuggingFaceAudioClient",
    "AsyncHuggingFaceImageClient",
    "AsyncHuggingFaceSpeechClient",
    "AsyncImageClient",
    "AsyncModelClient",
    "AsyncHuggingFaceTranscriptionClient",
    "AsyncOpenAISpeechClient",
    "AsyncOpenAITranscriptionClient",
    "AsyncRunner",
    "AsyncSpeechClient",
    "AsyncTranscriptionClient",
    "CLIChannel",
    "Chain",
    "Channel",
    "ChannelMessage",
    "DegenerateTurnError",
    "TruncatedTurnError",
    "WebChannel",
    "EvaluatorOptimizer",
    "MCPClient",
    "ModelConnectionError",
    "ContextOverflowError",
    "OrchestratorAgent",
    "Parallel",
    "PlanExecuteEvaluator",
    "RunHandle",
    "Router",
    "Scheduler",
    "SkillAgent",
    "audio_client",
    "chat",
    "client",
    "embed",
    "embedding_client",
    "generate_audio",
    "generate_image",
    "generate_speech",
    "image_client",
    "speech_client",
    "transcribe",
    "transcription_client",
    "HAS_A2A",
]

# Optional async A2A interop (requires the `a2a` extra).
from .a2a import HAS_A2A  # noqa: E402

if HAS_A2A:
    from .a2a import A2AConnectionError, RemoteAgent, build_a2a_app, serve_a2a  # noqa: E402

    __all__ += ["RemoteAgent", "A2AConnectionError", "serve_a2a", "build_a2a_app"]
