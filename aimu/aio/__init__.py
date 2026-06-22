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

from ._mcp_client import MCPClient
from ._model_client import AsyncModelClient, client, chat
from .fallback import AsyncFallbackClient
from .agent import Agent, AsyncRunner
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

try:
    from .providers.hf.audio import AsyncHuggingFaceAudioClient
except Exception:
    AsyncHuggingFaceAudioClient = None  # type: ignore[assignment,misc]

try:
    from .providers.hf.image import AsyncHuggingFaceImageClient
except Exception:
    AsyncHuggingFaceImageClient = None  # type: ignore[assignment,misc]

try:
    from .providers.gemini.image import AsyncGeminiImageClient
except Exception:
    AsyncGeminiImageClient = None  # type: ignore[assignment,misc]

try:
    from .providers.hf.speech import AsyncHuggingFaceSpeechClient
except Exception:
    AsyncHuggingFaceSpeechClient = None  # type: ignore[assignment,misc]

try:
    from .providers.openai.speech import AsyncOpenAISpeechClient
except Exception:
    AsyncOpenAISpeechClient = None  # type: ignore[assignment,misc]

try:
    from .providers.hf.transcription import AsyncHuggingFaceTranscriptionClient
except Exception:
    AsyncHuggingFaceTranscriptionClient = None  # type: ignore[assignment,misc]

try:
    from .providers.openai_transcription import AsyncOpenAITranscriptionClient
except Exception:
    AsyncOpenAITranscriptionClient = None  # type: ignore[assignment,misc]

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
    "Chain",
    "EvaluatorOptimizer",
    "MCPClient",
    "OrchestratorAgent",
    "Parallel",
    "PlanExecuteEvaluator",
    "Router",
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
