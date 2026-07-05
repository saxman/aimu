"""Async test helpers: mock async model client and live-backend dispatch.

Mirrors :class:`helpers.MockModelClient` and the ``create_real_*`` helpers but
for the async surface (``aimu.aio``).
"""

from __future__ import annotations

import time
from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest

from aimu.aio._base import AsyncBaseModelClient
from aimu.models import (
    HAS_ANTHROPIC,
    HAS_LLAMACPP,
    HAS_OPENAI_COMPAT,
    HuggingFaceClient,
    LlamaCppClient,
    OllamaClient,
    StreamChunk,
    StreamingContentType,
)

if HAS_LLAMACPP:
    from aimu.models import LlamaCppModel
if HAS_ANTHROPIC:
    from aimu.models import AnthropicClient
if HAS_OPENAI_COMPAT:
    from aimu.models import (
        GeminiClient,
        HFOpenAIClient,
        LMStudioOpenAIClient,
        OllamaOpenAIClient,
        OpenAIClient,
        VLLMOpenAIClient,
    )
else:
    LMStudioOpenAIClient = None
    OllamaOpenAIClient = None
    HFOpenAIClient = None
    VLLMOpenAIClient = None


class MockAsyncModelClient(AsyncBaseModelClient):
    """An async ModelClient stub whose chat() responses come from a fixed queue.

    ``responses`` entries follow the same convention as :class:`helpers.MockModelClient`:
    a plain ``str`` is a direct response; ``"tool"`` simulates one tool-call round
    by appending the relevant messages and consuming a follow-up response.
    """

    def __init__(self, responses: list):
        self.model = MagicMock()
        self.model.supports_tools = True
        self.model.supports_thinking = False
        self.model.supports_vision = False
        self.model.supports_audio = False
        self.model_kwargs = None
        self._system_message = None
        self.default_generate_kwargs = {}
        self.messages = []
        self.tools = []
        self.last_thinking = ""
        self._responses = list(responses)
        self._call_count = 0

    def _update_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}

    async def _chat(
        self, user_message=None, generate_kwargs=None, use_tools=True, stream=False, images=None, audio=None
    ):
        if stream:
            return self._chat_streamed(user_message, generate_kwargs, use_tools, images=images)
        # user_message=None → continuation: a turn on the current messages, append nothing.
        if user_message is not None:
            if audio:
                from aimu.models._internal.audio_input import _build_audio_content_blocks

                self.messages.append({"role": "user", "content": _build_audio_content_blocks(user_message, audio)})
            elif images:
                from aimu.models._internal.image_input import _build_user_content_blocks

                self.messages.append({"role": "user", "content": _build_user_content_blocks(user_message, images)})
            else:
                self.messages.append({"role": "user", "content": user_message})
        response = self._responses[self._call_count]
        self._call_count += 1

        # Single turn: record the model requesting a tool call and return. The Agent's tool-loop
        # engine executes it (appends the tool result); the client does not. Targets the first
        # advertised tool (set by the per-call ``tools=`` override), or a generic name if none.
        if response == "tool":
            name = getattr(self.tools[0], "__name__", "mock_tool") if self.tools else "mock_tool"
            self.messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"type": "function", "function": {"name": name, "arguments": {}}, "id": "x"}],
                }
            )
            return ""
        self.messages.append({"role": "assistant", "content": response})
        return response

    async def _chat_streamed(
        self, user_message=None, generate_kwargs=None, use_tools=True, images=None
    ) -> AsyncIterator[StreamChunk]:
        response = await self._chat(user_message, generate_kwargs, use_tools, images=images)
        yield StreamChunk(StreamingContentType.GENERATING, response)

    async def _generate(self, prompt, generate_kwargs=None, stream=False, images=None, audio=None):
        if stream:
            return self._generate_streamed(prompt, generate_kwargs)
        return await self._chat(prompt, generate_kwargs, images=images)

    async def _generate_streamed(self, prompt, generate_kwargs=None) -> AsyncIterator[StreamChunk]:
        text = await self._generate(prompt, generate_kwargs)
        yield StreamChunk(StreamingContentType.GENERATING, text)


# ---------------------------------------------------------------------------
# Live-backend dispatch (mirrors helpers.create_real_model_client / resolve_model_params)
# ---------------------------------------------------------------------------


def _resolve_async_client_for_type(client_type: str):
    """Return the *sync* client class for a given --client value.

    Async tests parametrize over model enum members (same enums as sync); the
    fixture below constructs the matching async client by inspecting the enum type.
    """
    if client_type == "ollama":
        return OllamaClient
    if client_type == "anthropic":
        if not HAS_ANTHROPIC:
            pytest.skip("anthropic package not installed")
        return AnthropicClient
    if client_type == "openai":
        if not HAS_OPENAI_COMPAT:
            pytest.skip("openai package not installed")
        return OpenAIClient
    if client_type == "gemini":
        if not HAS_OPENAI_COMPAT:
            pytest.skip("openai package not installed")
        return GeminiClient
    if client_type == "lmstudio_openai":
        return LMStudioOpenAIClient
    if client_type == "ollama_openai":
        return OllamaOpenAIClient
    if client_type == "hf_openai":
        return HFOpenAIClient
    if client_type == "vllm_openai":
        return VLLMOpenAIClient
    if client_type in ("hf", "huggingface"):
        return HuggingFaceClient
    if client_type == "llamacpp":
        if not HAS_LLAMACPP:
            pytest.skip("llama-cpp-python not installed")
        return LlamaCppClient
    return OllamaClient  # default


def resolve_async_model_params(config, default_params=None) -> list:
    """Pick the model enums for parametrizing the ``async_model_client`` fixture.

    Mirrors :func:`helpers.resolve_model_params` but defaults to async-supported
    providers only when ``--client=all``. For now, ``all`` means Ollama (since
    it's the easiest to validate locally); other providers can be added when
    we want to broaden the live-test footprint. Omitting ``--client`` skips live
    files and leaves mock-default files on their mock params.
    """
    client_option = config.getoption("--client", None)
    model = config.getoption("--model", "all")

    if client_option is None:
        return default_params if default_params is not None else []

    client_type = client_option.lower()

    if client_type == "all":
        if default_params is not None:
            return default_params
        params = list(OllamaClient.MODELS)
        if HAS_ANTHROPIC:
            params += list(AnthropicClient.MODELS)
        return params

    client = _resolve_async_client_for_type(client_type)
    if model == "all":
        return list(client.MODELS)
    return [client.MODELS[model]]


def create_real_async_model_client(request):
    """Dispatch ``request.param`` (a sync model enum value) to the matching async client.

    Mirrors :func:`helpers.create_real_model_client`. For in-process providers
    (HF, LlamaCpp) the async client is built by wrapping a fresh sync client
    (Decision 7; see docs/explanation/async-design.md).
    """
    from aimu.models.providers.ollama import OllamaModel

    model = request.param

    if model in OllamaClient.MODELS:
        time.sleep(2)  # let Ollama free memory from a prior model
        from aimu.aio.providers.ollama import AsyncOllamaClient

        # AsyncOllamaClient doesn't pre-pull (no sync `ollama.pull`); ensure the model
        # is present locally via the sync `ollama.pull` first if needed.
        import ollama

        try:
            ollama.pull(model.value)
        except ollama.ResponseError as exc:
            # Model can't be pulled in this environment (e.g. a 412 when the local
            # Ollama is too old for the model's manifest format). Skip rather than error.
            pytest.skip(f"Could not pull {model.value}: {exc}")
        client = AsyncOllamaClient(model, system_message="You are a helpful assistant.", model_keep_alive_seconds=2)
        # Force model.supports_vision attr access through enum so downstream tests
        # using `model_client.model.supports_*` work uniformly.
        assert isinstance(model, OllamaModel)
    elif HAS_ANTHROPIC and model in AnthropicClient.MODELS:
        from aimu.aio.providers.anthropic import AsyncAnthropicClient

        client = AsyncAnthropicClient(model, system_message="You are a helpful assistant.")
    elif HAS_OPENAI_COMPAT and model in OpenAIClient.MODELS:
        from aimu.aio.providers.openai_compat import AsyncOpenAIClient

        client = AsyncOpenAIClient(model, system_message="You are a helpful assistant.")
    elif HAS_OPENAI_COMPAT and model in GeminiClient.MODELS:
        from aimu.aio.providers.openai_compat import AsyncGeminiClient

        client = AsyncGeminiClient(model, system_message="You are a helpful assistant.")
    elif model in LMStudioOpenAIClient.MODELS:
        from aimu.aio.providers.openai_compat import AsyncLMStudioOpenAIClient

        client = AsyncLMStudioOpenAIClient(model, system_message="You are a helpful assistant.")
    elif model in OllamaOpenAIClient.MODELS:
        from aimu.aio.providers.openai_compat import AsyncOllamaOpenAIClient

        client = AsyncOllamaOpenAIClient(model, system_message="You are a helpful assistant.")
    elif model in HFOpenAIClient.MODELS:
        from aimu.aio.providers.openai_compat import AsyncHFOpenAIClient

        client = AsyncHFOpenAIClient(model, system_message="You are a helpful assistant.")
    elif model in VLLMOpenAIClient.MODELS:
        from aimu.aio.providers.openai_compat import AsyncVLLMOpenAIClient

        client = AsyncVLLMOpenAIClient(model, system_message="You are a helpful assistant.")
    elif model in HuggingFaceClient.MODELS:
        from aimu.aio.providers.hf.text import AsyncHuggingFaceClient

        sync_client = HuggingFaceClient(model, system_message="You are a helpful assistant.")
        client = AsyncHuggingFaceClient(sync_client)
    elif HAS_LLAMACPP and model in LlamaCppModel:
        from aimu.aio.providers.llamacpp import AsyncLlamaCppClient

        model_path = request.config.getoption("--model-path")
        if not model_path:
            pytest.skip("--model-path is required for llamacpp client tests")
        sync_client = LlamaCppClient(model, model_path=model_path, system_message="You are a helpful assistant.")
        client = AsyncLlamaCppClient(sync_client)
    else:
        raise ValueError(f"Unknown model: {model}")

    return client
