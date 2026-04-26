from __future__ import annotations

from typing import Any, Iterator, Optional, Union

from .base import BaseModelClient, Model, StreamChunk

# --- Optional provider imports ---

try:
    from .ollama import OllamaClient
    from .ollama.ollama_client import OllamaModel

    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False
    OllamaClient = None  # type: ignore[assignment,misc]
    OllamaModel = None  # type: ignore[assignment,misc]

try:
    from .hf import HuggingFaceClient
    from .hf.hf_client import HuggingFaceModel

    _HAS_HF = True
except ImportError:
    _HAS_HF = False
    HuggingFaceClient = None  # type: ignore[assignment,misc]
    HuggingFaceModel = None  # type: ignore[assignment,misc]

try:
    from .anthropic import AnthropicClient, AnthropicModel

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False
    AnthropicClient = None  # type: ignore[assignment,misc]
    AnthropicModel = None  # type: ignore[assignment,misc]

try:
    from .openai_compat import (
        OpenAIClient,
        OpenAIModel,
        GeminiClient,
        GeminiModel,
        LMStudioOpenAIClient,
        LMStudioOpenAIModel,
        OllamaOpenAIClient,
        OllamaOpenAIModel,
        HFOpenAIClient,
        HFOpenAIModel,
        VLLMOpenAIClient,
        VLLMOpenAIModel,
        LlamaServerOpenAIClient,
        LlamaServerOpenAIModel,
        SGLangOpenAIClient,
        SGLangOpenAIModel,
    )

    _HAS_OPENAI_COMPAT = True
except ImportError:
    _HAS_OPENAI_COMPAT = False
    OpenAIClient = GeminiClient = LMStudioOpenAIClient = OllamaOpenAIClient = None  # type: ignore[assignment,misc]
    HFOpenAIClient = VLLMOpenAIClient = LlamaServerOpenAIClient = SGLangOpenAIClient = None  # type: ignore[assignment,misc]
    OpenAIModel = GeminiModel = LMStudioOpenAIModel = OllamaOpenAIModel = None  # type: ignore[assignment,misc]
    HFOpenAIModel = VLLMOpenAIModel = LlamaServerOpenAIModel = SGLangOpenAIModel = None  # type: ignore[assignment,misc]

try:
    from .llamacpp import LlamaCppClient, LlamaCppModel

    _HAS_LLAMACPP = True
except ImportError:
    _HAS_LLAMACPP = False
    LlamaCppClient = None  # type: ignore[assignment,misc]
    LlamaCppModel = None  # type: ignore[assignment,misc]


class ModelClient(BaseModelClient):
    """
    Factory wrapper that instantiates the appropriate concrete client based on the
    model enum type passed in, then delegates all method calls to it.

    Usage::

        from aimu.models import ModelClient
        from aimu.models.ollama.ollama_client import OllamaModel

        client = ModelClient(OllamaModel.QWEN_3_8B)
        client.chat("Hello")

        # Provider-specific kwargs are passed through:
        client = ModelClient(LlamaCppModel.QWEN_3_8B, model_path="/path/to/model.gguf")
        client = ModelClient(OllamaModel.LLAMA_3_1_8B, model_keep_alive_seconds=120)
        client = ModelClient(LMStudioOpenAIModel.LLAMA_3_2_3B, base_url="http://myserver:1234/v1")
    """

    def __init__(self, model: Model, **kwargs: Any) -> None:
        # Dispatch to concrete client by model enum type.
        # isinstance checks are guarded by _HAS_* flags (short-circuit and) so
        # that a None model class from a missing optional dep never reaches isinstance().
        if _HAS_OLLAMA and isinstance(model, OllamaModel):
            self._client: BaseModelClient = OllamaClient(model, **kwargs)
        elif _HAS_HF and isinstance(model, HuggingFaceModel):
            self._client = HuggingFaceClient(model, **kwargs)
        elif _HAS_ANTHROPIC and isinstance(model, AnthropicModel):
            self._client = AnthropicClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, OpenAIModel):
            self._client = OpenAIClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, GeminiModel):
            self._client = GeminiClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, LMStudioOpenAIModel):
            self._client = LMStudioOpenAIClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, OllamaOpenAIModel):
            self._client = OllamaOpenAIClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, HFOpenAIModel):
            self._client = HFOpenAIClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, VLLMOpenAIModel):
            self._client = VLLMOpenAIClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, LlamaServerOpenAIModel):
            self._client = LlamaServerOpenAIClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, SGLangOpenAIModel):
            self._client = SGLangOpenAIClient(model, **kwargs)
        elif _HAS_LLAMACPP and isinstance(model, LlamaCppModel):
            self._client = LlamaCppClient(model, **kwargs)
        else:
            raise ValueError(
                f"No available client for model type {type(model).__name__!r}. "
                "Ensure the required optional dependency is installed (e.g. pip install aimu[ollama])."
            )

        # Mirror non-mutable attributes so callers can read them directly on this wrapper.
        # super().__init__() is intentionally not called — it would reset inner client state.
        self.model = self._client.model
        self.model_kwargs = self._client.model_kwargs
        self.default_generate_kwargs = self._client.default_generate_kwargs

    # --- Delegate mutable state to inner client so both stay in sync ---

    @property
    def messages(self) -> list[dict]:
        return self._client.messages

    @messages.setter
    def messages(self, value: list[dict]) -> None:
        self._client.messages = value

    @property
    def mcp_client(self) -> Any:
        return self._client.mcp_client

    @mcp_client.setter
    def mcp_client(self, value: Any) -> None:
        self._client.mcp_client = value

    @property
    def system_message(self) -> Optional[str]:
        return self._client.system_message

    @system_message.setter
    def system_message(self, message: str) -> None:
        self._client.system_message = message

    @property
    def last_thinking(self) -> str:
        return self._client.last_thinking

    @last_thinking.setter
    def last_thinking(self, value: str) -> None:
        self._client.last_thinking = value

    # --- Implement abstract methods by delegation ---

    def generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        include_thinking: bool = True,
    ) -> Union[str, Iterator[StreamChunk]]:
        return self._client.generate(prompt, generate_kwargs, stream=stream, include_thinking=include_thinking)

    def chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
    ) -> Union[str, Iterator[StreamChunk]]:
        return self._client.chat(user_message, generate_kwargs, use_tools=use_tools, stream=stream)

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        return self._client._update_generate_kwargs(generate_kwargs)
