"""Pins __all__ for the public namespaces.

Not a style check. AIMU promises a small surface, and 1.0 turns that into a compatibility
promise, so a name entering or leaving it should be a deliberate edit to this file rather
than a side effect of some other change.

The assertion is `REQUIRED <= actual <= REQUIRED | CONDITIONAL`, not set equality:
aimu.models builds __all__ from 15 conditional extends keyed on HAS_*, and aimu.aio and
aimu.agents each add four names under HAS_A2A, so the exported set legitimately differs
between a full install and a minimal one. Equality would encode this machine's extras
into the test.
"""

import aimu
import aimu.agents
import aimu.aio
import aimu.models
import aimu.tools
import pytest

# Captured from a fully-installed environment; see the derivation script in the
# v0.19 task-8 brief. REQUIRED = names present regardless of installed extras;
# CONDITIONAL = closed allowlist of names an optional dependency may add.
REQUIRED_AIMU = {
    "Agent",
    "AudioClient",
    "AudioModel",
    "AudioSpec",
    "BaseAudioClient",
    "BaseEmbeddingClient",
    "BaseImageClient",
    "BaseModelClient",
    "BaseSpeechClient",
    "ContextCompacted",
    "EmbeddingClient",
    "EmbeddingModel",
    "EmbeddingSpec",
    "EventSink",
    "FallbackClient",
    "FallbackExhaustedError",
    "GeminiImageClient",
    "GeminiImageModel",
    "GeminiImageSpec",
    "HAS_GEMINI_IMAGE",
    "HAS_HF_AUDIO",
    "HAS_HF_EMBEDDING",
    "HAS_HF_IMAGE",
    "HAS_HF_SPEECH",
    "HAS_OLLAMA_EMBEDDING",
    "HAS_OPENAI_EMBEDDING",
    "HAS_OPENAI_SPEECH",
    "HuggingFaceAudioClient",
    "HuggingFaceAudioModel",
    "HuggingFaceAudioSpec",
    "HuggingFaceImageClient",
    "HuggingFaceImageModel",
    "HuggingFaceImageSpec",
    "HuggingFaceSpeechClient",
    "HuggingFaceSpeechModel",
    "HuggingFaceSpeechSpec",
    "INERT_MESSAGE_KEYS",
    "ImageClient",
    "ImageModel",
    "ImageSpec",
    "Model",
    "ModelClient",
    "ModelSpec",
    "ModelTurnFinished",
    "ModelTurnStarted",
    "OpenAISpeechClient",
    "OpenAISpeechModel",
    "OpenAISpeechSpec",
    "PROVENANCE_CONTINUATION",
    "PROVENANCE_FINAL_ANSWER",
    "PROVENANCE_KEY",
    "PROVENANCE_PROACTIVE",
    "RequestPrepared",
    "RunEvent",
    "RunFinished",
    "RunStarted",
    "SpeechClient",
    "SpeechModel",
    "SpeechSpec",
    "StreamChunk",
    "StreamingContentType",
    "ToolApproval",
    "ToolCalled",
    "ToolContext",
    "ToolDenied",
    "TranscriptionClient",
    "TranscriptionModel",
    "TranscriptionSpec",
    "__version__",
    "agent",
    "aio",
    "approve_all",
    "audio_client",
    "available_audio_models",
    "available_embedding_models",
    "available_image_models",
    "available_speech_clients",
    "available_speech_models",
    "available_text_models",
    "available_transcription_models",
    "chat",
    "clear_hf_cache",
    "clear_llamacpp_cache",
    "client",
    "embed",
    "embedding_client",
    "emit",
    "extract_tool_calls",
    "generate_audio",
    "generate_image",
    "generate_json",
    "generate_speech",
    "image_client",
    "log_events",
    "parse_json_response",
    "pretty_print",
    "resolve_audio_model_string",
    "resolve_default_text_model_enum",
    "resolve_embedding_model_string",
    "resolve_image_model_enum",
    "resolve_image_model_string",
    "resolve_model_enum",
    "resolve_model_string",
    "resolve_speech_model_string",
    "resolve_transcription_model_string",
    "speech_client",
    "strip_inert_keys",
    "tool",
    "transcribe",
    "transcription_client",
}
CONDITIONAL_AIMU = set()

REQUIRED_MODELS = {
    "AudioClient",
    "AudioModel",
    "AudioSpec",
    "BaseAudioClient",
    "BaseEmbeddingClient",
    "BaseImageClient",
    "BaseModelClient",
    "BaseSpeechClient",
    "BaseTranscriptionClient",
    "ContextOverflowError",
    "EmbeddingClient",
    "EmbeddingModel",
    "EmbeddingSpec",
    "FallbackClient",
    "FallbackExhaustedError",
    "GeminiImageSpec",
    "HuggingFaceAudioSpec",
    "HuggingFaceEmbeddingSpec",
    "HuggingFaceImageSpec",
    "HuggingFaceSpeechSpec",
    "HuggingFaceTranscriptionSpec",
    "INERT_MESSAGE_KEYS",
    "ImageClient",
    "ImageModel",
    "ImageSpec",
    "Model",
    "ModelClient",
    "ModelConnectionError",
    "ModelSpec",
    "OllamaEmbeddingSpec",
    "OpenAIEmbeddingSpec",
    "OpenAISpeechSpec",
    "OpenAITranscriptionSpec",
    "PROVENANCE_CONTINUATION",
    "PROVENANCE_FINAL_ANSWER",
    "PROVENANCE_KEY",
    "PROVENANCE_PROACTIVE",
    "SpeechClient",
    "SpeechModel",
    "SpeechSpec",
    "StreamChunk",
    "StreamingContentType",
    "TranscriptionClient",
    "TranscriptionModel",
    "TranscriptionSpec",
    "available_audio_clients",
    "available_audio_models",
    "available_embedding_clients",
    "available_embedding_models",
    "available_image_clients",
    "available_image_models",
    "available_speech_clients",
    "available_speech_models",
    "available_text_clients",
    "available_text_models",
    "available_transcription_models",
    "extract_tool_calls",
    "generate_json",
    "parse_json_response",
    "resolve_audio_model_string",
    "resolve_default_text_model_enum",
    "resolve_embedding_model_string",
    "resolve_image_model_enum",
    "resolve_image_model_string",
    "resolve_model_enum",
    "resolve_model_string",
    "resolve_speech_model_string",
    "resolve_transcription_model_string",
    "strip_inert_keys",
}
CONDITIONAL_MODELS = {
    "AnthropicClient",
    "AnthropicModel",
    "GeminiClient",
    "GeminiImageClient",
    "GeminiImageModel",
    "GeminiModel",
    "HFOpenAIClient",
    "HFOpenAIModel",
    "HuggingFaceAudioClient",
    "HuggingFaceAudioModel",
    "HuggingFaceClient",
    "HuggingFaceEmbeddingClient",
    "HuggingFaceEmbeddingModel",
    "HuggingFaceImageClient",
    "HuggingFaceImageModel",
    "HuggingFaceModel",
    "HuggingFaceSpeechClient",
    "HuggingFaceSpeechModel",
    "HuggingFaceTranscriptionClient",
    "HuggingFaceTranscriptionModel",
    "LMStudioOpenAIClient",
    "LMStudioOpenAIModel",
    "LlamaCppClient",
    "LlamaCppModel",
    "LlamaServerOpenAIClient",
    "LlamaServerOpenAIModel",
    "OMLXOpenAIClient",
    "OMLXOpenAIModel",
    "OllamaClient",
    "OllamaEmbeddingClient",
    "OllamaEmbeddingModel",
    "OllamaModel",
    "OllamaOpenAIClient",
    "OllamaOpenAIModel",
    "OpenAIClient",
    "OpenAICompatClient",
    "OpenAIEmbeddingClient",
    "OpenAIEmbeddingModel",
    "OpenAIModel",
    "OpenAISpeechClient",
    "OpenAISpeechModel",
    "OpenAITranscriptionClient",
    "OpenAITranscriptionModel",
    "SGLangOpenAIClient",
    "SGLangOpenAIModel",
    "ToolCallFormat",
    "VLLMOpenAIClient",
    "VLLMOpenAIModel",
}

REQUIRED_AGENTS = {
    "Agent",
    "Chain",
    "DegenerateTurnError",
    "EvaluatorOptimizer",
    "HAS_A2A",
    "MessageHistory",
    "OrchestratorAgent",
    "Parallel",
    "PlanExecuteEvaluator",
    "Router",
    "Runner",
    "SkillAgent",
    "TruncatedTurnError",
}
CONDITIONAL_AGENTS = {
    "A2AConnectionError",
    "RemoteAgent",
    "build_a2a_app",
    "serve_a2a",
}

REQUIRED_TOOLS = {
    "MCPClient",
    "MCPConnectionError",
    "ToolApproval",
    "ToolArgumentError",
    "ToolContext",
    "ToolSignatureError",
    "approve_all",
    "builtin",
    "coerce_tool_arguments",
    "tool",
}
CONDITIONAL_TOOLS = set()

REQUIRED_AIO = {
    "Agent",
    "AsyncAudioClient",
    "AsyncEmbeddingClient",
    "AsyncFallbackClient",
    "AsyncGeminiImageClient",
    "AsyncHuggingFaceAudioClient",
    "AsyncHuggingFaceImageClient",
    "AsyncHuggingFaceSpeechClient",
    "AsyncHuggingFaceTranscriptionClient",
    "AsyncImageClient",
    "AsyncModelClient",
    "AsyncOpenAISpeechClient",
    "AsyncOpenAITranscriptionClient",
    "AsyncRunner",
    "AsyncSpeechClient",
    "AsyncTranscriptionClient",
    "CLIChannel",
    "Chain",
    "Channel",
    "ChannelMessage",
    "ContextOverflowError",
    "DegenerateTurnError",
    "EvaluatorOptimizer",
    "HAS_A2A",
    "MCPClient",
    "ModelConnectionError",
    "OrchestratorAgent",
    "Parallel",
    "PlanExecuteEvaluator",
    "Router",
    "RunHandle",
    "Scheduler",
    "SkillAgent",
    "TruncatedTurnError",
    "WebChannel",
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
}
CONDITIONAL_AIO = {
    "A2AConnectionError",
    "RemoteAgent",
    "build_a2a_app",
    "serve_a2a",
}


_CASES = [
    ("aimu", aimu, REQUIRED_AIMU, CONDITIONAL_AIMU),
    ("aimu.models", aimu.models, REQUIRED_MODELS, CONDITIONAL_MODELS),
    ("aimu.agents", aimu.agents, REQUIRED_AGENTS, CONDITIONAL_AGENTS),
    ("aimu.tools", aimu.tools, REQUIRED_TOOLS, CONDITIONAL_TOOLS),
    ("aimu.aio", aimu.aio, REQUIRED_AIO, CONDITIONAL_AIO),
]


@pytest.mark.parametrize("label,module,required,conditional", _CASES, ids=[c[0] for c in _CASES])
def test_surface_is_pinned(label, module, required, conditional):
    exported = set(module.__all__)
    missing = required - exported
    unexpected = exported - required - conditional
    assert not missing, f"{label} stopped exporting: {sorted(missing)}"
    assert not unexpected, f"{label} newly exports: {sorted(unexpected)}"


@pytest.mark.parametrize("label,module", [(c[0], c[1]) for c in _CASES], ids=[c[0] for c in _CASES])
def test_every_exported_name_resolves(label, module):
    """__all__ must not advertise a name that cannot be reached.

    Provider symbols legitimately resolve to None when their optional dependency is
    absent; the failure this catches is a name in __all__ with nothing behind it at all.
    """
    for name in module.__all__:
        getattr(module, name)


def test_conditionally_exported_provider_symbols_are_not_none():
    """A provider symbol only enters aimu.models.__all__ when its HAS_* flag is True
    (i.e. the dependency is installed), via the `if HAS_X: __all__.extend([...])`
    blocks in aimu/models/__init__.py. So any name in __all__ that is also one of the
    lazily-resolved provider symbols must resolve to a real object, never None -- None
    is the "dependency absent" outcome, which by construction shouldn't apply here.

    Deliberately not extended to aimu.aio: its __all__ lists provider symbols
    unconditionally (no HAS_*-gated __all__.extend), so None is a correct outcome there.
    """
    import aimu.models as m

    lazy = m._LAZY_PROVIDER_SYMBOLS
    unexpectedly_none = [n for n in m.__all__ if n in lazy and getattr(m, n) is None]
    assert unexpectedly_none == []
