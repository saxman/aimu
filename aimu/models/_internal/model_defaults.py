"""Default-model resolution for the ergonomic ``aimu.*`` entry points.

When a caller omits ``model=`` from :func:`aimu.chat` / :func:`aimu.client` (or the
image/audio/speech equivalents), these helpers resolve a default. The policy:

- **Text**: ``AIMU_LANGUAGE_MODEL`` env var first (deterministic); otherwise probe local
  providers for *already-available* models (running Ollama → HuggingFace cache → local
  OpenAI-compat servers), restricted to ids that match a provider ``Model`` enum so
  capability flags are known; otherwise raise.
- **Image / audio / speech / transcription / embedding**: env var only (``AIMU_IMAGE_MODEL`` /
  ``AIMU_AUDIO_MODEL`` / ``AIMU_SPEECH_MODEL`` / ``AIMU_TRANSCRIPTION_MODEL`` /
  ``AIMU_EMBEDDING_MODEL``); raise when unset. These are never auto-selected — a wrong silent
  pick is costly (an image-style swing, or a persisted vector store corrupted by a mismatched
  embedder). The unset error *does* list any locally available models (the ``available_*_models``
  discovery probes) to make the explicit choice easy.

AIMU never auto-selects a cloud provider and never *downloads* weights implicitly — every
HuggingFace probe is cache-only.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, NoReturn, Optional

if TYPE_CHECKING:
    from ..base import Model

log = logging.getLogger(__name__)

LANGUAGE_MODEL_ENV = "AIMU_LANGUAGE_MODEL"
IMAGE_MODEL_ENV = "AIMU_IMAGE_MODEL"
AUDIO_MODEL_ENV = "AIMU_AUDIO_MODEL"
SPEECH_MODEL_ENV = "AIMU_SPEECH_MODEL"
TRANSCRIPTION_MODEL_ENV = "AIMU_TRANSCRIPTION_MODEL"
EMBEDDING_MODEL_ENV = "AIMU_EMBEDDING_MODEL"

# Local OpenAI-compatible servers to probe at their default base_urls.
# (provider key matching aimu.models.model_client._provider_registry, base_url, Model enum attr).
# Cloud (openai, gemini) and ollama-openai (covered by the native Ollama probe) are excluded.
_OPENAI_COMPAT_PROBES = (
    ("lmstudio", "http://localhost:1234/v1", "LMStudioOpenAIModel"),
    ("vllm", "http://localhost:8000/v1", "VLLMOpenAIModel"),
    ("hf-openai", "http://localhost:8000/v1", "HFOpenAIModel"),
    ("llamaserver", "http://localhost:8080/v1", "LlamaServerOpenAIModel"),
    ("sglang", "http://localhost:30000/v1", "SGLangOpenAIModel"),
)


def _load_dotenv() -> None:
    """Load a project ``.env`` if python-dotenv is installed (matches provider clients)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _first(members: list) -> Optional["Model"]:
    """First member, preferring tool-capable ones. ``None`` for an empty list."""
    if not members:
        return None
    return next((m for m in members if getattr(m, "supports_tools", False)), members[0])


def _pick(members: list, installed: set[str]):
    """Filter ``members`` to those whose ``.value`` is installed, then prefer tool-capable.

    Retained as the selection primitive for the single-default string probes below.
    """
    return _first([m for m in members if m.value in installed])


def _provider_prefix(member: "Model") -> str:
    """Provider key (e.g. ``"lmstudio"``) for a text ``Model`` enum member.

    Inverse of :func:`aimu.models.model_client._provider_registry`. Lets the OpenAI-compat
    probe label a member that may have come from any of several probed servers.
    """
    from ..model_client import _provider_registry

    for prefix, (enum_cls, _client) in _provider_registry().items():
        if isinstance(member, enum_cls):
            return prefix
    raise ValueError(f"No installed provider matches model enum {member!r}")


# --- Raw local-availability scans (return all matching enum members; no selection) -----


def _ollama_installed_names() -> set[str]:
    """Model names pulled on a running Ollama server (empty set if unreachable)."""
    try:
        from .. import HAS_OLLAMA

        if not HAS_OLLAMA:
            return set()
        import ollama

        resp = ollama.list()
        raw = getattr(resp, "models", None)
        if raw is None and isinstance(resp, dict):
            raw = resp.get("models", [])
        names: set[str] = set()
        for entry in raw or []:
            name = getattr(entry, "model", None) or getattr(entry, "name", None)
            if name is None and isinstance(entry, dict):
                name = entry.get("model") or entry.get("name")
            if name:
                names.add(name)
        return names
    except Exception:  # server down, lib quirk, etc.
        return set()


def _hf_cached_repo_ids() -> set[str]:
    """Repo ids already in the local HuggingFace cache (empty set; never downloads)."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        return {repo.repo_id for repo in info.repos if getattr(repo, "repo_type", "model") == "model"}
    except Exception:
        return set()


def _hf_cached_modality_members(members: list, has_flag: bool) -> list:
    """Filter ``members`` to those whose ``.value`` repo id is in the local HF cache.

    Generic over modality: callers pass the modality enum's members plus its ``HAS_*``
    flag. Returns ``[]`` (without scanning) when the provider isn't installed.
    """
    if not has_flag:
        return []
    cached = _hf_cached_repo_ids()
    return [m for m in members if m.value in cached] if cached else []


def _ollama_members() -> list:
    """Every ``OllamaModel`` enum member installed on a running Ollama server."""
    try:
        from .. import HAS_OLLAMA, OllamaModel

        if not HAS_OLLAMA:
            return []
        names = _ollama_installed_names()
        return [m for m in OllamaModel if m.value in names] if names else []
    except Exception:
        return []


def _hf_cached_members() -> list:
    """Every ``HuggingFaceModel`` enum member already in the local cache (no download)."""
    try:
        from .. import HAS_HF, HuggingFaceModel

        return _hf_cached_modality_members(list(HuggingFaceModel), HAS_HF)
    except Exception:
        return []


def _openai_compat_members() -> list:
    """Every enum member served by a reachable local OpenAI-compat server (across all probes)."""
    try:
        from .. import HAS_OPENAI_COMPAT

        if not HAS_OPENAI_COMPAT:
            return []
        import openai

        import aimu.models as models_pkg
    except Exception:
        return []

    members: list = []
    for _provider, base_url, enum_name in _OPENAI_COMPAT_PROBES:
        enum = getattr(models_pkg, enum_name, None)
        if enum is None:
            continue
        try:
            probe = openai.OpenAI(base_url=base_url, api_key="not-needed", timeout=0.5, max_retries=0)
            served = {m.id for m in probe.models.list().data}
        except Exception:  # connection refused / not running — skip
            continue
        members.extend(m for m in enum if m.value in served)
    return members


# --- Single-default string probes (the public-default selection path) ------------------


def _ollama_installed_text_models() -> Optional[str]:
    """Pick an enum-matching model already installed on a running Ollama server."""
    member = _first(_ollama_members())
    return f"ollama:{member.value}" if member else None


def _hf_cached_text_models() -> Optional[str]:
    """Pick an enum-matching HuggingFace model already in the local cache (no download)."""
    member = _first(_hf_cached_members())
    return f"hf:{member.value}" if member else None


def _openai_compat_served_text_models() -> Optional[str]:
    """Pick an enum-matching model served by a running local OpenAI-compat server."""
    member = _first(_openai_compat_members())
    return f"{_provider_prefix(member)}:{member.value}" if member else None


def available_text_models(*, include_hf_cache: bool = True) -> list:
    """Return locally *available* text models as provider ``Model`` enum members.

    Discovery is download-free and cloud-free — it reports only what is loadable right
    now: models on a running Ollama server, models in the local HuggingFace cache, and
    models served by a reachable local OpenAI-compatible server. Order is provider
    priority (Ollama → HuggingFace cache → local servers), then enum-definition order
    within each provider.

    ``include_hf_cache=False`` skips the HuggingFace cache probe (the async surface cannot
    wrap an ``hf:`` in-process client directly).

    See :func:`resolve_default_text_model_enum` for the single auto-pick, and
    :func:`aimu.resolve_model_enum` to resolve a *named* model to an enum member.
    """
    members = list(_ollama_members())
    if include_hf_cache:
        members += _hf_cached_members()
    members += _openai_compat_members()
    return members


# --- Per-modality local discovery (surface choices; never auto-select) ----------------
#
# Unlike text, these helpers feed only the discovery surface (``aimu.available_*_models``)
# and the unset-env error message — they are never auto-selected. Image / audio / speech /
# transcription have no local server, so discovery is the HuggingFace cache only; embedding
# adds the Ollama probe (Ollama serves embedding models like text models). Cloud providers
# (OpenAI, Gemini) are never discovered.


def available_image_models() -> list:
    """Locally available image models as ``HuggingFaceImageModel`` members (HF cache only)."""
    from .. import HAS_HF_IMAGE, HuggingFaceImageModel

    return _hf_cached_modality_members(list(HuggingFaceImageModel or []), HAS_HF_IMAGE)


def available_audio_models() -> list:
    """Locally available audio models as ``HuggingFaceAudioModel`` members (HF cache only)."""
    from .. import HAS_HF_AUDIO, HuggingFaceAudioModel

    return _hf_cached_modality_members(list(HuggingFaceAudioModel or []), HAS_HF_AUDIO)


def available_speech_models() -> list:
    """Locally available speech models as ``HuggingFaceSpeechModel`` members (HF cache only)."""
    from .. import HAS_HF_SPEECH, HuggingFaceSpeechModel

    return _hf_cached_modality_members(list(HuggingFaceSpeechModel or []), HAS_HF_SPEECH)


def available_transcription_models() -> list:
    """Locally available transcription models as ``HuggingFaceTranscriptionModel`` members (HF cache only)."""
    from .. import HAS_HF_TRANSCRIPTION, HuggingFaceTranscriptionModel

    return _hf_cached_modality_members(list(HuggingFaceTranscriptionModel or []), HAS_HF_TRANSCRIPTION)


def _ollama_embedding_members() -> list:
    """Every ``OllamaEmbeddingModel`` member installed on a running Ollama server."""
    try:
        from .. import HAS_OLLAMA_EMBEDDING, OllamaEmbeddingModel

        if not HAS_OLLAMA_EMBEDDING:
            return []
        names = _ollama_installed_names()
        return [m for m in OllamaEmbeddingModel if m.value in names] if names else []
    except Exception:
        return []


def _hf_embedding_members() -> list:
    """Every ``HuggingFaceEmbeddingModel`` member already in the local HF cache."""
    try:
        from .. import HAS_HF_EMBEDDING, HuggingFaceEmbeddingModel

        return _hf_cached_modality_members(list(HuggingFaceEmbeddingModel or []), HAS_HF_EMBEDDING)
    except Exception:
        return []


def available_embedding_models() -> list:
    """Locally available embedding models (running Ollama → HuggingFace cache), as enum members."""
    return _ollama_embedding_members() + _hf_embedding_members()


def _modality_model_string(member) -> str:
    """Format a discovered modality member as ``"provider:model_id"``.

    Only HuggingFace and Ollama members are ever discovered (cloud providers are excluded),
    so the prefix is ``ollama:`` for an ``OllamaEmbeddingModel`` member and ``hf:`` otherwise.
    """
    try:
        from .. import OllamaEmbeddingModel

        if OllamaEmbeddingModel is not None and isinstance(member, OllamaEmbeddingModel):
            return f"ollama:{member.value}"
    except Exception:
        pass
    return f"hf:{member.value}"


# Maps each modality env var to the name of its discovery function. Resolved through module
# globals at call time (not captured as function refs) so tests can monkeypatch the helpers.
_MODALITY_DISCOVERY = {
    IMAGE_MODEL_ENV: "available_image_models",
    AUDIO_MODEL_ENV: "available_audio_models",
    SPEECH_MODEL_ENV: "available_speech_models",
    TRANSCRIPTION_MODEL_ENV: "available_transcription_models",
    EMBEDDING_MODEL_ENV: "available_embedding_models",
}


def _raise_no_default() -> NoReturn:
    from .. import available_text_clients

    providers = [c.__name__ for c in available_text_clients()]
    raise ValueError(
        "No model specified and no default could be resolved. "
        f"Set {LANGUAGE_MODEL_ENV}='provider:model_id', pass model=... explicitly, "
        "or install/start a local provider (a running Ollama server, a cached HuggingFace model, "
        "or a local OpenAI-compatible server). "
        f"Installed text providers: {providers or 'none'}."
    )


def resolve_default_text_model(*, include_hf_cache: bool = True) -> str:
    """Resolve a default ``"provider:model_id"`` string for text generation.

    Order: ``AIMU_LANGUAGE_MODEL`` → local already-available probes → ``ValueError``.
    Never selects a cloud provider and never downloads weights.

    ``include_hf_cache`` is set ``False`` by the async surface: an ``hf:`` default would
    need to wrap a sync client (in-process Decision 7), so the async path probes only
    Ollama and local OpenAI-compatible servers.
    """
    _load_dotenv()
    from ..model_client import resolve_model_string

    env_val = os.environ.get(LANGUAGE_MODEL_ENV)
    if env_val:
        resolve_model_string(env_val)  # validate now; raises a clear ValueError on a bad id
        return env_val

    probes = [_ollama_installed_text_models]
    if include_hf_cache:
        probes.append(_hf_cached_text_models)
    probes.append(_openai_compat_served_text_models)

    for probe in probes:
        picked = probe()
        if picked is not None:
            log.warning(
                "aimu: no model specified; auto-selected %r via local discovery. "
                "Set %s='provider:model_id' to pin a default.",
                picked,
                LANGUAGE_MODEL_ENV,
            )
            return picked

    _raise_no_default()


def resolve_default_text_model_enum(*, include_hf_cache: bool = True) -> "Model":
    """Like :func:`resolve_default_text_model`, but return the ``Model`` enum member.

    Order: ``AIMU_LANGUAGE_MODEL`` (parsed via :func:`resolve_model_string`) → the first
    locally available model (tool-capable preferred; see :func:`available_text_models`) →
    ``ValueError``. Never selects a cloud provider and never downloads weights.
    """
    _load_dotenv()
    from ..model_client import resolve_model_string

    env_val = os.environ.get(LANGUAGE_MODEL_ENV)
    if env_val:
        return resolve_model_string(env_val)

    member = _first(available_text_models(include_hf_cache=include_hf_cache))
    if member is not None:
        log.warning(
            "aimu: no model specified; auto-selected %r via local discovery. "
            "Set %s='provider:model_id' to pin a default.",
            member,
            LANGUAGE_MODEL_ENV,
        )
        return member

    _raise_no_default()


def resolve_default_modality_model(env_var: str) -> str:
    """Resolve a default model for image/audio/speech/etc. from ``env_var``; raise when unset.

    AIMU never auto-selects one of these (a wrong silent pick is costly: an image-style
    swing, or a persisted vector store corrupted by a mismatched embedder) and never
    downloads weights implicitly. So an unset env var is a clear, actionable error — but
    the message lists any locally available models to make the explicit choice easy.
    """
    _load_dotenv()
    val = os.environ.get(env_var)
    if val:
        return val

    hint = ""
    discover_name = _MODALITY_DISCOVERY.get(env_var)
    if discover_name:
        discover = globals().get(discover_name)
        try:
            available = [_modality_model_string(m) for m in discover()]
        except Exception:  # a discovery probe failing must not mask the real error
            available = []
        if available:
            hint = f" Locally available: {', '.join(available)}."

    raise ValueError(
        f"No model specified and {env_var} is not set. "
        f"Pass model=... explicitly or set {env_var}='provider:model_id'.{hint}"
    )
