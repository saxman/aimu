"""Default-model resolution for the ergonomic ``aimu.*`` entry points.

When a caller omits ``model=`` from :func:`aimu.chat` / :func:`aimu.client` (or the
image/audio/speech equivalents), these helpers resolve a default. The policy:

- **Text**: ``AIMU_LANGUAGE_MODEL`` env var first (deterministic); otherwise probe local
  providers for *already-available* models (running Ollama → HuggingFace cache → local
  OpenAI-compat servers), restricted to ids that match a provider ``Model`` enum so
  capability flags are known; otherwise raise.
- **Image / audio / speech**: env var only (``AIMU_IMAGE_MODEL`` / ``AIMU_AUDIO_MODEL`` /
  ``AIMU_SPEECH_MODEL``); raise when unset.

AIMU never auto-selects a cloud provider and never *downloads* weights implicitly — the
HuggingFace text probe is cache-only.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

LANGUAGE_MODEL_ENV = "AIMU_LANGUAGE_MODEL"
IMAGE_MODEL_ENV = "AIMU_IMAGE_MODEL"
AUDIO_MODEL_ENV = "AIMU_AUDIO_MODEL"
SPEECH_MODEL_ENV = "AIMU_SPEECH_MODEL"

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


def _pick(members: list, installed: set[str]):
    """Return the first enum member (priority order) whose ``.value`` is installed,
    preferring tool-capable members. ``None`` when nothing matches."""
    matches = [m for m in members if m.value in installed]
    if not matches:
        return None
    tool_capable = [m for m in matches if getattr(m, "supports_tools", False)]
    return (tool_capable or matches)[0]


def _ollama_installed_text_models() -> Optional[str]:
    """Pick an enum-matching model already installed on a running Ollama server."""
    try:
        from . import HAS_OLLAMA, OllamaModel

        if not HAS_OLLAMA:
            return None
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
        member = _pick(list(OllamaModel), names)
        return f"ollama:{member.value}" if member else None
    except Exception:  # server down, lib quirk, etc. — fall through to the next probe
        return None


def _hf_cached_text_models() -> Optional[str]:
    """Pick an enum-matching HuggingFace model already in the local cache (no download)."""
    try:
        from . import HAS_HF, HuggingFaceModel

        if not HAS_HF:
            return None
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        cached = {repo.repo_id for repo in info.repos if getattr(repo, "repo_type", "model") == "model"}
        member = _pick(list(HuggingFaceModel), cached)
        return f"hf:{member.value}" if member else None
    except Exception:
        return None


def _openai_compat_served_text_models() -> Optional[str]:
    """Pick an enum-matching model served by a running local OpenAI-compat server."""
    try:
        from . import HAS_OPENAI_COMPAT

        if not HAS_OPENAI_COMPAT:
            return None
        import openai

        from . import (  # noqa: F401 — imported for getattr lookup below
            HFOpenAIModel,
            LlamaServerOpenAIModel,
            LMStudioOpenAIModel,
            SGLangOpenAIModel,
            VLLMOpenAIModel,
        )
        import aimu.models as models_pkg
    except Exception:
        return None

    for provider, base_url, enum_name in _OPENAI_COMPAT_PROBES:
        enum = getattr(models_pkg, enum_name, None)
        if enum is None:
            continue
        try:
            probe = openai.OpenAI(base_url=base_url, api_key="not-needed", timeout=0.5, max_retries=0)
            served = {m.id for m in probe.models.list().data}
        except Exception:  # connection refused / not running — skip
            continue
        member = _pick(list(enum), served)
        if member:
            return f"{provider}:{member.value}"
    return None


def resolve_default_text_model(*, include_hf_cache: bool = True) -> str:
    """Resolve a default ``"provider:model_id"`` string for text generation.

    Order: ``AIMU_LANGUAGE_MODEL`` → local already-available probes → ``ValueError``.
    Never selects a cloud provider and never downloads weights.

    ``include_hf_cache`` is set ``False`` by the async surface: an ``hf:`` default would
    need to wrap a sync client (in-process Decision 7), so the async path probes only
    Ollama and local OpenAI-compatible servers.
    """
    _load_dotenv()
    from .model_client import resolve_model_string

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

    from . import available_text_clients

    providers = [c.__name__ for c in available_text_clients()]
    raise ValueError(
        "No model specified and no default could be resolved. "
        f"Set {LANGUAGE_MODEL_ENV}='provider:model_id', pass model=... explicitly, "
        "or install/start a local provider (a running Ollama server, a cached HuggingFace model, "
        "or a local OpenAI-compatible server). "
        f"Installed text providers: {providers or 'none'}."
    )


def resolve_default_modality_model(env_var: str) -> str:
    """Resolve a default model for image/audio/speech from ``env_var``; raise when unset.

    These modalities have no cheap local registry to probe and AIMU must never download
    weights implicitly, so an unset env var is a clear, actionable error.
    """
    _load_dotenv()
    val = os.environ.get(env_var)
    if not val:
        raise ValueError(
            f"No model specified and {env_var} is not set. "
            f"Pass model=... explicitly or set {env_var}='provider:model_id'."
        )
    return val
