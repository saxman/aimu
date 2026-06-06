"""Mock-only tests for default-model resolution (no backend / network required).

Covers ``aimu.models._internal.model_defaults`` (the text/modality resolvers and the local probes)
and the omitted-``model`` wiring on the top-level ``aimu.*`` entry points.
"""

from __future__ import annotations

import pytest

from aimu.models._internal import model_defaults as _defaults


class _FakeMember:
    """Stand-in for a provider Model enum member: ``.value`` + ``.supports_tools``."""

    def __init__(self, value: str, tools: bool):
        self.value = value
        self.supports_tools = tools


# --- _pick ---------------------------------------------------------------------------


def test_pick_prefers_tool_capable():
    members = [_FakeMember("plain", False), _FakeMember("tooly", True)]
    assert _defaults._pick(members, {"plain", "tooly"}).value == "tooly"


def test_pick_falls_back_to_first_when_no_tool_model():
    members = [_FakeMember("a", False), _FakeMember("b", False)]
    assert _defaults._pick(members, {"a", "b"}).value == "a"


def test_pick_returns_none_when_nothing_installed():
    members = [_FakeMember("a", False)]
    assert _defaults._pick(members, {"z"}) is None


# --- resolve_default_text_model: env var ----------------------------------------------


def test_text_env_var_takes_precedence(monkeypatch):
    monkeypatch.setenv("AIMU_LANGUAGE_MODEL", "fake:model")
    # Patch the validator so the test doesn't depend on which providers are installed.
    monkeypatch.setattr("aimu.models.model_client.resolve_model_string", lambda s: s)
    assert _defaults.resolve_default_text_model() == "fake:model"


def test_text_bad_env_var_raises(monkeypatch):
    monkeypatch.setenv("AIMU_LANGUAGE_MODEL", "bogus-provider:whatever")
    with pytest.raises(ValueError):
        _defaults.resolve_default_text_model()


# --- resolve_default_text_model: local probe fall-through -----------------------------


def test_text_probes_ollama_first(monkeypatch):
    monkeypatch.delenv("AIMU_LANGUAGE_MODEL", raising=False)
    monkeypatch.setattr(_defaults, "_ollama_installed_text_models", lambda: "ollama:qwen3:8b")
    monkeypatch.setattr(_defaults, "_hf_cached_text_models", lambda: pytest.fail("should not reach HF"))
    assert _defaults.resolve_default_text_model() == "ollama:qwen3:8b"


def test_text_ollama_down_falls_to_hf_cache(monkeypatch):
    monkeypatch.delenv("AIMU_LANGUAGE_MODEL", raising=False)
    monkeypatch.setattr(_defaults, "_ollama_installed_text_models", lambda: None)
    monkeypatch.setattr(_defaults, "_hf_cached_text_models", lambda: "hf:Qwen/Qwen3-8B")
    monkeypatch.setattr(_defaults, "_openai_compat_served_text_models", lambda: pytest.fail("should not reach servers"))
    assert _defaults.resolve_default_text_model() == "hf:Qwen/Qwen3-8B"


def test_text_falls_to_openai_compat_servers(monkeypatch):
    monkeypatch.delenv("AIMU_LANGUAGE_MODEL", raising=False)
    monkeypatch.setattr(_defaults, "_ollama_installed_text_models", lambda: None)
    monkeypatch.setattr(_defaults, "_hf_cached_text_models", lambda: None)
    monkeypatch.setattr(_defaults, "_openai_compat_served_text_models", lambda: "vllm:Qwen/Qwen3-8B")
    assert _defaults.resolve_default_text_model() == "vllm:Qwen/Qwen3-8B"


def test_text_async_skips_hf_cache(monkeypatch):
    monkeypatch.delenv("AIMU_LANGUAGE_MODEL", raising=False)
    monkeypatch.setattr(_defaults, "_ollama_installed_text_models", lambda: None)
    monkeypatch.setattr(_defaults, "_hf_cached_text_models", lambda: pytest.fail("async must not probe HF cache"))
    monkeypatch.setattr(_defaults, "_openai_compat_served_text_models", lambda: "lmstudio:foo")
    assert _defaults.resolve_default_text_model(include_hf_cache=False) == "lmstudio:foo"


def test_text_nothing_resolves_raises(monkeypatch):
    monkeypatch.delenv("AIMU_LANGUAGE_MODEL", raising=False)
    monkeypatch.setattr(_defaults, "_ollama_installed_text_models", lambda: None)
    monkeypatch.setattr(_defaults, "_hf_cached_text_models", lambda: None)
    monkeypatch.setattr(_defaults, "_openai_compat_served_text_models", lambda: None)
    with pytest.raises(ValueError, match="AIMU_LANGUAGE_MODEL"):
        _defaults.resolve_default_text_model()


# --- available_text_models / resolve_default_text_model_enum (enum discovery) ---------


def test_available_text_models_concatenates_in_priority_order(monkeypatch):
    ollama = [_FakeMember("o", True)]
    hf = [_FakeMember("h", False)]
    servers = [_FakeMember("s", True)]
    monkeypatch.setattr(_defaults, "_ollama_members", lambda: ollama)
    monkeypatch.setattr(_defaults, "_hf_cached_members", lambda: hf)
    monkeypatch.setattr(_defaults, "_openai_compat_members", lambda: servers)
    assert _defaults.available_text_models() == ollama + hf + servers


def test_available_text_models_skips_hf_cache_when_disabled(monkeypatch):
    monkeypatch.setattr(_defaults, "_ollama_members", lambda: [])
    monkeypatch.setattr(_defaults, "_hf_cached_members", lambda: pytest.fail("HF cache must be skipped"))
    monkeypatch.setattr(_defaults, "_openai_compat_members", lambda: [_FakeMember("s", False)])
    assert [m.value for m in _defaults.available_text_models(include_hf_cache=False)] == ["s"]


def test_default_enum_env_var_takes_precedence(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("AIMU_LANGUAGE_MODEL", "fake:model")
    monkeypatch.setattr("aimu.models.model_client.resolve_model_string", lambda s: sentinel)
    assert _defaults.resolve_default_text_model_enum() is sentinel


def test_default_enum_prefers_tool_capable_member(monkeypatch):
    monkeypatch.delenv("AIMU_LANGUAGE_MODEL", raising=False)
    plain, tooly = _FakeMember("plain", False), _FakeMember("tooly", True)
    monkeypatch.setattr(_defaults, "available_text_models", lambda **_: [plain, tooly])
    assert _defaults.resolve_default_text_model_enum() is tooly


def test_default_enum_nothing_resolves_raises(monkeypatch):
    monkeypatch.delenv("AIMU_LANGUAGE_MODEL", raising=False)
    monkeypatch.setattr(_defaults, "available_text_models", lambda **_: [])
    with pytest.raises(ValueError, match="AIMU_LANGUAGE_MODEL"):
        _defaults.resolve_default_text_model_enum()


# --- resolve_default_modality_model ---------------------------------------------------


def test_modality_env_var_returned(monkeypatch):
    monkeypatch.setenv("AIMU_IMAGE_MODEL", "hf:some/repo")
    assert _defaults.resolve_default_modality_model("AIMU_IMAGE_MODEL") == "hf:some/repo"


def test_modality_unset_raises(monkeypatch):
    monkeypatch.delenv("AIMU_AUDIO_MODEL", raising=False)
    with pytest.raises(ValueError, match="AIMU_AUDIO_MODEL"):
        _defaults.resolve_default_modality_model("AIMU_AUDIO_MODEL")


# --- entry-point wiring (omitted model invokes the resolver) --------------------------


def test_client_omitted_model_invokes_text_resolver(monkeypatch):
    import aimu

    called = {}

    def sentinel():
        called["yes"] = True
        raise RuntimeError("resolver invoked")

    monkeypatch.setattr("aimu.models._internal.model_defaults.resolve_default_text_model", sentinel)
    with pytest.raises(RuntimeError, match="resolver invoked"):
        aimu.client()
    assert called.get("yes")


def test_image_client_omitted_model_invokes_modality_resolver(monkeypatch):
    import aimu

    def sentinel(env_var):
        assert env_var == "AIMU_IMAGE_MODEL"
        raise RuntimeError("modality resolver invoked")

    monkeypatch.setattr("aimu.models._internal.model_defaults.resolve_default_modality_model", sentinel)
    with pytest.raises(RuntimeError, match="modality resolver invoked"):
        aimu.image_client()
