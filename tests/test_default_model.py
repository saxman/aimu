"""Mock-only tests for default-model resolution (no backend / network required).

Covers ``aimu.models._internal.model_defaults`` (the text/modality resolvers and the local probes)
and the omitted-``model`` wiring on the top-level ``aimu.*`` entry points.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest

import aimu.models
from aimu.models._internal import model_defaults as _defaults


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Neutralize the resolvers' ``_load_dotenv()`` call.

    The resolvers read a project ``.env`` on entry, which would repopulate an env var a
    test just cleared with ``monkeypatch.delenv`` (and leak a developer's real ``.env``
    into these mock-only tests). No-op it so every test controls the environment itself.
    """
    monkeypatch.setattr(_defaults, "_load_dotenv", lambda: None)


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
    monkeypatch.setattr("aimu.models.model_client.resolve_model", lambda s: s)
    assert _defaults.resolve_default_text_model() == "fake:model"


def test_text_bad_env_var_raises(monkeypatch):
    monkeypatch.setenv("AIMU_LANGUAGE_MODEL", "bogus-provider:whatever")
    with pytest.raises(ValueError):
        _defaults.resolve_default_text_model()


def test_text_env_var_accepts_an_endpoint_suffix(monkeypatch):
    """``AIMU_LANGUAGE_MODEL`` may carry the full extended string, endpoint included.

    The value is validated here and parsed for real by the client factory, so validation has to
    accept every form the factory does. Validating with the narrow ``provider:model_id`` resolver
    rejected ``@base_url`` outright, which made the endpoint reachable by an explicit
    ``aimu.client(...)`` argument but never by the env var.
    """
    monkeypatch.setenv("AIMU_LANGUAGE_MODEL", "ollama:qwen3.5:9b@http://gpu-box:11434")
    assert _defaults.resolve_default_text_model() == "ollama:qwen3.5:9b@http://gpu-box:11434"


def test_text_env_var_accepts_an_adhoc_id_with_flags(monkeypatch):
    """The ad-hoc form is part of the same extended syntax, so the env var takes it too."""
    monkeypatch.setenv("AIMU_LANGUAGE_MODEL", "lmstudio:some-local-build@http://box:1234;tools,vision")
    assert _defaults.resolve_default_text_model() == "lmstudio:some-local-build@http://box:1234;tools,vision"


def test_text_env_var_with_endpoint_still_rejects_an_unknown_id(monkeypatch):
    """An endpoint does not switch a curated-catalog provider into accepting any id."""
    monkeypatch.setenv("AIMU_LANGUAGE_MODEL", "ollama:no-such-tag:1b@http://gpu-box:11434")
    with pytest.raises(ValueError):
        _defaults.resolve_default_text_model()


# --- resolve_default_text_model: public export ----------------------------------------


def test_string_resolver_is_publicly_exported():
    """The extended-form resolver is importable, not just its enum twin.

    ``resolve_default_text_model_enum``'s own docstring sends a caller wanting ``@base_url`` or
    ``;flags`` to "the string resolver", so the pair has to be reachable from the same place the
    docstring is read from. Exporting only the enum half left the documented answer importable
    solely from ``aimu.models._internal``, and a host that needs the endpoint (to build a second
    client on the same default the first one got) had no supported way to ask for it.
    """
    import aimu

    assert aimu.resolve_default_text_model is _defaults.resolve_default_text_model
    assert aimu.models.resolve_default_text_model is _defaults.resolve_default_text_model


def test_exported_string_resolver_keeps_the_endpoint(monkeypatch):
    """The exported name is the lossless half: what comes back still carries the endpoint."""
    import aimu

    monkeypatch.setenv("AIMU_LANGUAGE_MODEL", "ollama:qwen3.5:9b@http://gpu-box:11434")
    assert aimu.resolve_default_text_model() == "ollama:qwen3.5:9b@http://gpu-box:11434"


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


def test_default_enum_env_var_endpoint_says_an_enum_cannot_carry_one(monkeypatch):
    """An endpoint is valid syntax the enum return type cannot express, so say that.

    A ``Model`` member names a catalogued id and nothing else, so this resolver has no way to
    hand back the endpoint. Rejecting is correct; reporting it as an unknown model id (which is
    what validating the unsplit string did) sends the reader looking for a typo in a valid id.
    """
    monkeypatch.setenv("AIMU_LANGUAGE_MODEL", "ollama:qwen3.5:9b@http://gpu-box:11434")
    with pytest.raises(ValueError, match="endpoint"):
        _defaults.resolve_default_text_model_enum()


def test_default_enum_env_var_takes_precedence(monkeypatch):
    from aimu.models.model_client import ResolvedModel

    sentinel = object()
    monkeypatch.setenv("AIMU_LANGUAGE_MODEL", "fake:model")
    # Patch the validator so the test doesn't depend on which providers are installed.
    monkeypatch.setattr(
        "aimu.models.model_client.resolve_model",
        lambda s: ResolvedModel(sentinel, "fake", None),
    )
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


def test_modality_unset_lists_locally_available(monkeypatch):
    monkeypatch.delenv("AIMU_IMAGE_MODEL", raising=False)
    monkeypatch.setattr(_defaults, "available_image_models", lambda: [_FakeMember("some/repo", False)])
    with pytest.raises(ValueError, match=r"Locally available: hf:some/repo"):
        _defaults.resolve_default_modality_model("AIMU_IMAGE_MODEL")


def test_modality_unset_no_hint_when_nothing_discovered(monkeypatch):
    monkeypatch.delenv("AIMU_IMAGE_MODEL", raising=False)
    monkeypatch.setattr(_defaults, "available_image_models", lambda: [])
    with pytest.raises(ValueError, match="AIMU_IMAGE_MODEL") as exc:
        _defaults.resolve_default_modality_model("AIMU_IMAGE_MODEL")
    assert "Locally available" not in str(exc.value)


def test_modality_discovery_failure_is_swallowed(monkeypatch):
    monkeypatch.delenv("AIMU_IMAGE_MODEL", raising=False)

    def boom():
        raise RuntimeError("scan blew up")

    monkeypatch.setattr(_defaults, "available_image_models", boom)
    with pytest.raises(ValueError) as exc:
        _defaults.resolve_default_modality_model("AIMU_IMAGE_MODEL")
    assert "Locally available" not in str(exc.value)


# --- per-modality local discovery (return enum members; never auto-select) ------------


def test_hf_cached_modality_members_filters_by_cache(monkeypatch):
    cached, uncached = _FakeMember("org/cached", False), _FakeMember("org/missing", False)
    monkeypatch.setattr(_defaults, "_hf_cached_repo_ids", lambda: {"org/cached"})
    assert _defaults._hf_cached_modality_members([cached, uncached], True) == [cached]


def test_hf_cached_modality_members_empty_when_flag_off(monkeypatch):
    monkeypatch.setattr(_defaults, "_hf_cached_repo_ids", lambda: pytest.fail("must not scan when flag off"))
    assert _defaults._hf_cached_modality_members([_FakeMember("a", False)], False) == []


def test_available_embedding_models_concatenates_ollama_then_hf(monkeypatch):
    ollama = [_FakeMember("nomic-embed-text", False)]
    hf = [_FakeMember("BAAI/bge-small-en-v1.5", False)]
    monkeypatch.setattr(_defaults, "_ollama_embedding_members", lambda: ollama)
    monkeypatch.setattr(_defaults, "_hf_embedding_members", lambda: hf)
    assert _defaults.available_embedding_models() == ollama + hf


def test_modality_model_string_defaults_to_hf_prefix():
    assert _defaults._modality_model_string(_FakeMember("org/repo", False)) == "hf:org/repo"


@pytest.mark.parametrize(
    "name",
    [
        "available_image_models",
        "available_audio_models",
        "available_speech_models",
        "available_transcription_models",
        "available_embedding_models",
    ],
)
def test_modality_discovery_functions_are_public(name):
    import aimu
    import aimu.models as models

    assert callable(getattr(aimu, name))
    assert getattr(aimu, name) is getattr(models, name)
    assert name in aimu.__all__
    assert name in models.__all__


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


# --- Port-8000 probe coexistence (vLLM / HF Transformers Serve / oMLX) -----------------
#
# Three probes share http://localhost:8000/v1. Discovery stays correct because each probe keeps
# only enum members whose ``.value`` appears in that server's /v1/models, and the id namespaces
# are disjoint: HF repo paths contain a '/', oMLX ids are bare --model-dir directory names.


def _fake_openai_serving(*ids):
    """Stand in for the ``openai`` module so every probe sees the same served-id list."""

    class _FakeModels:
        def list(self):
            return types.SimpleNamespace(data=[types.SimpleNamespace(id=i) for i in ids])

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.models = _FakeModels()

    # A real __spec__ so `importlib.util.find_spec("openai")` (which `installed()` -- and
    # now the lazy `aimu.models.__getattr__` -- calls) reports this stub as installed,
    # rather than raising ValueError on a spec-less sys.modules entry.
    stub = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    stub.__spec__ = importlib.machinery.ModuleSpec("openai", None)
    return stub


@pytest.mark.skipif(not aimu.models.HAS_OPENAI_COMPAT, reason="openai-compat providers not installed")
def test_omlx_id_discovered_and_not_claimed_by_vllm(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", _fake_openai_serving("Qwen3.6-35B-A3B-4bit"))
    found = {(type(m).__name__, m.name) for m in _defaults._openai_compat_members()}
    assert ("OMLXOpenAIModel", "QWEN_3_6_35B_4BIT") in found
    # An oMLX directory name is not a HuggingFace repo path, so the vLLM/HF-Serve probes on the
    # same port match nothing.
    assert not any(enum_name in ("VLLMOpenAIModel", "HFOpenAIModel") for enum_name, _ in found)


@pytest.mark.skipif(not aimu.models.HAS_OPENAI_COMPAT, reason="openai-compat providers not installed")
def test_vllm_id_not_claimed_by_omlx(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", _fake_openai_serving("Qwen/Qwen3-8B"))
    found = {(type(m).__name__, m.name) for m in _defaults._openai_compat_members()}
    assert ("VLLMOpenAIModel", "QWEN_3_8B") in found
    assert not any(enum_name == "OMLXOpenAIModel" for enum_name, _ in found)
