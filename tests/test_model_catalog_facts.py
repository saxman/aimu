# tests/test_model_catalog_facts.py
"""Unit tests for the shared intrinsic-facts table backing every local-runtime catalog."""

from __future__ import annotations

import pytest

from aimu.models._base.text import Model, ModelSpec
from aimu.models._catalog import MODEL_FACTS, ModelFacts, Wire, resolve_wire  # noqa: F401 (import parity w/ brief)


def test_wire_is_not_a_tuple():
    # enum unpacks tuple member values into multiple __init__ args, which would break
    # every catalog that assigns a bare Wire(...) to a member.
    assert not isinstance(Wire("x"), tuple)


def test_resolve_wire_applies_facts_and_id():
    spec = resolve_wire("QWEN_3_8B", Wire("qwen3:8b"))
    assert spec.id == "qwen3:8b"
    assert spec.tools is True
    assert spec.thinking is True
    assert spec.structured_output is False  # serving-path, not in the facts table


def test_resolve_wire_applies_serving_path_override():
    spec = resolve_wire("QWEN_3_8B", Wire("qwen3:8b", structured_output=True))
    assert spec.structured_output is True
    assert spec.tools is True  # untouched facts survive


def test_resolve_wire_applies_intrinsic_override():
    spec = resolve_wire("GEMMA_4_12B", Wire("gemma-4-12b.gguf", why="no mmproj projector", vision=False))
    assert spec.vision is False
    assert spec.thinking is True


def test_unknown_member_name_raises():
    with pytest.raises(KeyError, match="NO_SUCH_MODEL"):
        resolve_wire("NO_SUCH_MODEL", Wire("whatever"))


def test_unknown_override_key_raises():
    with pytest.raises(ValueError, match="banana"):
        resolve_wire("QWEN_3_8B", Wire("qwen3:8b", why="testing", banana=True))


def test_override_without_rationale_raises():
    # A silently unexplained override reads exactly like an applied fact.
    with pytest.raises(ValueError, match="why="):
        resolve_wire("GEMMA_4_12B", Wire("gemma-4-12b.gguf", vision=False))


def test_serving_path_flags_need_no_rationale():
    # structured_output/audio are properties of the runtime, not exceptions to a fact.
    spec = resolve_wire("QWEN_3_8B", Wire("qwen3:8b", structured_output=True, audio=True))
    assert (spec.structured_output, spec.audio) == (True, True)


def test_duplicate_ids_still_alias_like_modelspec():
    """Two members sharing an id must collapse to one, as they do with ModelSpec today.

    Wire needs id-only __eq__/__hash__ for this. Without them the duplicate stays live with a
    colliding _value_, lookup by value silently picks one, and
    test_model_catalog_consistency.py::test_no_silent_enum_aliases can never fail again.
    """

    class Dup(Model):
        QWEN_3_8B = Wire("same-id")
        QWEN_3_4B = Wire("same-id")

    assert [m.name for m in Dup] == ["QWEN_3_8B"]
    assert sorted(set(Dup.__members__) - {m.name for m in Dup}) == ["QWEN_3_4B"]


def test_facts_never_carry_serving_path_flags():
    assert not hasattr(ModelFacts(), "structured_output")
    assert not hasattr(ModelFacts(), "audio")


def test_generation_kwargs_are_copied_not_shared():
    a = resolve_wire("QWEN_3_8_27B", Wire("a"))
    b = resolve_wire("QWEN_3_8_27B", Wire("b"))
    a.generation_kwargs["temperature"] = 99.0
    assert b.generation_kwargs["temperature"] != 99.0


def test_enum_member_accepts_a_wire():
    class Cat(Model):
        QWEN_3_8B = Wire("qwen3:8b", structured_output=True)

    member = Cat.QWEN_3_8B
    assert member.value == "qwen3:8b"
    assert member.supports_tools is True
    assert member.supports_thinking is True
    assert member.supports_structured_output is True
    assert isinstance(member.spec, ModelSpec)


def test_enum_member_still_accepts_a_modelspec():
    class Cloud(Model):
        SOMETHING = ModelSpec("vendor-model-1", tools=True)

    assert Cloud.SOMETHING.value == "vendor-model-1"
    assert Cloud.SOMETHING.supports_tools is True


def test_wire_member_with_unknown_name_fails_at_class_creation():
    with pytest.raises(KeyError, match="NOT_A_REAL_MODEL"):

        class Bad(Model):
            NOT_A_REAL_MODEL = Wire("x")


def test_phi_4_mini_has_one_canonical_name():
    import aimu.models as models

    for attr in dir(models):
        enum = getattr(models, attr)
        if not attr.endswith("Model") or attr == "Model" or enum is None:
            continue
        try:
            names = {m.name for m in enum}
        except TypeError:
            continue
        assert "PHI_4_MINI" not in names, (
            f"{attr} still carries PHI_4_MINI; it is the same model as PHI_4_MINI_3_8B "
            f"and a duplicate name makes resolve_model_enum treat them as unrelated."
        )


def test_ollama_shim_catalog_matches_the_native_one():
    """The OpenAI shim fronts the same install with the same registry tags.

    Any name in one and not the other is drift, not a capability difference.
    """
    from aimu.models.providers.ollama import OllamaModel
    from aimu.models.providers.openai_compat import OllamaOpenAIModel

    native = {m.name: m.value for m in OllamaModel}
    shim = {m.name: m.value for m in OllamaOpenAIModel}
    assert shim == native


def test_hf_repo_runtimes_share_one_id_namespace():
    """vLLM, SGLang and HF Serve are all launched with a HuggingFace repo path.

    A model servable by one is servable by the others, so a name present in one and
    absent from another is drift. (Muse Glimmer is the documented exception: vLLM has
    dedicated tool/reasoning parsers for its framing and the other two do not.)
    """
    from aimu.models.providers.openai_compat import HFOpenAIModel, SGLangOpenAIModel, VLLMOpenAIModel

    catalogs = {"vLLM": VLLMOpenAIModel, "SGLang": SGLangOpenAIModel, "HF Serve": HFOpenAIModel}
    exceptions = {"MUSE_GLIMMER_30B"}
    sets = {name: {m.name for m in enum} - exceptions for name, enum in catalogs.items()}
    reference = sets["vLLM"]
    for name, members in sets.items():
        assert members == reference, f"{name} differs: {members ^ reference}"

    ids = {name: {m.name: m.value for m in enum if m.name not in exceptions} for name, enum in catalogs.items()}
    assert ids["SGLang"] == ids["vLLM"] and ids["HF Serve"] == ids["vLLM"]


GGUF_VISION_RATIONALE = "mmproj"


def test_gguf_catalogs_do_not_advertise_vision():
    """No GGUF path loads an mmproj projector by default, so none may claim vision.

    llama-cpp takes one via chat_handler=, and llama-server/LM Studio via their own flags,
    but the catalog describes the default path. Advertising vision would let a caller pass
    images that fail at request time.
    """
    from aimu.models.providers.llamacpp import LlamaCppModel
    from aimu.models.providers.openai_compat import LlamaServerOpenAIModel, LMStudioOpenAIModel

    for enum in (LlamaCppModel, LlamaServerOpenAIModel, LMStudioOpenAIModel):
        for member in enum:
            if member.name.endswith(("_4BIT", "_8BIT", "_BF16")):
                continue  # LM Studio's MLX entries are not a GGUF path
            assert member.supports_vision is False, f"{enum.__name__}.{member.name} claims vision"
            wire = getattr(member, "_wire", None)
            if wire and "vision" in wire.overrides:
                assert GGUF_VISION_RATIONALE in (wire.why or ""), (
                    f"{enum.__name__}.{member.name} overrides vision without naming the projector"
                )
