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


_INTRINSIC_FLAGS = ("supports_tools", "supports_thinking", "supports_vision", "thinking_levels", "thinking_optional")


def _cross_provider_bare_names(min_enums: int = 2) -> frozenset[str]:
    """Names carried as a bare member by at least *min_enums* distinct text-model catalogs.

    ``MODEL_FACTS`` exists to state a cross-provider identity once (see its module docstring):
    a name that earns a place there is expected to be reachable, bare, from more than one
    provider. A name that is a bare member in exactly **one** catalog is not serving that role
    -- it is a single-provider one-off -- and that is precisely the shape that can accidentally
    string-prefix an unrelated, longer sibling name. This module's own catalog has a real
    example: ``HuggingFaceModel.MAGISTRAL_SMALL`` (the 2509 revision, vision-capable, kept
    deliberately separate per ``_catalog.py``'s note on the two non-interchangeable Magistral
    releases) is a bare member in exactly one enum, and is a literal string-prefix of the
    unrelated ``MAGISTRAL_SMALL_24B`` (the 2506 revision, catalogued broadly elsewhere) --
    without this filter a naive prefix match would treat the latter as a "quant sibling" of the
    former and wrongly exempt it from the GGUF vision check below. Every genuine
    per-quantization base in these catalogs (``GPT_OSS_20B``, ``PHI_4_MINI_3_8B``, ``QWEN_3_32B``,
    ...) ships across many providers (Ollama, the HF-repo trio, the GGUF trio, ...), so requiring
    at least two is a real discriminator, not an arbitrary cutoff -- see
    ``test_cross_provider_bare_names_excludes_single_catalog_names`` below.
    """
    import aimu.models as models

    counts: dict[str, int] = {}
    for attr in dir(models):
        if not attr.endswith("Model") or attr == "Model":
            continue
        enum = getattr(models, attr)
        if enum is None:
            continue
        try:
            members = list(enum)
        except TypeError:
            continue
        if not members or not all(hasattr(members[0], flag) for flag in _INTRINSIC_FLAGS):
            continue
        for name in {m.name for m in enum}:
            counts[name] = counts.get(name, 0) + 1
    return frozenset(name for name, count in counts.items() if count >= min_enums)


def _quant_base(name: str, facts: dict) -> str | None:
    """If *name* is a per-quantization sibling of a ``MODEL_FACTS`` base, return that base.

    A quant member's name is its base model's ``MODEL_FACTS`` key plus ``"_"`` plus whatever
    suffix the quantization actually ships under. That suffix is **not** standardized upstream:
    most are ``_4BIT``/``_8BIT``/``_BF16``, but Task 11 alone added ``_FP16`` (Phi-4-mini's third
    precision has no bf16 build, only an "-mlx-fp16" one) and ``_MXFP4_Q4``/``_MXFP4_Q8``
    (gpt-oss ships natively quantized to mxfp4, so its mlx-community build is a Q4/Q8
    requantization of that format, not a plain bit-width one). An enumerated suffix allowlist
    needs a new entry every time upstream invents another naming scheme -- which happened three
    times in that one task alone -- and silently stops matching in the meantime. Matching on "is
    there a known base model name this is an extension of" instead needs no such upkeep: any
    future suffix, however it's spelled, is still recognised because the base name is still a
    prefix of it.

    Candidates are restricted to ``_cross_provider_bare_names()`` (see its docstring for why:
    plain string-prefix matching over the *entire* ``MODEL_FACTS`` table has a real
    false-positive, ``MAGISTRAL_SMALL`` prefixing ``MAGISTRAL_SMALL_24B``, that this filter
    excludes). The longest surviving candidate wins, so a more specific key beats a shorter one
    when both happen to match.
    """
    candidates = [
        key for key in facts if name != key and name.startswith(key + "_") and key in _cross_provider_bare_names()
    ]
    return max(candidates, key=len) if candidates else None


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
            if _quant_base(member.name, MODEL_FACTS) is not None:
                continue  # a per-quantization sibling is an MLX entry, not a GGUF one
            assert member.supports_vision is False, f"{enum.__name__}.{member.name} claims vision"
            wire = getattr(member, "_wire", None)
            if wire and "vision" in wire.overrides:
                assert GGUF_VISION_RATIONALE in (wire.why or ""), (
                    f"{enum.__name__}.{member.name} overrides vision without naming the projector"
                )


def test_hf_tools_true_requires_a_parse_route():
    """``supports_tools=True`` on a ``HuggingFaceModel`` member requires a way to parse the call.

    A tool call the in-process client cannot parse out of the model's raw output is
    indistinguishable from prose -- the model "calls" a tool and the caller never finds out.
    This is not hypothetical: mid-branch, ``HuggingFaceModel.PHI_4_MINI_3_8B`` briefly advertised
    ``tools=True`` on exactly this path (every catalog agreed on the same wrong value, so nothing
    keyed on cross-catalog agreement caught it -- it was caught twice by human inspection).

    Reading ``aimu/models/providers/hf/text.py::HuggingFaceClient.__init__``, a member has a
    parse route one of two ways: a ``tool_call_format`` other than ``ToolCallFormat.NA`` (the
    per-model regex/XML/JSON parsers), or the processor ``parse_response()`` path, which that
    method enables only for ``model.value.startswith("google/gemma-4")``. (``MAGISTRAL_SMALL``
    also loads via the processor branch, but its route to tool calls is its own
    ``ToolCallFormat.BRACKETED``, already covered by the first condition -- the processor branch
    itself does not set ``_uses_processor_parse_response`` for it.)

    Deliberately one-directional: the converse (a parse route with ``tools=False``) has a benign
    existing hit, ``DEEPSEEK_R1_8B`` (``ToolCallFormat.XML`` declared for symmetry with its
    Ollama/vLLM siblings' catalogs, weights that don't reliably call tools), so asserting the
    reverse would fail on a case that isn't a bug.
    """
    from aimu.models.providers.hf.text import HuggingFaceModel, ToolCallFormat

    for member in HuggingFaceModel:
        if not member.supports_tools:
            continue
        has_parse_route = member.tool_call_format != ToolCallFormat.NA or member.value.startswith("google/gemma-4")
        assert has_parse_route, (
            f"HuggingFaceModel.{member.name} claims tools=True but has no tool-call parse route "
            f"(tool_call_format is NA and its wire id does not start with 'google/gemma-4', the "
            f"only processor-parse-response path). Either give it a ToolCallFormat, or set "
            f"tools=False here via Wire(..., why=..., tools=False) if this serving path genuinely "
            f"cannot surface a tool call for these weights."
        )


def test_mlx_members_are_quant_suffixed_and_have_base_facts():
    """A quant-suffixed member is the same weights at a different precision.

    Its intrinsic facts must match the base model's, so a caller picking a quant does not
    silently get different declared capabilities.
    """
    from aimu.models.providers.openai_compat import OMLXOpenAIModel

    for member in OMLXOpenAIModel:
        base = _quant_base(member.name, MODEL_FACTS)
        if base is not None:
            assert MODEL_FACTS[member.name] == MODEL_FACTS[base], (
                f"{member.name} declares different facts from its base model {base}"
            )


def test_cross_provider_bare_names_excludes_single_catalog_names():
    """Pins the false-positive this task's fix exists to prevent.

    ``MAGISTRAL_SMALL`` (HuggingFaceModel's 2509 revision) and ``MAGISTRAL_SMALL_24B`` (the
    2506 revision, catalogued broadly elsewhere) are unrelated models that happen to share a
    string prefix. A naive "any MODEL_FACTS key is a valid base" match would treat the latter as
    a quantization of the former; ``_cross_provider_bare_names()`` must exclude
    ``MAGISTRAL_SMALL`` (a bare member of exactly one catalog) while still admitting genuinely
    cross-provider names like ``GPT_OSS_20B``.
    """
    names = _cross_provider_bare_names()
    assert "MAGISTRAL_SMALL" not in names
    assert "GPT_OSS_20B" in names
    assert "PHI_4_MINI_3_8B" in names


def test_quant_base_does_not_misidentify_an_unrelated_prefixed_model():
    """The regression this task's fix targets: a real member, not a synthetic string.

    ``MAGISTRAL_SMALL_24B`` is a bare model in its own right (catalogued across most local
    runtimes), not a quantization of the differently-versioned ``MAGISTRAL_SMALL``. Before the
    ``_cross_provider_bare_names()`` filter, plain prefix matching over all of ``MODEL_FACTS``
    misidentified it as one, which would have wrongly exempted it from
    ``test_gguf_catalogs_do_not_advertise_vision``'s vision check.
    """
    assert _quant_base("MAGISTRAL_SMALL_24B", MODEL_FACTS) is None


def test_quant_base_recognizes_every_task_11_irregular_suffix():
    """The members this fix was written for: none end in ``_4BIT``/``_8BIT``/``_BF16``."""
    for name, expected_base in (
        ("PHI_4_MINI_3_8B_FP16", "PHI_4_MINI_3_8B"),
        ("GPT_OSS_20B_MXFP4_Q4", "GPT_OSS_20B"),
        ("GPT_OSS_20B_MXFP4_Q8", "GPT_OSS_20B"),
    ):
        assert not name.endswith(("_4BIT", "_8BIT", "_BF16"))
        assert _quant_base(name, MODEL_FACTS) == expected_base
