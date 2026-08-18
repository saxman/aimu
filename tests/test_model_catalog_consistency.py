"""Cross-provider consistency of the text-model catalogs.

The same model shipped under multiple providers uses a provider-specific
``ModelSpec.id`` -- the *wire* identifier each server actually accepts
(``qwen3:8b`` for Ollama, ``Qwen/Qwen3-8B`` for vLLM/HF, ``qwen3-8b.gguf`` for
llama-server, ...). Those ids are not interchangeable and are correctly
different per provider.

The cross-provider *identity* is the enum-member **name** (``QWEN_3_8B``), which
is what ``resolve_model_enum`` uses to search every provider enum for a bare
name. This test enforces that a shared name describes the *same model*: its
intrinsic capability flags agree across providers. A silent drift (e.g. a new
provider entry that forgets ``vision=True``) would make bare-name resolution
hand back a member with the wrong capabilities depending on which provider won
the availability tiebreaker.

Two capability flags are deliberately NOT checked at all, because they describe
the *serving path* rather than the model and vary systematically:

* ``supports_structured_output`` -- Ollama grammar-enforces JSON for any model,
  so every ``OllamaModel`` member sets it while the same model under a raw
  vLLM/HF server does not.
* ``supports_audio`` -- Gemma 4 supports audio natively, but only the in-process
  HuggingFace path exposes audio input; the OpenAI-compat server catalogs leave
  it False by design (see the comments in ``providers/openai_compat.py``).

The remaining intrinsic flags (tools / thinking / vision / thinking_levels /
thinking_optional) are checked, with two escape hatches below:
``_INTENTIONAL_DIVERGENCES`` (a shared name whose flag *legitimately* differs by
serving path) and ``_SUSPECTED_OVERSIGHTS`` (a divergence that looks like a bug,
frozen so the suite is green while the fix is pending).
"""

from __future__ import annotations

from collections import defaultdict

import pytest

import aimu.models as models

# Intrinsic model capabilities: properties of the weights, not the serving path,
# so they must agree wherever a model name appears.
INTRINSIC_FLAGS = (
    "supports_tools",
    "supports_thinking",
    "supports_vision",
    "thinking_levels",
    "thinking_optional",
)

# Divergences that are CORRECT: the same model legitimately differs on an intrinsic
# flag because a specific serving path cannot expose that capability. These are not
# bugs and are not expected to be reconciled; each needs a rationale.
#
# Keyed (member_name, flag).
_INTENTIONAL_DIVERGENCES: dict[tuple[str, str], str] = {
    # The in-process HuggingFace and native-Ollama clients parse tool calls via a
    # per-model format; Gemma 3 has no format assigned, so those paths set tools=False.
    # OpenAI-compat servers (vLLM/SGLang/HF-serve/llama-server) parse tool calls
    # server-side, so they set tools=True for the same weights.
    (
        "GEMMA_3_12B",
        "supports_tools",
    ): "in-process HF/Ollama lack a tool-parse format; OpenAI-compat servers parse server-side",
    # llama-cpp vision needs an mmproj projector supplied via chat_handler=, which the
    # default GGUF path does not load; advertising vision=True there would let a caller
    # pass images that error out. Every other provider serves Gemma 4's vision natively.
    (
        "GEMMA_4_12B",
        "supports_vision",
    ): "llama-cpp default GGUF path loads no mmproj projector; other providers serve vision natively",
}

# Divergences that look like OVERSIGHTS pending confirmation. Frozen so this suite is
# green on a clean tree while still catching *new* drift. Remove an entry once the
# specs are reconciled (the stale-entry test below enforces that).
#
# Keyed (member_name, flag). Currently empty: the Llama 3.1/3.2 tool-calling divergence was
# resolved by live-testing tool use on Ollama (reliable) and enabling tools across the
# consumer runtimes; the Qwen 3.5/3.6 vision divergence was resolved by confirming the models
# are natively multimodal.
_SUSPECTED_OVERSIGHTS: set[tuple[str, str]] = set()


def _text_model_enums() -> dict[str, type]:
    """Every installed text-model enum exposed from ``aimu.models``.

    Enums are ``None`` when their provider's optional dep is absent; those are
    skipped, so the test enforces agreement across whatever is installed (all
    providers under the ``[all]`` dev extra).
    """
    enums: dict[str, type] = {}
    for attr in dir(models):
        if not attr.endswith("Model") or attr == "Model":
            continue
        obj = getattr(models, attr)
        if obj is None:
            continue
        try:
            members = list(obj)
        except TypeError:
            continue
        # Text-model members expose the capability properties; the modality enums
        # (ImageModel, AudioModel, ...) do not.
        if members and all(hasattr(members[0], flag) for flag in INTRINSIC_FLAGS):
            enums[attr] = obj
    return enums


def _shared_flag_values() -> dict[tuple[str, str], dict[str, bool]]:
    """Map (member_name, flag) -> {provider_enum: value} for names in >= 2 enums."""
    by_name: dict[str, dict[str, object]] = defaultdict(dict)
    for enum_name, enum in _text_model_enums().items():
        for member in enum:
            by_name[member.name][enum_name] = member

    result: dict[tuple[str, str], dict[str, bool]] = {}
    for member_name, providers in by_name.items():
        if len(providers) < 2:
            continue
        for flag in INTRINSIC_FLAGS:
            result[(member_name, flag)] = {enum_name: getattr(member, flag) for enum_name, member in providers.items()}
    return result


def test_text_model_enums_discovered():
    # Guard against the discovery heuristic silently matching nothing (which would
    # make every consistency assertion below vacuously pass).
    assert len(_text_model_enums()) >= 5


@pytest.mark.parametrize("enum_name", sorted(_text_model_enums()))
def test_no_silent_enum_aliases(enum_name):
    """No two members of a catalog may share a ``ModelSpec.id``.

    ``Model.__init__`` assigns ``_value_ = spec.id`` *before* enum's duplicate-value scan runs,
    and ``ModelSpec.__eq__``/``__hash__`` are id-only, so a second member with the same id
    silently becomes an **alias** of the first: it vanishes from iteration (and therefore from
    ``TOOL_MODELS``/``VISION_MODELS``, the discovery probes, and every check in this file), and
    its own ``ModelSpec`` -- including its capability flags -- is discarded. Nothing warns.

    The failure mode this guards is a catalog that carries both a bare and a per-quantization
    member for one model (e.g. ``OMLXOpenAIModel``'s ``QWEN_3_6_35B`` / ``QWEN_3_6_35B_4BIT``):
    giving the bare member the quantized member's id would delete the latter outright.
    """
    enum = _text_model_enums()[enum_name]
    canonical = {member.name for member in enum}
    aliases = sorted(set(enum.__members__) - canonical)
    assert not aliases, (
        f"{enum_name} member(s) {aliases} silently alias an earlier member sharing their "
        f"ModelSpec.id, so they are absent from iteration and their flags are discarded. "
        f"Give each member a distinct id."
    )


@pytest.mark.parametrize("key", sorted(_shared_flag_values()))
def test_shared_name_intrinsic_flags_agree(key):
    member_name, flag = key
    values = _shared_flag_values()[key]
    if len(set(values.values())) <= 1:
        return  # consistent
    if key in _INTENTIONAL_DIVERGENCES:
        return  # documented, correct serving-path difference
    if key in _SUSPECTED_OVERSIGHTS:
        pytest.xfail(f"known catalog divergence (suspected oversight): {member_name}.{flag} = {values}")
    per_provider = ", ".join(f"{enum}={val}" for enum, val in sorted(values.items()))
    pytest.fail(
        f"{member_name} disagrees on {flag} across providers: {per_provider}. "
        f"A shared enum name must describe the same model's intrinsic capabilities. "
        f"Fix the spec, or if this is intentional add {key!r} to _INTENTIONAL_DIVERGENCES "
        f"(with a rationale) or _SUSPECTED_OVERSIGHTS."
    )


# Expected ``think_opener_in_prompt`` per HuggingFace thinking model, verified against each
# repo's published chat template. True means the template appends a *bare* ``<think>`` to the
# generation prompt, so the model generates inside the thinking block and emits only the closing
# ``</think>``; the in-process client needs the flag to know not to expect a literal opener.
#
# Getting this wrong is silent: the non-streaming parser still splits correctly whenever
# ``</think>`` is present, so the flag only bites when a thinking block is *truncated* before its
# close (token budget exhausted), where a False-but-should-be-True entry returns raw reasoning as
# if it were the answer. That is why this is pinned here rather than left to review.
#
# To re-verify a member, check the tail of its chat template for what ``add_generation_prompt``
# emits. Note the distinctions that make several entries correctly False:
#   * ``QWEN_3_8B`` / ``SMOLLM3_3B`` emit only the *closed* ``<think>\n\n</think>`` (thinking off).
#   * ``GEMMA_4_*`` use ``<|channel>thought`` framing, not ``<think>`` tags at all.
#   * ``GPT_OSS_20B`` emits no opener.
_EXPECTED_THINK_OPENER: dict[str, bool] = {
    "QWEN_3_8_27B": True,  # template tail: {{- '<think>\n' }}
    "QWEN_3_8_27B_FP8": True,
    "QWEN_3_6_27B": True,  # same template shape as 3.5/3.8; was False until v0.13.2
    "QWEN_3_6_27B_FP8": True,
    "QWEN_3_5_9B": True,
    "QWEN_3_8B": False,  # closed <think>\n\n</think>\n\n only, and only when thinking is off
    "GEMMA_4_E4B": False,  # <|channel>thought framing
    "GEMMA_4_12B": False,
    "GPT_OSS_20B": False,  # no opener
    "DEEPSEEK_R1_8B": True,  # template tail: {{'<｜Assistant｜><think>\n'}}
    "SMOLLM3_3B": False,  # closed <think>\n\n</think>\n in non-think mode
}


def _hf_thinking_members() -> dict[str, object]:
    if not getattr(models, "HAS_HF", False):
        return {}
    return {member.name: member for member in models.HuggingFaceModel if member.supports_thinking}


@pytest.mark.skipif(not getattr(models, "HAS_HF", False), reason="requires the hf extra")
def test_hf_thinking_models_are_all_pinned():
    # A new thinking model must make a deliberate think_opener_in_prompt decision rather than
    # inheriting the False default unexamined, which is how the 3.6 entries went wrong.
    unpinned = sorted(set(_hf_thinking_members()) - set(_EXPECTED_THINK_OPENER))
    assert not unpinned, (
        f"HuggingFace thinking model(s) {unpinned} have no _EXPECTED_THINK_OPENER entry. "
        f"Check the repo's chat template for whether add_generation_prompt appends a bare "
        f"'<think>' and pin the expected value."
    )
    stale = sorted(set(_EXPECTED_THINK_OPENER) - set(_hf_thinking_members()))
    assert not stale, f"remove _EXPECTED_THINK_OPENER entries for models no longer in the catalog: {stale}"


@pytest.mark.skipif(not getattr(models, "HAS_HF", False), reason="requires the hf extra")
@pytest.mark.parametrize("member_name", sorted(_EXPECTED_THINK_OPENER))
def test_hf_think_opener_matches_template(member_name):
    members = _hf_thinking_members()
    if member_name not in members:
        pytest.skip(f"{member_name} not in catalog")
    expected = _EXPECTED_THINK_OPENER[member_name]
    actual = members[member_name].think_opener_in_prompt
    assert actual == expected, (
        f"HuggingFaceModel.{member_name}.think_opener_in_prompt is {actual}, expected {expected} "
        f"per its chat template. A True template with a False flag silently returns truncated "
        f"reasoning as the answer."
    )


def test_allowlisted_divergences_are_still_divergent():
    # Keep both allowlists honest: if an entry has been reconciled (no longer diverges),
    # it should be removed rather than lingering as noise.
    shared = _shared_flag_values()
    allowlisted = set(_INTENTIONAL_DIVERGENCES) | _SUSPECTED_OVERSIGHTS
    stale = [key for key in allowlisted if key in shared and len(set(shared[key].values())) <= 1]
    assert not stale, f"remove reconciled entries from the allowlists: {stale}"
