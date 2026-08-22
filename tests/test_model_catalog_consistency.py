"""Cross-provider consistency of the text-model catalogs.

The same model shipped under multiple providers uses a provider-specific
``ModelSpec.id`` -- the *wire* identifier each server actually accepts
(``qwen3:8b`` for Ollama, ``Qwen/Qwen3-8B`` for vLLM/HF, ``qwen3-8b.gguf`` for
llama-server, ...). Those ids are not interchangeable and are correctly
different per provider.

The cross-provider *identity* is the enum-member **name** (``QWEN_3_8B``), which
is what ``resolve_model_enum`` uses to search every provider enum for a bare
name. A shared name must describe the *same model*, but that agreement is now
**structural** rather than asserted here: intrinsic capability flags (tools /
thinking / vision / thinking_levels / thinking_optional) live once, in
``aimu/models/_catalog.py``'s ``MODEL_FACTS`` table, keyed on the member name.
Every catalog entry is a ``Wire`` that resolves against that one entry, so
there is no per-provider restatement left to drift out of sync -- see that
module's docstring for the intrinsic-vs-serving-path rationale in full.

What this file still guards, because ``MODEL_FACTS`` agreement alone can't
catch it:

* The deliberate exceptions -- a serving path that overrides an intrinsic fact
  because it cannot deliver the capability the weights have (e.g. llama-cpp's
  GGUF path lacking an mmproj projector for Gemma 4's vision). Each such
  override lives on the member as ``Wire(..., why=...)``, and
  ``test_overrides_are_explained`` pins the exact set below, so a new one is a
  reviewed act rather than a silent addition.
* Enum-internal footguns unrelated to ``MODEL_FACTS`` (``test_no_silent_enum_aliases``).
* The HuggingFace chat-template pins (``think_opener_in_prompt``), which have
  nothing to do with cross-provider agreement and are verified against each
  repo's own template.
"""

from __future__ import annotations

import pytest

import aimu.models as models

# Intrinsic model capabilities: properties of the weights, not the serving path. Used only
# to recognize a text-model enum during discovery (every text-model member exposes these).
INTRINSIC_FLAGS = (
    "supports_tools",
    "supports_thinking",
    "supports_vision",
    "thinking_levels",
    "thinking_optional",
)


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


def test_overrides_are_explained():
    """Every intrinsic-flag override carries a rationale, and the set is pinned.

    Divergence used to be recorded in _INTENTIONAL_DIVERGENCES here, one step removed from
    the member it described. It now lives on the member as Wire(..., why=...), which
    resolve_wire enforces. This test pins the *set* so a new override is a reviewed act.
    """
    from aimu.models._catalog import SERVING_PATH_FLAGS

    found = set()
    for enum_name, enum in _text_model_enums().items():
        for member_name, raw in enum.__members__.items():
            wire = getattr(raw, "_wire", None)
            if wire is None:
                continue
            for flag in set(wire.overrides) - SERVING_PATH_FLAGS:
                assert wire.why, f"{enum_name}.{member_name} overrides {flag} with no why="
                found.add((enum_name, member_name, flag))

    assert found == EXPECTED_OVERRIDES, (
        f"override set changed.\n  added: {sorted(found - EXPECTED_OVERRIDES)}\n"
        f"  removed: {sorted(EXPECTED_OVERRIDES - found)}"
    )


EXPECTED_OVERRIDES = {
    ("HFOpenAIModel", "GEMMA_3_12B", "tools"),
    ("LlamaServerOpenAIModel", "GEMMA_3_12B", "tools"),
    ("SGLangOpenAIModel", "GEMMA_3_12B", "tools"),
    ("VLLMOpenAIModel", "GEMMA_3_12B", "tools"),
    ("LlamaCppModel", "GEMMA_4_12B", "vision"),
}


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
