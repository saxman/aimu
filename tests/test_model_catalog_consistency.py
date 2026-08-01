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

The remaining intrinsic flags (tools / thinking / vision) are checked, with two
escape hatches below: ``_INTENTIONAL_DIVERGENCES`` (a shared name whose flag
*legitimately* differs by serving path) and ``_SUSPECTED_OVERSIGHTS`` (a
divergence that looks like a bug, frozen so the suite is green while the fix is
pending).
"""

from __future__ import annotations

from collections import defaultdict

import pytest

import aimu.models as models

# Intrinsic model capabilities: properties of the weights, not the serving path,
# so they must agree wherever a model name appears.
INTRINSIC_FLAGS = ("supports_tools", "supports_thinking", "supports_vision")

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


def test_allowlisted_divergences_are_still_divergent():
    # Keep both allowlists honest: if an entry has been reconciled (no longer diverges),
    # it should be removed rather than lingering as noise.
    shared = _shared_flag_values()
    allowlisted = set(_INTENTIONAL_DIVERGENCES) | _SUSPECTED_OVERSIGHTS
    stale = [key for key in allowlisted if key in shared and len(set(shared[key].values())) <= 1]
    assert not stale, f"remove reconciled entries from the allowlists: {stale}"
