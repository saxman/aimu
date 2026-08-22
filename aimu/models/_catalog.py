"""Intrinsic model capabilities, stated once and shared by every local-runtime catalog.

A model's ``tools`` / ``thinking`` / ``vision`` flags and its card-specified sampling
profiles are properties of the *weights*: they are the same model whether it is served by
Ollama, vLLM, or llama.cpp. Restating them per provider is how they drift, so they live
here, keyed on the cross-provider enum-member name that ``resolve_model_enum`` matches on.

What is deliberately NOT here is anything that is a property of the *runtime*:
``structured_output`` (Ollama grammar-enforces JSON for any model; a raw vLLM server does
not) and ``audio`` (only the in-process HuggingFace path exposes audio input). Those are
declared per catalog, as is any case where a serving path cannot deliver an intrinsic
capability -- see ``Wire``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

from ._base.text import ModelSpec


@dataclass(frozen=True)
class ModelFacts:
    """What a model is, independent of who serves it."""

    tools: bool = False
    thinking: bool = False
    vision: bool = False
    thinking_levels: bool = False
    thinking_optional: bool = True
    generation_kwargs: Optional[dict] = None
    nonthinking_generation_kwargs: Optional[dict] = None


# Flags a catalog declares for itself rather than overriding: they describe the runtime.
SERVING_PATH_FLAGS = frozenset({"structured_output", "audio"})


class Wire:
    """One catalog's entry for a model: the id that server accepts, plus its own flags.

    ``Wire`` is deliberately not a tuple or NamedTuple: ``enum`` unpacks a tuple member
    value into multiple ``__init__`` arguments, which would break every catalog.

    Overriding an *intrinsic* flag requires ``why=``, because a serving path that cannot
    deliver a capability the weights have is an exception a reader needs explained at the
    point it applies. Declaring a serving-path flag does not, since that is the catalog
    stating its own capability rather than contradicting a fact.
    """

    __slots__ = ("id", "why", "overrides")

    def __init__(self, id: str, why: Optional[str] = None, **overrides: Any):
        self.id = id
        self.why = why
        self.overrides = overrides

    def __repr__(self) -> str:
        return f"Wire({self.id!r}, why={self.why!r}, **{self.overrides!r})"

    # Id-only, exactly like ModelSpec. This is load-bearing, not cosmetic: enum detects a
    # duplicate member by comparing values, so a Wire without these would make two members
    # sharing an id stop aliasing -- which sounds like an improvement but is not. It would
    # leave two live members with the same ``_value_``, so lookup by value silently returns
    # one of them, and it would make test_no_silent_enum_aliases vacuous rather than failing.
    # Keeping ModelSpec's semantics keeps that guard meaningful and this refactor inert.
    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Wire):
            return self.id == other.id
        return NotImplemented


_SPEC_FIELDS = {f.name for f in fields(ModelSpec)} - {"id"}


def resolve_wire(name: str, wire: Wire) -> ModelSpec:
    """Build the ``ModelSpec`` for catalog member *name* from the shared facts + *wire*."""
    try:
        facts = MODEL_FACTS[name]
    except KeyError:
        raise KeyError(
            f"{name} has no MODEL_FACTS entry. Add its intrinsic capabilities to "
            f"aimu/models/_catalog.py before cataloguing it under a provider."
        ) from None

    unknown = sorted(set(wire.overrides) - _SPEC_FIELDS)
    if unknown:
        raise ValueError(
            f"{name}: unknown override(s) {unknown}. A silently ignored override reads "
            f"exactly like an applied one. Valid keys: {sorted(_SPEC_FIELDS)}."
        )

    contradicted = sorted(set(wire.overrides) - SERVING_PATH_FLAGS)
    if contradicted and not wire.why:
        raise ValueError(
            f"{name}: overriding intrinsic flag(s) {contradicted} requires why=, naming what "
            f"about this serving path cannot deliver the capability the weights have."
        )

    # Build from facts first, then let overrides win, so an override (serving-path or
    # intrinsic-with-why) replaces rather than collides with the corresponding fact.
    resolved: dict[str, Any] = {
        "tools": facts.tools,
        "thinking": facts.thinking,
        "vision": facts.vision,
        "thinking_levels": facts.thinking_levels,
        "thinking_optional": facts.thinking_optional,
        "generation_kwargs": dict(facts.generation_kwargs) if facts.generation_kwargs else None,
        "nonthinking_generation_kwargs": (
            dict(facts.nonthinking_generation_kwargs) if facts.nonthinking_generation_kwargs else None
        ),
    }
    resolved.update(wire.overrides)
    return ModelSpec(id=wire.id, **resolved)


# Sampling profiles, copied verbatim (with their comments) from
# aimu/models/providers/ollama.py:25-50, the only place they were previously declared as
# shared constants. They record which model card each value came from and when it was
# verified; that provenance does not change by moving where the constant lives.
_GEMMA_KWARGS = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
# Qwen 3.6 27B and 3.8 share a thinking-mode profile; 3.5 and 3.6 35B-A3B each differ only in
# presence_penalty (1.5 rather than 0.0), per their own cards.
# Values are from each model card's thinking-mode row, verified 2026-08-17.
_QWEN_THINKING_KWARGS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}
_QWEN_3_5_THINKING_KWARGS = {**_QWEN_THINKING_KWARGS, "presence_penalty": 1.5}
# Qwen 3.6 35B-A3B's card gives a different presence_penalty for its "general tasks"
# thinking row (1.5) than the 27B's (0.0), verified against both cards 2026-08-17.
_QWEN_3_6_35B_THINKING_KWARGS = {**_QWEN_THINKING_KWARGS, "presence_penalty": 1.5}
# Every Qwen 3.5 / 3.6 / 3.8 card specifies the same instruct-mode row.
_QWEN_INSTRUCT_KWARGS = {
    "temperature": 0.7,
    "top_p": 0.80,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}
_MUSE_GLIMMER_KWARGS = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}


MODEL_FACTS: dict[str, ModelFacts] = {
    # Alibaba. Qwen 3.5/3.6/3.8 are a unified vision-language family: vision is built into the
    # base weights rather than shipped as a separate -VL variant, so every serving path that can
    # run the weights at all can serve images from them.
    "QWEN_3_8_27B": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        thinking_levels=True,
        generation_kwargs=_QWEN_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    # Quantization variants of the same weights; the intrinsic facts (and the card's sampling
    # profile) are identical to the bare QWEN_3_8_27B entry above.
    "QWEN_3_8_27B_4BIT": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        thinking_levels=True,
        generation_kwargs=_QWEN_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_8_27B_8BIT": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        thinking_levels=True,
        generation_kwargs=_QWEN_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_8_27B_BF16": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        thinking_levels=True,
        generation_kwargs=_QWEN_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_8_27B_FP8": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        thinking_levels=True,
        generation_kwargs=_QWEN_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_6_35B": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        generation_kwargs=_QWEN_3_6_35B_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_6_35B_4BIT": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        generation_kwargs=_QWEN_3_6_35B_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_6_35B_8BIT": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        generation_kwargs=_QWEN_3_6_35B_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_6_35B_BF16": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        generation_kwargs=_QWEN_3_6_35B_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_6_27B": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        generation_kwargs=_QWEN_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    # Quantization variants of QWEN_3_6_27B (MLX); intrinsic facts identical to the bare entry.
    "QWEN_3_6_27B_4BIT": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        generation_kwargs=_QWEN_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_6_27B_8BIT": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        generation_kwargs=_QWEN_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_6_27B_BF16": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        generation_kwargs=_QWEN_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_6_27B_FP8": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        generation_kwargs=_QWEN_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_5_9B": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        generation_kwargs=_QWEN_3_5_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    # Quantization variants of QWEN_3_5_9B (MLX); intrinsic facts identical to the bare entry.
    "QWEN_3_5_9B_4BIT": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        generation_kwargs=_QWEN_3_5_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_5_9B_8BIT": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        generation_kwargs=_QWEN_3_5_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_5_9B_BF16": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        generation_kwargs=_QWEN_3_5_THINKING_KWARGS,
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,
    ),
    "QWEN_3_32B": ModelFacts(tools=True, thinking=True),
    # Quantization variants of QWEN_3_32B (MLX); intrinsic facts identical to the bare entry.
    "QWEN_3_32B_4BIT": ModelFacts(tools=True, thinking=True),
    "QWEN_3_32B_8BIT": ModelFacts(tools=True, thinking=True),
    "QWEN_3_32B_BF16": ModelFacts(tools=True, thinking=True),
    # Plain "Qwen 3" (not the 3.5/3.6/3.8 vision-language family, hence vision=False). Only the
    # HuggingFace catalog ever declared a sampling profile for it; transcribed verbatim from
    # aimu/models/providers/hf/text.py.
    "QWEN_3_8B": ModelFacts(
        tools=True,
        thinking=True,
        generation_kwargs={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
    ),
    # Quantization variants of QWEN_3_8B (MLX); intrinsic facts identical to the bare entry.
    "QWEN_3_8B_4BIT": ModelFacts(
        tools=True,
        thinking=True,
        generation_kwargs={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
    ),
    "QWEN_3_8B_8BIT": ModelFacts(
        tools=True,
        thinking=True,
        generation_kwargs={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
    ),
    "QWEN_3_8B_BF16": ModelFacts(
        tools=True,
        thinking=True,
        generation_kwargs={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
    ),
    "QWEN_3_4B": ModelFacts(tools=True, thinking=True),
    # Quantization variants of QWEN_3_4B (MLX); intrinsic facts identical to the bare entry.
    "QWEN_3_4B_4BIT": ModelFacts(tools=True, thinking=True),
    "QWEN_3_4B_8BIT": ModelFacts(tools=True, thinking=True),
    "QWEN_3_4B_BF16": ModelFacts(tools=True, thinking=True),
    # Google
    "GEMMA_4_E4B": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    # Quantization variants of GEMMA_4_E4B (MLX); intrinsic facts identical to the bare entry.
    "GEMMA_4_E4B_4BIT": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    "GEMMA_4_E4B_8BIT": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    "GEMMA_4_E4B_BF16": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    # GEMMA_4_12B: LlamaCppModel declares vision=False (no default mmproj projector for this
    # size), disagreeing with every other catalog's vision=True. That is a serving-path
    # limitation, not an intrinsic fact, so the fact is vision=True and LlamaCppModel's entry
    # will carry a Wire(..., why="no mmproj projector", vision=False) override (Task 4).
    "GEMMA_4_12B": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    # Quantization variants of GEMMA_4_12B (MLX); intrinsic facts identical to the bare entry.
    "GEMMA_4_12B_4BIT": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    "GEMMA_4_12B_8BIT": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    "GEMMA_4_12B_BF16": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    "GEMMA_4_26B": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    # Quantization variants of GEMMA_4_26B (MLX); intrinsic facts identical to the bare entry.
    "GEMMA_4_26B_4BIT": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    "GEMMA_4_26B_8BIT": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    "GEMMA_4_26B_BF16": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    "GEMMA_4_31B": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    # Quantization variants of GEMMA_4_31B (MLX); intrinsic facts identical to the bare entry.
    "GEMMA_4_31B_4BIT": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    "GEMMA_4_31B_8BIT": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    "GEMMA_4_31B_BF16": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS),
    # GEMMA_3_12B: the OpenAI-compat local-server catalogs (HFOpenAI/LlamaServerOpenAI/SGLangOpenAI/
    # VLLMOpenAI) declare tools=True, disagreeing with the in-process HuggingFace/Ollama catalogs'
    # tools=False (neither in-process path has a tool-parse format for this model). The fact is
    # tools=False; the four OpenAI-compat catalogs will carry a Wire(..., why=..., tools=True)
    # override (Task 4).
    "GEMMA_3_12B": ModelFacts(tools=False, thinking=False, vision=True),
    # Quantization variants of GEMMA_3_12B (MLX). Facts mirror the bare entry (tools=False
    # intrinsically); the MLX-serving catalogs override tools=True on the Wire, same as their
    # non-quantized OpenAI-compat siblings, since that override is a serving-path fact, not an
    # intrinsic one.
    "GEMMA_3_12B_4BIT": ModelFacts(tools=False, thinking=False, vision=True),
    "GEMMA_3_12B_8BIT": ModelFacts(tools=False, thinking=False, vision=True),
    "GEMMA_3_12B_BF16": ModelFacts(tools=False, thinking=False, vision=True),
    # Zhipu
    "GLM_4_7_FLASH_31B_Q4": ModelFacts(thinking=True),
    # Quantization variants of GLM_4_7_FLASH_31B_Q4 (MLX); intrinsic facts identical to the
    # bare entry.
    "GLM_4_7_FLASH_31B_Q4_4BIT": ModelFacts(thinking=True),
    "GLM_4_7_FLASH_31B_Q4_8BIT": ModelFacts(thinking=True),
    "GLM_4_7_FLASH_31B_Q4_BF16": ModelFacts(thinking=True),
    # OpenAI (open-weight)
    "GPT_OSS_20B": ModelFacts(
        tools=True, thinking=True, generation_kwargs={"temperature": 1.0, "top_p": 1.0, "top_k": 0}
    ),
    # gpt-oss ships natively quantized to mxfp4, so mlx-community re-packages it as Q4/Q8
    # requantizations of that native format rather than plain 4bit/8bit/bf16 builds -- hence
    # the MXFP4_Q4/MXFP4_Q8 member names (no bf16 build exists). Facts mirror the bare entry.
    "GPT_OSS_20B_MXFP4_Q4": ModelFacts(
        tools=True, thinking=True, generation_kwargs={"temperature": 1.0, "top_p": 1.0, "top_k": 0}
    ),
    "GPT_OSS_20B_MXFP4_Q8": ModelFacts(
        tools=True, thinking=True, generation_kwargs={"temperature": 1.0, "top_p": 1.0, "top_k": 0}
    ),
    # Meta
    "LLAMA_3_1_8B": ModelFacts(tools=True),
    # Quantization variants of LLAMA_3_1_8B (MLX); intrinsic facts identical to the bare entry.
    "LLAMA_3_1_8B_4BIT": ModelFacts(tools=True),
    "LLAMA_3_1_8B_8BIT": ModelFacts(tools=True),
    "LLAMA_3_1_8B_BF16": ModelFacts(tools=True),
    "LLAMA_3_2_3B": ModelFacts(tools=True),
    # Quantization variants of LLAMA_3_2_3B (MLX); intrinsic facts identical to the bare entry.
    "LLAMA_3_2_3B_4BIT": ModelFacts(tools=True),
    "LLAMA_3_2_3B_8BIT": ModelFacts(tools=True),
    "LLAMA_3_2_3B_BF16": ModelFacts(tools=True),
    # Mistral
    "MAGISTRAL_SMALL": ModelFacts(tools=True, generation_kwargs={"top_p": 0.95, "temperature": 0.7}),
    "MAGISTRAL_SMALL_24B": ModelFacts(tools=True, thinking=True),
    # mlx-community publishes only a bf16 build under the plain naming convention (a 4bit-DWQ
    # variant also exists, but DWQ is a distinct quantization method, not this catalog's plain
    # 4bit/8bit convention, so it is not catalogued here). Facts mirror the bare entry.
    "MAGISTRAL_SMALL_24B_BF16": ModelFacts(tools=True, thinking=True),
    "MINISTRAL_3_14B": ModelFacts(tools=True),
    # Quantization variants of MINISTRAL_3_14B (MLX); intrinsic facts identical to the bare entry.
    "MINISTRAL_3_14B_4BIT": ModelFacts(tools=True),
    "MINISTRAL_3_14B_8BIT": ModelFacts(tools=True),
    "MINISTRAL_3_14B_BF16": ModelFacts(tools=True),
    "MISTRAL_7B": ModelFacts(tools=True),
    # mlx-community publishes only 4bit/8bit for this model (no bf16 build confirmed). Facts
    # mirror the bare entry.
    "MISTRAL_7B_4BIT": ModelFacts(tools=True),
    "MISTRAL_7B_8BIT": ModelFacts(tools=True),
    "MISTRAL_NEMO_12B": ModelFacts(tools=True, generation_kwargs={"temperature": 0.3}),
    # Community MLX conversion. Quantization variants share the base model's intrinsic facts
    # and sampling profile.
    "MUSE_GLIMMER_30B": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_MUSE_GLIMMER_KWARGS),
    "MUSE_GLIMMER_30B_4BIT": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_MUSE_GLIMMER_KWARGS),
    "MUSE_GLIMMER_30B_8BIT": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_MUSE_GLIMMER_KWARGS),
    "MUSE_GLIMMER_30B_BF16": ModelFacts(tools=True, thinking=True, vision=True, generation_kwargs=_MUSE_GLIMMER_KWARGS),
    # NVIDIA
    "NEMOTRON_3_NANO_30B": ModelFacts(tools=True, thinking=True),
    # Quantization variants of NEMOTRON_3_NANO_30B (MLX). mlx-community's confirmed matching
    # 4Bit/8Bit/BF16 trio for this model carries an extra "-MLX-" infix and Titlecase quant
    # token in its own repo naming (see OMLXOpenAIModel); the member names here stay the plain
    # _4BIT/_8BIT/_BF16 convention regardless. Facts mirror the bare entry.
    "NEMOTRON_3_NANO_30B_4BIT": ModelFacts(tools=True, thinking=True),
    "NEMOTRON_3_NANO_30B_8BIT": ModelFacts(tools=True, thinking=True),
    "NEMOTRON_3_NANO_30B_BF16": ModelFacts(tools=True, thinking=True),
    "NEMOTRON_CASCADE_2_30B": ModelFacts(tools=True, thinking=True),
    # Quantization variants of NEMOTRON_CASCADE_2_30B (MLX); intrinsic facts identical to the
    # bare entry.
    "NEMOTRON_CASCADE_2_30B_4BIT": ModelFacts(tools=True, thinking=True),
    "NEMOTRON_CASCADE_2_30B_8BIT": ModelFacts(tools=True, thinking=True),
    "NEMOTRON_CASCADE_2_30B_BF16": ModelFacts(tools=True, thinking=True),
    "NEMOTRON_H_8B": ModelFacts(tools=True),
    # Microsoft. PHI_4_MINI and PHI_4_MINI_3_8B were two catalog names for one model, collapsed
    # onto PHI_4_MINI_3_8B in Task 6. tools=True per microsoft/Phi-4-mini-instruct's chat
    # template (tokenizer_config.json emits a <|tool|>{tools}<|/tool|> block for a system
    # message carrying tools, i.e. the model is trained for function calling) and Ollama's own
    # registry page for phi4-mini, which carries the "tools" capability badge. Ollama's prior
    # tools=False here was a stale entry, not a serving-path limitation.
    "PHI_4_14B": ModelFacts(),
    # Quantization variants of PHI_4_14B (MLX); intrinsic facts identical to the bare entry.
    "PHI_4_14B_4BIT": ModelFacts(),
    "PHI_4_14B_8BIT": ModelFacts(),
    "PHI_4_14B_BF16": ModelFacts(),
    "PHI_4_MINI_3_8B": ModelFacts(tools=True),
    # mlx-community publishes 4bit/8bit plus a distinctly-named "-mlx-fp16" build (no plain
    # bf16 repo), hence the FP16 (not BF16) member suffix. Facts mirror the bare entry.
    "PHI_4_MINI_3_8B_4BIT": ModelFacts(tools=True),
    "PHI_4_MINI_3_8B_8BIT": ModelFacts(tools=True),
    "PHI_4_MINI_3_8B_FP16": ModelFacts(tools=True),
    # HuggingFace
    "SMOLLM2_1_7B": ModelFacts(),
    # No mlx-community *quantized* build exists for this model (only the unquantized
    # mlx-community/SmolLM2-1.7B-Instruct repo, confirmed present but out of scope for Task
    # 11's quant-suffixed fill), so no MLX quant member is catalogued for it.
    "SMOLLM3_3B": ModelFacts(tools=True, thinking=True, generation_kwargs={"temperature": 0.6, "top_p": 0.95}),
    # DeepSeek
    "DEEPSEEK_R1_7B": ModelFacts(thinking=True),
    # Quantization variants of DEEPSEEK_R1_7B (MLX); intrinsic facts identical to the bare entry.
    "DEEPSEEK_R1_7B_4BIT": ModelFacts(thinking=True),
    "DEEPSEEK_R1_7B_8BIT": ModelFacts(thinking=True),
    "DEEPSEEK_R1_7B_BF16": ModelFacts(thinking=True),
    "DEEPSEEK_R1_8B": ModelFacts(thinking=True, generation_kwargs={"temperature": 0.6}),
    # Quantization variants of DEEPSEEK_R1_8B (MLX); intrinsic facts identical to the bare entry.
    "DEEPSEEK_R1_8B_4BIT": ModelFacts(thinking=True, generation_kwargs={"temperature": 0.6}),
    "DEEPSEEK_R1_8B_8BIT": ModelFacts(thinking=True, generation_kwargs={"temperature": 0.6}),
    "DEEPSEEK_R1_8B_BF16": ModelFacts(thinking=True, generation_kwargs={"temperature": 0.6}),
}
