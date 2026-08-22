# Add a new model

To make a new model usable with `ModelClient`, add a member to that provider's `Model` enum. What the member's value looks like depends on which kind of catalog it is:

- **Cloud catalogs** (`AnthropicModel`, `OpenAIModel`, `GeminiModel`): each hosted model ships under exactly one provider, so the member's value is a plain `ModelSpec` carrying the model id and every capability flag directly. See [Basic case](#basic-case) below.
- **Local-runtime catalogs** (`OllamaModel`, `HuggingFaceModel`, `LlamaCppModel`, and the seven `*OpenAIModel` local-server catalogs): the same weights are often reachable through several of these, so a model's *intrinsic* capabilities (`tools`/`thinking`/`vision`, the thinking-control fields, and card sampling profiles) are declared exactly once, in a shared `MODEL_FACTS` table, keyed on the cross-provider enum-member name. Each catalog's member value is a `Wire(id)` -- just the wire id that runtime's server accepts -- which resolves against `MODEL_FACTS` at class-construction time. See [Local-runtime catalogs](#local-runtime-catalogs) below.

!!! note "AIMU ships curated models only"
    Every modality requires the model to be a member of its provider enum (or a hand-built spec, see [Custom models](#custom-models)). Passing an arbitrary `"provider:some/unknown-repo"` string **raises** `ValueError`, rather than silently fabricating a spec with guessed capabilities. Capability flags (tools/thinking/vision, `pipeline_class`, `supports_negative_prompt`, voice/step defaults, and so on) can't be inferred reliably from a repo name, and a wrong guess causes hard-to-debug runtime failures. So the catalog is intentional: to use a new model, add it to the enum here.

## The same model under multiple providers

One model is often reachable through several local-runtime providers (Qwen3 8B runs on Ollama, vLLM, HuggingFace, llama.cpp, ...). AIMU keeps two things separate:

- **The wire id** (`Wire.id` on a local-runtime catalog, `ModelSpec.id` on a cloud catalog) is sent to that provider's server. It is legitimately different per provider and must match what the server accepts: `qwen3:8b` (Ollama), `Qwen/Qwen3-8B` (vLLM / HF-serve / SGLang), `qwen3-8b.gguf` (llama-server), `qwen3-8b` (LlamaCpp / LM Studio). Do **not** try to normalise these; a mismatched id just fails the request.
- **The enum-member name** (`QWEN_3_8B`) is the *cross-provider identity*. It is the same across every provider enum, and it is what [`resolve_model_enum`](../reference/api/models.md) searches when a caller passes a bare name (e.g. `"QWEN_3_8B"`), and what a local-runtime catalog's `Wire` looks up in `MODEL_FACTS`. Keep the name identical across providers for the same model so bare-name resolution finds every serving option.

### Keep capability flags consistent across providers

For local-runtime catalogs, this is now **structural** rather than something you have to remember: the *intrinsic* capability flags (`tools`, `thinking`, `vision`, `thinking_levels`, `thinking_optional`) live once in `MODEL_FACTS` (`aimu/models/_catalog.py`), keyed on the enum-member name, and every catalog's `Wire(id)` resolves against that same entry. Adding `QWEN_3_8B` to a new local-runtime catalog with `Wire("qwen3-8b")` picks up `thinking=True` automatically; there is no separate flag to restate or forget. `tests/test_model_catalog_consistency.py::test_every_local_runtime_member_has_a_wire` fails the suite if a local-runtime catalog member is left as a bare `ModelSpec` instead (which would silently exempt it from this guarantee).

Two flags stay **outside** `MODEL_FACTS` and are declared per catalog, because they genuinely describe the **serving path**, not the model:

- **`structured_output`**: Ollama grammar-enforces JSON for any model, so every `OllamaModel` member sets it; a raw vLLM/HF server does not. Declare it directly on the `Wire`, e.g. `Wire("qwen3:8b", structured_output=True)` -- no `why=` needed, since a serving-path flag doesn't contradict a fact.
- **`audio`**: some models (e.g. Gemma 4) support audio natively, but only certain serving paths expose audio input (the in-process HuggingFace client does; the OpenAI-compat server catalogs leave it `False` by design). Same syntax: `Wire(id, audio=True)`.

An *intrinsic* flag may still legitimately differ when a specific serving path cannot expose the capability, even though the model has it. Two live examples:

- `GEMMA_3_12B` has `tools=False` on the in-process HuggingFace and native-Ollama clients (they parse tool calls via a per-model format, and Gemma 3 has none assigned) but `tools=True` on the OpenAI-compat servers (they parse tool calls server-side).
- `GEMMA_4_12B` has `vision=False` on LlamaCpp (the default GGUF path loads no `mmproj` projector; vision needs one passed via `chat_handler=`) but `vision=True` on every other provider.

When you introduce a divergence like these, override the flag on that catalog's `Wire` with a `why=` naming what the serving path can (or can't) deliver relative to the shared fact -- e.g. `GEMMA_3_12B`'s intrinsic fact is `tools=False` (no `ToolCallFormat` assigned for the in-process HF path), so the OpenAI-compat server catalogs override it: `Wire("google/gemma-3-12b-it", why="OpenAI-compat servers parse tool calls server-side", tools=True)`. `Wire` **enforces this at import time**: overriding an intrinsic flag without `why=` raises `ValueError` before the module even finishes loading, so a silent capability mismatch can't ship. `tests/test_model_catalog_consistency.py::test_overrides_are_explained` additionally pins the exact set of overrides in `EXPECTED_OVERRIDES`, so a new one is a reviewed, deliberate addition to that set rather than something that just starts passing.

## Basic case

Cloud catalogs (`AnthropicModel`, `OpenAIModel`, `GeminiModel`) are single-provider, so each enum value is a plain `ModelSpec` carrying the id and every capability flag directly:

```python
# aimu/models/providers/anthropic.py
from ..base import Model, ModelSpec

class AnthropicModel(Model):
    CLAUDE_SONNET_4_6 = ModelSpec("claude-sonnet-4-6", tools=True, thinking=True, vision=True)
    # Add a new model:
    CLAUDE_OPUS_5 = ModelSpec("claude-opus-5", tools=True, thinking=True, vision=True)
```

That's it. The model id (`"claude-opus-5"`) is what gets sent to the provider; the capability flags drive `is_tool_using_model`, `is_thinking_model`, `is_vision_model`, and the derived `TOOL_MODELS` / `THINKING_MODELS` / `VISION_MODELS` classproperties.

A bare `ModelSpec` defaults the Anthropic reasoning request to `ThinkingStyle.ENABLED` (`budget_tokens`). New Claude reasoning models (Opus 4.7+ and Fable 5) require the adaptive request shape and **400 on the `enabled` form**, so define those with a `ThinkingStyle.ADAPTIVE` extra (see [Anthropic provider-specific extras](#anthropic-provider-specific-extras) below).

## Local-runtime catalogs

`OllamaModel`, `HuggingFaceModel`, `LlamaCppModel`, and the seven `*OpenAIModel` local-server catalogs (`OllamaOpenAIModel`, `LMStudioOpenAIModel`, `VLLMOpenAIModel`, `SGLangOpenAIModel`, `HFOpenAIModel`, `LlamaServerOpenAIModel`, `OMLXOpenAIModel`) don't carry a `ModelSpec` directly. Instead, each member is a `Wire(id)` that resolves against the shared `MODEL_FACTS` table (`aimu/models/_catalog.py`), keyed on the cross-provider enum-member name. Adding a model to one of these catalogs is a two-step process:

**1. Add the model's intrinsic facts to `MODEL_FACTS`, once** -- skip this step if another local-runtime catalog already carries the same enum-member name (the facts already exist):

```python
# aimu/models/_catalog.py
MODEL_FACTS: dict[str, ModelFacts] = {
    ...
    "QWEN_3_8B": ModelFacts(
        tools=True,
        thinking=True,
        generation_kwargs={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
    ),
}
```

**2. Add a `Wire(id)` to the provider catalog** -- just the wire id that runtime's server accepts; the capability flags and sampling profile resolve from `MODEL_FACTS` automatically:

```python
# aimu/models/providers/ollama.py
from ..._catalog import Wire

class OllamaModel(Model):
    QWEN_3_8B = Wire("qwen3:8b")
```

Only pass a keyword to `Wire` when this serving path can't deliver an intrinsic capability the facts declare (with `why=`, see [Keep capability flags consistent across providers](#keep-capability-flags-consistent-across-providers) above), or when declaring a serving-path-only flag (`structured_output`, `audio`; no `why=` needed).

## With custom generation kwargs

Some models work best with specific sampling parameters. `generation_kwargs` is an intrinsic property of the weights (the card's tuned sampling row), so on a **local-runtime catalog** it belongs in `MODEL_FACTS`, not on the individual `Wire`s -- every catalog serving that name picks it up automatically:

```python
# aimu/models/_catalog.py
MODEL_FACTS: dict[str, ModelFacts] = {
    ...
    "QWEN_3_8B": ModelFacts(
        tools=True,
        thinking=True,
        generation_kwargs={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
    ),
}
```

On a **cloud catalog**, where there is no shared table, set it directly on the `ModelSpec`:

```python
# aimu/models/providers/anthropic.py
CLAUDE_OPUS_5 = ModelSpec(
    "claude-opus-5",
    tools=True,
    thinking=True,
    generation_kwargs={"temperature": 1.0},
)
```

`generation_kwargs` is merged per key on every `chat()` / `generate()` call, never used as a
replacement. Four tiers apply, lowest precedence first, identically on every provider:

| Tier | Source | Set by |
|------|--------|--------|
| 1 | `Client.DEFAULT_GENERATE_KWARGS` | the library, for parameters nobody else sets (`max_tokens`) |
| 2 | `ModelSpec.generation_kwargs` (or `nonthinking_generation_kwargs`) | **you, here** -- via `MODEL_FACTS` on a local-runtime catalog, or directly on a cloud catalog's `ModelSpec` |
| 3 | `client.default_generate_kwargs` | the user, for every call on that client instance |
| 4 | `chat(generate_kwargs={...})` | the user, for one call |

Tier 1 sits *under* your profile deliberately, so the library's own `temperature=0.1` cannot
quietly beat a card's tuned value. Tiers 3 and 4 sit over it because they are explicit
instructions from the caller. The chain is applied by `_resolve_generate_kwargs()` (shared via `_GenerateKwargsMixin`, backed
by `merge_generate_kwargs()` in `aimu/models/_internal/generate_kwargs.py`),
so a profile you add here takes effect wherever the model is served. (Before v0.15 an explicit
`generate_kwargs` *replaced* the profile wholesale on the Ollama and HuggingFace paths, silently
discarding the model's tuned sampling; until v0.15.1 the Anthropic, OpenAI-compatible, and
llama.cpp paths ignored the profile entirely.)

### Thinking-control fields

Three optional fields describe how a reasoning model can be *steered*, and back the portable
[`thinking=`](control-thinking.md) parameter. Leaving them at their defaults is always safe.
They are intrinsic (properties of the weights), so on a local-runtime catalog they belong in
`MODEL_FACTS` alongside `tools`/`thinking`/`vision`:

```python
# aimu/models/_catalog.py
MODEL_FACTS: dict[str, ModelFacts] = {
    ...
    "QWEN_3_8_27B": ModelFacts(
        tools=True,
        thinking=True,
        vision=True,
        thinking_levels=True,                                   # accepts low / medium / high
        generation_kwargs=_QWEN_THINKING_KWARGS,                # sampling while reasoning
        nonthinking_generation_kwargs=_QWEN_INSTRUCT_KWARGS,     # sampling with reasoning off
    ),
}
```

(On a cloud catalog, set the same fields directly on the `ModelSpec`, as in [With custom generation kwargs](#with-custom-generation-kwargs) above.)

- **`thinking_levels`** (default `False`): the model accepts an effort level. **Under-declare
  this.** A missing declaration degrades to a warning and a correct answer, because the
  resolver drops the level and the model answers at its default effort. A wrong declaration
  risks emitting a value the backend rejects: Qwen 3.8's chat template calls
  `raise_exception` on an effort outside `{xhigh, medium, low}`, and Google's endpoint accepts
  `minimal/low/medium/high/none` but not `xhigh`. Set it only once you have checked what the
  model's template or API actually accepts.
- **`thinking_optional`** (default `True`): set `False` when the model *always* reasons and
  cannot be turned off, as `GEMINI_2_5_PRO` cannot. `thinking=False` against such a model
  warns and proceeds, so the caller is billed for reasoning they asked to skip.
- **`nonthinking_generation_kwargs`** (default `None`): the sampling profile the model card
  specifies for instruct mode, selected automatically when thinking resolves off. When it is
  `None`, `generation_kwargs` applies in both modes. Qwen's cards differ meaningfully between
  the two: 3.8 wants `temperature=1.0, top_p=0.95, presence_penalty=0.0` while reasoning and
  `temperature=0.7, top_p=0.80, presence_penalty=1.5` in instruct mode.

`ModelSpec.__post_init__` rejects `thinking_levels=True` or `thinking_optional=False` on a
member whose `thinking` is `False`, since neither means anything without reasoning.

Declaring the flag is only half the job: the provider client also has to be able to express it
on the wire. See [Control thinking effort](control-thinking.md) for the per-provider mechanisms
and [Add or update a provider](add-new-provider.md) for the reserved-key contract.

## HuggingFace provider-specific extras

`HuggingFaceModel` is a local-runtime catalog, so its members are `Wire`s like any other -- but it also carries extra positional values for the tool-call response format and a thinking-template flag:

```python
# aimu/models/providers/hf/text.py
from ..._catalog import Wire

class HuggingFaceModel(Model):
    QWEN_3_8B = (
        Wire("Qwen/Qwen3-8B"),
        ToolCallFormat.XML,                    # tool_call_format
    )
    QWEN_3_5_9B = (
        Wire("Qwen/Qwen3.5-9B"),
        ToolCallFormat.XML,
        True,                                  # think_opener_in_prompt
    )
```

`tools=True`/`thinking=True` for both still come from their `MODEL_FACTS["QWEN_3_8B"]` / `MODEL_FACTS["QWEN_3_5_9B"]` entries, exactly as in any other local-runtime catalog; `Wire` here just happens to sit inside a 2- or 3-tuple instead of standing alone as the whole member value.

Pick `ToolCallFormat.XML` / `JSON_OBJECT` / `JSON_ARRAY` / `BRACKETED` / `NA` based on how the base model emits tool calls. See existing entries in `aimu/models/providers/hf/text.py` for examples per model family.

## Anthropic provider-specific extras

`AnthropicModel` carries an optional `ThinkingStyle` extra that selects how the `thinking` request is built. Omit it (bare `ModelSpec`) for `ENABLED`/budget-style models; pass `ThinkingStyle.ADAPTIVE` for models that require adaptive thinking:

```python
# aimu/models/providers/anthropic.py
class AnthropicModel(Model):
    # Adaptive: thinking={"type": "adaptive", "display": "summarized"}; sampling params dropped.
    # Required by Opus 4.7+ and Fable 5 (the enabled form 400s here).
    CLAUDE_OPUS_4_8 = (ModelSpec("claude-opus-4-8", tools=True, thinking=True, vision=True), ThinkingStyle.ADAPTIVE)
    # Enabled (default): thinking={"type": "enabled", "budget_tokens": N}; the model always thinks.
    CLAUDE_HAIKU_4_5 = ModelSpec("claude-haiku-4-5", tools=True, thinking=True, vision=True)
```

Rule of thumb: Opus 4.7 and later, and Fable 5, are adaptive-only; Opus 4.6, Sonnet 4.6, and Haiku 4.5 use the budget form. Confirm a new model's support with the Models API (`client.models.retrieve(id).capabilities["thinking"]["types"]`); `adaptive.supported` / `enabled.supported` tell you which shape to pick.

## Image, audio, and speech models

The non-text modalities follow the same pattern with their own spec type and enum. Add a member to the provider enum in the relevant client module:

```python
# aimu/models/providers/hf/image.py: diffusers text-to-image
class HuggingFaceImageModel(ImageModel):
    SD_1_5 = HuggingFaceImageSpec("runwayml/stable-diffusion-v1-5", max_prompt_tokens=77)
    FLUX_2_KLEIN_4B = HuggingFaceImageSpec(
        "black-forest-labs/FLUX.2-klein-4B",
        pipeline_class="Flux2KleinPipeline",
        img2img_pipeline_class="Flux2KleinPipeline",
        img2img_uses_strength=False,        # unified pipeline conditions on the image directly
        supports_negative_prompt=False,     # Flux2KleinPipeline.__call__ has no negative_prompt param
        default_steps=4,
        max_prompt_tokens=512,              # T5-XXL encoder
    )

# aimu/models/providers/gemini/image.py: Gemini Nano Banana (cloud)
class GeminiImageModel(ImageModel):
    NANO_BANANA = GeminiImageSpec("gemini-2.5-flash-image")   # supports_negative_prompt defaults False

# aimu/models/providers/hf/audio.py: music / sound generation
class HuggingFaceAudioModel(AudioModel):
    MUSICGEN_SMALL = HuggingFaceAudioSpec("facebook/musicgen-small", pipeline_type="musicgen")
    AUDIOLDM2 = HuggingFaceAudioSpec("cvssp/audioldm2", pipeline_type="audioldm2", default_steps=200)

# aimu/models/providers/hf/speech.py: text-to-speech
class HuggingFaceSpeechModel(SpeechModel):
    BARK = HuggingFaceSpeechSpec("suno/bark", pipeline_type="bark", default_voice="v2/en_speaker_6")
```

Capability fields differ per modality; set the ones that matter for the model:

- **`HuggingFaceImageSpec`**: `pipeline_class`, `img2img_pipeline_class`, `img2img_uses_strength`, `supports_negative_prompt`, `default_steps`/`default_guidance`/`default_width`/`default_height`, `default_negative_prompt`, `max_prompt_tokens`.
- **`GeminiImageSpec`**: `supports_negative_prompt` (defaults `False`), `default_aspect_ratio`, `default_image_size`, `image_config_kwargs`.
- **`HuggingFaceAudioSpec`**: `pipeline_type` (`"musicgen"` / `"audioldm2"` / `"stable_audio"`), `default_duration_s`, `default_steps`.
- **`HuggingFaceSpeechSpec`** / **`OpenAISpeechSpec`**: `pipeline_type` (HF: `"tts_pipeline"` / `"speecht5"` / `"bark"`), `default_voice`, `default_speed`.

Because these specs carry behaviour the runtime depends on (e.g. `pipeline_class`, `supports_negative_prompt`), the enum is the single source of truth: the string form (`"hf:<repo>"`, `"gemini:<id>"`) resolves to the **same** spec object as the enum member, and an unknown id raises.

## Custom models

For a one-off model not worth adding to the catalog, construct the spec yourself and pass the object (not a string) to the client. This is an explicit, deliberate escape hatch where you supply every capability flag:

```python
from aimu.models import ImageClient
from aimu.models.base import HuggingFaceImageSpec

spec = HuggingFaceImageSpec("my-org/my-diffusion-model", pipeline_class="StableDiffusionPipeline")
client = ImageClient(spec)   # accepted: you've stated the capabilities explicitly
```

If the model is genuinely best-of-class, prefer adding it to the enum so everyone benefits.

## Verify

```python
from aimu.models import AnthropicModel

assert AnthropicModel.CLAUDE_OPUS_5.value == "claude-opus-5"
assert AnthropicModel.CLAUDE_OPUS_5.supports_tools
assert AnthropicModel.CLAUDE_OPUS_5 in AnthropicModel.TOOL_MODELS
```

Then run the model-client tests against the new entry:

```bash
pytest tests/test_models.py --client=anthropic --model=CLAUDE_OPUS_5
```

The model client is auto-pulled (Ollama, HuggingFace) or hits the cloud API on first use.

## See also

- [`aimu.models.ModelSpec`](../reference/api/models.md#aimu.models.ModelSpec): the dataclass fields
- [Model matrix](../reference/model-matrix.md): every shipped model with capability flags
