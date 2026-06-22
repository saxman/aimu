# Add a new model

To make a new model usable with `ModelClient`, add a member to that provider's `Model` enum. The member's value is a `ModelSpec` with the model id and capability flags.

!!! note "AIMU ships curated models only"
    Every modality requires the model to be a member of its provider enum (or a hand-built spec, see [Custom models](#custom-models)). Passing an arbitrary `"provider:some/unknown-repo"` string **raises** `ValueError`, rather than silently fabricating a spec with guessed capabilities. Capability flags (tools/thinking/vision, `pipeline_class`, `supports_negative_prompt`, voice/step defaults, and so on) can't be inferred reliably from a repo name, and a wrong guess causes hard-to-debug runtime failures. So the catalog is intentional: to use a new model, add it to the enum here.

## Basic case

For most providers (`OllamaModel`, `AnthropicModel`, `OpenAIModel`, etc.) the enum value is a single `ModelSpec`:

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

## With custom generation kwargs

Some models work best with specific sampling parameters:

```python
QWEN_3_8B = ModelSpec(
    "qwen3:8b",
    tools=True,
    thinking=True,
    generation_kwargs={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
)
```

`generation_kwargs` is merged on top of the client's defaults whenever `chat()` or `generate()` is called without an explicit `generate_kwargs` argument.

## HuggingFace provider-specific extras

`HuggingFaceModel` carries extra positional values for the tool-call response format and a thinking-template flag:

```python
# aimu/models/providers/hf/text.py
class HuggingFaceModel(Model):
    QWEN_3_8B = (
        ModelSpec("Qwen/Qwen3-8B", tools=True, thinking=True),
        ToolCallFormat.XML,                    # tool_call_format
    )
    QWEN_3_5_9B = (
        ModelSpec("Qwen/Qwen3.5-9B", tools=True, thinking=True),
        ToolCallFormat.XML,
        True,                                  # think_opener_in_prompt
    )
```

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
