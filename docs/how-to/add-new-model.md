# Add a new model

To make a new model usable with `ModelClient`, add a member to that provider's `Model` enum. The member's value is a `ModelSpec` with the model id and capability flags.

## Basic case

For most providers (`OllamaModel`, `AnthropicModel`, `OpenAIModel`, etc.) the enum value is a single `ModelSpec`:

```python
# aimu/models/anthropic/anthropic_client.py
from ..base import Model, ModelSpec

class AnthropicModel(Model):
    CLAUDE_SONNET_4_6 = ModelSpec("claude-sonnet-4-6", tools=True, thinking=True, vision=True)
    # Add a new model:
    CLAUDE_OPUS_5 = ModelSpec("claude-opus-5", tools=True, thinking=True, vision=True)
```

That's it. The model id (`"claude-opus-5"`) is what gets sent to the provider; the capability flags drive `is_tool_using_model`, `is_thinking_model`, `is_vision_model`, and the derived `TOOL_MODELS` / `THINKING_MODELS` / `VISION_MODELS` classproperties.

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
# aimu/models/hf/hf_client.py
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

Pick `ToolCallFormat.XML` / `JSON_OBJECT` / `JSON_ARRAY` / `BRACKETED` / `NA` based on how the base model emits tool calls. See existing entries in `aimu/models/hf/hf_client.py` for examples per model family.

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

- [`aimu.models.ModelSpec`](../reference/api/models.md#aimu.models.ModelSpec) — the dataclass fields
- [Model matrix](../reference/model-matrix.md) — every shipped model with capability flags
