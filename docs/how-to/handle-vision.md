# Handle vision input

Pass `images=[...]` to `chat()` (stateful) or `generate()` (stateless one-shot) on a vision-capable model. AIMU normalises every input to OpenAI content blocks internally and adapts them per-provider.

## Basic usage

```python
import aimu

client = aimu.client("openai:gpt-4o-mini")
client.chat("What's in this image?", images=["./cat.jpg"])

# Multiple images
client.chat("Compare these two photos.", images=["a.png", "b.png"])
```

## Stateless single-turn (`generate`)

Use `generate(images=...)` for a one-shot vision call that **does not touch conversation history** — ideal for "look once and answer" tasks (captioning, scoring, classification) where you don't want the image and reply accumulating in `self.messages`:

```python
client = aimu.client("openai:gpt-4o-mini")

# Each call is independent — no history kept, no reset() needed between calls.
caption = client.generate("Caption this image in one sentence.", images=["./cat.jpg"])
score = client.generate("Rate this image's quality 1-10.", images=["./cat.jpg"])

assert client.messages == []   # generate() never mutates message state
```

`chat(images=...)` is the stateful path — the turn persists so you can ask follow-ups. `generate(images=...)` is the stateless path — each call stands alone. Both accept the same image forms and raise `ValueError` for a non-vision model. (Before, one-shot vision required a `client.reset()` + `chat()` dance to avoid polluting history; `generate(images=...)` removes that.)

## Accepted image forms

Each item in `images=[...]` may be any of:

- **File path** as `str` or `pathlib.Path` — read from disk and base64-encoded
- **Raw `bytes`** — encoded directly
- **`http(s)://` URL** — passed through to providers that support it
- **`data:image/...;base64,...` URL** — passed through

```python
from pathlib import Path

client.chat("describe", images=[
    "./local.jpg",                                   # path string
    Path("./other.png"),                             # Path
    b"\x89PNG\r\n\x1a\n...",                         # bytes
    "https://example.com/cat.jpg",                   # URL
    "data:image/png;base64,iVBORw0KGgoAAAANS...",    # data URL
])
```

## Vision-capable models

Each `Model` enum exposes a `VISION_MODELS` classproperty listing members where `supports_vision=True`. Passing `images=` to a non-vision model raises `ValueError` up front.

Common vision-capable models:

- **OpenAI**: GPT-4o, GPT-4.1, o3, o4-mini
- **Anthropic**: Claude 4.x (Sonnet, Opus, Haiku)
- **Google Gemini**: 1.5 Pro/Flash, 2.0 Flash, 2.5 Pro/Flash
- **Ollama**: Gemma 3, Gemma 4
- **HuggingFace**: Gemma 3, Gemma 4, Qwen 3.5/3.6 VL variants

See the [model matrix](../reference/model-matrix.md) for the full list.

## Per-provider adaptation

AIMU keeps `self.messages` in OpenAI format always (`image_url` content blocks) and adapts at request time:

| Provider | Adaptation |
|---|---|
| OpenAI / Gemini / OpenAI-compatible | Pass-through (the SDK speaks `image_url` natively) |
| Anthropic | Rewrites `image_url` → Anthropic `image` blocks (`base64` source for data URLs, `url` source for http(s)) |
| Ollama (native API) | Extracts to Ollama's message-level `images=[<bare base64>]` field. **Only inline base64 supported**; http(s) URLs raise `ValueError` |
| HuggingFace (with processor) | Decodes to PIL images and passes via `AutoProcessor`; `image_url` blocks are rewritten to `{"type": "image"}` placeholders for the chat template |
| llama-cpp | Pass-through; requires loading an `mmproj` projector via the `chat_handler=` constructor kwarg |

## Vision + agents

`images=` is forwarded through every agent and workflow. Only the *initial* turn carries images; continuation prompts (after a tool call) are text-only.

- `Agent.run("task", images=[...])` — attaches to the initial turn
- `Chain.run("task", images=[...])` — forwards to step 0
- `Router.run("task", images=[...])` — forwards to the dispatched handler
- `Parallel.run("task", images=[...])` — forwards to every worker
- `EvaluatorOptimizer.run("task", images=[...])` — forwards only to the initial generator turn

## See also

- Notebook [04 - Vision](https://github.com/saxman/aimu/blob/main/notebooks/04%20-%20Vision.ipynb)
- [Model matrix](../reference/model-matrix.md) — `supports_vision` column
