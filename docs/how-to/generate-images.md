# Generate images

AIMU has a parallel image-generation surface to the text chat client. The shape mirrors text: a base ABC (`BaseImageClient`), a factory class (`ImageClient`), per-provider concrete clients, and one-line entry points (`aimu.image_client()` / `aimu.generate_image()`). Two providers ship today:

- **HuggingFace `diffusers`** for local generation (Stable Diffusion 1.5 / XL / 3.5, FLUX dev / schnell)
- **Google Nano Banana** (`gemini-2.5-flash-image`) via the cloud Gemini API

## Install

Pick the providers you want. Both are opt-in extras:

```bash
pip install -e '.[hf]'         # HuggingFace diffusers — local, GPU-friendly
pip install -e '.[google]'     # Google Nano Banana — cloud, fast first call
pip install -e '.[hf,google]'  # both
```

The `[hf]` extra brings text + image (it already ships `torch` and `transformers` for the HuggingFace text client, plus `diffusers`, `safetensors`, and `Pillow` for image). `[google]` adds `google-genai` and `Pillow`.

Set `GOOGLE_API_KEY` (env or `.env`) to use Nano Banana.

## One-shot

```python
import aimu

# Local diffusers (downloads weights on first call)
path = aimu.generate_image(
    "a watercolor of a fox in a snowy forest",
    model="hf:runwayml/stable-diffusion-v1-5",
    format="path",
)

# Cloud Nano Banana
img = aimu.generate_image(
    "a watercolor of a fox in a snowy forest",
    model="gemini:nano-banana",
    aspect_ratio="16:9",
)
```

`format=` selects how the image is returned: `"pil"` (default, `PIL.Image`), `"path"` (saves PNG, returns path string), `"bytes"` (raw PNG), or `"data_url"` (base64-encoded for inline embedding).

## Streaming progress

Pass `stream=True` to get an iterator of `IMAGE_GENERATING` chunks, one per denoising step (HF) or one start/done pair per image (Gemini, which has no per-step API):

```python
import aimu

client = aimu.image_client("hf:runwayml/stable-diffusion-v1-5")

for chunk in client.generate("a fox", stream=True, num_inference_steps=25):
    c = chunk.content
    if c["final"]:
        print(f"\nDone — saved to {c['result']}" if c.get("result") else "\nDone")
    else:
        print(f"Step {c['step']}/{c['total_steps']}", end="\r")
```

### Intermediate-image previews — `preview_every=N`

Decoding latents to a PIL image at every step adds ~50–200 ms per call on GPU, so previews are opt-in. Pass `preview_every=N` to decode every Nth step (and always the final):

```python
for chunk in client.generate(
    "a fox", stream=True, num_inference_steps=25, preview_every=5
):
    c = chunk.content
    if c["image"] is not None:
        c["image"].save(f"preview_step_{c['step']}.png")
```

Gemini Nano Banana ignores `preview_every` — its cloud API has no intermediate latents.

### Streaming the built-in tool through an agent

The built-in `generate_image` tool is itself a generator. When dispatched through `agent.run(stream=True)`, its progress chunks flow through the agent's own stream — no side channel:

```python
from aimu.agents import Agent
from aimu.tools import builtin

agent = Agent(text_client, tools=[builtin.generate_image])

for chunk in agent.run("draw me a fox", stream=True):
    if chunk.is_image_progress():
        c = chunk.content
        print(f"  {c['step']}/{c['total_steps']}")
    elif chunk.is_tool_call():
        print(f"  Done: {chunk.content['response']}")
    elif chunk.is_text():
        print(chunk.content, end="")
```

To opt into intermediate previews from the agent path, build a bound tool with `make_image_tool(client, preview_every=N)` and pass that to the agent instead of `builtin.generate_image`.

## Direct client — reuse weights / API client

A fresh client per call reloads weights for HuggingFace; for Nano Banana it just rebuilds the API client. For repeated generation, build once and reuse:

```python
from aimu import image_client

client = image_client("hf:stabilityai/stable-diffusion-xl-base-1.0")
for prompt in ["a fox", "a deer", "an owl"]:
    img = client.generate(prompt, width=1024, height=1024)
    img.save(f"{prompt}.png")
```

You can also pass an enum member for IDE autocomplete when browsing available models:

```python
from aimu import image_client
from aimu.models import HuggingFaceImageModel, GeminiImageModel

client = image_client(HuggingFaceImageModel.SDXL_BASE)
client = image_client(GeminiImageModel.NANO_BANANA)
```

Per-provider knobs differ:

- **HuggingFace**: `negative_prompt`, `width`, `height`, `num_inference_steps`, `guidance_scale`, `seed`. Defaults come from the `HuggingFaceImageSpec` (e.g. SD 1.5 → 25 steps, 512×512, guidance 7.5; FLUX schnell → 4 steps, 1024×1024).
- **Gemini Nano Banana**: `aspect_ratio` (e.g. `"1:1"`, `"16:9"`), `image_size`. No `seed` — the API doesn't expose one.

Both share: `num_images=N`, `format=`, `output_dir=`.

## As an agent tool

The built-in `generate_image` tool lets any chat LLM call image generation when the user asks. The LLM decides *when* to call it; the tool saves a PNG and returns the path so it appears in the conversation history.

```python
import os
os.environ["AIMU_IMAGE_MODEL"] = "gemini:nano-banana"  # or any "hf:..." string

from aimu import client
from aimu.agents import Agent
from aimu.tools import builtin

text_client = client("anthropic:claude-sonnet-4-6")
agent = Agent(
    text_client,
    system_message=(
        "You can generate images with the generate_image tool. When the user "
        "asks for an image, write a vivid prompt and call the tool. Tell the "
        "user where the image was saved."
    ),
    tools=[builtin.generate_image],
)

response = agent.run("Please make me a watercolor of a fox in a snowy forest.")
```

The tool uses a lazy module-level singleton — first call constructs an image client based on `AIMU_IMAGE_MODEL` (default: SDXL base), subsequent calls reuse it.

### Per-agent override — `make_image_tool`

When you want a different model from the global singleton (or several agents in one process shouldn't share a pipeline), bind a tool to a specific client:

```python
from aimu import image_client
from aimu.tools.builtin import make_image_tool

fast = image_client("hf:black-forest-labs/FLUX.1-schnell")  # 4-step model
fast_tool = make_image_tool(fast)

agent = Agent(text_client, tools=[fast_tool])
```

`make_image_tool()` returns a fresh `@tool`-decorated callable bound to the supplied client; the singleton stays untouched.

## Skill integration (deeper, optional)

For `SkillAgent` users, drop a `SKILL.md` under `.agents/skills/image-generation/` with prompt-engineering guidance. Skills are filesystem-discovered (not shipped with AIMU); see [notebooks/15 - Image Generation.ipynb](https://github.com/saxman/aimu/tree/main/notebooks) for a copyable example.

## Async

The async surface mirrors the sync surface one-for-one. Because image providers either load weights in-process (HuggingFace) or hold an SDK client (Gemini), the factory follows the wrap pattern established for HF / LlamaCpp text clients — build a sync client first and pass it to `aio.image_client()`:

```python
import asyncio
from aimu import aio, image_client

sync = image_client("hf:runwayml/stable-diffusion-v1-5")
async_client = aio.image_client(sync)

async def make_two():
    a, b = await asyncio.gather(
        async_client.generate("a watercolor of a fox", width=512, height=512),
        async_client.generate("a watercolor of a deer", width=512, height=512),
    )
    return a, b
```

Async `generate()` routes through `asyncio.to_thread` so the event loop stays free during inference. On a single GPU, sibling `gather`-ed HF calls don't truly overlap on the device (CUDA streams + the GIL serialise), but the event loop stays free for other coroutines.

For Gemini, the cloud SDK could be called natively async — kept as a follow-up for shape consistency with the in-process wrap pattern.

## See also

- [Reference: provider matrix](../reference/provider-matrix.md) — image vs text provider tables.
- [Reference: env vars](../reference/env-vars.md) — `GOOGLE_API_KEY`, `AIMU_IMAGE_MODEL`.
- [Notebook 15 — Image Generation](https://github.com/saxman/aimu/tree/main/notebooks) — runnable end-to-end demo.
- [Add a custom tool](add-custom-tool.md) — to build your own image-generation tool variant.
- [Generate audio](generate-audio.md) — the audio surface, which mirrors this one.
