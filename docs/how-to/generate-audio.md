# Generate audio

AIMU has a parallel audio-generation surface to the text and image clients. The shape mirrors image generation: a base ABC (`BaseAudioClient`), a factory class (`AudioClient`), per-provider concrete clients, and one-line entry points (`aimu.audio_client()` / `aimu.generate_audio()`). One provider ships today:

- **HuggingFace** for local generation (MusicGen small/medium/large, AudioLDM2, Stable Audio Open)

The scope is music and sound generation, not text-to-speech.

## Install

```bash
pip install -e '.[hf]'
```

The `[hf]` extra already ships `torch` and `transformers` for the HuggingFace text client. Audio adds `diffusers` (for AudioLDM2 / Stable Audio) and `soundfile` (for WAV encoding). All come in one extra — no separate install step.

## One-shot

```python
import aimu

# Save a WAV file and return its path
path = aimu.generate_audio(
    "a lo-fi hip-hop beat with soft piano",
    model="hf:facebook/musicgen-small",
    format="path",
)

# Return raw WAV bytes
data = aimu.generate_audio(
    "gentle ocean waves on a beach",
    model="hf:cvssp/audioldm2",
    format="bytes",
)
```

`format=` selects how the audio is returned:
- `"numpy"` (default) — `(sample_rate: int, audio: np.ndarray)` tuple, the native representation
- `"path"` — saves a WAV file, returns the path string
- `"bytes"` — raw WAV-encoded bytes
- `"data_url"` — `data:audio/wav;base64,...` for inline embedding

## Choosing a model

Three pipeline families ship in the `HuggingFaceAudioModel` enum:

| Enum member | Repo id | Pipeline | Notes |
|---|---|---|---|
| `MUSICGEN_SMALL` | `facebook/musicgen-small` | `musicgen` | Fastest; 300M params |
| `MUSICGEN_MEDIUM` | `facebook/musicgen-medium` | `musicgen` | Balanced quality |
| `MUSICGEN_LARGE` | `facebook/musicgen-large` | `musicgen` | Best quality; 3.3B params |
| `AUDIOLDM2` | `cvssp/audioldm2` | `audioldm2` | Diffusion model; general sound |
| `STABLE_AUDIO_OPEN` | `stabilityai/stable-audio-open-1.0` | `stable_audio` | Diffusion model; 44.1 kHz stereo |

**MusicGen** (`musicgen`) is token-autoregressive — it generates directly at 32 kHz without a diffusion loop, so there are no inference steps. Duration maps to a token count (~50 tokens/s).

**AudioLDM2** and **Stable Audio Open** (`audioldm2` / `stable_audio`) use latent diffusion — they accept `num_inference_steps` and support step-by-step streaming progress.

## Direct client

Build a client once, reuse it for multiple generations:

```python
from aimu import audio_client

client = audio_client("hf:facebook/musicgen-small")
for prompt in ["lo-fi jazz loop", "driving electronic beat", "calm acoustic guitar"]:
    sr, audio = client.generate(prompt, duration_s=5.0, format="numpy")
    print(f"  {prompt}: {audio.shape}, {sr} Hz")
```

You can also pass an enum member for IDE autocomplete:

```python
from aimu import audio_client
from aimu.models import HuggingFaceAudioModel

client = audio_client(HuggingFaceAudioModel.MUSICGEN_MEDIUM)
```

On a multi-GPU box, target a specific card with `model_kwargs={"device": "cuda:1"}` (default placement is `cuda:0`). These models are small enough to fit one card, so — unlike the image client — there is no sharding default.

### Per-call kwargs

| Kwarg | Applies to | Default | Notes |
|---|---|---|---|
| `duration_s` | all | 10.0 | Seconds of audio to generate (max 30) |
| `num_inference_steps` | diffusers (`audioldm2`, `stable_audio`) | 200 | More steps → higher quality, slower |
| `seed` | all | None | Reproducible output for MusicGen; passed to diffusers scheduler for others |
| `num_audio` | all | 1 | Generate multiple clips in one call |
| `format` | all | `"numpy"` | Output format (see above) |
| `output_dir` | all | None | Directory for `format="path"`; defaults to `<output>/audio/` |

```python
# Fast smoke test: 3 seconds, 10 diffusion steps
clip = client.generate(
    "rainstorm on a tin roof",
    duration_s=3.0,
    num_inference_steps=10,
    format="path",
)

# Two clips, same call
clips = client.generate("lo-fi beat", num_audio=2, format="numpy")
for sr, audio in clips:
    print(audio.shape, sr)
```

## Streaming progress

Pass `stream=True` to get an iterator of `AUDIO_GENERATING` chunks. For MusicGen, one final chunk is emitted when generation completes. For diffusers models, one chunk per step plus a final chunk:

```python
from aimu import audio_client

client = audio_client("hf:cvssp/audioldm2")

for chunk in client.generate("summer thunderstorm", stream=True, num_inference_steps=20):
    c = chunk.content
    if c["final"]:
        print(f"\nDone — saved to {c['result']}")
    else:
        print(f"Step {c['step']}/{c['total_steps']}", end="\r")
```

`chunk.content` shape for `AUDIO_GENERATING`:

```
{
    "step": int,            # current step (1-based; 0 before any step)
    "total_steps": int,     # total steps (1 for MusicGen)
    "final": bool,          # True on the last chunk per clip
    "result": str | bytes | tuple | None,  # encoded output on final chunk, None otherwise
    "duration_s": float,    # requested duration
}
```

Use `chunk.is_audio_progress()` to dispatch cleanly.

### Streaming through an agent

The built-in `generate_audio` tool is a generator. Its progress chunks flow through `agent.run(stream=True)`:

```python
from aimu.agents import Agent
from aimu.tools import builtin

agent = Agent(text_client, tools=[builtin.generate_audio])

for chunk in agent.run("compose a 10-second lo-fi beat", stream=True):
    if chunk.is_audio_progress():
        c = chunk.content
        if not c["final"]:
            print(f"  {c['step']}/{c['total_steps']}", end="\r")
        else:
            print(f"\n  Saved: {c['result']}")
    elif chunk.is_text():
        print(chunk.content, end="")
```

## As an agent tool

The built-in `generate_audio` tool lets any chat LLM call audio generation when the user asks. The LLM decides when to call it; the tool saves a WAV and returns the path.

```python
import os
os.environ["AIMU_AUDIO_MODEL"] = "hf:facebook/musicgen-small"

from aimu import client
from aimu.agents import Agent
from aimu.tools import builtin

text_client = client("anthropic:claude-sonnet-4-6")
agent = Agent(
    text_client,
    system_message=(
        "You can generate audio with the generate_audio tool. When the user asks "
        "for music or sound effects, call the tool and tell them where the file was saved."
    ),
    tools=[builtin.generate_audio],
)

response = agent.run("Make me a short lo-fi beat I can study to.")
```

The tool uses a lazy module-level singleton — first call constructs an audio client from `AIMU_AUDIO_MODEL` (default: MusicGen small), subsequent calls reuse it.

### Per-agent override — `make_audio_tool`

When you need a different model from the global singleton, or multiple agents in one process should use different pipelines:

```python
from aimu import audio_client
from aimu.tools.builtin import make_audio_tool

medium = audio_client("hf:facebook/musicgen-medium")
medium_tool = make_audio_tool(medium, duration_s=15.0, num_inference_steps=100)

agent = Agent(text_client, tools=[medium_tool])
```

`make_audio_tool()` returns a fresh `@tool`-decorated callable bound to the supplied client; the singleton stays untouched. Pass `duration_s=` to fix the generation length, and `num_inference_steps=N` to override the default denoising step count (diffusers models only — AudioLDM2, Stable Audio; ignored by MusicGen).

## Async

The async surface mirrors the sync surface one-for-one. HuggingFace models load weights in-process, so the factory follows the wrap pattern — build a sync client first and pass it to `aio.audio_client()`:

```python
import asyncio
from aimu import aio, audio_client

sync = audio_client("hf:facebook/musicgen-small")
async_client = aio.audio_client(sync)

async def make_two():
    a, b = await asyncio.gather(
        async_client.generate("lo-fi beat", duration_s=5.0),
        async_client.generate("ambient wind", duration_s=5.0),
    )
    return a, b
```

`generate()` routes through `asyncio.to_thread` so the event loop stays free during inference. On a single GPU, sibling `gather`-ed calls don't truly overlap (CUDA streams and the GIL serialize), but the event loop stays free for other coroutines.

```python
# Async one-shot
from aimu import aio

path = await aio.generate_audio(
    "driving electronic beat",
    model="hf:facebook/musicgen-small",
    format="path",
    duration_s=10.0,
)
```

## See also

- [Reference: model matrix](../reference/model-matrix.md) — audio model table with pipeline types and defaults.
- [Reference: stream phases](../reference/stream-phases.md) — `AUDIO_GENERATING` chunk shape.
- [Reference: env vars](../reference/env-vars.md) — `AIMU_AUDIO_MODEL`.
- [Notebook 16 — Audio Generation](https://github.com/saxman/aimu/tree/main/notebooks) — runnable end-to-end demo.
- [Generate images](generate-images.md) — the image surface, which this mirrors.
- [Generate speech](generate-speech.md) — the TTS surface (`aimu.speech_client()` / `aimu.generate_speech()`). Audio and speech are separate modalities: audio generates music/sounds from descriptive prompts; speech converts literal text to spoken audio.
