# Generate speech

AIMU has a parallel speech-generation surface to the text, image, and audio clients. The shape mirrors audio generation: a base ABC (`BaseSpeechClient`), a factory class (`SpeechClient`), per-provider concrete clients, and one-line entry points (`aimu.speech_client()` / `aimu.generate_speech()`). Two providers ship today:

- **HuggingFace** for local TTS (MMS-TTS-ENG, BARK)
- **OpenAI** for cloud TTS (tts-1, tts-1-hd; requires `OPENAI_API_KEY`)

Speech is distinct from audio: the audio surface (`aimu.audio_client()`) generates music and sounds from a descriptive prompt; the speech surface converts literal text to spoken audio.

## Install

**OpenAI TTS** (recommended for low latency):

```bash
pip install -e '.[openai_compat]'
export OPENAI_API_KEY=sk-...
```

**HuggingFace TTS** (local, no API key):

```bash
pip install -e '.[hf]'
```

The `[hf]` extra already ships `torch`, `transformers`, and `soundfile` for the HuggingFace text and audio clients. TTS adds no new dependencies.

## One-shot

```python
import aimu

# Save a WAV file and return its path (default format)
path = aimu.generate_speech("Hello, world!", model="openai:tts-1")

# Return raw WAV bytes
wav_bytes = aimu.generate_speech("Good morning.", model="openai:tts-1", format="bytes")

# Local HuggingFace TTS
sr, audio = aimu.generate_speech(
    "Hello from HuggingFace!",
    model="hf:facebook/mms-tts-eng",
    format="numpy",
)
```

`format=` selects how the audio is returned:
- `"path"` (default): saves a WAV file, returns the path string
- `"bytes"`: raw WAV-encoded bytes
- `"numpy"`: `(sample_rate: int, audio: np.ndarray)` tuple
- `"data_url"`: `data:audio/wav;base64,...` for inline embedding

## Choosing a model

### OpenAI

| Enum member | Model id | Notes |
|---|---|---|
| `TTS_1` | `tts-1` | Fast, standard quality. Recommended for live narration |
| `TTS_1_HD` | `tts-1-hd` | Slower, higher quality |

Voices (pass `voice=` to `generate`): `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`. Default: `alloy`.

### HuggingFace

| Enum member | Repo id | Pipeline | Notes |
|---|---|---|---|
| `MMS_TTS_ENG` | `facebook/mms-tts-eng` | `tts_pipeline` | Fast; English only |
| `SPEECHT5` | `microsoft/speecht5_tts` | `speecht5` | Good quality English; uses speaker embeddings (xvectors) |
| `BARK` | `suno/bark` | `bark` | Zero-shot voice codes; multilingual |

**SpeechT5** uses x-vector speaker embeddings for voice control. The default embedding (CMU Arctic index 7306) is loaded automatically from `datasets` on first call. To use a different speaker, pass `voice="N"` where N is an integer index into the `Matthijs/cmu-arctic-xvectors` dataset (0–1132):

```python
sr, audio = client.generate("Hello!", voice="42", format="numpy")
```

**BARK** voice codes: `v2/en_speaker_6`, `v2/en_speaker_9`, etc. Pass as `voice=` to `generate`.

Ad-hoc models via `"hf:<repo_id>"`: pipeline type inferred from known repo prefixes, defaults to `tts_pipeline`.

## Direct client

Build a client once, reuse it for multiple synthesis calls (weights load on first call):

```python
from aimu import speech_client

client = speech_client("openai:tts-1")
for line in ["Good morning.", "How can I help you today?", "Have a great day!"]:
    path = client.generate(line, voice="nova", format="path")
    print(path)
```

You can also pass an enum member for IDE autocomplete:

```python
from aimu import speech_client
from aimu.models import OpenAISpeechModel

client = speech_client(OpenAISpeechModel.TTS_1_HD)
```

For local HuggingFace TTS on a multi-GPU box, target a specific card with `model_kwargs={"device": "cuda:1"}`. TTS models are small, so there is no sharding default. (OpenAI TTS is a cloud API, so placement doesn't apply.)

### Per-call kwargs

| Kwarg | Notes |
|---|---|
| `voice` | Voice name (provider-specific; see above). Defaults to spec default (`"alloy"` for OpenAI). |
| `speed` | Playback speed multiplier. Default: `1.0`. |
| `num_audio` | Generate multiple clips in one call (default: `1`). |
| `format` | Output format (see above). Default: `"path"`. |
| `output_dir` | Directory for `format="path"`; defaults to `<output>/speech/`. |

```python
# Two clips, same call
clips = client.generate("Hello!", num_audio=2, format="bytes")
for wav in clips:
    print(len(wav), "bytes")
```

## Streaming progress

Pass `stream=True` to get an iterator of `SPEECH_GENERATING` chunks. For OpenAI, chunks arrive as HTTP bytes stream with `total_chunks=None`. For HuggingFace (single-pass), one final chunk is emitted:

```python
from aimu import speech_client

client = speech_client("openai:tts-1")

for chunk in client.generate("The quick brown fox jumps over the lazy dog.", stream=True):
    c = chunk.content
    if c["final"]:
        print(f"\nDone, saved to {c['result']}")
    else:
        print(f"Chunk {c['chunk_index']}", end="\r")
```

`chunk.content` shape for `SPEECH_GENERATING`:

```
{
    "chunk_index": int,          # 1-based index of this chunk
    "total_chunks": int | None,  # None for OpenAI (unknown upfront); 1 for HF single-pass
    "final": bool,               # True on the last chunk
    "result": str | bytes | ..., # encoded output on final chunk, None otherwise
}
```

Use `chunk.is_speech_progress()` to dispatch cleanly.

### Streaming through an agent

The built-in `generate_speech` tool is a generator. Its progress chunks flow through `agent.run(stream=True)`:

```python
from aimu.agents import Agent
from aimu.tools import builtin

agent = Agent(text_client, tools=[builtin.generate_speech])

for chunk in agent.run("Say hello in a warm, welcoming tone.", stream=True):
    if chunk.is_speech_progress():
        c = chunk.content
        if c["final"]:
            print(f"\nSaved: {c['result']}")
    elif chunk.is_text():
        print(chunk.content, end="")
```

## As an agent tool

The built-in `generate_speech` tool lets any chat LLM call TTS when the user asks. The LLM decides when to call it; the tool saves a WAV and returns the path.

```python
import os
os.environ["AIMU_SPEECH_MODEL"] = "hf:microsoft/speecht5_tts"

from aimu import client
from aimu.agents import Agent
from aimu.tools import builtin

text_client = client("anthropic:claude-sonnet-4-6")
agent = Agent(
    text_client,
    system_message=(
        "You can speak responses using the generate_speech tool. "
        "When the user asks you to say something aloud, call the tool."
    ),
    tools=[builtin.generate_speech],
)

response = agent.run("Please say 'Welcome to AIMU' in a friendly voice.")
```

The tool uses a lazy module-level singleton: the first call constructs a speech client from `AIMU_SPEECH_MODEL` (default: `"hf:microsoft/speecht5_tts"`), subsequent calls reuse it.

### Per-agent override: `make_speech_tool`

When you need a specific voice or speed from the global singleton, or multiple agents should use different voices:

```python
from aimu import speech_client
from aimu.tools.builtin import make_speech_tool

client_nova = speech_client("openai:tts-1")
nova_tool = make_speech_tool(client_nova, voice="nova", speed=1.1)

agent = Agent(text_client, tools=[nova_tool])
```

`make_speech_tool()` returns a fresh `@tool`-decorated callable bound to the supplied client; the singleton stays untouched.

## Async

The async surface mirrors the sync surface. Build a sync client first and pass it to `aio.speech_client()`:

```python
import asyncio
from aimu import aio, speech_client

sync = speech_client("openai:tts-1")
async_client = aio.speech_client(sync)

async def narrate(lines):
    results = await asyncio.gather(*[
        async_client.generate(line, format="bytes")
        for line in lines
    ])
    return results
```

`generate()` routes through `asyncio.to_thread` so the event loop stays free during API calls.

```python
# Async one-shot: pass a sync client as model
from aimu import aio, speech_client

sync = speech_client("openai:tts-1")
wav = await aio.generate_speech("Hello!", model=sync, format="bytes")
```

## See also

- [Reference: stream phases](../reference/stream-phases.md): `SPEECH_GENERATING` chunk shape.
- [Reference: env vars](../reference/env-vars.md): `AIMU_SPEECH_MODEL`.
- [Generate audio](generate-audio.md): the music/sound surface, which this mirrors.
- [Notebook 17: Speech](https://github.com/saxman/aimu/blob/main/notebooks/20%20-%20Speech.ipynb): runnable TTS demo.
- [Transcribe audio](transcribe-audio.md): the speech-to-text (ASR) surface.
