# Handle audio input

Pass `audio=[...]` to `chat()` (stateful) or `generate()` (stateless one-shot) on an audio-capable model. AIMU normalises every input to OpenAI `input_audio` content blocks internally and adapts them per-provider.

## Basic usage

```python
import aimu

client = aimu.client("openai:gpt-4o")
client.chat("Transcribe this recording.", audio=["./interview.wav"])

# Multiple clips
client.chat("Compare the speakers in these two clips.", audio=["./a.wav", "./b.wav"])
```

## Stateless single-turn (`generate`)

Use `generate(audio=...)` for a one-shot audio call that **does not touch conversation history**:

```python
client = aimu.client("openai:gpt-4o")

# Each call is independent: no history kept, no reset() needed between calls.
lang = client.generate("What language is spoken here?", audio=["./clip.mp3"])
tone = client.generate("Describe the speaker's tone.", audio=["./clip.mp3"])

assert client.messages == []   # generate() never mutates message state
```

`chat(audio=...)` is the stateful path: the turn persists so you can ask follow-ups. `generate(audio=...)` is the stateless path: each call stands alone.

## Accepted audio forms

Each item in `audio=[...]` may be any of:

- **File path** as `str` or `pathlib.Path`: read from disk and base64-encoded
- **Raw `bytes`**: encoded directly (defaults to `wav` format)
- **`https://` URL**: fetched eagerly (the `input_audio` block has no remote-URL field)
- **`data:audio/...;base64,...` URL**: passed through; format extracted from MIME type

Supported format strings (inferred from extension or MIME type): `wav`, `mp3`, `ogg`, `flac`, `m4a`, `webm`.

```python
from pathlib import Path

client.chat("Analyse these clips", audio=[
    "./local.wav",                                     # path string
    Path("./other.mp3"),                               # Path
    b"RIFF...",                                        # raw bytes (wav assumed)
    "https://example.com/clip.wav",                    # URL (fetched)
    "data:audio/flac;base64,AAAA...",                  # data URL
])
```

## Audio-capable models

Each `Model` enum exposes an `AUDIO_MODELS` classproperty listing members where `supports_audio=True`. Passing `audio=` to a non-audio model raises `ValueError` up front.

```python
from aimu.models.providers.openai.text import OpenAIClient
print(OpenAIClient.AUDIO_MODELS)
# [<OpenAIModel.GPT_4O_MINI: ...>, <OpenAIModel.GPT_4O: ...>, ...]
```

Supported audio-capable text models:

| Provider | Models |
|---|---|
| OpenAI | GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano |
| Google Gemini | 2.0 Flash, 2.0 Flash Lite, 2.5 Pro, 2.5 Flash |
| HuggingFace | Gemma 4 E4B, Gemma 4 12B, Nemotron-H-8B |
| Ollama | None yet (Gemma 4 weights support audio; add `audio=True` once Ollama API exposes audio input) |

## Mutual exclusivity with `images=`

`audio=` and `images=` are mutually exclusive per turn. Passing both raises `ValueError`:

```python
# Raises ValueError: "Pass either images= or audio= per call, not both."
client.chat("describe", images=["photo.jpg"], audio=["clip.wav"])
```

## Multi-turn conversations

Audio blocks persist in `self.messages` as `input_audio` content, so the model can refer back to earlier turns:

```python
client = aimu.client("openai:gpt-4o")
client.chat("What is the main topic of this recording?", audio=["./meeting.wav"])
client.chat("Summarise the action items mentioned.")  # follow-up, no audio needed
```

## Async surface

`audio=` is available on `aio.chat()` and `aio.generate()` with the same signature:

```python
from aimu import aio
import asyncio

async def main():
    client = aio.client("openai:gpt-4o")
    result = await client.generate("Transcribe this.", audio=["./clip.wav"])
    print(result)

asyncio.run(main())
```

## Per-provider adaptation

AIMU keeps `self.messages` in OpenAI format always (`input_audio` content blocks) and adapts at request time:

| Provider | Adaptation |
|---|---|
| OpenAI / Gemini / OpenAI-compatible | Pass-through (`input_audio` is already OpenAI format) |
| Anthropic | Rewrites to Anthropic `{"type": "audio", "source": {"type": "base64", ...}}` blocks |
| HuggingFace | Decodes to float32 numpy arrays via `soundfile`; passes as `audio=`/`sampling_rate=16000` to the processor |
| Ollama | Raises `ValueError` (API does not yet support audio input) |

## See also

- Notebook [05 - Audio Input](https://github.com/saxman/aimu/blob/main/notebooks/05-audio-input.qmd)
- [Handle vision input](handle-vision.md): parallel surface for image input
- [Model matrix](../reference/model-matrix.md): `supports_audio` column
