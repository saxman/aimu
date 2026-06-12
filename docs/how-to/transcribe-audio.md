# Transcribe audio

Pass an audio clip to `transcription_client().transcribe()` or the one-shot
`aimu.transcribe()` to get the spoken text as a string.

## Basic usage

```python
import aimu

# One-shot
text = aimu.transcribe("./clip.wav", model="openai:whisper-1")

# Reusable client (better for many files)
client = aimu.transcription_client("openai:whisper-1")
text = client.transcribe("./clip.wav")
```

## Accepted audio forms

Each `audio` argument accepts the same forms as `audio=` on `chat()`:

- **File path** as `str` or `pathlib.Path` -- read from disk
- **Raw `bytes`** -- WAV format assumed
- **`https://` URL** -- fetched eagerly
- **`data:audio/...;base64,...` URL** -- passed through

Supported format strings: `wav`, `mp3`, `ogg`, `flac`, `m4a`, `webm`.

## Structured output with timestamps

Pass `response_format="verbose_json"` to get segments with start/end timestamps:

```python
result = client.transcribe("./clip.wav", response_format="verbose_json")
# {
#   "text": "Hello world.",
#   "language": "en",
#   "duration": 2.5,
#   "segments": [{"start": 0.0, "end": 2.5, "text": "Hello world."}]
# }
```

`response_format="verbose_json"` requires a model with `supports_timestamps=True`.
All models in the catalog support it. `response_format="json"` returns `{"text": "..."}`.

## Language hint

```python
text = client.transcribe("./clip.wav", language="fr")  # BCP-47 code
```

`None` (default) triggers auto-detection.

## Available models

| Provider | Enum | Model ID |
|---|---|---|
| OpenAI | `OpenAITranscriptionModel.WHISPER_1` | `whisper-1` |
| OpenAI | `OpenAITranscriptionModel.GPT_4O_TRANSCRIBE` | `gpt-4o-transcribe` |
| OpenAI | `OpenAITranscriptionModel.GPT_4O_MINI_TRANSCRIBE` | `gpt-4o-mini-transcribe` |
| HuggingFace | `HuggingFaceTranscriptionModel.WHISPER_TINY` | `openai/whisper-tiny` |
| HuggingFace | `HuggingFaceTranscriptionModel.WHISPER_LARGE_V3` | `openai/whisper-large-v3` |
| HuggingFace | `HuggingFaceTranscriptionModel.DISTIL_WHISPER_LARGE_V3` | `distil-whisper/distil-large-v3` |

Requires `OPENAI_API_KEY` for OpenAI models. HuggingFace models run locally
(need the `[hf]` extra).

## Default model via env var

```bash
export AIMU_TRANSCRIPTION_MODEL="openai:whisper-1"
```

Then `aimu.transcription_client()` and `aimu.transcribe()` resolve the model
without an explicit argument.

## Async surface

```python
from aimu import aio
import asyncio

async def main():
    sync_client = aimu.transcription_client("openai:whisper-1")
    async_client = aio.transcription_client(sync_client)
    text = await async_client.transcribe("./clip.wav")
    print(text)

asyncio.run(main())
```

## As an agent tool

```python
from aimu.tools.builtin import transcribe_audio
from aimu.agents import Agent

agent = Agent(text_client, tools=[transcribe_audio])
# agent can now call transcribe_audio(audio_path="./clip.wav")
```

For a custom client: `make_transcription_tool(client)` returns a bound version.

## See also

- [Handle audio input](handle-audio-input.md) -- `audio=` on text models for analysis/QA without dedicated transcription
- Notebook [19 - Transcription](https://github.com/saxman/aimu/blob/main/notebooks/19%20-%20Transcription.ipynb)
