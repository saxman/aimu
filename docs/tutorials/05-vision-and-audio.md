# Vision & audio input

In ~10 minutes you'll send images — and audio — to a capable model, from a single `chat()` call up
through agents and workflows.

This is the last tutorial; it fills in the remaining *input modalities*. (For output modalities —
generating images, speech, or music — see the how-to guides.)

## Setup

You need a vision-capable model. The fastest free option is **Ollama with Gemma 4**:

```bash
ollama pull gemma4:e4b
```

Or use any cloud model that supports vision (GPT-4o, Claude 4.x, Gemini 2.x).

```python
import aimu

client = aimu.client("ollama:gemma4:e4b")
client.is_vision_model   # True
```

## 1. Send an image

Pass `images=[...]` to `chat()`. Each item can be a file path, raw bytes, an http(s) URL, or a
`data:image/...` URL:

```python
response = client.chat("What's in this image?", images=["./cat.jpg"])
print(response)
```

Multiple images? Just add to the list:

```python
client.chat("Compare these.", images=["a.png", "b.png"])
```

For a one-shot "look once and answer" call that *doesn't* keep history, use `generate(images=...)` —
the stateless sibling of `chat()`:

```python
caption = client.generate("Caption this image.", images=["./cat.jpg"])  # client.messages stays empty
```

`client.messages` keeps everything in OpenAI content-block format internally; each provider adapts at
request time. See [how-to: handle vision](../how-to/handle-vision.md) for the per-provider details.

## 2. Send audio

Audio input mirrors images one-for-one. Pass `audio=[...]` to an **audio-capable** model (OpenAI
GPT-4o / GPT-4.1, Google Gemini 2.x, or HuggingFace Gemma 4 — note the Ollama build of Gemma 4 is
vision-only), with the same input forms (file path, bytes, `https://` URL, or a `data:audio/...`
URL):

```python
client = aimu.client("openai:gpt-4o")   # an audio-capable model
print(client.chat("Transcribe and summarise this clip.", audio=["./note.mp3"]))
```

`images=` and `audio=` are mutually exclusive per turn (passing both raises `ValueError`). Both are
available on `chat()` (stateful) and `generate()` (stateless). See
[how-to: handle audio input](../how-to/handle-audio-input.md).

> Transcription and speech are separate, dedicated surfaces: use `aimu.transcribe()` for
> speech-to-text with a Whisper-class model, and `aimu.generate_speech()` for text-to-speech. `audio=`
> here is for an audio-capable *chat* model reasoning over a clip.

## 3. Images through an agent

`images=` threads through `Agent.run()`. Only the initial task turn carries the image; the agent's
continuation turns are text-only:

```python
from aimu.agents import Agent

agent = Agent(client, "You are a helpful image analyst.", name="vision-bot")
print(agent.run("Describe the scene, then list any text you can read.", images=["./photo.jpg"]))
```

(Streaming works here too — the same `stream=True` iterator from the
[Streaming tutorial](04-streaming.md).)

## 4. Images through workflows

`images=` is forwarded through every workflow's `run()`, to the turns where it makes sense:

| Workflow | Where the image goes |
|---|---|
| `Chain.run(task, images=[...])` | step 0 only (later steps see step 0's text) |
| `Router.run(task, images=[...])` | the dispatched handler |
| `Parallel.run(task, images=[...])` | every worker |
| `EvaluatorOptimizer.run(task, images=[...])` | the initial generator turn only |

```python
from aimu.agents import Chain

chain = Chain.from_client(client, [
    "Describe what you see in detail.",
    "Summarise the description in one sentence.",
])
print(chain.run("Tell me about this photo.", images=["./photo.jpg"]))
```

Step 0 sees the image; step 1 sees only step 0's text output.

## What's next

You've completed the tutorials. You now know:

- The top-level API: `aimu.chat()`, `aimu.client()`, `aimu.agent()`, and model strings.
- The agent loop and the `@tool` decorator.
- The code-controlled workflows: Chain, Router, Parallel, EvaluatorOptimizer.
- The full `StreamChunk` streaming API.
- Vision and audio input into chats, agents, and workflows.

For specific tasks, browse the [how-to guides](../how-to/index.md) — including
[generating images](../how-to/generate-images.md), [speech](../how-to/generate-speech.md), and
[transcription](../how-to/transcribe-audio.md). For the design intent behind any of this, see the
[explanation pages](../explanation/index.md).
