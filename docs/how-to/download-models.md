# Download models

AIMU does not download Ollama models behind your back. Building a client does no network I/O, and a
model the server does not have fails at the first request with an error that says how to install it.
Downloading is something you ask for, in one of two ways: `aimu.pull()` ahead of time, or `pull=True`
on the client.

```python
import aimu

aimu.pull("ollama:qwen3.5:9b")            # download now, then build clients freely
client = aimu.client("ollama:qwen3.5:9b")  # no network I/O here
```

## See what is installed

`aimu.available_text_models()` lists the catalogued models that are loadable right now, as enum
members: models on a running Ollama server, in the local HuggingFace cache, and on reachable local
OpenAI-compatible servers. It downloads nothing and never contacts a cloud provider.

```python
from aimu.models import OllamaModel

installed = aimu.available_text_models()
pullable = [m for m in OllamaModel if m not in installed]  # catalogued, not on the server yet
```

The Ollama probe reads `OLLAMA_HOST` (else `127.0.0.1:11434`). To list a remote server for one call,
name it:

```python
aimu.available_text_models(ollama_host="http://gpu-box:11434")
aimu.available_embedding_models(ollama_host="http://gpu-box:11434")
```

## Download ahead of time with `aimu.pull()`

`aimu.pull()` takes an enum member or a `"provider:model_id"` string.

```python
aimu.pull("ollama:qwen3.5:9b")
aimu.pull("ollama:nomic-embed-text", host="http://gpu-box:11434")  # on a remote server
aimu.pull("hf:BAAI/bge-small-en-v1.5")
```

What it does depends on the provider:

- **Ollama** runs Ollama's own pull on the target server. Only missing or changed layers download, so
  pulling a model that is already current costs one round trip to the registry and nothing else. It
  is also how you update a model to the registry's newest build.
- **HuggingFace** downloads everything that model's client loads, into the local HuggingFace cache.
  Afterwards the client builds and runs with no network access (`HF_HUB_OFFLINE=1` works). Some
  clients load more than their own repo: SpeechT5 also fetches a vocoder and a speaker-embedding
  dataset, and `aimu.pull()` fetches those too. Only the weight format the loader reads is downloaded,
  not the ONNX, GGUF, or TensorFlow copies a repo may also carry.

The async twin is `await aimu.aio.pull(...)`, with the same arguments.

### What it refuses

- **A bare name** such as `"QWEN_3_8B"`. The same name is catalogued under several providers, so name
  the one to download from: `"ollama:qwen3:8b"`.
- **Cloud models** (Anthropic, OpenAI, Gemini): there is nothing to download.
- **llama.cpp**: it loads a GGUF file you supply with `model_path=`.
- **Local OpenAI-compatible servers** (LM Studio, vLLM, SGLang, llama-server, oMLX, HF Transformers
  Serve): each loads models its own way, such as `lms get` for LM Studio or launch arguments for vLLM.
  For `ollama-openai`, pull with the native provider: `aimu.pull("ollama:<id>")`.

## Download on construction with `pull=True`

```python
client = aimu.client("ollama:qwen3.5:9b", pull=True)
embedder = aimu.embedding_client("ollama:nomic-embed-text", pull=True)
```

This follows the same rule as `aimu.pull()`, so it costs a registry round trip on **every**
construction. That matters wherever clients are built often: `make_subagent_tool` builds a fresh
client per spawn. Prefer `aimu.pull()` once at startup.

The async client cannot wait in its constructor, so `aio.client(..., pull=True)` pulls before its
first request instead. Concurrent first requests share one pull, and a pull that fails is retried by
the next request.

## What gets logged

A WARNING is logged when a pull actually starts downloading, naming the model, the server, and the
size of the layer. Ollama has no way to ask "is there an update?" before pulling, so the warning
comes when bytes start moving, not before. Progress follows at INFO, and a completion line gives the
total downloaded. A model that was already up to date logs only at DEBUG.

## When a model is not installed

```python
client = aimu.client("ollama:qwen3:4b")
client.chat("hi")
# aimu.ModelNotInstalledError: Model 'qwen3:4b' is not installed on the Ollama server at
# http://127.0.0.1:11434. Install it with aimu.pull("ollama:qwen3:4b") (or `ollama pull qwen3:4b`
# on that server), or construct the client with pull=True.
```

`ModelNotInstalledError` is a distinct type, so it can go in `FallbackClient(retry_on=...)` to fall
back to a model that is installed. The server's own error is kept on `__cause__`.

A sub-agent spawned by `make_subagent_tool` whose model is missing does not raise into the parent's
run. The parent model receives a tool result saying the specialist could not start and that retrying
will not help, and a WARNING names the model and the fix for whoever runs the program.

## See also

- [Switch providers](switch-providers.md): `host=` for a remote Ollama server
- [Embed text](use-embeddings.md)
- [Spawn sub-agents](spawn-subagents.md)
