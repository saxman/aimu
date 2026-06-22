# Getting started

This is a 15-minute walkthrough from a fresh install to a working agent. By the end you'll know the three core entry points (`aimu.chat()`, `aimu.client()`, and `Agent`) and how to swap providers without changing call sites.

## 1. Install

Pick a backend. For this tutorial we'll use **Ollama** because it's local and free:

```bash
pip install aimu[ollama]
```

Then pull a small tool-capable model:

```bash
ollama pull qwen3.5:9b
```

For cloud providers instead, use `pip install aimu[anthropic]` (or `[openai_compat]`) and set the corresponding API key in your environment. Every example below works identically; only the model string changes.

## 2. Your first chat

```python
import aimu

response = aimu.chat("What is the capital of France?", model="ollama:qwen3.5:9b")
print(response)
```

You should see something like *"The capital of France is Paris."*

`aimu.chat()` is a one-shot: it builds a fresh client, sends one message, returns the response, and is done. There's no client object to manage.

### Omitting the model

You can leave out `model=` entirely:

```python
response = aimu.chat("What is the capital of France?")
```

When the model is omitted, AIMU resolves a default in this order:

1. The **`AIMU_LANGUAGE_MODEL`** env var (a `"provider:model_id"` string). Set it in your project's `.env` to pin a default: `AIMU_LANGUAGE_MODEL=ollama:qwen3.5:9b`.
2. An **already-available local model**: a running Ollama server, a model already in your HuggingFace cache, or a running local OpenAI-compatible server (LM Studio, vLLM, llama-server, SGLang). The chosen model is logged.
3. Otherwise a `ValueError` listing how to fix it.

AIMU never auto-selects a cloud provider (no surprise API bills) and never downloads weights implicitly. Passing `model=` explicitly (as every example here does) is always the clearest, most reproducible choice.

## 3. Multi-turn conversation

For a conversation, build a reusable client with `aimu.client()`:

```python
import aimu

client = aimu.client("ollama:qwen3.5:9b", system="You are concise.")

client.chat("My favourite colour is blue.")
print(client.chat("What did I just tell you?"))
# 'You told me your favourite colour is blue.'
```

`client.chat()` accumulates history in `client.messages`. The system message is set at construction time and locks once the first chat is sent. Call `client.reset()` if you need to start over.


## 4. Streaming

For UIs and progress visibility, pass `stream=True`. You get an iterator of `StreamChunk` objects:

```python
for chunk in aimu.chat("Tell me a short story", model="ollama:qwen3.5:9b", stream=True):
    if chunk.is_text():
        print(chunk.content, end="", flush=True)
```

`chunk.is_text()` is `True` for both `THINKING` and `GENERATING` phases. For finer control see [how-to: stream output](../how-to/stream-output.md).

## 5. Swap providers

The whole point of `ModelClient` is provider-agnostic code. Switch by changing the model string:

```python
# Local Ollama
aimu.chat("hi", model="ollama:qwen3.5:9b")

# Anthropic (needs ANTHROPIC_API_KEY)
aimu.chat("hi", model="anthropic:claude-sonnet-4-6")

# OpenAI (needs OPENAI_API_KEY)
aimu.chat("hi", model="openai:gpt-4o-mini")

# Google Gemini (needs GOOGLE_API_KEY)
aimu.chat("hi", model="gemini:gemini-2.5-flash")
```

The rest of your code is unchanged. See [how-to: switch providers](../how-to/switch-providers.md) for the full list.

## 6. Your first agent

So far we've called `chat()` directly. An `Agent` adds a tool-using loop on top: it keeps calling `chat()` until the model stops invoking tools.

First, declare a tool:

```python
import aimu

@aimu.tool
def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a letter in a word."""
    return word.lower().count(letter.lower())
```

The `@aimu.tool` decorator inspects the signature and docstring to build a tool spec for the model. The function itself is unchanged.

Then wrap a client in an `Agent`:

```python
from aimu.agents import Agent

client = aimu.client("ollama:qwen3.5:9b")
agent = Agent(client, "You are a helpful assistant.", tools=[letter_counter])

print(agent.run("How many r's are in 'strawberry'?"))
# The model calls letter_counter(word="strawberry", letter="r"), gets 3,
# and responds with the answer.
```

The agent's loop: send the user's message → if the model called tools, dispatch them and continue → repeat until the model returns text without calling tools.

## What's next

You've now used the three load-bearing APIs:

- `aimu.chat()` for one-shots
- `aimu.client()` for conversations
- `Agent` for autonomous tool-using loops

The next tutorials build on each:

- **[First agent with tools](02-first-agent-with-tools.md)**: deeper on `@tool` and built-in tools.
- **[Workflows](03-workflows.md)**: code-controlled patterns (chain / router / parallel) when you want the *flow* fixed.
- **[Vision and streaming](04-vision-and-streaming.md)**: image input plus the full `StreamChunk` API.

Or jump into [how-to guides](../how-to/index.md) for specific tasks.
