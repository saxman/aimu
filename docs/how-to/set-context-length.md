# Set the context length

The context window is how much of a conversation a model can see at once. On a local backend it is
a runtime choice you pay for in memory, not a fixed property of the weights, and the default is
often smaller than the model supports. AIMU exposes it as one portable `context_length` key in
`generate_kwargs`, so a call site does not have to know which provider spells it `num_ctx`.

```python
import aimu

client = aimu.client("ollama:qwen3.8:27b")

# Every call on this client
client.default_generate_kwargs = {"context_length": 32768}
client.chat("Summarise the attached transcript.")

# Just this call
client.chat("And now the long one.", generate_kwargs={"context_length": 131072})
```

It layers like any other generation parameter, so tier 4 (the per-call dict) beats tier 3 (the
client default). See [the four tiers](../reference/provider-matrix.md#generation-parameters).

## What it is not

`context_length` sizes the window the model reads. `max_tokens` caps what it writes. They are
independent: a 32k window with `max_tokens=512` reads a long conversation and answers briefly.

## Cancelling a client default

`None` means unset, which is the only way back to the backend's own sizing once a client default
is in place:

```python
client.default_generate_kwargs = {"context_length": 32768}
client.chat("this one can be small", generate_kwargs={"context_length": None})
```

## Where each provider takes it

Only Ollama's native API accepts a context length per request. Everywhere else the window is sized
out of band, so AIMU drops the key and logs a warning naming where to set it instead — rather than
raising, so moving a working client default to another provider never breaks the call.

| Provider | Per request? | Where it comes from |
|---|---|---|
| `ollama` | **yes** — `num_ctx` | the request, or `OLLAMA_CONTEXT_LENGTH` on the server |
| `llamacpp` | no | `LlamaCppClient(..., n_ctx=N)`, at load time |
| `ollama-openai` | no | `OLLAMA_CONTEXT_LENGTH` on the server |
| `llamaserver`, `vllm`, `sglang` | no | server launch (`--ctx-size`, `--max-model-len`) |
| `lmstudio`, `hf-openai`, `omlx` | no | server launch (LM Studio's context-length setting) |
| `hf` | no | the weights' own `max_position_embeddings` |
| `anthropic`, `openai`, `gemini` | no | fixed by the vendor |

The warning fires once per client, not once per call, so an agent loop does not repeat it every
round.

## Raising it to fix a failing request

A request that no longer fits raises
[`ContextOverflowError`](../reference/provider-matrix.md) on Ollama, and a turn cut off before it
produced anything raises `TruncatedTurnError`. Both are worth reading as "the window is too small
for this conversation": raise `context_length`, shorten the history, or advertise fewer tools.

`TruncatedTurnError` fires on every provider as of v0.27.0. Before that only Ollama reported the
signal it reads (`client.last_output_truncated`), so on the other backends a cut-off turn came back
as a bare empty string. `client.last_stop_reason` carries the provider's own word for how the turn
ended (`"length"`, `"max_tokens"`, `"stop"`, ...) if you want to inspect it directly; `None` means
the provider said nothing, which is not the same as "finished normally".

```python
from aimu.models import ContextOverflowError

try:
    agent.run(task)
except ContextOverflowError:
    client.default_generate_kwargs["context_length"] = 65536
    agent.run(task)
```

A larger window costs memory for the KV cache, and on a local backend an over-large value fails at
load rather than degrading, so raise it deliberately rather than setting the maximum by default.

## Adding it to a new provider

A client declares what it does with the key as one entry in `GENERATE_KWARG_SUPPORT`, the table the
shared merge reads for every portable generation parameter (see
[Add a new provider](add-new-provider.md)):

```python
class MyClient(BaseModelClient):
    GENERATE_KWARG_SUPPORT = {
        # ...the other seven portable keys...
        "context_length": "num_ctx",  # the backend's own name for it
        # ...or, when it cannot be set per request, the remedy to name in the warning:
        # "context_length": Unsupported("Set it when starting the server (--ctx-size)."),
    }
```

The entry is required; a test fails if a client leaves any portable key undeclared, because an
undeclared key is forwarded unchanged, and a backend that cannot take it either rejects the whole
request or discards the value with nothing said.
