# Get structured output

Pass `schema=` to `chat()` or `generate()` to get a validated, typed object back instead of
a string. `schema` may be a **dataclass** (dependency-free) or a **Pydantic v2 model**.

```python
import aimu
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

client = aimu.client("openai:gpt-4.1")
person = client.chat("Extract the person: Ada Lovelace, 36.", schema=Person)
# person -> Person(name="Ada Lovelace", age=36)
```

## How it works: native, with a parse fallback

AIMU uses the best method the model supports, automatically:

- **Native enforcement** when the model has `supports_structured_output=True`: the provider
  constrains generation to the schema (OpenAI `response_format`, Ollama `format=`, Anthropic
  forced-tool, or Anthropic `output_config.format` on the models that reject a forced tool). Check it with `client.supports_structured_output`.
- **Prompt-and-parse** otherwise: the schema is appended to the prompt and the response is
  parsed.

Either way you get a validated instance or a `ValueError` (parsing failed). The choice is
based on the model's static capability, not on catching a runtime error, so a genuine
provider failure surfaces rather than silently downgrading.

```python
aimu.client("openai:gpt-4.1").supports_structured_output      # True  (native)
aimu.client("ollama:qwen3.5:9b").supports_structured_output   # True  (Ollama format=)
aimu.client("hf:Qwen/Qwen3-8B").supports_structured_output    # False (prompt-and-parse)
```

## Works on `generate()` too

```python
person = aimu.client("ollama:llama3.2:3b").generate("Ada Lovelace, 36", schema=Person)
```

Or one-shot:

```python
person = aimu.generate_json(client, "Ada Lovelace, 36", schema=Person)  # parse + retry helper
```

(`generate_json` is the older prompt-and-parse convenience with retries; `schema=` on
`chat`/`generate` is the newer path that prefers native enforcement.)

## Pydantic

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float
    paid: bool

invoice = client.chat("Acme, $1,250.00, unpaid", schema=Invoice)  # -> Invoice
```

Pydantic is optional; dataclasses work without it.

## With tools

`schema=` composes with tool calling on OpenAI-compatible and parse-path providers: tools
run in the loop, and the final answer obeys the schema.

**Anthropic is the exception.** On most of its catalog the native structured output *is* a
forced tool, which conflicts with offering action tools. Fable 5.1/5.2, Opus 5.5, and Sonnet 5.5
reject a forced tool, so they use native JSON outputs (`output_config.format`) instead, but AIMU's
structured turn on them still advertises no tools. Either way, combining `schema=` with active
`tools=` raises a `ValueError`. Drop the tools (or `use_tools=False`), or use a provider whose
`response_format` composes with tools.

Native JSON outputs are stricter about the schema than the forced tool. AIMU adds
`additionalProperties: false` to every object that leaves it unset, as the API requires. It does
not remove constraints the API rejects, such as `minimum` or `maxLength` on a Pydantic `Field`;
the API's 400 names them.

## Streaming structured output

`schema=` combines with `stream=True`. The call returns an iterator of `StreamChunk`s so you can
show thinking / generation live, and the validated object is delivered two ways:

- a terminal `StreamChunk` with `phase == DONE` whose `content` is `{"result": <object>}`, and
- `client.last_structured`, set once the stream is fully consumed (mirrors `client.last_usage`).

```python
for chunk in client.chat("Is 17 prime? Think, then answer.", schema=Verdict, stream=True):
    if chunk.phase is StreamingContentType.THINKING:
        print(chunk.content, end="")      # reasoning tokens, live
    elif chunk.is_done():
        verdict = chunk.content["result"]  # validated Verdict
# ...or just: verdict = client.last_structured   (after the loop)
```

An `include=` filter still applies to the thinking/generation phases, but the terminal result chunk
is **always** emitted so the payload is never filtered away.

**Anthropic usually streams the JSON, not thinking.** On most of its catalog the structured mode
is a forced tool, which the API forbids alongside extended thinking, so the stream yields the answer
JSON as it is built (GENERATING chunks) with no THINKING, and a `thinking=` level is warned about and
dropped. The models on native JSON outputs (Fable 5.1/5.2, Opus 5.5, Sonnet 5.5) keep thinking,
stream it, and honour `thinking=`. Ollama and OpenAI-compatible thinking models stream thinking
normally.

## Constraints

- **`self.messages` stays plain.** The typed object is a return value only; the assistant turn
  is stored as the plain JSON string, so conversation history remains provider-portable.

## Async

Identical on the async surface:

```python
from aimu import aio

async def main():
    client = aio.client("openai:gpt-4.1")
    person = await client.chat("Ada Lovelace, 36", schema=Person)
```

## Which models enforce natively

`supports_structured_output=True` (native) on the OpenAI, Gemini, Ollama (all models), and
Anthropic catalogs; `Client.STRUCTURED_MODELS` lists them per provider. HuggingFace and
llama-cpp use the prompt-and-parse path. See [parse helpers](../reference/api/aimu.md) for
`parse_json_response` / `generate_json`, which back the coercion.

## See also

- [Stream output](stream-output.md): the general streaming surface; `schema=` combines with it
- [Switch providers](switch-providers.md): `supports_structured_output` varies by provider
