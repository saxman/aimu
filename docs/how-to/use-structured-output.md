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

- **Native enforcement** when the model has `supports_structured_output=True` — the provider
  constrains generation to the schema (OpenAI `response_format`, Ollama `format=`, Anthropic
  forced-tool). Check it with `client.supports_structured_output`.
- **Prompt-and-parse** otherwise — the schema is appended to the prompt and the response is
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

Pydantic is optional — dataclasses work without it.

## With tools

`schema=` composes with tool calling on OpenAI-compatible and parse-path providers: tools
run in the loop, and the final answer obeys the schema.

**Anthropic is the exception.** Its native structured output *is* a forced tool, which
conflicts with offering action tools, so combining `schema=` with active `tools=` raises a
`ValueError`. Drop the tools (or `use_tools=False`), or use a provider whose `response_format`
composes with tools.

## Constraints

- **`schema=` and `stream=True` are mutually exclusive** — a typed object can't be streamed
  incrementally; passing both raises `ValueError`.
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

- [Stream output](stream-output.md) — note that streaming and `schema=` don't combine
- [Switch providers](switch-providers.md) — `supports_structured_output` varies by provider
