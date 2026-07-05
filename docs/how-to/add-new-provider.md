# Add or update a provider

A **provider** is a backend AIMU talks to (Ollama, Anthropic, an OpenAI-compatible server, …). Adding one means writing a client class and wiring it into the factory. This is different from [adding a new *model*](add-new-model.md), which is just a new member on an existing provider's `Model` enum.

Provider clients live under [`aimu/models/providers/`](https://github.com/saxman/aimu/tree/main/aimu/models/providers), one module per provider. The public surface is unchanged by where the file sits: users still reach your client through `aimu.client("yourprovider:model-id")`, `ModelClient(YourModel.X)`, or `from aimu.models import YourClient`.

## 1. Decide the file layout

- **Flat module**: `providers/<name>.py`. The default, used by `anthropic.py`, `ollama.py`, `llamacpp.py`.
- **Subpackage**: `providers/<name>/<modality>.py`, **only** when the provider ships *more than one* standalone modality client (the rule that gives `hf/`, `openai/`, `gemini/` their `text.py` / `image.py` / …). A single-client provider stays flat.

## 2. Write the client

There are two paths depending on the backend's API.

### Path A: OpenAI-compatible endpoint (easiest)

If the backend speaks the OpenAI REST API, you don't reimplement chat/streaming/tools. Subclass `OpenAICompatClient` and supply a `base_url` + a `Model` enum. For a *local inference server*, add it alongside the others in [`providers/openai_compat.py`](https://github.com/saxman/aimu/blob/main/aimu/models/providers/openai_compat.py):

```python
# aimu/models/providers/openai_compat.py
class MyServerOpenAIModel(Model):
    QWEN_3_8B = ModelSpec("qwen3-8b", tools=True, thinking=True)


class MyServerOpenAIClient(OpenAICompatClient):
    MODELS = MyServerOpenAIModel

    def __init__(self, model: MyServerOpenAIModel, base_url: str = "http://localhost:9000/v1", **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)
```

A *cloud* provider with its own identity and ≥2 modalities gets a subpackage instead. See how `OpenAIClient` lives in `providers/openai/text.py` and subclasses `OpenAICompatClient` via `from ..openai_compat import OpenAICompatClient`.

### Path B: native SDK

For a backend with its own SDK/wire format, subclass `BaseModelClient` and implement the three abstract methods plus the capability classproperties. `self.messages` always stays in OpenAI format; adapt to the provider's format at request time (never mutate `self.messages`). Reuse the shared helpers in [`aimu/models/_internal/`](https://github.com/saxman/aimu/tree/main/aimu/models/_internal), such as `streaming` and `image_input`:

```python
# aimu/models/providers/myprovider.py
from ..base import BaseModelClient, Model, ModelSpec, StreamChunk, StreamingContentType, classproperty


class MyProviderModel(Model):
    BIG = ModelSpec("big-v1", tools=True, thinking=True, vision=True)


class MyProviderClient(BaseModelClient):
    MODELS = MyProviderModel

    @classproperty
    def TOOL_MODELS(cls):      # noqa: N805
        return [m for m in cls.MODELS if m.supports_tools]

    @classproperty
    def THINKING_MODELS(cls):  # noqa: N805
        return [m for m in cls.MODELS if m.supports_thinking]

    @classproperty
    def VISION_MODELS(cls):    # noqa: N805
        return [m for m in cls.MODELS if m.supports_vision]

    def __init__(self, model, model_kwargs=None, system_message=None):
        super().__init__(model, model_kwargs, system_message)
        # ... construct the backend SDK client ...

    def _update_generate_kwargs(self, generate_kwargs=None): ...
    def _generate(self, prompt, generate_kwargs=None, stream=False, images=None): ...
    def _chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None): ...
```

`chat()` / `generate()` (and the `include=` stream filter) are concrete on the base; you only implement `_chat` / `_generate`. Use `self._chat_setup(...)` to build the request; when the model returns tool calls, call `self._record_tool_calls(tool_calls, content)` to parse and store them on the assistant message. `_chat` only *records* tool calls — it never executes them. Tool execution and the call-model → run-tools → repeat loop belong to the agent's tool-loop engine (`aimu.agents._tool_loop`), not the provider. See [`providers/anthropic.py`](https://github.com/saxman/aimu/blob/main/aimu/models/providers/anthropic.py) for a full native example (including the OpenAI↔Anthropic format adapters).

!!! note "Provider-local helpers"
    A helper used by *one* provider family lives with it (e.g. `providers/hf/_device.py`, `providers/_thinking.py`), not in `_internal/`. Put anything only your provider needs next to your provider.

## 3. Wire the factory

In [`aimu/models/model_client.py`](https://github.com/saxman/aimu/blob/main/aimu/models/model_client.py), three small edits make `ModelClient` and `"provider:id"` strings work:

```python
# 1. Guarded import (so a missing optional dep degrades gracefully)
try:
    from .providers.myprovider import MyProviderClient, MyProviderModel
    _HAS_MYPROVIDER = True
except Exception:
    _HAS_MYPROVIDER = False
    MyProviderClient = MyProviderModel = None  # type: ignore[assignment,misc]

# 2. Registry entry in _provider_registry()  ->  enables "myprovider:big-v1" strings
if _HAS_MYPROVIDER:
    registry["myprovider"] = (MyProviderModel, MyProviderClient)

# 3. Dispatch in ModelClient.__init__  ->  enables ModelClient(MyProviderModel.BIG)
elif _HAS_MYPROVIDER and isinstance(model, MyProviderModel):
    self._client = MyProviderClient(model, **kwargs)
```

## 4. Export it (with graceful degradation)

In [`aimu/models/__init__.py`](https://github.com/saxman/aimu/blob/main/aimu/models/__init__.py), re-export the client + enum under a `try/except` that sets a `HAS_MYPROVIDER` flag and `None` fallbacks, then add the names to `__all__` when the flag is set. This keeps `import aimu` working on a minimal install. (Path A local-server subclasses are exported the same way, alongside the other `openai_compat` names.)

## 5. Mirror the async surface

Add `aimu/aio/providers/myprovider.py`:

- **Native** providers subclass `AsyncBaseModelClient` (async `_chat` / `_generate`; `asyncio.TaskGroup` for concurrent tool calls).
- **In-process** providers (those that load weights, like HF/LlamaCpp) instead *wrap a sync client* so they don't load weights twice (Decision 7, see [async design](../explanation/async-design.md)).

Then wire `aimu/aio/_model_client.py` (same three edits as step 3) and export from `aimu/aio/__init__.py`.

## 6. Tests

- **Mock** coverage: `tests/test_models_api.py` (no backend needed). This is the canary that catches wiring/import breaks.
- **Live** coverage flows through `tests/test_models.py`; add your provider to the dispatch in `tests/conftest.py` / `tests/helpers.py` so `--client=myprovider` resolves. Live tests are opt-in; bare `pytest` skips them.

## Verify

```python
from aimu.models import ModelClient, resolve_model_string

assert resolve_model_string("myprovider:big-v1").supports_tools
client = ModelClient("myprovider:big-v1")
print(client.chat("hello"))
```

```bash
pytest tests/test_models_api.py            # mock wiring (always)
pytest tests/test_models.py --client=myprovider   # live (needs the backend)
```

## See also

- [Add a new model](add-new-model.md): register a model on an *existing* provider
- [Architecture](../explanation/architecture.md): the `BaseModelClient` contract and factory pattern
- [Use async (`aio`)](use-async.md): the async surface your mirror plugs into
