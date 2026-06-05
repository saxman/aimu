# Contributing

Thanks for considering a contribution.

## Dev setup

Clone and install with all extras:

```bash
git clone https://github.com/saxman/aimu
cd aimu
pip install -e '.[all,dev,notebooks,docs]'
```

Or with [uv](https://docs.astral.sh/uv/) (faster):

```bash
uv sync --all-extras
```

For gated HuggingFace models, log in once:

```bash
hf auth login
```

## Run the tests

The default suite is mock-based and needs no backend:

```bash
pytest tests/ --ignore=tests/test_models.py
```

To run against a real backend:

```bash
pytest tests/test_models.py --client=ollama --model=QWEN_3_5_9B
pytest tests/test_models.py --client=anthropic
pytest tests/test_models.py --client=llamacpp --model-path=/path/to/model.gguf
```

See [CLAUDE.md](https://github.com/saxman/aimu/blob/main/CLAUDE.md) for the full list of `--client` options.

## Linting

Ruff is configured in `pyproject.toml` (line length 120):

```bash
ruff check .
ruff format .
```

## Build the docs locally

```bash
pip install -e '.[docs]'
mkdocs serve
```

The site rebuilds on file changes. Browse at `http://127.0.0.1:8000`.

To check for broken links and missing pages:

```bash
mkdocs build --strict
```

## Coding conventions

- **Plain Python over framework primitives.** No `Runnable` protocol, no `BaseTool`, no LCEL `|`, no Pydantic for tool args. See [design principles](explanation/design-principles.md) for the full list of what's deliberately out of scope.
- **OpenAI message dicts as the only data model.** No `Message` class. Provider-specific formats are adapted at request time, never persisted.
- **Loud failures.** Bad input raises with an actionable message. Silent skips and `try/except: pass` are reviewed carefully.

## Adding a new provider

Write a client module under `aimu/models/providers/` (a flat `providers/<name>.py`, or a `providers/<name>/<modality>.py` subpackage only if the provider ships several standalone modality clients), subclass `BaseModelClient` (or `OpenAICompatClient` for OpenAI-compatible endpoints), wire it into the `ModelClient` factory and `_provider_registry()`, export it from `aimu/models/__init__.py` under a `HAS_*` flag, and mirror it on the async surface.

See [how-to: add or update a provider](how-to/add-new-provider.md) for the full step-by-step, and an existing provider (e.g. `providers/anthropic.py`) for the pattern.

## Adding a new model to an existing provider

Add a member to the provider's `Model` enum with a `ModelSpec(id, tools=..., thinking=..., vision=...)` value. The `TOOL_MODELS` / `THINKING_MODELS` / `VISION_MODELS` lists derive automatically. See [how-to: add a new model](how-to/add-new-model.md).

## Pull requests

- One concern per PR. Splitting a refactor across multiple PRs makes review tractable.
- Include or update tests. Each module has a dedicated test file (`tests/test_models_api.py`, `tests/test_tool_decorator.py`, `tests/test_workflow_chain.py`, etc.) — add new behavioural tests next to the existing ones for the surface you're touching.
- Update docs when adding a public API. New `@tool`-decorated function? Add it to `builtin.<group>`. New workflow? Add a tutorial or how-to.
- Run `ruff check .` and `pytest` before pushing.

## Reporting bugs

File an issue with:

- A minimal reproducer (under 20 lines if possible).
- The full traceback.
- AIMU version, Python version, and which provider you were using.

## Questions

Open a GitHub discussion. For Claude Code agents working in this repo, [CLAUDE.md](https://github.com/saxman/aimu/blob/main/CLAUDE.md) is the canonical engineering reference.
