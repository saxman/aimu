# Reference

Information-oriented documentation: exhaustive, dry, accurate. Read it like a dictionary.

## API reference

Auto-generated from docstrings. One page per package:

- [`aimu`](api/aimu.md): top-level `chat()`, `client()`, model-string parser
- [`aimu.aio`](api/aio.md): async mirror of the public surface (`AsyncModelClient`, async `Agent`/`Chain`/`Parallel`/…, `aio.MCPClient`)
- [`aimu.models`](api/models.md): `ModelClient`, `BaseModelClient`, `ModelSpec`, `StreamChunk`, provider clients
- [`aimu.agents`](api/agents.md): `Agent`, `SkillAgent`, `OrchestratorAgent`, `Chain`, `Router`, `Parallel`, `EvaluatorOptimizer`
- [`aimu.tools`](api/tools.md): `@tool` decorator, `MCPClient`, `builtin` tool groups
- [`aimu.skills`](api/skills.md): `AgentSkill`, `SkillManager`, MCP server builder
- [`aimu.memory`](api/memory.md): `SemanticMemoryStore`, `DocumentStore`
- [`aimu.prompts`](api/prompts.md): `PromptCatalog`, `PromptTuner` and concrete tuners, `Scorer`
- [`aimu.evals`](api/evals.md): `Benchmark`, `DeepEvalModel`, `DeepEvalScorer`
- [`aimu.history`](api/history.md): `ConversationManager`

## Other reference

- [Provider matrix](provider-matrix.md): every provider with extras, env vars, defaults
- [Model matrix](model-matrix.md): every model enum member with capability flags
- [Stream phases](stream-phases.md): what each `StreamingContentType` value means
- [Environment variables](env-vars.md): every env var AIMU reads
- [CLI](cli.md): runnable `python -m` entry points (MCP servers, etc.)
