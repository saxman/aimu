# Changelog

## v0.4 (unreleased) — API redesign

Breaking changes across four areas, plus the new documentation site.

### Top-level API

- **New** `aimu.chat(user_message, *, model, ...)` — one-shot chat with a model string or enum.
- **New** `aimu.client(model, *, system=None, **kwargs)` — one-line `ModelClient` factory.
- **New** `aimu.resolve_model_string("provider:model_id")` — model-string parser.
- **New** `ModelClient` now accepts a `"provider:model_id"` string in addition to enum members.

### Model clients

- **New** `ModelSpec` frozen dataclass replaces positional enum tuples. All `Model` enums migrated.
- **New** `client.reset(system_message="__keep__")` clears history and unlocks the system-message setter.
- **Breaking** `system_message` is immutable after the first `chat()` call. The setter raises `RuntimeError`; call `reset()` to unlock.
- **New** `include=[...]` stream filter on `chat()` and `generate()` selects phases (`"thinking"`, `"tool_calling"`, `"generating"`, `"done"`).
- **Internal** Abstract methods renamed `chat → _chat`, `generate → _generate`. Concrete `chat`/`generate` on the base class apply the `include` filter and delegate.

### Agents

- **Breaking** `Agent` constructor signature changed: `Agent(model_client, system_message=None, name=None, tools=None, ...)`. `system_message` is the second positional argument; `name` is optional (auto-derived).
- **Breaking** `AgenticModelClient` removed from the public API. Use `agent.as_model_client()` instead.
- **Breaking** `OrchestratorAgent._setup_orchestrator` renamed to `_init_orchestrator`.
- **New** `OrchestratorAgent.assemble(client, system_message, workers=[...])` factory builds an orchestrator without subclassing.
- **New** Workflow factories: `Chain.of(client, prompts)`, `Router.of(client, classifier_prompt, handlers)`, `Parallel.of(client, worker_prompts, aggregator_prompt=)`.
- **Breaking** `AgentChunk` and `ChainChunk` collapsed into `StreamChunk` (back-compat aliases kept). `chunk.agent_name → chunk.agent`; `chunk.step → chunk.iteration`.

### Tools

- **New** `@tool` raises `ToolSignatureError` at decoration time on unsupported signatures (`*args`/`**kwargs`, params with no type hint and no default).
- **New** `Optional[T]` and `T | None` unwrap to the inner type in tool specs.
- **New** Built-in tool subgroups: `builtin.web`, `builtin.fs`, `builtin.compute`, `builtin.misc`.
- **New** `MCPClient` raises `MCPConnectionError` (rather than silently failing) on construction or call failure. Added `.ping()` method.

### Skills

- **Breaking** `SkillManager` raises `SkillLoadError` on malformed `SKILL.md` (instead of silently skipping).
- **Breaking** `SkillManager.get_skill_body()` raises `SkillNotFoundError` on unknown skill name (instead of returning a sentinel string).
- **New** Skill catalogue prompt includes script-derived tool names inline.
- **New** `Skill` renamed to `AgentSkill` (back-compat alias kept).
- **New** Skills logged at `INFO` on discovery.

### Documentation

- New documentation site built with MkDocs Material and hosted on GitHub Pages.
- Diátaxis structure: tutorials, how-to guides, reference, explanation.
- README slimmed to landing-page size.

## Earlier versions

This is the first formal changelog entry. Prior versions tracked changes via git history; consult `git log` on GitHub for v0.3.x and earlier.
