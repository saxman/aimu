# How-to guides

Task-oriented recipes. Each guide answers a specific question: how do I do *this*?

If you're new to AIMU, start with the [tutorials](../tutorials/index.md) instead — those build a working mental model. How-to guides assume you already know the basics and want the steps for a particular task.

## Working with models

- [Switch providers](switch-providers.md) — change backends without changing call sites
- [Add a new model](add-new-model.md) — register a model enum member
- [Stream output](stream-output.md) — `stream=True`, phase filtering, helpers
- [Use async (`aio`)](use-async.md) — embed AIMU in async apps; `asyncio.TaskGroup`-backed `Parallel`
- [Handle vision input](handle-vision.md) — pass images via `images=`
- [Generate images](generate-images.md) — `aimu.image_client()` / `generate_image()` with HuggingFace `diffusers` + Google Nano Banana
- [Generate audio](generate-audio.md) — `aimu.audio_client()` / `generate_audio()` with HuggingFace MusicGen, AudioLDM2, and Stable Audio
- [Generate speech](generate-speech.md) — `aimu.speech_client()` / `generate_speech()` for TTS with HuggingFace MMS-TTS/BARK or OpenAI tts-1/tts-1-hd

## Tools

- [Add a custom tool](add-custom-tool.md) — `@tool` decorator rules and patterns
- [Use MCP tools](use-mcp-tools.md) — cross-process tools via FastMCP

## Agents and workflows

- [Use skills](use-skills.md) — `SkillAgent` and the `SKILL.md` format
- [Build an orchestrator](build-orchestrator.md) — `OrchestratorAgent.assemble` or subclass
- [Plan, execute, evaluate, replan](plan-execute-evaluate.md) — `PlanExecuteEvaluator` for tasks with measurable success criteria

## Memory and persistence

- [Persist conversations](persist-conversations.md) — `ConversationManager`
- [Use semantic memory](use-semantic-memory.md) — `SemanticMemoryStore`
- [Use document memory](use-document-memory.md) — `DocumentStore`

## Prompts and evaluation

- [Tune prompts](tune-prompts.md) — hill-climbing optimisation against labelled data
- [Benchmark models](benchmark-models.md) — multi-model comparison harness
- [Integrate DeepEval](integrate-deepeval.md) — use DeepEval metrics as scorers / judges
