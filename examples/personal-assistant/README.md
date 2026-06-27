# Personal Assistant

A reference single-user personal assistant in the style of
[OpenClaw](https://github.com/openclaw/openclaw) and
[Hermes Agent](https://hermes-agent.nousresearch.com/), built entirely from AIMU primitives.
It shows how the library composes into an always-on assistant without the library itself
becoming an application.

It wires together:

- **A channel** (`aimu.aio.CLIChannel`) for input/output. The `Channel` ABC is transport-only,
  so a network adapter (Telegram, Slack) is a drop-in replacement.
- **A `SkillAgent`** that answers each message and can **author new skills** at runtime via the
  `author_skill` tool (`aimu.skills.make_skill_authoring_tool`): the self-improvement loop.
- **A `Scheduler`** (`aimu.aio.Scheduler`) for proactive messages (reminders / check-ins).
- **`ConversationManager`** for history that survives restarts.

## Run

```bash
python examples/personal-assistant/assistant.py \
    --model anthropic:claude-sonnet-4-6 \
    --reminder-seconds 30
```

Omit `--model` to use `AIMU_LANGUAGE_MODEL` or a locally available model. Chat at the `>`
prompt; press Ctrl-D to exit.

## What to try

- **Proactive message.** With `--reminder-seconds 30`, an unprompted suggestion appears ~30s
  after startup, pushed through the channel by the scheduler.
- **Self-authored skill.** Teach the assistant a procedure: *"Remember how I like my standup
  formatted: three bullets, Yesterday, Today, Blockers."* It calls `author_skill`, which writes
  a `SKILL.md` under `--skills-dir` and refreshes the skill manager. In a later conversation the
  assistant can `activate_skill` to recall it.
- **Self-authored *script*.** Ask it to automate something: *"Write a shell script that shows my
  disk usage, then run it."* It calls `add_skill_script` to drop a `.py` or `.sh` file into a
  skill's `scripts/` directory; `reload_skills()` makes the new `{skill}__{stem}` tool callable in
  the same turn, so it runs the script and reports the output.
- **Persistence.** Restart with the same `--history` path and the prior conversation is restored.

By default the assistant keeps all of its state (authored skills and conversation history) under
`<output>/personal-assistant/` (where `<output>` is `aimu.paths.output`), so a run leaves nothing
in your working directory. Override with `--skills-dir` and `--history`.

## Layout

| File | Role |
|------|------|
| `assistant.py` | CLI entry point (argparse) + `asyncio.run` |
| `_assistant_common.py` | `Assistant` / `AssistantConfig` (the testable daemon core) |
| `tests/test_assistant.py` | Mock-only tests (fake channel + mock model) |

## Security

Authored skill scripts run as **real subprocesses with your user privileges, no sandbox**, exactly
like OpenClaw and Hermes Agent. Real capability is the point of a personal assistant, but it means a
prompt-injected or mistaken model can run arbitrary code on your machine. Only run this with a model
and inputs you trust. The daemon prints a notice to this effect on startup. For untrusted code,
`builtin.execute_python` is a sandboxed alternative (no filesystem or subprocess access). Script
execution also blocks the event loop for up to its 30-second timeout.

## Notes & limits

- **Single user.** A *personal* assistant serves one person, so there is no multi-user session
  keying; `ConversationManager` holds one conversation.
- **Scheduler is in-memory.** Jobs are Python callables registered in code, not durable across
  restarts. Durable cron-like scheduling belongs in a wrapper above the library.
