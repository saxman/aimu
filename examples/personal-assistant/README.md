# Personal Assistant

A reference single-user personal assistant in the style of
[OpenClaw](https://github.com/openclaw/openclaw) and
[Hermes Agent](https://hermes-agent.nousresearch.com/), built entirely from AIMU primitives.
It shows how the library composes into an always-on assistant without the library itself
becoming an application.

It wires together:

- **A channel** for input/output: `aimu.aio.CLIChannel` (terminal) or the example's `WebChannel`
  (a browser WebSocket, see [Web front end](#web-front-end)). The `Channel` ABC is transport-only,
  so each is a drop-in replacement, and a network adapter (Telegram, Slack) would be too.
- **A `SkillAgent`** that answers each message and can **author new skills** at runtime via the
  `author_skill` tool (`aimu.skills.make_skill_authoring_tool`): the self-improvement loop.
- **AIMU's built-in tools** (`aimu.tools.builtin`) selected by group via `--tools` (web search,
  filesystem reads, a calculator + sandboxed Python, date/time by default).
- **Remote MCP servers** by URL via `--mcp` (repeatable) or the runtime `add_mcp_server` tool, so the
  assistant can use a hosted service's tools (`aimu.aio.MCPClient.connect(url=...)`).
- **Persistent memory** (on by default): a `SemanticMemoryStore` for facts about the user
  (`store_memory` / `search_memories`) and a `DocumentStore` for longer documents the user provides
  (`save_document` / `search_documents`). Both persist under the output dir, so memory survives restarts
  and spans conversations. Disable with `--no-memory`.
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

To use a remote MCP service's tools, pass its URL (repeatable; `--mcp-bearer TOKEN` for an
authenticated server):

```bash
python examples/personal-assistant/assistant.py \
    --mcp https://mcp.example.com/sse --mcp-bearer "$MY_TOKEN"
```

## Web front end

The same assistant is also served over a browser WebSocket (install the web stack with
`pip install aimu[web]`):

```bash
python examples/personal-assistant/web_assistant.py \
    --model ollama:qwen3:8b \
    --reminder-seconds 20
# then open http://127.0.0.1:8000
```

This is a small [Starlette](https://www.starlette.io/) server plus `web_channel.py`'s
`WebChannel` (a `Channel` over the browser socket). The `Assistant` loop is **unchanged**: only the
channel differs, the whole point of the `Channel` ABC. Replies stream token-by-token, the model's
reasoning and tool calls render as distinct blocks as they happen, and proactive scheduler messages
arrive in the browser unprompted (something a request/response UI like Streamlit/Gradio can't do
without polling). `--host` / `--port` accept the usual flags; single-user, so a second simultaneous
connection is rejected.

## What to try

- **Built-in tools.** The assistant gets AIMU's built-in tools by default (`--tools web,fs,compute,misc`):
  ask *"what's the weather in Paris?"*, *"what's today's date?"*, or *"what is 17 times 23?"*. Select
  groups with `--tools` (`web`, `fs`, `compute`, `misc`, `image`, `audio`, `speech`, `transcription`,
  or `all` / `none`); the generative groups need their `AIMU_*_MODEL` env var. Note `compute`'s
  sandboxed `execute_python` (safe, ephemeral compute) is **complementary** to the full-access skill
  scripts below (persistent, reusable automation), not a replacement.
- **See the model work.** Both the CLI and web UI show each turn's reasoning and tool calls, not
  just the final answer (on by default; pass `--no-show-thinking` / `--no-show-tools` to hide them).
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
- **Remote MCP tools.** Start with `--mcp <url>` to give the assistant a hosted service's tools, or
  ask it mid-session: *"connect to the MCP server at https://..."* (it calls `add_mcp_server`, and the
  new tools are callable in the same turn). Added servers last for the session; `--mcp` reconnects each
  boot.
- **Memory.** Tell it a durable fact, *"my standup is at 9am"*; restart and ask *"when is my standup?"*
  and it recalls it from the `SemanticMemoryStore`. Paste a longer note and say *"save this as a
  document"*, then later *"search my documents for X"* (the `DocumentStore`). `--no-memory` disables both.
- **Persistence.** Restart with the same `--history` path and the prior conversation is restored.

By default the assistant keeps all of its state (authored skills, conversation history, and memory)
under `<output>/personal-assistant/` (where `<output>` is `aimu.paths.output`), so a run leaves nothing
in your working directory: skills under `skills/`, history in `history.json`, semantic facts under
`memory/`, and documents under `documents/`. Override the skills/history locations with `--skills-dir`
and `--history`.

## Layout

| File | Role |
|------|------|
| `assistant.py` | CLI entry point (argparse) + `asyncio.run` |
| `_assistant_common.py` | `Assistant` / `AssistantConfig` (the testable daemon core) |
| `web_assistant.py` | Web entry point: a Starlette + `uvicorn` WebSocket server (needs `aimu[web]`) |
| `web_channel.py` | `WebChannel`: a `Channel` adapter over a browser WebSocket |
| `static/index.html` | Dependency-free chat page (streaming replies + proactive messages) |
| `tests/test_assistant.py` | Mock-only tests (fake channel + mock model) |
| `tests/test_web_assistant.py` | Mock-only WebChannel + WebSocket round-trip tests |

## Security

Authored skill scripts run as **real subprocesses with your user privileges, no sandbox**, exactly
like OpenClaw and Hermes Agent. Real capability is the point of a personal assistant, but it means a
prompt-injected or mistaken model can run arbitrary code on your machine. Only run this with a model
and inputs you trust. The daemon prints a notice to this effect on startup. For untrusted code,
`builtin.execute_python` is a sandboxed alternative (no filesystem or subprocess access). Script
execution also blocks the event loop for up to its 30-second timeout.

Remote MCP servers are a similar trust surface: their tools run whatever the service exposes, so only
connect (`--mcp` or `add_mcp_server`) to services you trust. A future tool-approval hook will let an
app gate individual tool calls.

## Notes & limits

- **Single user.** A *personal* assistant serves one person, so there is no multi-user session
  keying; `ConversationManager` holds one conversation.
- **Scheduler is in-memory.** Jobs are Python callables registered in code, not durable across
  restarts. Durable cron-like scheduling belongs in a wrapper above the library.
