# Build a personal assistant

A personal AI assistant (in the style of [OpenClaw](https://github.com/openclaw/openclaw) or
[Hermes Agent](https://hermes-agent.nousresearch.com/)) is a long-running process that talks to
you over a chat channel, acts unprompted (reminders, check-ins), and grows its own skills. AIMU
ships three async-first primitives for exactly the pieces that go beyond a single LLM call:

| Need | Primitive |
|---|---|
| Talk over a transport (terminal, chat platform) | `aimu.aio.Channel` ABC + `aimu.aio.CLIChannel` |
| Act proactively on a schedule | `aimu.aio.Scheduler` |
| Author new skills at runtime | `aimu.skills.write_skill` / `make_skill_authoring_tool` |

These are **library primitives**, not an app. A complete single-user daemon that wires them
together lives in [`examples/personal-assistant/`](https://github.com/saxman/aimu/tree/main/examples/personal-assistant).
Model-agnostic LLMs, multi-agent routing, voice I/O, and persistent memory already exist
elsewhere in AIMU, so they are not duplicated here.

!!! note "Single user by design"
    A *personal* assistant serves one person, so there is no multi-user session keying.
    [`ConversationManager`](persist-conversations.md) holds the one conversation. Multi-user
    routing belongs in a wrapper above the library.

## Channels

A `Channel` is a tiny transport ABC: receive inbound messages, send replies.

```python
from aimu.aio import CLIChannel, ChannelMessage

channel = CLIChannel()

async for msg in channel.receive():        # async generator over inbound messages
    print("got:", msg.text)
    await channel.send("Thanks!", reply_to=msg)
```

`ChannelMessage` is plain transport data (`text`, `sender`, `channel`, `images`, `metadata`),
distinct from LLM conversation state; `text` and `images` map straight onto `agent.run(...)`.
`send()` accepts either a finished string or an `AsyncIterator[StreamChunk]` to relay a streamed
reply token-by-token.

The `reply_to` argument is the seam for network adapters: a Telegram/Slack `Channel` uses it to
route a reply to the right chat. `CLIChannel` is single-user, so it ignores it. A network adapter
is a drop-in subclass (constructed via a `connect()` classmethod factory, like
[`aio.MCPClient.connect`](use-mcp-tools.md)); none ships yet, keeping heavy SDKs out of core.

## Scheduler

`Scheduler` runs interval and one-shot async jobs concurrently under one `asyncio.TaskGroup`. A
job is a zero-argument coroutine factory; bind context (an agent, a channel) with a closure.

```python
import functools
from aimu.aio import Scheduler

async def check_in(agent, channel):
    reply = await agent.run("Give the user one short, useful suggestion.")
    await channel.send(reply)

scheduler = Scheduler()
scheduler.every(3600, functools.partial(check_in, agent, channel), name="hourly")
scheduler.at(30, functools.partial(check_in, agent, channel))   # one-shot, 30s after start
await scheduler.run()   # blocks until scheduler.stop()
```

A job that raises is logged and, for interval jobs, retried on the next tick: one misbehaving
reminder must not kill the daemon. `run()` is single-use; build a fresh `Scheduler` to run again.
Job persistence across restarts is intentionally out of scope (jobs are Python callables).

## Skill authoring (self-improvement)

The assistant can write a new skill when it works out a repeatable procedure, so it remembers how
next time. `make_skill_authoring_tool` returns an async `@tool` the agent calls; pass it to a
[`SkillAgent`](use-skills.md) so authored skills are discoverable in the same run.

```python
from aimu.skills import SkillManager, make_skill_authoring_tool
from aimu import aio

skills_dir = ".agents/skills"
manager = SkillManager(skill_dirs=[skills_dir])
author_skill = make_skill_authoring_tool(manager, skills_dir)

client = aio.client("anthropic:claude-sonnet-4-6", system="You are a personal assistant. "
                    "When the user teaches you a repeatable procedure, call author_skill to save it.")
agent = aio.SkillAgent(client, tools=[author_skill], skill_manager=manager, name="assistant")
```

`author_skill(name, description, body)` calls `write_skill(...)` (which validates the name as a
slug, refuses to clobber, and round-trips through the parser so a malformed file fails loudly)
then `manager.refresh()` to invalidate the discovery cache.

After `refresh()`, the new skill is reachable via `activate_skill`. To also refresh the skill
catalog injected into an in-flight system prompt (so a just-authored skill appears mid-conversation),
call `agent.reload_skills()` (see the next section); the personal-assistant example does this
automatically when a script is authored.

You can also call `write_skill` directly (no agent):

```python
from aimu.skills import write_skill
write_skill("format-standup", "Format a standup update.",
            "# Standup\n\nThree bullets: Yesterday, Today, Blockers.", skills_dir=".agents/skills")
```

## Scripts in skills (author and run code)

A skill can bundle executable helper scripts. AIMU registers every `scripts/*.py` and `scripts/*.sh`
in a skill as a `{skill}__{stem}` tool that runs the script as a subprocess (`.py` via the current
Python, `.sh` via `bash`), with an optional `args` string forwarded to the script's arguments and a
30-second timeout. This lets an assistant **automate a procedure as code** and run it as a tool, the
self-improving pattern Hermes Agent uses.

`make_skill_script_tool(agent, manager, skills_dir)` returns an async `add_skill_script` tool. After
writing the script it calls `agent.reload_skills()`, which rebuilds the skills server and appends the
new `{skill}__{stem}` tool to the agent's tool list, so the assistant can author **and run** a script
within the same turn:

```python
from aimu.skills import SkillManager, make_skill_authoring_tool, make_skill_script_tool
from aimu import aio

skills_dir = ".agents/skills"
manager = SkillManager(skill_dirs=[skills_dir])
client = aio.client("anthropic:claude-sonnet-4-6", system="You are a personal assistant.")
agent = aio.SkillAgent(client, skill_manager=manager, name="assistant")
agent.tools = [
    make_skill_authoring_tool(manager, skills_dir),     # author_skill(name, description, body)
    make_skill_script_tool(agent, manager, skills_dir), # add_skill_script(skill_name, filename, content)
]
```

You can also attach scripts non-interactively with `write_skill(..., scripts={...})`:

```python
from aimu.skills import write_skill
write_skill(
    "disk", "Report disk usage.", "# Disk\n\nRun the usage script.",
    skills_dir=skills_dir,
    scripts={"usage.sh": "#!/usr/bin/env bash\ndf -h\n"},   # .sh files are marked executable
)
```

!!! danger "Scripts run with full access (no sandbox)"
    Skill scripts execute as real subprocesses with your user privileges, exactly like OpenClaw and
    Hermes Agent. There is no sandbox. Only run an assistant that can author/run scripts with a model
    and inputs you trust. (`builtin.execute_python` is the sandboxed alternative for untrusted code:
    no filesystem or subprocess access.) The subprocess also blocks the event loop for up to its
    30-second timeout.

## Wire it into a daemon

The pieces compose into one process: a top-level `asyncio.TaskGroup` runs the channel listener
and the scheduler concurrently; each inbound message runs the agent and streams the reply back;
history is persisted after each turn.

```python
import asyncio
from aimu import aio
from aimu.aio import CLIChannel, Scheduler
from aimu.history import ConversationManager

async def main():
    channel = CLIChannel()
    scheduler = Scheduler()
    conversation = ConversationManager("assistant_history.json", use_last_conversation=True)
    # ... build agent with author_skill as above; restore prior messages with agent.restore(...) ...

    async def serve():
        try:
            async for msg in channel.receive():
                reply = await agent.run(msg.text, stream=True, images=msg.images)
                await channel.send(reply, reply_to=msg)
                conversation.update_conversation([dict(m) for m in agent.model_client.messages])
        finally:
            scheduler.stop()        # channel closed (EOF) -> stop the scheduler so run() returns

    async with asyncio.TaskGroup() as tg:
        tg.create_task(serve())
        tg.create_task(scheduler.run())

asyncio.run(main())
```

The reference app in `examples/personal-assistant/` is this, fleshed out: serialize reactive and
proactive turns with a lock (they share one agent), an argparse entry point, and mock-only tests.
It also gives the agent a small fixed set of AIMU [built-in tools](add-custom-tool.md)
(`builtin.web + builtin.misc`). The built-ins are sync functions, and the async agent dispatches
them through `asyncio.to_thread`, so they attach to `agent.tools` with no wrapping. The example is
kept intentionally minimal; selectable tool groups, [remote MCP servers](use-mcp-tools.md), and
[persistent memory](use-semantic-memory.md) are capabilities AIMU ships but the example leaves out.

Run it:

```bash
python examples/personal-assistant/assistant.py \
    --model anthropic:claude-sonnet-4-6 --reminder-seconds 30
```

## Web front end

Because the assistant loop consumes any `Channel`, a browser UI is just a different `Channel`
implementation: nothing in `Assistant` changes. The example ships a `WebChannel` (a `Channel` over a
browser WebSocket) and a small [Starlette](https://www.starlette.io/) server
(`examples/personal-assistant/web_assistant.py`). `WebChannel.send` relays a streamed reply as
`token` frames + a `done` frame, and a proactive (scheduler-pushed) message as a `message` frame the
page styles distinctly. The server runs a pump task (reading the socket into the channel) and
`assistant.run()` concurrently under one `asyncio.TaskGroup`; on disconnect the pump feeds a sentinel
that ends `receive()`, stops the scheduler, and tears down cleanly.

Because the server owns the scheduler via `assistant.run()`, proactive messages reach the browser
unprompted, the property a request/response UI (Streamlit/Gradio) can't provide without polling.

The agent's stream carries `THINKING` and `TOOL_CALLING` chunks alongside `GENERATING`, so both
channels can surface the model's reasoning and tool calls, not just the final answer. `CLIChannel`
takes opt-in `show_thinking` / `show_tools` flags (off by default to keep the library channel
minimal); the example-local `WebChannel` emits `thinking` / `tool` frames the page renders as
distinct blocks. The personal-assistant example turns both on, threading
`AssistantConfig.show_thinking` / `show_tools` into each channel.

```bash
pip install aimu[web]
python examples/personal-assistant/web_assistant.py --model ollama:qwen3:8b --reminder-seconds 20
# open http://127.0.0.1:8000
```

The `WebChannel` lives in the example (not the library): a web server is app-layer, and keeping it
out of `aimu.aio` keeps the library free of a web-framework dependency. It is a worked example of
extending the `Channel` ABC, the same seam a Telegram or Slack adapter would use.

## See also

- [Use skills](use-skills.md): the `SkillAgent` and `SKILL.md` format the authoring tool writes to
- [Use async (`aio`)](use-async.md): the async surface these primitives live on
- [Persist conversations](persist-conversations.md): `ConversationManager`
- Add capabilities the example leaves out: [use MCP tools](use-mcp-tools.md), [semantic memory](use-semantic-memory.md), [document memory](use-document-memory.md)
- [`aimu.aio` API reference](../reference/api/aio.md) · [`aimu.skills` API reference](../reference/api/skills.md)
