# Defer tools until an agent needs them

Every tool an agent holds costs its schema on every model call, whether the turn uses the tool or not. On a
small context window that cost decides how many tools an agent can have at all. A **tool group** lets an
agent hold tools without advertising them: it lists what it can load, loads a group by name, and the group's
tools join its tool list from the next step on.

This is the pattern behind Anthropic's tool search (`defer_loading`) and Claude Code's deferred tools.

## Give an agent a deferred group

```python
from aimu import aio
from aimu.tools import ToolGroup, builtin

web = ToolGroup(
    "web",
    "Search the web and read pages.",
    build=lambda: list(builtin.web),
    guidance="Cite the pages you used.",
)
agent = aio.Agent(client, "You are a helpful assistant.", tool_groups=[web])
```

The agent now carries two extra tools:

- `available_tools(filter="")` lists every group with its state: `loadable`, `loaded`, or `resident`.
- `load_tools(name)` builds the group, appends its tools, and returns the group's tool names and `guidance`.

A `build` that succeeds runs once per agent instance, when the group is loaded, so a group that opens a store or
loads a model costs nothing until an agent asks for it. On `aimu.aio` it runs in a worker thread.

## List tools the agent already holds

A group with no `build` is **resident**: you already put its tools in `tools=`, and the group exists only so
`available_tools` shows the agent its whole picture.

```python
memory = ToolGroup("memory", "Remember facts across conversations.")
agent = aio.Agent(client, tools=memory_tools, tool_groups=[memory, web])
```

An agent whose groups are all resident gets no extra tools.

## What loading does and does not do

- **It only appends.** Loaded tools go after everything else the agent holds, and nothing is unloaded, so a
  provider's prompt cache stays valid.
- **It never shadows.** A loaded tool whose name is taken is skipped, and the result says which group holds
  the name.
- **It is per instance.** Two agents built from one group list load independently, which is what keeps
  concurrent sub-agents of one type apart.
- **It goes through your gate.** `load_tools` is an ordinary tool, so a `tool_approval` policy sees it like
  any other call.

A model that calls a deferred tool without loading it gets a "not found" result that names
`available_tools` and `load_tools`.

## When a build fails

If `build` raises, or returns something other than a list of named tool callables, the load fails
cleanly: `load_tools` returns `Could not load '<name>': <error>`, the failure is logged at WARNING, and the
group stays loadable, so a later load retries the build. On `aimu.aio`, a caller cancelled while a build is
running stops waiting, but the build keeps going, and the next load of that group joins it instead of
starting a second one.

## Restore loads without a model call

A host rebuilding an agent for a resumed conversation can replay its loads directly:

```python
for name in previously_loaded:
    await agent.load_tool_group(name)  # the same text load_tools returns
```

`load_tool_group` is the host acting directly, so it does not pass through `tool_approval`; only the
model's `load_tools` call does.

## Give a sub-agent groups

A typed spawn spec takes `"tool_groups"`; each spawn builds a fresh agent, so its loads are its own:

```python
from aimu.aio.tools.builtin import make_async_subagent_tool

spawn = make_async_subagent_tool(model, agent_types={
    "researcher": {"system_message": "Research the task.", "tool_groups": [web]},
})
```
