# System message lifecycle

`system_message` is a small attribute that controls a lot of behaviour. This page explains how it behaves before and during a conversation, why assigning it mid-conversation re-conditions the model in place, and why `reset()` still exists.

## Two phases, one always-live setter

There is no lock. The setter works at any time; what it *does* depends on whether a conversation has started.

| Phase | Trigger | Behaviour of the setter |
|---|---|---|
| **Seed** | Fresh client, no chat sent yet (`messages` empty) | Stores `_system_message`. It is prepended to `messages` on the first `chat()`. |
| **Swap** | A conversation is underway (`messages` non-empty) | Rewrites the `{"role": "system"}` entry in `messages` in place: replacing its content, inserting one at index 0 if absent, or removing it when set to `None`. |

```python
client = aimu.client("ollama:qwen3.5:9b", system="You are terse.")
client.chat("Hi")                          # seed prepended to messages

client.system_message = "You are a pirate." # swaps messages[0] in place
client.chat("Tell me about the sea")        # model now answers in the new persona, with history intact
```

Changing a chat's persona mid-conversation is the motivating use case: the model is re-conditioned on the new prompt for every subsequent turn while the full conversation history is preserved.

## `messages` is the source of truth

The active system prompt the model sees at request time is the system entry **inside `messages`**, not `self._system_message`. `_system_message` is only a seed, consulted once (when `messages` is empty) to populate that entry on the first chat. This holds across every provider: the OpenAI-compatible and Ollama clients send the message list as-is, and the Anthropic client scans the list for the `role == "system"` entry and lifts it into its top-level `system=` param. So rewriting that one entry (which is exactly what the setter does) is the correct, provider-portable way to change the active prompt. See [`aimu/models/_internal/chat_state.py`](https://github.com/saxman/aimu/blob/main/aimu/models/_internal/chat_state.py).

## The deliberate tradeoffs

Allowing mid-conversation mutation has two consequences, both accepted on purpose:

- **The transcript becomes counterfactual.** Prior assistant turns were generated under the *old* prompt but now sit beneath the new system entry. For a persona swap this is the intended, seamless behaviour; if you need an honest record of where the switch happened, append a steering user turn instead, or `reset()` to start fresh.
- **No guard against silent cross-agent mutation.** Earlier versions raised `RuntimeError` if you reassigned `system_message` after a conversation started, which incidentally caught the case of one agent mutating a `ModelClient` whose conversation another agent owns. That guard is gone. Don't share a single live-conversation client across agents that each set `system_message`; give each agent its own client (as the prebuilt orchestrator agents do, one `ModelClient(model_client.model)` per worker).

## `reset()`: change the prompt *and* drop history

When you want a clean slate rather than a seamless swap, `reset()` clears `messages`:

```python
client.reset()                       # clears messages, preserves system_message (default)
client.reset(system_message=None)    # clears messages and clears system_message
client.reset(system_message="new")   # clears messages and replaces system_message
```

The default (preserve) exists because reset between agent runs or benchmark rows usually keeps the same role. Pass `system_message=...` explicitly when you want to change it.

## How agents and workflows interact with this

Agents that take a `system_message` argument apply it via `reset()` before each run. From [`aimu/agents/agent.py`](https://github.com/saxman/aimu/blob/main/aimu/agents/agent.py):

```python
def _prepare_run(self) -> None:
    if self.reset_messages_on_run or self.system_message is not None:
        self.model_client.reset(system_message=self.system_message)
```

`_prepare_run()` only touches the system message. The agent's tools (and `deps` / `tool_approval`) are the agent's own state and are handed to the tool-loop engine per run — passed through to `chat(..., tools=...)` on each turn — rather than persisted on the client.

This is why `Chain.from_client(client, [prompt1, prompt2])` works even though all steps share one client: each step's `Agent` resets at the start of its turn and applies its own `system_message`.

`SkillAgent` relies on the swap directly: it appends the discovered skill catalog to the active system prompt by assigning `system_message`, which rewrites the in-history entry without wiping the conversation.

Same story for `Benchmark`: it calls `client.reset()` between rows so each row starts with a clean conversation but the same role.

## See also

- [How-to: switch providers](../how-to/switch-providers.md): `system="..."` at construction time
- [`aimu.models.BaseModelClient.reset`](../reference/api/models.md#aimu.models.BaseModelClient): API reference
