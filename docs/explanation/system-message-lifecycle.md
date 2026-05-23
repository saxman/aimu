# System message lifecycle

`system_message` is a small attribute that controls a lot of behaviour. This page explains its three-state lifecycle, the failure mode that motivated the lock, and why `reset()` exists.

## Three states

| State | Trigger | Behaviour of the setter |
|---|---|---|
| **Mutable** | Fresh client, no chat sent yet | Assignment works; mutates `_system_message` and updates the system role in `messages` if present. |
| **Locked** | After the first `chat()` call | Assignment raises `RuntimeError`. |
| **Unlocked** | After `client.reset()` | Back to mutable. |

```python
client = aimu.client("ollama:qwen3.5:9b", system="v1")
client.chat("Hi")                          # state transitions from Mutable to Locked

client.system_message = "v2"               # ❌ RuntimeError
client.reset()                              # state goes back to Mutable
client.system_message = "v2"               # ✅ works
```

## The bug this prevents

Before the lock, `client.system_message = "..."` after some chats had run did one of two things depending on the existing message history:

- If a `{"role": "system"}` message was at the start of `messages`, the setter mutated that message in place — silently changing the model's instructions mid-conversation, with no indication in the message trail.
- If no system message existed yet, the setter inserted a `system` message at the start, changing the meaning of the existing conversation.

Both behaviours are surprising and hard to debug. The lock turns the silent failure into a `RuntimeError` with an actionable message:

> system_message is immutable after the conversation starts. Call client.reset() to clear messages, then assign a new system_message.

## Why `reset()` instead of allowing mutation

The convention is: the system message is part of the *contract* with the model. If you want to change the contract, you should also clear the chat history — otherwise the prior turns happened under different instructions and reasoning about the trajectory becomes harder.

`reset()` enforces this by always clearing `messages` along with unlocking the lock:

```python
client.reset()                       # clears messages, preserves system_message (default)
client.reset(system_message=None)    # clears messages and clears system_message
client.reset(system_message="new")   # clears messages and replaces system_message
```

The default — preserve — exists because reset between agent runs or benchmark rows usually keeps the same role. Pass `system_message=...` explicitly when you want to change it.

## How agents and workflows interact with this

Agents that take a `system_message` argument apply it via `reset()` before each run. From [`aimu/agents/agent.py`](https://github.com/saxman/aimu/blob/main/aimu/agents/agent.py):

```python
def _prepare_run(self) -> None:
    if self.reset_messages_on_run or self.system_message is not None:
        self.model_client.reset(system_message=self.system_message)
    if self.tools:
        self.model_client.tools = list(self.tools)
```

This is why `Chain.of(client, [prompt1, prompt2])` works even though all three steps share one client: each step's `Agent` resets at the start of its turn and applies its own `system_message`. After the reset the lock is back to mutable until the next chat.

Same story for `Benchmark`: it calls `client.reset()` between rows so each row starts with a clean conversation but the same role.

## Implementation

The lock is a private boolean:

```python
class BaseModelClient(ABC):
    _system_message: str | None
    _system_message_locked: bool      # False after construction; True after first chat
```

The setter checks the flag and either mutates `_system_message` or raises. `_chat_setup()` (called by every concrete `_chat()`) sets the flag to `True` after appending the first user message. `reset()` sets it back to `False`.

## See also

- [How-to: switch providers](../how-to/switch-providers.md) — `system="..."` at construction time
- [`aimu.models.BaseModelClient.reset`](../reference/api/models.md#aimu.models.BaseModelClient) — API reference
