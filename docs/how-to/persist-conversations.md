# Persist conversations

`ConversationManager` saves chat message history to disk so it survives process restarts. It's backed by [TinyDB](https://tinydb.readthedocs.io): a single JSON file, no server.

## Basic usage

```python
import aimu
from aimu.history import ConversationManager

manager = ConversationManager("conversations.json", use_last_conversation=True)

client = aimu.client("ollama:qwen3.5:9b")
client.messages = manager.messages   # load the last conversation

client.chat("What is the capital of France?")
manager.update_conversation(client.messages)   # save after every turn
```

`use_last_conversation=True` loads the most recently updated conversation. Pass `False` (the default) to start fresh and call `create_new_conversation()` explicitly.

## Multiple conversations

```python
manager = ConversationManager("conversations.json")

# Start a new one
doc_id, messages = manager.create_new_conversation()
client.messages = messages

client.chat("Hello.")
manager.update_conversation(client.messages)
```

`create_new_conversation()` returns the document id (for retrieval) and an empty messages list.

## Use with agents

`Agent` and the workflows mutate `model_client.messages` during `run()`. Persist after the run:

```python
from aimu.agents import Agent

agent = Agent(client, "Be helpful.")
agent.run("Plan my trip.")
manager.update_conversation(client.messages)
```

## Distinguish agent-loop turns from user input

An `Agent` reaches its answer by injecting `{"role": "user", ...}` turns the human never typed: a continuation prompt between tool-calling rounds, and (if configured) a final-answer prompt when the loop hits `max_iterations`. In a live stream these are *input*, so they never reach the screen. But when you **replay** a stored history they look identical to real user messages.

AIMU marks them with an inert `provenance` key so your display can tell them apart. Genuine user input and ordinary assistant turns are left untagged, so the check is a plain `.get()`:

```python
import aimu

for message in client.messages:
    provenance = message.get(aimu.PROVENANCE_KEY)
    if provenance == aimu.PROVENANCE_CONTINUATION:
        render_caption("↻ agent continuation")      # or skip it entirely
    elif provenance == aimu.PROVENANCE_FINAL_ANSWER:
        render_caption("✓ final-answer wrap-up")
    elif provenance == aimu.PROVENANCE_PROACTIVE:     # scheduler / proactive push
        render_bubble(message, badge="proactive")
    else:
        render_bubble(message)                        # genuine user or assistant turn
```

The key is one of `PROVENANCE_CONTINUATION`, `PROVENANCE_FINAL_ANSWER`, or `PROVENANCE_PROACTIVE` (all exported from `aimu` and `aimu.models`). It is inert with respect to the model: it is stripped from every provider request, exactly like the [`thinking` key](../explanation/thinking-and-context.md#the-provenance-key-distinguishing-framework-injected-turns). The Streamlit examples in `examples/web/` use it to mute or hide injected turns when re-rendering history.

## See also

- [`aimu.history.ConversationManager`](../reference/api/history.md)
- [Thinking and context](../explanation/thinking-and-context.md#the-provenance-key-distinguishing-framework-injected-turns): the inert-key mechanism behind `provenance`
- Notebook [02 - Conversations](https://github.com/saxman/aimu/blob/main/notebooks/02%20-%20Conversations.ipynb)
