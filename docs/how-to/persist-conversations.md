# Persist conversations

`ConversationManager` saves chat message history to disk so it survives process restarts. It's backed by [TinyDB](https://tinydb.readthedocs.io) — a single JSON file, no server.

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

## See also

- [`aimu.history.ConversationManager`](../reference/api/history.md)
- Notebook [06 - Conversations](https://github.com/saxman/aimu/blob/main/notebooks/06%20-%20Conversations.ipynb)
