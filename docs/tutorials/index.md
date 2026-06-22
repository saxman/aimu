# Tutorials

Hand-held end-to-end walkthroughs. Goal: get you to a working result so you trust the library and know where things live.

Work through them in order; each builds on the previous. Total time end-to-end: about 50 minutes.

1. **[Getting started](01-getting-started.md)** *(~15 min)*: install AIMU, send your first message, build your first agent.
2. **[First agent with tools](02-first-agent-with-tools.md)** *(~10 min)*: `@tool` decorator and the agentic loop.
3. **[Workflows](03-workflows.md)** *(~15 min)*: `Chain.of`, `Router.of`, `Parallel.of` for code-controlled patterns.
4. **[Vision and streaming](04-vision-and-streaming.md)** *(~10 min)*: image input plus the `StreamChunk` API.

Once you've finished, the [how-to guides](../how-to/index.md) cover specific tasks; the [explanation pages](../explanation/index.md) cover the *why* behind the design.


## What you'll need

- Python 3.10+
- One model backend. The tutorials use **Ollama** because it's local and free, but every snippet works with any provider; just swap the model string:

  ```python
  aimu.chat("hi", model="ollama:qwen3.5:9b")        # local
  aimu.chat("hi", model="anthropic:claude-sonnet-4-6")  # cloud
  aimu.chat("hi", model="openai:gpt-4o-mini")          # cloud
  ```

For Ollama: install from [ollama.com](https://ollama.com) and run `ollama pull qwen3.5:9b` once. For cloud providers, set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` in your environment or a `.env` file.
