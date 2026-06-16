# News summarizer

One task — *"summarize recent AI technical news"* — solved **four ways**, so the script doubles as
a tour of AIMU's autonomous-agent and code-controlled-workflow surfaces. The `--method` flag picks
which pattern wires the same building blocks together:

| `--method` | Pattern | AIMU class |
|---|---|---|
| `agent` | Autonomous agent — the LLM drives search → read → summarize | [`Agent`](../../aimu/agents/agent.py) |
| `chain` | Prompt chaining — research → summarize → format, each step feeds the next | [`Chain`](../../aimu/agents/workflows/chain.py) |
| `parallel` | Parallelization — one worker per topic runs concurrently, then an aggregator merges | [`Parallel`](../../aimu/agents/workflows/parallel.py) |
| `orchestrator` | Orchestrator-workers — an orchestrator LLM treats "searcher" and "summarizer" agents as tools | [`OrchestratorAgent`](../../aimu/agents/orchestrator_agent.py) |

## Run

```bash
python examples/news-summarizer/news_summarizer.py --method agent
python examples/news-summarizer/news_summarizer.py --method chain --model ollama:qwen3.5:9b
python examples/news-summarizer/news_summarizer.py --method parallel --output-dir ./my-run
python examples/news-summarizer/news_summarizer.py --list
```

Each run writes a Markdown `summary.md` and a `trace.json` under `output/news/<timestamp>/`
(unless overridden with `--output-dir`).

> **Requires** a tool-calling model and a reachable SearXNG instance for web search
> (`SEARXNG_BASE_URL`).

## Learn more

- [07 - Agents](../../notebooks/07%20-%20Agents.ipynb) — the `Agent` loop and `as_model_client()`
- [09 - Workflows](../../notebooks/09%20-%20Workflows.ipynb) — Chain, Router, Parallel, EvaluatorOptimizer
- [Build an orchestrator](../../docs/how-to/build-orchestrator.md) — the orchestrator-workers pattern
