# Build an orchestrator

`OrchestratorAgent` is the base for the *orchestrator + worker-tools* pattern: an orchestrator `Agent` dispatches to specialist worker agents via tool calls. The LLM decides which workers to call, in what order, and how many times.

Two construction paths:

1. **`OrchestratorAgent.assemble(...)`** — no subclass. Each worker is auto-wrapped as a `@tool` named after the worker.
2. **Subclass and call `self._init_orchestrator(...)`** — full control over each dispatch tool's name, description, and signature.

## Path 1: `assemble`

When each worker can be dispatched by forwarding the user's task verbatim, `assemble()` removes all boilerplate:

```python
import aimu
from aimu.agents import Agent, OrchestratorAgent

client = aimu.client("ollama:qwen3.5:9b")

researcher = Agent(client, "Research the topic. Return key facts.", name="researcher")
critic     = Agent(client, "Critique the research for accuracy and gaps.", name="critic")
writer     = Agent(client, "Write a final summary from the research and critique.", name="writer")

orchestrator = OrchestratorAgent.assemble(
    client,
    system_message=(
        "Use researcher to gather facts, critic to check them, "
        "writer to produce the final summary."
    ),
    workers=[researcher, critic, writer],
)

print(orchestrator.run("How does retrieval-augmented generation work?"))
```

Each worker becomes a callable `@tool` named after the worker's `name` (sanitised). The tool description is the first line of the worker's `system_message`.

## Path 2: subclass

When you need custom tool names, multi-argument tools, or pre/post-processing:

```python
import aimu
from aimu.agents import Agent, OrchestratorAgent
from aimu.models import ModelClient

class CodeReviewAgent(OrchestratorAgent):
    def __init__(self, model_client):
        security = Agent(ModelClient(model_client.model),
                         "Find security issues. Be specific.", name="security")
        perf     = Agent(ModelClient(model_client.model),
                         "Find performance issues. Be specific.", name="perf")

        @aimu.tool
        def review_security(code: str) -> str:
            """Review the given code for security vulnerabilities."""
            return security.run(code)

        @aimu.tool
        def review_performance(code: str) -> str:
            """Review the given code for performance issues."""
            return perf.run(code)

        self._init_orchestrator(
            model_client,
            name="code-review-agent",
            system_message="Use both review tools, then synthesise the findings.",
            tools=[review_security, review_performance],
            concurrent_tool_calls=True,
        )
```

`_init_orchestrator()` does three things: assigns the tool functions to `model_client.tools`, sets `concurrent_tool_calls`, and constructs the inner orchestrator `Agent`. The base class provides `run()` and `messages` — the subclass needs nothing else.

## Concurrent worker dispatch

Pass `concurrent_tool_calls=True` so the orchestrator dispatches multiple tools in parallel when the model returns several tool calls in one turn. This is essential when workers do I/O (web search, API calls) and idle most of their time waiting.

The prebuilt agents in `aimu.agents.prebuilt` enable this automatically when `worker_tools=` is provided.

## Prebuilt orchestrators

`aimu.agents.prebuilt` includes three working orchestrators you can use directly or copy as a starting point:

- `ResearchReportAgent` — orchestrator + `research_overview`, `find_examples`, `find_counterpoints`
- `CodeReviewAgent` — orchestrator + `review_security`, `review_performance`, `review_readability`
- `ContentCreationAgent` — orchestrator + `research_topic`, `create_outline`, `write_section`

```python
from aimu.agents.prebuilt import ResearchReportAgent
from aimu.tools import builtin

agent = ResearchReportAgent(client, worker_tools=builtin.web)
print(agent.run("What is retrieval-augmented generation?"))
```

## See also

- [Tutorial: workflows](../tutorials/03-workflows.md) — Chain, Router, Parallel, EvaluatorOptimizer
- [`aimu.agents.OrchestratorAgent`](../reference/api/agents.md#aimu.agents.OrchestratorAgent)
- Notebook [11 - Agent Examples](https://github.com/saxman/aimu/blob/main/notebooks/11%20-%20Agent%20Examples.ipynb)
