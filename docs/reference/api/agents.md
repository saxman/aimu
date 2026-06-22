# `aimu.agents`

Agents and code-controlled workflows.

## Hierarchy

::: aimu.agents.Runner

::: aimu.agents.MessageHistory

## Agents

::: aimu.agents.Agent

::: aimu.agents.SkillAgent

::: aimu.agents.OrchestratorAgent

## Workflows

::: aimu.agents.Chain

::: aimu.agents.Router

::: aimu.agents.Parallel

::: aimu.agents.EvaluatorOptimizer

::: aimu.agents.PlanExecuteEvaluator

## A2A interop

Optional Agent2Agent protocol support (requires the `a2a` extra: `pip install 'aimu[a2a]'`).
Available only when `aimu.agents.HAS_A2A` is `True`. See [A2A vs MCP](../../explanation/a2a-vs-mcp.md)
and [Connect agents (A2A)](../../how-to/connect-agents-a2a.md).

::: aimu.agents.RemoteAgent

::: aimu.agents.serve_a2a

::: aimu.agents.build_a2a_app

::: aimu.agents.A2AConnectionError
