from .agent import Agent
from .base import MessageHistory, Runner
from .orchestrator_agent import OrchestratorAgent
from .skill_agent import SkillAgent
from .workflows.chain import Chain
from .workflows.evaluator import EvaluatorOptimizer
from .workflows.parallel import Parallel
from .workflows.plan_execute_evaluator import PlanExecuteEvaluator
from .workflows.router import Router

__all__ = [
    # Base
    "MessageHistory",
    "Runner",
    # Agents (autonomous)
    "Agent",
    "OrchestratorAgent",
    "SkillAgent",
    # Workflows (code-controlled)
    "Chain",
    "EvaluatorOptimizer",
    "Parallel",
    "PlanExecuteEvaluator",
    "Router",
    # A2A interop (optional; requires the `a2a` extra)
    "HAS_A2A",
]

# Optional A2A interop. Guarded so a missing `a2a` extra doesn't break the package import;
# RemoteAgent / serve_a2a are only bound when the extra is installed (HAS_A2A reflects this).
from .a2a import HAS_A2A  # noqa: E402

if HAS_A2A:
    from .a2a import A2AConnectionError, RemoteAgent, build_a2a_app, serve_a2a  # noqa: E402

    __all__ += ["RemoteAgent", "A2AConnectionError", "serve_a2a", "build_a2a_app"]
