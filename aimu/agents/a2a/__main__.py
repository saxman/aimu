"""Serve an AIMU agent over A2A from the command line.

Mirrors ``python -m aimu.tools.mcp`` (which serves built-in *tools*); this serves a whole
*agent*. Builds an :class:`~aimu.agents.Agent` around a model and exposes it via A2A::

    python -m aimu.agents.a2a --model anthropic:claude-sonnet-4-6 \\
        --system "You are a helpful research assistant." --port 9000

Then connect from another process with ``aimu.agents.RemoteAgent.connect("http://localhost:9000")``.
"""

from __future__ import annotations

import argparse

from aimu.agents.agent import Agent
from aimu.agents.a2a.server import serve_a2a


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m aimu.agents.a2a", description="Serve an AIMU agent over A2A.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model string (e.g. 'anthropic:claude-sonnet-4-6'). "
        "Defaults to AIMU_LANGUAGE_MODEL / a locally available model.",
    )
    parser.add_argument("--system", default=None, help="System message for the agent.")
    parser.add_argument("--name", default=None, help="Agent name (advertised in the A2A card).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    import aimu

    client = aimu.client(args.model, system=args.system)
    agent = Agent(client, system_message=args.system, name=args.name or "aimu-agent")
    serve_a2a(agent, host=args.host, port=args.port, name=args.name, description=args.system)


if __name__ == "__main__":
    main()
