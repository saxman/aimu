"""Optional A2A (Agent2Agent) interop for AIMU runners.

Mirrors the MCP pattern (``aimu.tools.MCPClient`` / ``python -m aimu.tools.mcp``) at the
*agent* level: MCP exposes tools across a process boundary, A2A exposes whole agents.

- :class:`RemoteAgent` — consume a remote A2A agent as a local ``Runner``.
- :func:`serve_a2a` / :func:`build_a2a_app` — expose any AIMU ``Runner`` as an A2A server.

Requires the ``a2a`` extra (``pip install 'aimu[a2a]'``). ``HAS_A2A`` reports availability.
"""

from __future__ import annotations

try:
    from .client import A2AConnectionError, RemoteAgent
    from .server import build_a2a_app, serve_a2a

    HAS_A2A = True
    __all__ = ["RemoteAgent", "A2AConnectionError", "serve_a2a", "build_a2a_app", "HAS_A2A"]
except ImportError:
    HAS_A2A = False
    __all__ = ["HAS_A2A"]
