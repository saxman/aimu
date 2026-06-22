"""Optional async A2A interop for AIMU async runners (twin of :mod:`aimu.agents.a2a`)."""

from __future__ import annotations

try:
    from aimu.agents.a2a.client import A2AConnectionError

    from .client import RemoteAgent
    from .server import build_a2a_app, serve_a2a

    HAS_A2A = True
    __all__ = ["RemoteAgent", "A2AConnectionError", "serve_a2a", "build_a2a_app", "HAS_A2A"]
except ImportError:
    HAS_A2A = False
    __all__ = ["HAS_A2A"]
