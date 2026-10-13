"""The server a request went to, in a form safe to store.

``RequestPrepared.provider`` is a client class name, so two OpenAI-compatible servers are the same
record without this. Credentials can sit in a URL (``https://user:key@host``, ``?api_key=``), and a
recorded request is persisted, so userinfo, query, and fragment are dropped here, once, rather than
trusted to every host that stores one.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit


def recordable_endpoint(url: Any) -> Optional[str]:
    """``url`` as scheme, host, port, and path only; ``None`` for no URL."""
    if url is None:
        return None
    parts = urlsplit(str(url))
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path.rstrip("/"), "", ""))
