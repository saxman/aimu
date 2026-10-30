"""Which models the chat apps offer, and which one they start with. Shared by all three apps.

Ollama clients do not download a model when constructed, so the menu offers only Ollama models that
are already installed. The apps' automatic default never lands on a client whose constructor
downloads weights (HuggingFace, llama.cpp): a chat app that starts a multi-gigabyte download because
nothing else was set up is the failure AIMU removed from Ollama. Picking such a model from the menu
is still allowed; that is the user's explicit choice.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

# Client classes whose constructor loads (and so downloads) weights in-process.
DOWNLOADS_ON_CONSTRUCTION = ("HuggingFaceClient", "LlamaCppClient")

NO_OLLAMA_MODELS = (
    'No Ollama models are installed. Install one with `ollama pull qwen3.5:9b` or `aimu.pull("ollama:qwen3.5:9b")`, '
    "then reload."
)

NO_READY_MODEL = (
    "No model is ready to use without a download. Install an Ollama model with `ollama pull qwen3.5:9b` "
    'or `aimu.pull("ollama:qwen3.5:9b")`, or set up a cloud provider\'s API key, then reload.'
)


def model_key(member: Any) -> tuple[str, str]:
    """A hashable identity for an enum member. The same member name exists under several providers."""
    return (type(member).__name__, member.name)


def model_options(client_cls: Any, installed: set[tuple[str, str]], *, ollama_client: Any) -> list:
    """The tool-capable models to offer for ``client_cls``: installed only, for Ollama."""
    if ollama_client is None or client_cls is not ollama_client:
        return list(client_cls.TOOL_MODELS)
    return [m for m in client_cls.TOOL_MODELS if model_key(m) in installed]


def default_client_and_model(
    clients: Iterable[Any],
    installed: set[tuple[str, str]],
    *,
    ollama_client: Any,
    preferred: Optional[Any] = None,
) -> tuple[Optional[Any], Optional[Any]]:
    """The first client with a model ready to use, and its model; ``(None, None)`` when there is none.

    Ollama comes first when installed, with ``preferred`` chosen if it is installed. Clients whose
    constructor downloads weights are never chosen automatically.
    """
    clients = list(clients)
    ordered = [c for c in clients if c is ollama_client] + [c for c in clients if c is not ollama_client]
    for client_cls in ordered:
        if client_cls.__name__ in DOWNLOADS_ON_CONSTRUCTION:
            continue
        options = model_options(client_cls, installed, ollama_client=ollama_client)
        if options:
            return client_cls, preferred if preferred in options else options[0]
    return None, None
