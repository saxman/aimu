from importlib import import_module
from typing import TYPE_CHECKING

# base and document_store are chromadb-free, so they stay eager.
from .base import MemoryStore
from .document_store import DocumentStore

if TYPE_CHECKING:
    from .semantic_store import SemanticMemoryStore

# SemanticMemoryStore pulls in chromadb (the `memory` extra). It is loaded on first
# access so that `from aimu.memory import DocumentStore` works without that dependency.
_LAZY = {
    "SemanticMemoryStore": ".semantic_store",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, __name__), name)


__all__ = ["MemoryStore", "SemanticMemoryStore", "DocumentStore"]
