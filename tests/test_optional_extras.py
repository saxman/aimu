"""A missing optional extra must name itself in the error.

These simulate absence by making the dependency unimportable, so they pass whether or not
the extra is installed in the running environment.
"""

import builtins
import importlib
import sys

import pytest


def _evict(monkeypatch):
    for mod in [m for m in sys.modules if m.startswith("aimu.memory") or m.startswith("aimu.prompts")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)


def _without(monkeypatch, blocked: str):
    """Simulate `blocked` being entirely uninstalled (mirrors a real ModuleNotFoundError,
    whose `.name` is the module that could not be found)."""
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == blocked or name.startswith(blocked + "."):
            raise ImportError(f"No module named {blocked!r}", name=blocked)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    _evict(monkeypatch)


def _broken(monkeypatch, blocked: str, unrelated_missing: str):
    """Simulate `blocked` being installed but failing to import for its own, unrelated
    reasons (e.g. an ABI clash in one of its dependencies). The resulting ImportError's
    `.name` is the unrelated dependency, not `blocked`."""
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == blocked or name.startswith(blocked + "."):
            raise ImportError(f"No module named {unrelated_missing!r}", name=unrelated_missing)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    _evict(monkeypatch)


def test_memory_without_chromadb_names_the_extra(monkeypatch):
    _without(monkeypatch, "chromadb")
    with pytest.raises(ImportError, match=r"\[memory\]"):
        importlib.import_module("aimu.memory.semantic_store")


def test_prompt_catalog_without_sqlalchemy_names_the_extra(monkeypatch):
    _without(monkeypatch, "sqlalchemy")
    with pytest.raises(ImportError, match=r"\[prompts\]"):
        importlib.import_module("aimu.prompts.catalog")


def test_memory_with_chromadb_broken_by_unrelated_dependency_propagates(monkeypatch):
    """chromadb installed but failing on its own (e.g. onnxruntime) must not be
    reported as "missing the [memory] extra" -- that advice would not help."""
    _broken(monkeypatch, "chromadb", "onnxruntime")
    with pytest.raises(ImportError, match="onnxruntime") as excinfo:
        importlib.import_module("aimu.memory.semantic_store")
    assert "[memory]" not in str(excinfo.value)


def test_prompt_catalog_with_sqlalchemy_broken_by_unrelated_dependency_propagates(monkeypatch):
    """sqlalchemy installed but failing on its own (e.g. greenlet) must not be
    reported as "missing the [prompts] extra" -- that advice would not help."""
    _broken(monkeypatch, "sqlalchemy", "greenlet")
    with pytest.raises(ImportError, match="greenlet") as excinfo:
        importlib.import_module("aimu.prompts.catalog")
    assert "[prompts]" not in str(excinfo.value)


def test_document_store_importable_without_chromadb(monkeypatch):
    """DocumentStore is a plain path-based store that never touches chromadb; it must
    stay reachable through `aimu.memory` even when chromadb can't be imported at all."""
    _without(monkeypatch, "chromadb")
    monkeypatch.delitem(sys.modules, "aimu.memory", raising=False)
    module = importlib.import_module("aimu.memory")
    assert module.DocumentStore is not None
