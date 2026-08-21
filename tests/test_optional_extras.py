"""A missing optional extra must name itself in the error.

These simulate absence by making the dependency unimportable, so they pass whether or not
the extra is installed in the running environment.
"""

import builtins
import importlib
import sys

import pytest


def _without(monkeypatch, blocked: str):
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == blocked or name.startswith(blocked + "."):
            raise ImportError(f"No module named {blocked!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    for mod in [m for m in sys.modules if m.startswith("aimu.memory") or m.startswith("aimu.prompts")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)


def test_memory_without_chromadb_names_the_extra(monkeypatch):
    _without(monkeypatch, "chromadb")
    with pytest.raises(ImportError, match=r"\[memory\]"):
        importlib.import_module("aimu.memory.semantic_store")


def test_prompt_catalog_without_sqlalchemy_names_the_extra(monkeypatch):
    _without(monkeypatch, "sqlalchemy")
    with pytest.raises(ImportError, match=r"\[prompts\]"):
        importlib.import_module("aimu.prompts.catalog")
