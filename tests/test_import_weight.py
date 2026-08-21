"""Guards on what a bare ``import aimu`` costs.

Every assertion here runs ``import aimu`` in a *subprocess*. The pytest process has
already imported torch and transformers by way of other test modules, so an in-process
``sys.modules`` check would pass without proving anything.
"""

import json
import subprocess
import sys

_PROBE = """
import sys, json
import aimu
json.dump(sorted(sys.modules), sys.stdout)
"""


def loaded_modules() -> set[str]:
    """Top-level module names present after a bare ``import aimu`` in a fresh interpreter."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    return {name.split(".")[0] for name in json.loads(proc.stdout)}


def test_import_aimu_does_not_load_sentence_transformers():
    assert "sentence_transformers" not in loaded_modules()


def test_provider_entry_reports_availability_without_importing():
    """`.available` must answer from find_spec, not by importing the provider."""
    from aimu.models._internal.factory import ProviderEntry

    entry = ProviderEntry(
        prefix="hf",
        module="aimu.models.providers.hf.embedding",
        enum_name="HuggingFaceEmbeddingModel",
        client_name="HuggingFaceEmbeddingClient",
        requires="sentence_transformers",
        install_hint="needs [hf]",
    )
    assert entry.available is True

    enum_cls, client_cls = entry.load()
    assert enum_cls.__name__ == "HuggingFaceEmbeddingModel"
    assert client_cls.__name__ == "HuggingFaceEmbeddingClient"


def test_unknown_dependency_is_unavailable_not_an_error():
    from aimu.models._internal.factory import ProviderEntry

    entry = ProviderEntry(
        prefix="nope",
        module="aimu.models.providers.nope",
        enum_name="X",
        client_name="Y",
        requires="a_module_that_does_not_exist",
        install_hint="never",
    )
    assert entry.available is False


def test_import_aimu_does_not_load_torch_or_transformers():
    loaded = loaded_modules()
    assert "torch" not in loaded
    assert "transformers" not in loaded


def test_has_flags_do_not_import_their_providers():
    """Reading every HAS_* flag must stay import-free."""
    probe = """
import sys, json
import aimu.models as m
[getattr(m, n) for n in dir(m) if n.startswith("HAS_")]
json.dump(sorted({k.split(".")[0] for k in sys.modules}), sys.stdout)
"""
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    loaded = set(json.loads(proc.stdout))
    assert "torch" not in loaded
    assert "sentence_transformers" not in loaded


def test_absent_provider_symbol_is_none_not_an_error():
    """Today's contract: a provider symbol whose dep is missing evaluates to None."""
    probe = """
import sys, json
from unittest import mock
import importlib.util as u

real = u.find_spec
def fake(name, package=None):
    if name == "ollama":
        return None
    return real(name, package)

with mock.patch.object(u, "find_spec", fake):
    import aimu.models as m
    json.dump({"flag": m.HAS_OLLAMA, "sym": m.OllamaModel is None}, sys.stdout)
"""
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    out = json.loads(proc.stdout)
    assert out["flag"] is False
    assert out["sym"] is True
