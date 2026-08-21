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
