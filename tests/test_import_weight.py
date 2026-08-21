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
