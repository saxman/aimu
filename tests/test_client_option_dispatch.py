"""The ``--client`` test option maps to the client class it names.

This guards the test harness itself. ``_resolve_client`` used to end in
``return OllamaClient  # default``, so a ``--client`` value with no branch silently ran the
*Ollama* catalog against Ollama and reported a pass. ``llamaserver_openai`` and
``sglang_openai`` were documented as supported while doing exactly that, and a real member of
either catalog would then have died in ``create_real_model_client`` with "Unknown model".

Two properties keep that from recurring: every documented option resolves to its own class,
and an unrecognised option raises instead of falling back.
"""

from __future__ import annotations

import pytest

import helpers
import helpers_aio
from aimu.models import HAS_OPENAI_COMPAT

# --client value -> the class name it must resolve to. Sync and async share the option
# vocabulary; the async helper returns the *sync* class (it wraps or maps it downstream).
_OPENAI_COMPAT_OPTIONS = {
    "lmstudio_openai": "LMStudioOpenAIClient",
    "ollama_openai": "OllamaOpenAIClient",
    "hf_openai": "HFOpenAIClient",
    "vllm_openai": "VLLMOpenAIClient",
    "llamaserver_openai": "LlamaServerOpenAIClient",
    "sglang_openai": "SGLangOpenAIClient",
    "omlx_openai": "OMLXOpenAIClient",
}


@pytest.mark.skipif(not HAS_OPENAI_COMPAT, reason="openai-compat providers not installed")
@pytest.mark.parametrize("option,expected", sorted(_OPENAI_COMPAT_OPTIONS.items()))
def test_local_server_option_resolves_to_its_own_client(option, expected):
    assert helpers._resolve_client(option, config=None).__name__ == expected
    assert helpers_aio._resolve_async_client_for_type(option).__name__ == expected


def test_ollama_option_is_explicit():
    # 'ollama' must be a real branch, not the residue of a default-return.
    assert helpers._resolve_client("ollama", config=None).__name__ == "OllamaClient"
    assert helpers_aio._resolve_async_client_for_type("ollama").__name__ == "OllamaClient"


@pytest.mark.parametrize("bogus", ["sglang", "llamaserver", "omlx", "not-a-provider", ""])
def test_unknown_option_raises_instead_of_defaulting(bogus):
    # Near-miss values (the provider key rather than the test option) are the realistic typo,
    # and are exactly what used to pass silently.
    with pytest.raises(ValueError, match="Unknown --client value"):
        helpers._resolve_client(bogus, config=None)
    with pytest.raises(ValueError, match="Unknown --client value"):
        helpers_aio._resolve_async_client_for_type(bogus)
