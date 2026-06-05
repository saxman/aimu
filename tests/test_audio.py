"""Live audio-generation tests parametrized by provider — opt in via CLI flags.

Mirrors :mod:`tests/test_images.py` for the audio modality. Same shape:
``pytest_generate_tests`` reads the CLI flags, ``audio_client`` is a session-scoped
fixture that yields a real :class:`BaseAudioClient`, and each test runs against
every parametrized model.

Usage::

    # Run live MusicGen Small tests
    pytest tests/test_audio.py --audio-client=hf --audio-model=MUSICGEN_SMALL

    # Run all HuggingFace audio models
    pytest tests/test_audio.py --audio-client=hf

    # Run every model on every installed provider
    pytest tests/test_audio.py --audio-client=all

Tests are **skipped entirely** when ``--audio-client`` is omitted, so the file
is safe in default CI runs. Expect each test to take 10–120 seconds depending
on model size and hardware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from helpers import create_real_audio_client, resolve_audio_model_params

from aimu.models import BaseAudioClient


def pytest_generate_tests(metafunc):
    if "audio_client" not in metafunc.fixturenames:
        return
    params = resolve_audio_model_params(metafunc.config)
    metafunc.parametrize("audio_client", params, indirect=True, scope="session")


@pytest.fixture(scope="session")
def audio_client(request) -> Iterable[BaseAudioClient]:
    yield from create_real_audio_client(request)


def _provider_kwargs(client: BaseAudioClient) -> dict:
    """Provider-specific kwargs to keep live runs fast.

    MusicGen: 3s is enough for a smoke test (default is 10s).
    AudioLDM2 / Stable Audio: reduce inference steps to 10.
    """
    from aimu.models.providers.hf.audio import HuggingFaceAudioClient

    if not isinstance(client, HuggingFaceAudioClient):
        return {}
    ptype = client.spec.pipeline_type
    if ptype == "musicgen":
        return {"duration_s": 3.0}
    return {"duration_s": 3.0, "num_inference_steps": 10}


# ---------------------------------------------------------------------------
# Cross-provider smoke tests
# ---------------------------------------------------------------------------


def test_class_properties(audio_client):
    assert isinstance(audio_client, BaseAudioClient)
    assert audio_client.spec is not None
    assert isinstance(audio_client.spec.id, str) and audio_client.spec.id


def test_generate_returns_numpy(audio_client):
    """generate() with format='numpy' returns (sample_rate, ndarray)."""
    result = audio_client.generate("a simple drum beat", format="numpy", **_provider_kwargs(audio_client))
    assert isinstance(result, tuple) and len(result) == 2
    sr, audio = result
    assert isinstance(sr, int) and sr > 0
    assert isinstance(audio, np.ndarray) and audio.size > 0


def test_generate_format_path(audio_client, tmp_path):
    """format='path' writes a real WAV file and returns its path."""
    out = audio_client.generate(
        "ambient wind sounds",
        format="path",
        output_dir=tmp_path,
        **_provider_kwargs(audio_client),
    )
    assert isinstance(out, str)
    p = Path(out)
    assert p.exists() and p.suffix == ".wav" and p.parent == tmp_path
    # WAV magic bytes
    assert p.read_bytes().startswith(b"RIFF")


def test_generate_format_bytes(audio_client):
    """format='bytes' returns a valid WAV byte string."""
    data = audio_client.generate(
        "soft piano melody",
        format="bytes",
        **_provider_kwargs(audio_client),
    )
    assert isinstance(data, bytes)
    assert data.startswith(b"RIFF")
    assert len(data) > 100


def test_generate_num_audio_returns_list(audio_client):
    """num_audio > 1 returns a list of the requested length."""
    results = audio_client.generate(
        "electronic beat",
        num_audio=2,
        format="numpy",
        **_provider_kwargs(audio_client),
    )
    assert isinstance(results, list)
    assert len(results) == 2
    for sr, audio in results:
        assert isinstance(sr, int) and sr > 0
        assert isinstance(audio, np.ndarray) and audio.size > 0


# ---------------------------------------------------------------------------
# Provider-specific behaviour (auto-skipped on wrong provider/pipeline_type)
# ---------------------------------------------------------------------------


def test_musicgen_seed_reproducible(audio_client):
    """Same seed → similar output length. MusicGen only."""
    from aimu.models.providers.hf.audio import HuggingFaceAudioClient

    if not isinstance(audio_client, HuggingFaceAudioClient):
        pytest.skip("Seed test only applies to HuggingFace audio")
    if audio_client.spec.pipeline_type != "musicgen":
        pytest.skip("Seed reproducibility only applies to MusicGen")

    kwargs = _provider_kwargs(audio_client)
    a_sr, a_audio = audio_client.generate("lo-fi jazz loop", seed=99, format="numpy", **kwargs)
    b_sr, b_audio = audio_client.generate("lo-fi jazz loop", seed=99, format="numpy", **kwargs)
    assert a_sr == b_sr
    # Same seed + same prompt should produce same number of samples.
    assert a_audio.shape == b_audio.shape


def test_diffusers_num_inference_steps(audio_client):
    """num_inference_steps parameter accepted by AudioLDM2 / Stable Audio."""
    from aimu.models.providers.hf.audio import HuggingFaceAudioClient

    if not isinstance(audio_client, HuggingFaceAudioClient):
        pytest.skip("num_inference_steps only applies to HuggingFace audio")
    if audio_client.spec.pipeline_type == "musicgen":
        pytest.skip("MusicGen is token-autoregressive; num_inference_steps does not apply")

    result = audio_client.generate(
        "rainstorm",
        format="numpy",
        duration_s=3.0,
        num_inference_steps=5,
    )
    assert isinstance(result, tuple)
    _, audio = result
    assert audio.size > 0
