"""Live image-generation tests parametrized by provider — opt in via CLI flags.

Mirrors :mod:`tests/test_models.py` for the image modality. Same shape:
``pytest_generate_tests`` reads the CLI flags, ``image_client`` is a session-scoped
fixture that yields a real :class:`BaseImageClient`, and each test runs against
every parametrized model.

Usage:

- Run live HuggingFace SD 1.5 tests::

      pytest tests/test_images.py --image-client=hf --image-model=SD_1_5

- Run live Google Nano Banana tests (requires ``GOOGLE_API_KEY``)::

      pytest tests/test_images.py --image-client=gemini --image-model=NANO_BANANA

- Run every model on a single provider::

      pytest tests/test_images.py --image-client=hf

- Run every model on every installed provider (heavy — pulls multiple HF weights)::

      pytest tests/test_images.py --image-client=all

Tests are **skipped entirely** when ``--image-client`` is omitted, so the file is
safe in default CI runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest
from PIL import Image

from helpers import create_real_image_client, resolve_image_model_params

from aimu.models import BaseImageClient


def pytest_generate_tests(metafunc):
    if "image_client" not in metafunc.fixturenames:
        return
    params = resolve_image_model_params(metafunc.config)
    metafunc.parametrize("image_client", params, indirect=True, scope="session")


@pytest.fixture(scope="session")
def image_client(request) -> Iterable[BaseImageClient]:
    yield from create_real_image_client(request)


def _provider_kwargs(client: BaseImageClient) -> dict:
    """Provider-specific kwargs that keep live runs fast and deterministic.

    Diffusers benefits from explicit small dims + low step count; Gemini's
    ``generate_content`` infers size and has no step concept, so an empty dict
    is correct there.
    """
    from aimu.models.providers.hf.image import HuggingFaceImageClient

    if isinstance(client, HuggingFaceImageClient):
        return {"width": 256, "height": 256, "num_inference_steps": 4}
    return {}


# ---------------------------------------------------------------------------
# Cross-provider smoke tests
# ---------------------------------------------------------------------------


def test_class_properties(image_client):
    """Each concrete image client implements the base contract."""
    assert isinstance(image_client, BaseImageClient)
    assert image_client.spec is not None
    assert isinstance(image_client.spec.id, str) and image_client.spec.id


def test_generate_returns_pil_image(image_client):
    """generate() default format returns a PIL Image."""
    img = image_client.generate("a simple red ball on a white background", **_provider_kwargs(image_client))
    assert isinstance(img, Image.Image)


def test_generate_format_path(image_client, tmp_path):
    """format='path' writes a real PNG and returns its path."""
    out = image_client.generate(
        "a simple blue square",
        format="path",
        output_dir=tmp_path,
        **_provider_kwargs(image_client),
    )
    assert isinstance(out, str)
    p = Path(out)
    assert p.exists() and p.suffix == ".png" and p.parent == tmp_path
    decoded = Image.open(p)
    assert decoded.size[0] > 0 and decoded.size[1] > 0


def test_generate_format_bytes(image_client):
    """format='bytes' returns a valid PNG byte string."""
    data = image_client.generate(
        "a simple green triangle",
        format="bytes",
        **_provider_kwargs(image_client),
    )
    assert isinstance(data, bytes)
    assert data.startswith(b"\x89PNG")
    assert len(data) > 100


def test_generate_num_images_returns_list(image_client):
    """num_images > 1 returns a list of the requested length."""
    imgs = image_client.generate(
        "a yellow star",
        num_images=2,
        **_provider_kwargs(image_client),
    )
    assert isinstance(imgs, list)
    assert len(imgs) == 2
    assert all(isinstance(i, Image.Image) for i in imgs)


# ---------------------------------------------------------------------------
# Provider-specific behaviour (auto-skipped on the wrong provider)
# ---------------------------------------------------------------------------


def test_hf_seed_reproducible(image_client):
    """Same seed → same bytes. HuggingFace diffusers only — Gemini has no seed API."""
    from aimu.models.providers.hf.image import HuggingFaceImageClient

    if not isinstance(image_client, HuggingFaceImageClient):
        pytest.skip("Seed reproducibility only applies to HuggingFace diffusers")

    kwargs = _provider_kwargs(image_client)
    a = image_client.generate("a yellow star", seed=123, format="bytes", **kwargs)
    b = image_client.generate("a yellow star", seed=123, format="bytes", **kwargs)
    assert a[:8] == b[:8]  # PNG signature
    assert abs(len(a) - len(b)) < 0.1 * max(len(a), len(b))


def test_gemini_aspect_ratio_accepted(image_client):
    """aspect_ratio kwarg threads through to the Gemini ImageConfig.

    HuggingFace ignores aspect_ratio — its API takes explicit width/height — so
    this test is Gemini-only.
    """
    from aimu.models.providers.gemini.image import GeminiImageClient

    if not isinstance(image_client, GeminiImageClient):
        pytest.skip("aspect_ratio is a Gemini-only knob")

    img = image_client.generate("a watercolor of a fox", aspect_ratio="1:1")
    assert isinstance(img, Image.Image)
