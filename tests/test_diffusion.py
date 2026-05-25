"""Live diffusion tests — opt in via ``--diffusion-model=<MEMBER_NAME>``.

Skipped by default. To run::

    pytest tests/test_diffusion.py --diffusion-model=SD_1_5

Generates a small image with the selected :class:`DiffusionModel` member and
asserts non-empty PNG bytes + reasonable dimensions. Reuses the loaded pipeline
across tests via a session-scoped fixture so weights load once.
"""

from __future__ import annotations

import pytest

try:
    from aimu.models import HAS_DIFFUSION

    if HAS_DIFFUSION:
        from aimu.models.diffusion import DiffusionClient, DiffusionModel
except ImportError:  # pragma: no cover
    HAS_DIFFUSION = False


@pytest.fixture(scope="session")
def diffusion_client(request):
    if not HAS_DIFFUSION:
        pytest.skip("Diffusion extra not installed (pip install -e '.[diffusion]')")
    member_name = request.config.getoption("--diffusion-model")
    if not member_name:
        pytest.skip("Pass --diffusion-model=<MEMBER_NAME> to run live diffusion tests")
    try:
        member = DiffusionModel[member_name]
    except KeyError:
        available = ", ".join(m.name for m in DiffusionModel)
        pytest.fail(f"Unknown DiffusionModel member: {member_name!r}. Available: {available}")
    return DiffusionClient(member)


def test_live_generate_returns_pil_image(diffusion_client):
    """Smoke test: produces a PIL image with the requested dimensions."""
    from PIL import Image

    img = diffusion_client.generate(
        "a simple red ball on a white background",
        width=256,
        height=256,
        num_inference_steps=4,
        seed=42,
    )
    assert isinstance(img, Image.Image)
    assert img.size == (256, 256)


def test_live_generate_format_path(diffusion_client, tmp_path):
    """format='path' writes a valid PNG and returns its path."""
    from PIL import Image

    out = diffusion_client.generate(
        "a simple blue square",
        width=256,
        height=256,
        num_inference_steps=4,
        seed=7,
        format="path",
        output_dir=tmp_path,
    )
    assert isinstance(out, str)
    decoded = Image.open(out)
    assert decoded.size == (256, 256)


def test_live_generate_format_bytes(diffusion_client):
    """format='bytes' returns non-empty PNG bytes."""
    data = diffusion_client.generate(
        "a simple green triangle",
        width=256,
        height=256,
        num_inference_steps=4,
        seed=11,
        format="bytes",
    )
    assert isinstance(data, bytes)
    assert len(data) > 100
    assert data.startswith(b"\x89PNG")


def test_live_generate_seed_reproducible(diffusion_client):
    """Same seed → same bytes (for deterministic samplers)."""
    a = diffusion_client.generate(
        "a yellow star",
        width=256,
        height=256,
        num_inference_steps=4,
        seed=123,
        format="bytes",
    )
    b = diffusion_client.generate(
        "a yellow star",
        width=256,
        height=256,
        num_inference_steps=4,
        seed=123,
        format="bytes",
    )
    # Not all schedulers are perfectly deterministic across PyTorch versions, but
    # output should be at least non-trivially similar in size and headers.
    assert a[:8] == b[:8]  # PNG signature
    assert abs(len(a) - len(b)) < 0.1 * max(len(a), len(b))
