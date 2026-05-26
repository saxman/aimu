"""Mock-only unit tests for the built-in ``generate_image`` tool and ``make_image_tool``.

Verifies the lazy singleton wiring, the tool spec shape, env-var override, and
the per-agent factory escape hatch. No diffusers install required.
"""

from __future__ import annotations

# Install the diffusers stub before importing aimu.tools.builtin's image bits.
from test_images_api import _install_diffusers_stub  # noqa: F401 — side effect

_install_diffusers_stub()

import importlib  # noqa: E402

import aimu.models  # noqa: E402

if not aimu.models.HAS_HF_IMAGE:
    aimu.models.hf_image = importlib.import_module("aimu.models.hf_image")
    aimu.models.HAS_HF_IMAGE = True
    aimu.models.HuggingFaceImageClient = aimu.models.hf_image.HuggingFaceImageClient
    aimu.models.HuggingFaceImageModel = aimu.models.hf_image.HuggingFaceImageModel

import aimu  # noqa: E402

aimu.HAS_HF_IMAGE = True
aimu.HuggingFaceImageClient = aimu.models.HuggingFaceImageClient
aimu.HuggingFaceImageModel = aimu.models.HuggingFaceImageModel

from aimu.tools import builtin  # noqa: E402


# ---------------------------------------------------------------------------
# Tool spec shape
# ---------------------------------------------------------------------------


def test_generate_image_has_tool_spec():
    spec = builtin.generate_image.__tool_spec__
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "generate_image"
    assert "prompt" in spec["function"]["parameters"]["properties"]
    assert spec["function"]["parameters"]["properties"]["prompt"]["type"] == "string"
    assert spec["function"]["parameters"]["required"] == ["prompt"]


def test_generate_image_is_sync():
    """The sync built-in must be a regular def, not async — async lives under aimu.aio.tools."""
    assert builtin.generate_image.__tool_is_async__ is False


def test_generate_image_in_image_subgroup():
    assert builtin.generate_image in builtin.image
    assert builtin.generate_image in builtin.ALL_TOOLS


# ---------------------------------------------------------------------------
# Singleton + env var
# ---------------------------------------------------------------------------


def test_lazy_singleton_constructed_once(monkeypatch):
    """_get_image_client should cache a single image client and reuse it."""
    monkeypatch.setattr(builtin, "_image_client", None)
    monkeypatch.setenv("AIMU_IMAGE_MODEL", "hf:test/repo")

    constructed: list[str] = []

    def fake_image_client(model_str):
        constructed.append(model_str)
        from aimu.models.hf_image import HuggingFaceImageClient

        return HuggingFaceImageClient(model_str)

    monkeypatch.setattr("aimu.image_client", fake_image_client)

    c1 = builtin._get_image_client()
    c2 = builtin._get_image_client()
    assert c1 is c2
    assert constructed == ["hf:test/repo"]


def test_singleton_honours_env_var(monkeypatch, tmp_path):
    """AIMU_IMAGE_MODEL env var should pick the singleton's model."""
    monkeypatch.setattr(builtin, "_image_client", None)
    monkeypatch.setenv("AIMU_IMAGE_MODEL", "hf:my/custom-repo")

    c = builtin._get_image_client()
    assert c.spec.id == "my/custom-repo"


def test_tool_calls_singleton_and_returns_path(monkeypatch, tmp_path):
    """Calling generate_image() exercises the lazy path end-to-end."""
    from aimu.models.hf_image import HuggingFaceImageClient, HuggingFaceImageModel

    monkeypatch.setattr(builtin, "_image_client", HuggingFaceImageClient(HuggingFaceImageModel.SD_1_5))
    # Reroute saves into tmp_path so we don't litter the repo's output dir.
    monkeypatch.setattr(
        builtin._get_image_client(),
        "generate",
        lambda prompt, format: f"{tmp_path}/fake.png" if format == "path" else None,
    )

    result = builtin.generate_image("a cat")
    assert result.endswith("fake.png")


# ---------------------------------------------------------------------------
# make_image_tool — per-agent factory escape hatch
# ---------------------------------------------------------------------------


def test_make_image_tool_returns_new_tool_bound_to_supplied_client():
    from aimu.models.hf_image import HuggingFaceImageClient, HuggingFaceImageModel

    client = HuggingFaceImageClient(HuggingFaceImageModel.FLUX_SCHNELL)
    bound_tool = builtin.make_image_tool(client)

    assert bound_tool is not builtin.generate_image
    assert bound_tool.__tool_spec__["function"]["name"] == "generate_image"
    assert bound_tool.__tool_is_async__ is False


def test_make_image_tool_uses_its_client_not_singleton(monkeypatch, tmp_path):
    from aimu.models.hf_image import HuggingFaceImageClient, HuggingFaceImageModel

    # Singleton should remain untouched; the bound tool should call its own client.
    sentinel = "singleton-should-not-be-touched"
    monkeypatch.setattr(builtin, "_image_client", sentinel)

    custom = HuggingFaceImageClient(HuggingFaceImageModel.FLUX_SCHNELL)
    bound_tool = builtin.make_image_tool(custom)

    monkeypatch.setattr(
        custom,
        "generate",
        lambda prompt, format: f"{tmp_path}/custom.png" if format == "path" else None,
    )

    result = bound_tool("a fox")
    assert result.endswith("custom.png")
    # Singleton wasn't constructed (would have failed type-checks on the sentinel).
    assert builtin._image_client is sentinel
