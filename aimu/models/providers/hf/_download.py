"""Download-only counterpart of the HuggingFace loaders, for ``aimu.pull``.

Each HuggingFace client declares what its loader fetches as a list of :class:`HubDownload` entries
(``_hub_downloads(spec)``, defined next to the loader it describes). :func:`fetch` downloads one
entry into the same caches the loader reads, so after every declared entry is fetched the client
builds and runs with no network. ``tests/test_hf_downloads.py`` runs each loader against recording
stand-ins and fails when it requests a repo its declaration does not name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

logger = logging.getLogger(__name__)

# Formats ``from_pretrained`` never reads for a PyTorch load. A whole-repo snapshot would otherwise
# fetch them too, and some repos carry several copies of the weights.
_ALWAYS_IGNORED = [
    "*.onnx",
    "*.onnx_data",
    "onnx/*",
    "*.gguf",
    "*.msgpack",
    "*.h5",
    "*.tflite",
    "*.ot",
    "openvino/*",
    "coreml/*",
]

# PyTorch pickles, which transformers skips when the repo also carries safetensors.
_PICKLED_WEIGHTS = ["*.bin", "*.pt", "*.pth"]


@dataclass(frozen=True)
class HubDownload:
    """One thing a loader fetches from the HuggingFace Hub.

    ``kind`` picks the downloader: ``"snapshot"`` for a transformers or sentence-transformers repo,
    ``"diffusers"`` for a diffusers pipeline (``options`` carries ``variant`` / ``use_safetensors``,
    applied by diffusers' own file selection), ``"dataset"`` for a dataset the loader reads with
    ``datasets.load_dataset`` (``options`` carries ``split``).
    """

    repo_id: str
    kind: Literal["snapshot", "diffusers", "dataset"] = "snapshot"
    options: Mapping[str, Any] = field(default_factory=dict)


def weight_patterns(repo_id: str) -> tuple[Optional[list[str]], list[str]]:
    """``(allow_patterns, ignore_patterns)`` for a snapshot matching what ``from_pretrained`` reads.

    Lists the repo first, because pickled weights are only redundant when safetensors are present:
    a repo that ships nothing but ``pytorch_model.bin`` still needs it.
    """
    from huggingface_hub import HfApi

    files = [sibling.rfilename for sibling in HfApi().model_info(repo_id).siblings or []]
    ignore = list(_ALWAYS_IGNORED)
    if any(name.endswith(".safetensors") for name in files):
        ignore += _PICKLED_WEIGHTS
    return None, ignore


def fetch(download: HubDownload) -> None:
    """Download one declared entry (only what is missing or changed) into the loader's cache."""
    logger.warning("Downloading %s %r from the HuggingFace Hub.", download.kind, download.repo_id)
    if download.kind == "snapshot":
        import huggingface_hub

        allow, ignore = weight_patterns(download.repo_id)
        huggingface_hub.snapshot_download(download.repo_id, allow_patterns=allow, ignore_patterns=ignore)
    elif download.kind == "diffusers":
        from diffusers import DiffusionPipeline

        DiffusionPipeline.download(download.repo_id, **download.options)
    elif download.kind == "dataset":
        try:
            import datasets
        except ImportError as exc:
            raise ImportError(
                f"{download.repo_id!r} is a dataset the loader reads with the `datasets` package, which no "
                "aimu extra installs. Run `pip install datasets`."
            ) from exc

        datasets.load_dataset(download.repo_id, **download.options)
    else:
        raise ValueError(f"Unknown HubDownload kind {download.kind!r} for {download.repo_id!r}.")
