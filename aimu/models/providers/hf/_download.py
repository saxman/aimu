"""Download-only counterpart of the HuggingFace loaders, for ``aimu.pull``.

Each HuggingFace client declares what its loader fetches as a list of :class:`HubDownload` entries
(``_hub_downloads(spec)``, defined next to the loader it describes). :func:`fetch` downloads one
entry into the same caches the loader reads, so after every declared entry is fetched the client
builds and runs with no network. ``tests/test_hf_downloads.py`` runs each loader against recording
stand-ins and fails when it requests a repo its declaration does not name.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

logger = logging.getLogger(__name__)

# Directories that hold a second copy of a model for another runtime (or the original release format),
# never read by a transformers / sentence-transformers load.
_FOREIGN_DIRS = ("onnx/", "openvino/", "coreml/", "original/", "metal/")

_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".msgpack",
    ".ot",
    ".onnx",
    ".onnx_data",
    ".gguf",
    ".tflite",
)

# Formats a PyTorch load never reads, dropped even from a directory with no standard weight name.
_NON_TORCH_SUFFIXES = (".h5", ".msgpack", ".ot", ".onnx", ".onnx_data", ".gguf", ".tflite")

_SAFETENSORS_SHARD = re.compile(r"^model-\d+-of-\d+\.safetensors$")
_PYTORCH_SHARD = re.compile(r"^pytorch_model-\d+-of-\d+\.bin$")


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


def select_files(files: list[str]) -> list[str]:
    """The files of a repo listing that a transformers / sentence-transformers load reads.

    Repos routinely carry several copies of the same weights (fp32 next to fp16, pickles next to
    safetensors, the original release format, an audiocraft checkpoint), and a whole-repo snapshot
    downloads all of them. Weights are chosen per directory the way transformers chooses them: the
    sharded safetensors named by ``model.safetensors.index.json``, else ``model.safetensors``, else the
    sharded ``pytorch_model`` bins, else ``pytorch_model.bin``. A directory with weights under none of
    those names is kept whole (minus non-PyTorch formats), since over-fetching beats leaving the loader
    something to download. Everything that is not a weight file is kept.
    """
    kept = [name for name in files if not name.startswith(_FOREIGN_DIRS)]
    weights_by_dir: dict[str, list[str]] = {}
    for name in kept:
        if name.endswith(_WEIGHT_SUFFIXES):
            directory, _, filename = name.rpartition("/")
            weights_by_dir.setdefault(directory, []).append(filename)

    chosen: set[str] = set()
    for directory, filenames in weights_by_dir.items():
        prefix = f"{directory}/" if directory else ""
        present = {prefix + name for name in kept if name.startswith(prefix) and "/" not in name[len(prefix) :]}
        if f"{prefix}model.safetensors.index.json" in present:
            picked = [f for f in filenames if _SAFETENSORS_SHARD.match(f)]
        elif "model.safetensors" in filenames:
            picked = ["model.safetensors"]
        elif f"{prefix}pytorch_model.bin.index.json" in present:
            picked = [f for f in filenames if _PYTORCH_SHARD.match(f)]
        elif "pytorch_model.bin" in filenames:
            picked = ["pytorch_model.bin"]
        else:
            picked = [f for f in filenames if not f.endswith(_NON_TORCH_SUFFIXES)]
        chosen.update(prefix + f for f in picked)

    return [name for name in kept if not name.endswith(_WEIGHT_SUFFIXES) or name in chosen]


def weight_patterns(repo_id: str) -> tuple[Optional[list[str]], Optional[list[str]]]:
    """``(allow_patterns, ignore_patterns)`` for a snapshot of exactly what ``from_pretrained`` reads.

    Lists the repo first and allows the files :func:`select_files` picks, by name.
    """
    from huggingface_hub import HfApi

    files = [sibling.rfilename for sibling in HfApi().model_info(repo_id).siblings or []]
    return select_files(files), None


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
