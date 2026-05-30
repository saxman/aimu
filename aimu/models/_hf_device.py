"""Shared device-placement helpers for in-process HuggingFace clients.

The image (diffusers), audio, and speech clients all load weights in-process and
need the same placement logic: an optional single-GPU ``device`` hint, a
``cuda``/``mps``/CPU fallback, and — for large-model modalities only — a
multi-GPU capability check before defaulting to ``device_map`` sharding.

The text client is intentionally *not* a consumer: ``transformers`` handles
multi-GPU natively via ``device_map="auto"`` (already its default), which also
spills to CPU/disk. Diffusers, by contrast, only supports ``device_map="balanced"``
at the pipeline level, so the image client defaults to that when sharding helps.

``device`` (e.g. ``"cuda:1"``) is an AIMU-level hint for placing a whole model on
one device. It is not a ``from_pretrained`` argument, so callers pop it out before
forwarding the rest of ``model_kwargs`` to the loader. ``device_map`` *is* a real
loader argument and stays in the kwargs.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def pop_device_hint(kwargs: dict) -> Optional[str]:
    """Remove and return the AIMU ``device`` placement hint from a kwargs dict.

    Returns ``None`` when no hint was supplied. Mutates ``kwargs`` in place so the
    remaining keys are safe to forward to ``from_pretrained`` / ``pipeline``.
    """
    return kwargs.pop("device", None)


def multi_gpu_with_accelerate() -> bool:
    """True when ``device_map`` sharding is both possible and worthwhile.

    Requires more than one visible CUDA device and an installed ``accelerate``
    (the loaders need it for ``device_map``). Returns ``False`` on MPS, CPU, a
    single GPU, or when torch/accelerate are unavailable.
    """
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return False
    return importlib.util.find_spec("accelerate") is not None


def resolve_device(hint: Optional[str] = None) -> str:
    """Resolve a placement target: the explicit ``hint`` or the best accelerator.

    Falls back ``cuda`` → ``mps`` → ``cpu`` when no hint is given. Returns ``"cpu"``
    if torch is unavailable.
    """
    if hint is not None:
        return hint
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def move_to_device(obj: Any, hint: Optional[str] = None) -> Any:
    """Move a model/pipeline to ``hint`` (or the autodetected accelerator).

    Thin wrapper over ``obj.to(resolve_device(hint))``. Use this only on the
    single-device path — when ``device_map`` is set, the loader owns placement and
    ``.to()`` must not be called.
    """
    return obj.to(resolve_device(hint))
