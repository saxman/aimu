"""Shared device-placement helpers for in-process HuggingFace clients.

The image (diffusers), audio, and speech clients all load weights in-process and
need placement logic: an optional single-GPU ``device`` hint, a ``cuda``/``mps``/CPU
fallback, and — for the image client — **memory-aware auto-placement** that inspects
how many GPUs are present and how much memory each has *free* before deciding where
to put a freshly loaded pipeline (see :func:`auto_place_pipeline`).

The text client is intentionally *not* a consumer: ``transformers`` handles
multi-GPU natively via ``device_map="auto"`` (already its default), which also
spills to CPU/disk.

``device`` (e.g. ``"cuda:1"``) is an AIMU-level hint for placing a whole model on
one device. It is not a ``from_pretrained`` argument, so callers pop it out before
forwarding the rest of ``model_kwargs`` to the loader. ``device_map`` *is* a real
loader argument and stays in the kwargs. Either one, when set explicitly, overrides
auto-placement.
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


def has_accelerate() -> bool:
    """True when the ``accelerate`` package (required for CPU offload) is installed."""
    return importlib.util.find_spec("accelerate") is not None


def cuda_free_memory() -> list[tuple[int, int]]:
    """Per-GPU free memory as ``[(device_index, free_bytes), ...]``, most-free first.

    Uses ``torch.cuda.mem_get_info`` so the figures reflect memory actually free
    *right now* — accounting for other processes on the card (e.g. a local LLM
    server). Returns an empty list when CUDA/torch are unavailable.
    """
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    infos = [(i, torch.cuda.mem_get_info(i)[0]) for i in range(torch.cuda.device_count())]
    infos.sort(key=lambda pair: pair[1], reverse=True)
    return infos


def default_torch_dtype() -> Any:
    """The memory-efficient ``torch.dtype`` for the active accelerator.

    CUDA → ``bfloat16`` (``float16`` if the GPU lacks bf16 support); MPS → ``float16``;
    CPU → ``float32``. bf16 is preferred on capable GPUs because it halves VRAM versus
    fp32 without fp16's overflow issues (e.g. black-image VAE decodes). Returns ``None``
    if torch is unavailable, so callers can simply omit the argument.
    """
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


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


_GIB = 1024**3
_MEMORY_MARGIN = 1.15  # headroom over raw weights for activations / workspace


def _pipeline_component_sizes(pipe: Any) -> list[int]:
    """Byte size of each ``nn.Module`` component of a diffusers pipeline (params + buffers)."""
    try:
        import torch.nn as nn
    except ImportError:
        return []
    sizes = []
    for component in getattr(pipe, "components", {}).values():
        if isinstance(component, nn.Module):
            params = sum(p.numel() * p.element_size() for p in component.parameters())
            buffers = sum(b.numel() * b.element_size() for b in component.buffers())
            sizes.append(params + buffers)
    return sizes


def auto_place_pipeline(pipe: Any) -> Any:
    """Place a freshly loaded (CPU-resident) diffusers pipeline using real free VRAM.

    Inspects how many GPUs are visible and how much memory each has *free* right now,
    measures the pipeline's loaded size, then picks the cheapest strategy that fits:

    1. whole pipeline fits the freest single GPU  → move it there (fastest);
    2. else the largest single component fits a GPU → ``enable_model_cpu_offload`` on it
       (components stream to GPU as needed; peak ≈ largest component);
    3. else                                        → ``enable_sequential_cpu_offload``
       (per-layer streaming; slowest but fits almost anything).

    Falls back to the simple ``cuda``/``mps``/CPU move when there is no CUDA, no
    measurable size, or ``accelerate`` is missing (offload requires it). Logs the
    decision at INFO so the chosen plan is visible.
    """
    gpus = cuda_free_memory()
    if not gpus:
        return move_to_device(pipe, None)  # MPS / CPU

    idx, free = gpus[0]
    sizes = _pipeline_component_sizes(pipe)
    model_bytes = sum(sizes)
    largest = max(sizes, default=0)

    if model_bytes == 0 or model_bytes * _MEMORY_MARGIN <= free:
        logger.info(
            "Auto-placement: pipeline (%.1f GiB) → cuda:%d (%.1f GiB free).",
            model_bytes / _GIB, idx, free / _GIB,
        )
        return pipe.to(f"cuda:{idx}")

    if not has_accelerate():
        logger.warning(
            "Auto-placement: pipeline (%.1f GiB) exceeds free VRAM on every GPU and "
            "`accelerate` is not installed for CPU offload; pinning to cuda:%d may OOM. "
            "Install accelerate, free GPU memory, or pass model_kwargs={'device': ...}.",
            model_bytes / _GIB, idx,
        )
        return pipe.to(f"cuda:{idx}")

    if largest * _MEMORY_MARGIN <= free:
        logger.info(
            "Auto-placement: pipeline (%.1f GiB) too large for one GPU; model CPU offload "
            "on cuda:%d (largest component %.1f GiB, %.1f GiB free).",
            model_bytes / _GIB, idx, largest / _GIB, free / _GIB,
        )
        pipe.enable_model_cpu_offload(gpu_id=idx)
        return pipe

    logger.info(
        "Auto-placement: tight VRAM (%.1f GiB free on cuda:%d); sequential CPU offload.",
        free / _GIB, idx,
    )
    pipe.enable_sequential_cpu_offload(gpu_id=idx)
    return pipe
