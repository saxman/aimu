"""Audio output encoding for audio-generation clients.

Sibling to :mod:`aimu.models._internal.image_output` (which encodes images for diffusion output);
this module encodes ``(sample_rate, np.ndarray)`` pairs produced by audio pipelines into
one of ``"numpy"`` / ``"path"`` / ``"bytes"`` / ``"data_url"``.

The single helper :func:`encode_audio` is shared by the sync and async audio clients so
format-conversion logic lives in one place and is unit-testable without loading model weights.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Optional, Union

OutputFormat = Literal["numpy", "path", "bytes", "data_url"]
VALID_FORMATS = ("numpy", "path", "bytes", "data_url")


def _to_wav_bytes(audio: Any, sample_rate: int) -> bytes:
    """Encode a numpy audio array to WAV bytes via soundfile."""
    import soundfile as sf

    buf = BytesIO()
    sf.write(buf, audio.T if audio.ndim == 2 else audio, sample_rate, format="WAV")
    return buf.getvalue()


def encode_audio(
    audio: Any,
    sample_rate: int,
    format: str = "path",
    *,
    prompt: str = "",
    output_dir: Optional[Path] = None,
) -> Union[tuple, str, bytes]:
    """Encode a single audio array into the requested output format.

    Args:
        audio: A ``numpy.ndarray`` — shape ``(samples,)`` for mono or
            ``(channels, samples)`` for multi-channel (e.g. stereo from Stable Audio Open).
        sample_rate: Sampling rate in Hz (e.g. 32000 for MusicGen, 44100 for Stable Audio).
        format: One of ``"numpy"`` (return ``(sample_rate, audio)`` unchanged),
            ``"bytes"`` (WAV-encoded bytes), ``"data_url"`` (``data:audio/wav;base64,...``),
            or ``"path"`` (save WAV to ``output_dir`` and return the file path string).
        prompt: The originating prompt — used only to seed the filename hash when
            ``format="path"``. Not stored anywhere else.
        output_dir: Directory to save into when ``format="path"``. Defaults to
            ``<repo>/output/audio/``; created on demand.

    Returns:
        ``(sample_rate, audio)`` tuple, ``bytes``, or path/data-url ``str``.

    Raises:
        ValueError: If ``format`` is not one of :data:`VALID_FORMATS`.
    """
    if format == "numpy":
        return (sample_rate, audio)

    if format not in VALID_FORMATS:
        raise ValueError(f"Unknown format: {format!r}. Expected one of {VALID_FORMATS}.")

    data = _to_wav_bytes(audio, sample_rate)

    if format == "bytes":
        return data
    if format == "data_url":
        return "data:audio/wav;base64," + base64.b64encode(data).decode("ascii")

    # format == "path"
    if output_dir is None:
        from aimu.paths import output as default_output_root

        output_dir = default_output_root / "audio"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_hash = hashlib.sha1(f"{prompt}{ts}".encode("utf-8")).hexdigest()[:8]
    path = output_dir / f"{ts}-{short_hash}.wav"
    path.write_bytes(data)
    return str(path)
