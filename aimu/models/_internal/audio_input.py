"""Audio normalization and per-provider conversion helpers for audio input.

The canonical format inside ``self.messages`` is OpenAI's ``input_audio`` block::

    {
        "role": "user",
        "content": [
            {"type": "text", "text": "..."},
            {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}},
        ],
    }

This mirrors OpenAI's actual API block type for audio, consistent with the "plain data =
OpenAI message format" design principle (analogous to ``image_url`` for vision).

Providers that don't speak OpenAI's format natively (Anthropic, HuggingFace) convert at
request time without mutating ``self.messages``. Ollama raises an error because its API
has no audio input support.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Union

AudioInput = Union[str, bytes, Path]

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)

# Formats accepted by OpenAI's input_audio spec.
_SUPPORTED_FORMATS = {"wav", "mp3", "ogg", "flac", "m4a", "webm"}

_EXT_TO_FORMAT = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".ogg": "ogg",
    ".flac": "flac",
    ".m4a": "m4a",
    ".webm": "webm",
}


def _normalize_audio(audio: AudioInput) -> tuple[str, str]:
    """Normalize a single audio input to ``(base64_data, format_string)``.

    Accepts:
      - raw bytes                               → base64-encoded; format defaults to ``"wav"``
      - ``data:audio/<fmt>;base64,<b64>``       → parsed directly
      - file path (str or ``pathlib.Path``)     → read and base64-encoded; format from extension
      - http(s):// URL                          → fetched eagerly (``input_audio`` has no remote-URL field)
    """
    if isinstance(audio, bytes):
        return base64.b64encode(audio).decode("ascii"), "wav"

    if isinstance(audio, Path):
        return _path_to_base64(audio)

    if isinstance(audio, str):
        if audio.startswith("data:audio/"):
            return _parse_audio_data_url(audio)
        if audio.startswith(("http://", "https://")):
            return _fetch_url(audio)
        return _path_to_base64(Path(audio))

    raise TypeError(f"Unsupported audio input type: {type(audio).__name__}")


def _path_to_base64(path: Path) -> tuple[str, str]:
    fmt = _EXT_TO_FORMAT.get(path.suffix.lower(), "wav")
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii"), fmt


def _parse_audio_data_url(url: str) -> tuple[str, str]:
    m = _DATA_URL_RE.match(url)
    if not m:
        raise ValueError(f"Not a base64 audio data URL: {url[:64]!r}")
    mime = m.group("mime")  # e.g. "audio/wav"
    fmt = mime.split("/")[-1] if "/" in mime else "wav"
    if fmt not in _SUPPORTED_FORMATS:
        fmt = "wav"
    return m.group("data"), fmt


def _fetch_url(url: str) -> tuple[str, str]:
    """Fetch an http(s) audio URL and return (base64_data, format)."""
    import urllib.request

    # Guess format from the URL path before fetching.
    path_part = url.split("?")[0]
    fmt = _EXT_TO_FORMAT.get(Path(path_part).suffix.lower(), "wav")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    return base64.b64encode(data).decode("ascii"), fmt


def _build_audio_content_blocks(text: str, audio: list[AudioInput]) -> list[dict]:
    """Build a multi-modal user message content list with ``input_audio`` blocks."""
    blocks: list[dict] = [{"type": "text", "text": text}]
    for clip in audio:
        b64, fmt = _normalize_audio(clip)
        blocks.append({"type": "input_audio", "input_audio": {"data": b64, "format": fmt}})
    return blocks


def _openai_audio_blocks_to_anthropic(blocks: list[dict]) -> list[dict]:
    """Convert an OpenAI-format content block list containing ``input_audio`` entries to Anthropic format.

    Text blocks and unrecognised blocks are passed through unchanged. ``input_audio`` blocks
    become Anthropic ``audio`` blocks with a ``base64`` source.

    Note: verify the exact Anthropic audio block structure against the live API at
    implementation time — the shape used here matches the documented Anthropic pattern.
    """
    out: list[dict] = []
    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif btype == "input_audio":
            ia = block["input_audio"]
            out.append(
                {
                    "type": "audio",
                    "source": {
                        "type": "base64",
                        "media_type": f"audio/{ia['format']}",
                        "data": ia["data"],
                    },
                }
            )
        else:
            out.append(block)
    return out


def _extract_audio_arrays(messages: list[dict]) -> list:
    """Walk messages and return all audio clips decoded to float32 numpy arrays (in order).

    Used by the HuggingFace processor path. Requires ``soundfile`` (in the ``[hf]`` extra).
    """
    import io

    import soundfile as sf

    arrays = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "input_audio":
                    raw = base64.b64decode(block["input_audio"]["data"])
                    audio_arr, _ = sf.read(io.BytesIO(raw), dtype="float32")
                    arrays.append(audio_arr)
    return arrays


def _replace_audio_with_placeholder(messages: list[dict]) -> list[dict]:
    """Return a copy of ``messages`` with ``input_audio`` blocks replaced by ``{"type": "audio"}``.

    HuggingFace VL/audio chat templates use ``{"type": "audio"}`` placeholder blocks; the
    actual arrays are passed separately as ``audio=`` to the processor.
    """
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_blocks = [{"type": "audio"} if b.get("type") == "input_audio" else b for b in content]
            out.append({**msg, "content": new_blocks})
        else:
            out.append(msg)
    return out


def _adapt_messages_for_ollama(messages: list[dict]) -> list[dict]:
    """Raise ``ValueError`` if any message contains ``input_audio`` blocks.

    Ollama's API has no audio input support. Once Ollama adds it, this function
    will be updated to perform the necessary format conversion.
    """
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "input_audio":
                    raise ValueError(
                        "Audio input is not yet supported by the Ollama API. "
                        "Use the HuggingFace client for local audio-capable models."
                    )
    return messages
