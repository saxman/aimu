"""OpenAI-backed text-to-speech (TTS) client.

``OpenAISpeechClient`` calls ``openai.audio.speech.create()`` (tts-1 / tts-1-hd).
Uses the same ``openai`` SDK already required by the ``[openai_compat]`` extra --
no new dependency.

Auth: reads ``OPENAI_API_KEY`` from env (same as :class:`OpenAIClient`).

Output standardisation: ``response_format="pcm"`` requests raw 16-bit signed
integers at 24 kHz. The client decodes them to float32 numpy via
``np.frombuffer(..., dtype=np.int16) / 32768.0`` and passes the result through
:func:`aimu.models._internal.audio_output.encode_audio` -- the same helper used by every
other audio/speech client.

Usage::

    from aimu.models import OpenAISpeechClient, OpenAISpeechModel

    client = OpenAISpeechClient(OpenAISpeechModel.TTS_1)
    path = client.generate("Hello, world!")

    sr, audio = client.generate("Hello", format="numpy")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Union

from dotenv import load_dotenv

from ...base import BaseSpeechClient, OpenAISpeechSpec, SpeechModel, StreamChunk, StreamingContentType

logger = logging.getLogger(__name__)

_OPENAI_PCM_SAMPLE_RATE = 24_000  # OpenAI PCM is always 24 kHz mono


class OpenAISpeechModel(SpeechModel):
    """Catalog of OpenAI TTS models.

    Each member's value is an :class:`OpenAISpeechSpec`. Voices are passed at
    call time via ``voice=``; each spec carries a ``default_voice``.
    """

    TTS_1 = OpenAISpeechSpec("tts-1", default_voice="alloy", default_speed=1.0)
    TTS_1_HD = OpenAISpeechSpec("tts-1-hd", default_voice="alloy", default_speed=1.0)


class OpenAISpeechClient(BaseSpeechClient):
    """Text-to-speech client backed by the OpenAI API (tts-1 / tts-1-hd).

    Requests ``response_format="pcm"`` (raw 16-bit signed integers at 24 kHz)
    and decodes to float32 numpy so the shared :func:`encode_audio` helper can
    write WAV files without a separate format layer.

    ``stream=True`` enables HTTP byte streaming: progress chunks carry the byte
    index; the final chunk carries the encoded output after all bytes are received.
    """

    MODELS = OpenAISpeechModel

    def __init__(
        self,
        model: Union[OpenAISpeechModel, OpenAISpeechSpec, str],
        model_kwargs: Optional[dict] = None,
    ):
        if isinstance(model, str):
            # Accept "openai:tts-1" string form
            if ":" in model:
                provider, _, model_id = model.partition(":")
                if provider != "openai":
                    raise ValueError(
                        f"Only 'openai:' provider is supported for OpenAISpeechClient. Got provider: {provider!r}"
                    )
                # Resolve to a known enum member; arbitrary ids are not supported via string.
                for member in OpenAISpeechModel:
                    if member.value == model_id:
                        model = member
                        break
                else:
                    available = sorted(m.value for m in OpenAISpeechModel)
                    raise ValueError(
                        f"Unknown OpenAI speech model id {model_id!r}. AIMU supports curated "
                        f"models only; pass a known id, an OpenAISpeechModel member, or a hand-built "
                        f"OpenAISpeechSpec for a custom model. Available ids: {available}"
                    )
            else:
                raise ValueError(
                    f"OpenAI speech model string must be in 'openai:model_id' form "
                    f"(e.g. 'openai:tts-1'). Got: {model!r}"
                )

        if isinstance(model, OpenAISpeechModel):
            spec = model.spec
        elif isinstance(model, OpenAISpeechSpec):
            spec = model
        else:
            raise TypeError(
                f"OpenAISpeechClient expects an OpenAISpeechModel member, "
                f"OpenAISpeechSpec, or 'openai:<model_id>' string. Got: {type(model).__name__}"
            )

        super().__init__(model=model, model_kwargs=model_kwargs)
        self.spec = spec

        load_dotenv()
        from openai import OpenAI

        api_key = (model_kwargs or {}).get("api_key") or os.environ.get("OPENAI_API_KEY", "not-set")
        self._client = OpenAI(api_key=api_key)

    def _generate(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        num_audio: int = 1,
        **_ignored: Any,
    ) -> list:
        import numpy as np

        resolved_voice = voice or self.spec.default_voice
        resolved_speed = float(speed) if speed is not None else self.spec.default_speed

        results = []
        for _ in range(num_audio):
            response = self._client.audio.speech.create(
                model=self.spec.id,
                voice=resolved_voice,
                input=text,
                speed=resolved_speed,
                response_format="pcm",
            )
            pcm_bytes = response.content
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            results.append((_OPENAI_PCM_SAMPLE_RATE, audio))
        return results

    def _generate_streamed(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        num_audio: int = 1,
        format: str = "path",
        output_dir: Optional[Any] = None,
        **_ignored: Any,
    ):
        """Stream HTTP byte chunks from the OpenAI TTS API.

        Accumulates raw PCM bytes across all HTTP chunks (chunk boundaries do not
        align to sample boundaries), decodes to float32 numpy once all bytes are
        received, then emits a single final :attr:`StreamingContentType.SPEECH_GENERATING`
        chunk with the encoded output.
        """
        import numpy as np

        from ..._internal.audio_output import encode_audio

        resolved_voice = voice or self.spec.default_voice
        resolved_speed = float(speed) if speed is not None else self.spec.default_speed

        for _ in range(num_audio):
            all_bytes = bytearray()
            chunk_index = 0

            with self._client.audio.with_streaming_response.speech.create(
                model=self.spec.id,
                voice=resolved_voice,
                input=text,
                speed=resolved_speed,
                response_format="pcm",
            ) as response:
                for byte_chunk in response.iter_bytes(chunk_size=4096):
                    all_bytes.extend(byte_chunk)
                    chunk_index += 1
                    yield StreamChunk(
                        StreamingContentType.SPEECH_GENERATING,
                        {
                            "chunk_index": chunk_index,
                            "total_chunks": None,
                            "final": False,
                            "result": None,
                        },
                    )

            audio = np.frombuffer(bytes(all_bytes), dtype=np.int16).astype(np.float32) / 32768.0
            encoded = encode_audio(audio, _OPENAI_PCM_SAMPLE_RATE, format=format, prompt=text, output_dir=output_dir)
            yield StreamChunk(
                StreamingContentType.SPEECH_GENERATING,
                {
                    "chunk_index": chunk_index + 1,
                    "total_chunks": chunk_index + 1,
                    "final": True,
                    "result": encoded,
                },
            )
