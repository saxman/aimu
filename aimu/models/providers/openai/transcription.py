"""OpenAI-backed speech-to-text (ASR) client.

``OpenAITranscriptionClient`` calls ``openai.audio.transcriptions.create()``
(whisper-1 / gpt-4o-transcribe / gpt-4o-mini-transcribe). Uses the same ``openai``
SDK already required by the ``[openai_compat]`` extra.

Auth: reads ``OPENAI_API_KEY`` from env (same as :class:`OpenAIClient`).

Input: ``_normalize_audio()`` handles paths, bytes, data URLs, and https:// URLs.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Optional, Union

from dotenv import load_dotenv

from ...base import BaseTranscriptionClient, OpenAITranscriptionSpec, TranscriptionModel


class OpenAITranscriptionModel(TranscriptionModel):
    """Catalog of OpenAI speech-to-text models."""

    WHISPER_1 = OpenAITranscriptionSpec("whisper-1", supports_timestamps=True, supports_translation=True)
    GPT_4O_TRANSCRIBE = OpenAITranscriptionSpec("gpt-4o-transcribe", supports_timestamps=True)
    GPT_4O_MINI_TRANSCRIBE = OpenAITranscriptionSpec("gpt-4o-mini-transcribe", supports_timestamps=True)


class OpenAITranscriptionClient(BaseTranscriptionClient):
    """Speech-to-text client backed by the OpenAI API.

    Accepts ``whisper-1``, ``gpt-4o-transcribe``, and ``gpt-4o-mini-transcribe``.
    """

    MODELS = OpenAITranscriptionModel

    def __init__(
        self,
        model: Union[OpenAITranscriptionModel, OpenAITranscriptionSpec, str],
        model_kwargs: Optional[dict] = None,
    ):
        if isinstance(model, str):
            if ":" in model:
                provider, _, model_id = model.partition(":")
                if provider != "openai":
                    raise ValueError(
                        f"Only 'openai:' provider is supported for OpenAITranscriptionClient. "
                        f"Got provider: {provider!r}"
                    )
                for member in OpenAITranscriptionModel:
                    if member.value == model_id:
                        model = member
                        break
                else:
                    available = sorted(m.value for m in OpenAITranscriptionModel)
                    raise ValueError(
                        f"Unknown OpenAI transcription model id {model_id!r}. AIMU supports curated "
                        f"models only; pass a known id, an OpenAITranscriptionModel member, or a "
                        f"hand-built OpenAITranscriptionSpec for a custom model. "
                        f"Available ids: {available}"
                    )
            else:
                raise ValueError(
                    f"OpenAI transcription model string must be in 'openai:model_id' form "
                    f"(e.g. 'openai:whisper-1'). Got: {model!r}"
                )

        if isinstance(model, OpenAITranscriptionModel):
            spec = model.spec
        elif isinstance(model, OpenAITranscriptionSpec):
            spec = model
        else:
            raise TypeError(
                f"OpenAITranscriptionClient expects an OpenAITranscriptionModel member, "
                f"OpenAITranscriptionSpec, or 'openai:<model_id>' string. Got: {type(model).__name__}"
            )

        super().__init__(model=model, model_kwargs=model_kwargs)
        self.spec = spec

        load_dotenv()
        from openai import OpenAI

        api_key = (model_kwargs or {}).get("api_key") or os.environ.get("OPENAI_API_KEY", "not-set")
        self._client = OpenAI(api_key=api_key)

    def _transcribe(
        self,
        audio: Any,
        *,
        language: Optional[str] = None,
        response_format: str = "text",
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str | dict:
        from aimu.models._internal.audio_input import _normalize_audio

        b64, fmt = _normalize_audio(audio)
        raw_bytes = base64.b64decode(b64)
        file_obj = io.BytesIO(raw_bytes)
        file_obj.name = f"audio.{fmt}"

        kwargs: dict[str, Any] = {
            "model": self.spec.id,
            "file": file_obj,
            "response_format": response_format,
        }
        if language is not None:
            kwargs["language"] = language
        if prompt is not None:
            kwargs["prompt"] = prompt
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self._client.audio.transcriptions.create(**kwargs)

        if response_format == "text":
            return str(response)
        return response.model_dump()
