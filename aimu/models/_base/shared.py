"""Cross-modality base types shared by every client surface.

``StreamChunk`` / ``StreamingContentType`` are the single streaming vocabulary used
by text chat, image / audio / speech generation, streaming tools, and workflow runs.
``classproperty`` is the read-only class-level property descriptor used by the
capability classproperties (``TOOL_MODELS`` etc.).
"""

import logging
from enum import Enum
from typing import NamedTuple, Optional, Union

logger = logging.getLogger(__name__)


class StreamingContentType(str, Enum):
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    GENERATING = "generating"
    IMAGE_GENERATING = "image_generating"
    AUDIO_GENERATING = "audio_generating"
    SPEECH_GENERATING = "speech_generating"
    DONE = "done"


class StreamChunk(NamedTuple):
    """A single chunk yielded by ``client.chat(stream=True)``, ``Agent.run(stream=True)``,
    ``image_client.generate(stream=True)``, or any streaming tool / workflow.

    Fields:
        phase:     content type of this chunk (THINKING, TOOL_CALLING, GENERATING,
                   IMAGE_GENERATING, AUDIO_GENERATING, SPEECH_GENERATING, DONE)
        content:   shape depends on phase:
                   - ``str`` for THINKING / GENERATING (token).
                   - ``dict {"name", "arguments", "response"}`` for TOOL_CALLING
                     (``arguments`` is the dict the model passed to the tool).
                   - ``dict {"step", "total_steps", "image", "final", "result"}`` for
                     IMAGE_GENERATING — ``step`` is 1-indexed, ``image`` is an optional
                     ``PIL.Image`` (None unless ``preview_every`` opted in this step),
                     ``final=True`` marks the terminal chunk for one image, and ``result``
                     carries the encoded output (path / bytes / data-url per ``format=``)
                     on the final chunk.
                   - ``dict {"step", "total_steps", "final", "result", "duration_s"}`` for
                     AUDIO_GENERATING — ``step`` is 1-indexed (1 of 1 for non-diffusers
                     models), ``final=True`` marks the terminal chunk per audio item, and
                     ``result`` carries the encoded output on the final chunk.
                   - ``dict {"chunk_index", "total_chunks", "final", "result"}`` for
                     SPEECH_GENERATING — ``total_chunks`` is ``None`` for streaming
                     providers where the total is unknown upfront (OpenAI); ``1`` for
                     single-pass providers (HuggingFace). ``final=True`` marks the
                     terminal chunk; ``result`` carries the encoded output on the final
                     chunk only.
                   - ``str`` for DONE (usually empty).
        agent:     name of the agent that produced this chunk, or ``None`` for a plain
                   ``client.chat()`` / ``client.generate()`` call. Set automatically by
                   ``Agent`` and workflow runners.
        iteration: zero-based iteration index inside the agent loop, or ``0`` for plain chat.

    Use ``chunk.is_text()`` / ``chunk.is_tool_call()`` / ``chunk.is_image_progress()`` /
    ``chunk.is_audio_progress()`` / ``chunk.is_speech_progress()`` to dispatch on phase
    without repeating the equality check in user code.
    """

    phase: StreamingContentType
    content: Union[str, dict]
    agent: Optional[str] = None
    iteration: int = 0

    def is_text(self) -> bool:
        """True if this chunk carries text (THINKING or GENERATING)."""
        return self.phase in (StreamingContentType.THINKING, StreamingContentType.GENERATING)

    def is_tool_call(self) -> bool:
        """True if this chunk carries a tool-call result."""
        return self.phase == StreamingContentType.TOOL_CALLING

    def is_image_progress(self) -> bool:
        """True if this chunk carries image-generation progress (IMAGE_GENERATING)."""
        return self.phase == StreamingContentType.IMAGE_GENERATING

    def is_audio_progress(self) -> bool:
        """True if this chunk carries audio-generation progress (AUDIO_GENERATING)."""
        return self.phase == StreamingContentType.AUDIO_GENERATING

    def is_speech_progress(self) -> bool:
        """True if this chunk carries speech-generation progress (SPEECH_GENERATING)."""
        return self.phase == StreamingContentType.SPEECH_GENERATING


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self.func(cls)
