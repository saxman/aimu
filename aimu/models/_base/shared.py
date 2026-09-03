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


class ModelConnectionError(RuntimeError):
    """Raised when a model client cannot reach its inference server (e.g. the server is down or the
    ``base_url`` is unreachable). Wraps the underlying provider/transport error, which is preserved on
    ``__cause__`` so callers can walk the chain for the specific reason (e.g. "Connection refused").
    Mirrors :class:`aimu.tools.client.MCPConnectionError`."""


class ContextOverflowError(RuntimeError):
    """Raised when a request's messages no longer fit the model's context window: either a networked
    backend rejected the request outright, or an in-process backend with no server to reject it ran a
    pre-flight token count against the model's own known window and declined to even attempt the call.
    The input-side counterpart of :class:`aimu.agents.TruncatedTurnError`, which reports an *output*
    that ran out of room. On a networked backend, the provider's own error is preserved on
    ``__cause__``; on an in-process pre-flight check there is no such error to chain, and ``__cause__``
    is ``None`` by design -- do not assume every instance of this error carries one."""


class ModelRefusalError(RuntimeError):
    """Raised when a model's safety classifiers decline a request rather than answer it.

    Not a transport or validation failure: the API returns HTTP 200 with a ``stop_reason`` of
    ``"refusal"`` and **no answer content**, so a client that only reads the content blocks returns
    an empty string and nothing tells the caller their request was declined. In an agent loop that
    is indistinguishable from a degenerate turn, so the continuation nudge fires and the run burns
    its iterations getting refused again.

    ``category`` is the provider's own classifier label (an open set -- e.g. ``"cyber"``,
    ``"bio"``) and may be ``None``; ``explanation`` is its prose, when it supplies any. Being a
    distinct type, it composes with :class:`aimu.models.FallbackClient`'s ``retry_on``, which is
    the documented recovery: a refusal on one model is often answered by another."""

    def __init__(self, message: str, *, category: Optional[str] = None, explanation: Optional[str] = None):
        super().__init__(message)
        self.category = category
        self.explanation = explanation


class StreamingContentType(str, Enum):
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    CONTINUING = "continuing"
    GENERATING = "generating"
    IMAGE_GENERATING = "image_generating"
    AUDIO_GENERATING = "audio_generating"
    SPEECH_GENERATING = "speech_generating"
    DONE = "done"


class StreamChunk(NamedTuple):
    """A single chunk yielded by ``client.chat(stream=True)``, ``Agent.run(stream=True)``,
    ``image_client.generate(stream=True)``, or any streaming tool / workflow.

    Fields:
        phase:     content type of this chunk (THINKING, TOOL_CALLING, CONTINUING, GENERATING,
                   IMAGE_GENERATING, AUDIO_GENERATING, SPEECH_GENERATING, DONE)
        content:   shape depends on phase:
                   - ``str`` for THINKING / GENERATING (token).
                   - ``dict {"name", "arguments", "response"}`` for TOOL_CALLING
                     (``arguments`` is the dict the model passed to the tool).
                   - ``dict {"kind", "prompt"}`` for CONTINUING: the loop is about to inject a
                     prompt of its own, and the round that follows is that injected turn.
                     ``kind`` is ``PROVENANCE_CONTINUATION`` (a nudge after an empty turn, tools
                     still enabled) or ``PROVENANCE_FINAL_ANSWER`` (the forced wrap-up at the round
                     cap, tools disabled), the same two values the injected message is tagged with
                     in ``client.messages``. ``prompt`` is the text actually sent, so an agent
                     configured with its own ``continuation_prompt`` / ``final_answer_prompt``
                     reports that rather than the built-in default. The value travels under three
                     names: ``PROVENANCE_KEY`` (``"provenance"``) on the message, ``kind`` here on
                     the chunk, and ``reason`` in ``WebChannel``'s ``loop`` frame.
                   - ``dict {"step", "total_steps", "image", "final", "result"}`` for
                     IMAGE_GENERATING: ``step`` is 1-indexed, ``image`` is an optional
                     ``PIL.Image`` (None unless ``preview_every`` opted in this step),
                     ``final=True`` marks the terminal chunk for one image, and ``result``
                     carries the encoded output (path / bytes / data-url per ``format=``)
                     on the final chunk.
                   - ``dict {"step", "total_steps", "final", "result", "duration_s"}`` for
                     AUDIO_GENERATING: ``step`` is 1-indexed (1 of 1 for non-diffusers
                     models), ``final=True`` marks the terminal chunk per audio item, and
                     ``result`` carries the encoded output on the final chunk.
                   - ``dict {"chunk_index", "total_chunks", "final", "result"}`` for
                     SPEECH_GENERATING: ``total_chunks`` is ``None`` for streaming
                     providers where the total is unknown upfront (OpenAI); ``1`` for
                     single-pass providers (HuggingFace). ``final=True`` marks the
                     terminal chunk; ``result`` carries the encoded output on the final
                     chunk only.
                   - ``str`` for DONE (usually empty), **or** ``dict {"result": <object>}`` for the
                     terminal chunk of a streamed structured-output call (``schema=`` + ``stream=True``),
                     where ``result`` is the validated dataclass / Pydantic instance.
        agent:     name of the agent that produced this chunk, or ``None`` for a plain
                   ``client.chat()`` / ``client.generate()`` call. Set automatically by
                   ``Agent`` and workflow runners.
        iteration: zero-based iteration index inside the agent loop, or ``0`` for plain chat.

    Use ``chunk.is_text()`` / ``chunk.is_tool_call()`` / ``chunk.is_continuing()`` /
    ``chunk.is_image_progress()`` / ``chunk.is_audio_progress()`` / ``chunk.is_speech_progress()`` /
    ``chunk.is_done()`` to dispatch on phase without repeating the equality check in user code.
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

    def is_continuing(self) -> bool:
        """True if this chunk announces a round the loop injected (CONTINUING).

        Named for the phase, not for one of its two kinds: it is also True for the forced wrap-up,
        which tells the model to stop rather than to keep working. Read ``content["kind"]`` to tell
        the two apart.
        """
        return self.phase == StreamingContentType.CONTINUING

    def is_image_progress(self) -> bool:
        """True if this chunk carries image-generation progress (IMAGE_GENERATING)."""
        return self.phase == StreamingContentType.IMAGE_GENERATING

    def is_audio_progress(self) -> bool:
        """True if this chunk carries audio-generation progress (AUDIO_GENERATING)."""
        return self.phase == StreamingContentType.AUDIO_GENERATING

    def is_speech_progress(self) -> bool:
        """True if this chunk carries speech-generation progress (SPEECH_GENERATING)."""
        return self.phase == StreamingContentType.SPEECH_GENERATING

    def is_done(self) -> bool:
        """True if this chunk is the terminal DONE marker.

        For a streamed structured-output call (``schema=`` + ``stream=True``) the DONE
        chunk's ``content`` is ``{"result": <validated object>}``.
        """
        return self.phase == StreamingContentType.DONE


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self.func(cls)
