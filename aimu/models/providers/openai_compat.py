"""Generic OpenAI-compatible client base plus the local-inference-server subclasses.

``OpenAICompatClient`` speaks the OpenAI REST API against any ``base_url``. The
subclasses below are thin: each sets a default ``base_url`` and a ``Model`` enum of
server-appropriate ids. They cover *local* servers that merely expose an OpenAI-compatible
endpoint (Ollama, LM Studio, vLLM, HF-Serve, llama-server, SGLang, oMLX).

The cloud-brand providers that also use this protocol live in their own provider
subpackages, since they have multiple modalities and first-class identities:
``aimu.models.providers.openai`` (GPT/o-series + TTS) and
``aimu.models.providers.gemini`` (text + image).
"""

import json
import logging
import re
from typing import Any, Iterator, Optional, Union

import openai

from ..base import (
    BaseModelClient,
    Model,
    ModelConnectionError,
    StreamChunk,
    StreamingContentType,
    classproperty,
)
from .._catalog import Wire
from .._internal.audio_input import _build_audio_content_blocks
from .._internal.generate_kwargs import Unsupported
from .._internal.image_input import _build_user_content_blocks
from .._internal.message_meta import strip_inert_keys
from .._internal.sdk_config import sdk_client_kwargs
from .._internal.thinking import QWEN_REASONING_EFFORT, pop_thinking
from .._internal.usage import usage_from_openai
from ._thinking import _ThinkingParser, _split_thinking

logger = logging.getLogger(__name__)


def _guarded_create(sdk_client, **kwargs):
    """Call the chat-completions endpoint, translating a server-unreachable failure into
    ``ModelConnectionError``. The SDK's ``APIConnectionError`` message is generic ("Connection
    error."); the specific cause (e.g. "Connection refused") is preserved on ``__cause__``."""
    try:
        return sdk_client.chat.completions.create(**kwargs)
    except openai.APIConnectionError as exc:
        raise ModelConnectionError(str(exc)) from exc


def _guard_stream(stream) -> Iterator:
    """Yield from a streaming response, translating a mid-stream connection drop into
    ``ModelConnectionError`` (the connection can fail while consuming, not only on create)."""
    try:
        yield from stream
    except openai.APIConnectionError as exc:
        raise ModelConnectionError(str(exc)) from exc


# The local inference servers that read extra sampling fields off an OpenAI request (vLLM, SGLang,
# llama-server, LM Studio, oMLX, HF Transformers Serve) take the three non-OpenAI knobs that way, so
# they are declared supported here and routed into ``extra_body`` by the hook below, following the
# sampling extensions vLLM and SGLang document. Two members do not inherit that unchanged: llama-server
# spells the repetition knob ``repeat_penalty`` (see LLAMASERVER_OPENAI_GENERATE_KWARGS), and Ollama's
# OpenAI shim reads none of the three at all (see OLLAMA_OPENAI_GENERATE_KWARGS).
#
# How far to trust this table, per member: the two exceptions above were each checked against their own
# backend's reference, which is what earned them a table. LM Studio, oMLX, and HF Transformers Serve
# inherit the family verdict *unconfirmed* -- no reference for them was consulted. The cell most in
# doubt is ``repetition_penalty`` for an llama.cpp-engine server: LM Studio runs llama.cpp, and
# llama-server wants ``repeat_penalty``, so LM Studio plausibly does too. It is left as the family's
# spelling because acting on that inference without the reference is what put the two exceptions here
# in the first place; confirm it before changing it.
#
# The cloud endpoints reject the three and declare so themselves: see CLOUD_OPENAI_GENERATE_KWARGS. The
# OpenAI API has no context-length parameter at all, because a local server's window is sized when the
# server starts.
OPENAI_COMPAT_GENERATE_KWARGS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "presence_penalty": "presence_penalty",
    "repetition_penalty": "repetition_penalty",
    "max_tokens": "max_tokens",
    "context_length": Unsupported(
        "Set it when starting the server (llama-server --ctx-size, vLLM --max-model-len, "
        "LM Studio's context-length setting)."
    ),
}

# The portable keys the OpenAI schema has no top-level place for, which a local server takes as an
# extra request field instead. Named portably rather than as wire names: the hook routes whatever
# spelling each client's own GENERATE_KWARG_SUPPORT renamed a key into, and routes nothing for a key
# the client declared unsupported.
_EXTRA_BODY_PORTABLE_KWARGS = ("top_k", "min_p", "repetition_penalty")

# The cloud endpoints (OpenAI, and Google's OpenAI-compatible surface) accept only the OpenAI set, and
# their context window is the vendor's to decide. Google's compatibility reference documents no
# top-level top_k and no place for it under extra_body, so it is declared unsupported here: a parameter
# the endpoint rejects fails the whole request, where a dropped one only stops applying. That verdict is
# the safe reading of an open question, not a verified impossibility: the same reference does document
# extra_body={"generation_config": ...}, and the native Gemini API carries topK inside generationConfig,
# so that route may well work. It could be neither confirmed nor disproved without a live key, so
# revisit Gemini's top_k with one in hand rather than treating the question as settled.
CLOUD_OPENAI_GENERATE_KWARGS = {
    **OPENAI_COMPAT_GENERATE_KWARGS,
    "top_k": Unsupported("This endpoint accepts only the OpenAI parameter set, which has no top_k."),
    "min_p": Unsupported("This endpoint accepts only the OpenAI parameter set, which has no min_p."),
    "repetition_penalty": Unsupported(
        "This endpoint accepts only the OpenAI parameter set. Use presence_penalty or frequency_penalty instead."
    ),
    "context_length": Unsupported("This model's context window is fixed by the provider."),
}


class OpenAICompatClient(BaseModelClient):
    MODELS = Model

    GENERATE_KWARG_SUPPORT = OPENAI_COMPAT_GENERATE_KWARGS

    DEFAULT_GENERATE_KWARGS = {
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    # ``chat_template_kwargs`` is a Qwen / vLLM template convention rather than part of the
    # OpenAI API, so cloud subclasses whose servers would reject or ignore it opt out.
    _SUPPORTS_CHAT_TEMPLATE_KWARGS = True

    def __init__(
        self,
        model: Model,
        base_url: str,
        api_key: str = "not-needed",
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        super().__init__(model, model_kwargs, system_message)
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key, **sdk_client_kwargs(timeout, max_retries))

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_thinking]

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_tools]

    @classproperty
    def VISION_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_vision]

    @classproperty
    def AUDIO_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_audio]

    @classproperty
    def STRUCTURED_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_structured_output]

    def _rewrite_generate_kwargs(self, kwargs: dict) -> dict:
        return self._apply_resolved_thinking(self._route_extra_body_kwargs(kwargs))

    def _route_extra_body_kwargs(self, kwargs: dict) -> dict:
        """Move the non-OpenAI sampling knobs into ``extra_body``, where a local server reads them.

        Reshaping a request for one API is exactly what this hook is for, which is why it is not in the
        declared table: the table says whether a key survives and under what name, and this says where
        it goes. Taking the names back off the table is what keeps the two halves agreeing -- a
        subclass that renames one (llama-server's ``repeat_penalty``) or declares it unsupported
        (Ollama's OpenAI shim, and the cloud endpoints) needs no second edit here. It matters that they
        agree: the OpenAI SDK's ``create()`` takes no arbitrary keywords, so a renamed key left at the
        top level would raise ``TypeError`` rather than reach the server.
        """
        support = self.GENERATE_KWARG_SUPPORT
        routed = [support[key] for key in _EXTRA_BODY_PORTABLE_KWARGS if isinstance(support.get(key), str)]
        present = {key: kwargs.pop(key) for key in routed if key in kwargs}
        if not present:
            return kwargs
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body.update(present)
        kwargs["extra_body"] = extra_body
        return kwargs

    def _apply_resolved_thinking(self, kwargs: dict) -> dict:
        """Translate a resolved thinking request into OpenAI-compatible request fields."""
        resolved = pop_thinking(kwargs)
        if resolved is None:
            return kwargs

        if resolved.level is not None and self.model.thinking_levels:
            kwargs["reasoning_effort"] = QWEN_REASONING_EFFORT[resolved.level]

        if self._SUPPORTS_CHAT_TEMPLATE_KWARGS:
            extra_body = dict(kwargs.get("extra_body") or {})
            template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            template_kwargs["enable_thinking"] = resolved.enabled
            extra_body["chat_template_kwargs"] = template_kwargs
            kwargs["extra_body"] = extra_body
        elif not resolved.enabled:
            self._warn_once(
                f"{self.model.value}: this provider has no way to disable reasoning; thinking=False ignored."
            )

        return kwargs

    @staticmethod
    def _with_response_format(generate_kwargs: dict, response_format: Optional[dict]) -> dict:
        """Wrap a JSON Schema dict in OpenAI's ``response_format`` envelope.

        Uses ``strict: False`` so arbitrary user schemas (optional fields, defaults) don't
        trip OpenAI strict-mode's subset rules; the schema still constrains generation and
        the base coerces/validates the result.
        """
        if not response_format:
            return generate_kwargs
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(response_format.get("title", "Response")))[:64] or "Response"
        envelope = {"type": "json_schema", "json_schema": {"name": name, "schema": response_format, "strict": False}}
        return {**generate_kwargs, "response_format": envelope}

    def _iter_stream(self, stream) -> Iterator[StreamChunk]:
        """Iterate a completion stream, yielding StreamChunks and updating self.last_thinking.

        Usage is captured from the terminal chunk emitted when the request sets
        ``stream_options={"include_usage": True}``: it carries ``usage`` and an empty
        ``choices`` list, so ``self.last_usage`` is populated once the stream is fully
        consumed (``None`` if the server reports no usage).
        """
        self.last_thinking = ""
        self.last_usage = None
        parser = _ThinkingParser() if self.is_thinking_model else None

        for chunk in _guard_stream(stream):
            if getattr(chunk, "usage", None):
                self.last_usage = usage_from_openai(chunk)
            if not chunk.choices:  # terminal usage chunk (empty choices) or keep-alive
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                self.last_thinking += reasoning
                yield StreamChunk(StreamingContentType.THINKING, reasoning)
            if delta.content is None:
                continue
            logger.debug("LLM raw chunk: %s", chunk)
            if parser:
                for phase, text in parser.feed(delta.content):
                    if phase == StreamingContentType.THINKING:
                        self.last_thinking += text
                        yield StreamChunk(StreamingContentType.THINKING, text)
                    else:
                        yield StreamChunk(StreamingContentType.GENERATING, text)
            else:
                yield StreamChunk(StreamingContentType.GENERATING, delta.content)

    def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs = self._resolve_generate_kwargs(generate_kwargs)
        generate_kwargs = self._with_response_format(generate_kwargs, response_format)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs, images=images, audio=audio)

        if images:
            content_in = _build_user_content_blocks(prompt, images)
        elif audio:
            content_in = _build_audio_content_blocks(prompt, audio)
        else:
            content_in = prompt
        response = _guarded_create(
            self._client,
            model=self.model.value,
            messages=[{"role": "user", "content": content_in}],
            **generate_kwargs,
        )
        logger.debug("LLM raw response: %s", response)
        msg = response.choices[0].message
        self.last_usage = usage_from_openai(response)
        content = msg.content or ""

        self.last_thinking = ""
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            self.last_thinking = reasoning
        elif self.is_thinking_model:
            self.last_thinking, content = _split_thinking(content)

        return content

    def _generate_streamed(
        self,
        prompt: str,
        generate_kwargs: dict[str, Any],
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Iterator[StreamChunk]:
        if images:
            content_in = _build_user_content_blocks(prompt, images)
        elif audio:
            content_in = _build_audio_content_blocks(prompt, audio)
        else:
            content_in = prompt
        stream = _guarded_create(
            self._client,
            model=self.model.value,
            messages=[{"role": "user", "content": content_in}],
            stream=True,
            stream_options={"include_usage": True},
            **generate_kwargs,
        )
        yield from self._iter_stream(stream)

    def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools, images=images, audio=audio)
        generate_kwargs = self._with_response_format(generate_kwargs, response_format)

        if stream:
            return self._chat_streamed(generate_kwargs, tools)

        response = _guarded_create(
            self._client,
            model=self.model.value,
            messages=strip_inert_keys(self.messages),
            tools=tools if tools else openai.NOT_GIVEN,
            **generate_kwargs,
        )
        logger.debug("LLM raw response: %s", response)
        msg = response.choices[0].message
        self.last_usage = usage_from_openai(response)
        self.last_thinking = ""
        # Servers that strip <think> tags server-side (llama-server, vLLM/SGLang reasoning parsers)
        # return reasoning in a separate reasoning_content field; prefer it when present, else fall
        # back to parsing inline tags out of content.
        reasoning = getattr(msg, "reasoning_content", None)

        # Single turn: if the model called tools, execute them and return. The model's response
        # to the tool results comes on the next chat() call (the loop lives in Agent).
        if msg.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": json.loads(tc.function.arguments)} for tc in msg.tool_calls
            ]
            text = msg.content or ""
            if reasoning:
                self.last_thinking = reasoning
            elif self.is_thinking_model:
                self.last_thinking, text = _split_thinking(text)
            msgs_before = len(self.messages)
            self._record_tool_calls(tool_calls, content=text)
            if self.last_thinking:
                self.messages[msgs_before]["thinking"] = self.last_thinking
            return text

        content = msg.content or ""
        if reasoning:
            self.last_thinking = reasoning
        elif self.is_thinking_model:
            self.last_thinking, content = _split_thinking(content)

        self._append_message({"role": "assistant", "content": content})
        if self.last_thinking:
            self.messages[-1]["thinking"] = self.last_thinking
        return content

    def _chat_streamed(self, generate_kwargs: dict[str, Any], tools: list) -> Iterator[StreamChunk]:
        stream = _guard_stream(
            _guarded_create(
                self._client,
                model=self.model.value,
                messages=strip_inert_keys(self.messages),
                stream=True,
                stream_options={"include_usage": True},
                tools=tools if tools else openai.NOT_GIVEN,
                **generate_kwargs,
            )
        )

        # Yield content/thinking chunks as they arrive (incremental streaming) while accumulating
        # any tool-call deltas separately. In the OpenAI streaming protocol content and tool_call
        # deltas don't require buffering: prose the model emits alongside a tool call is streamable,
        # and the accumulated tool_calls are recorded once the stream ends. (Draining the whole
        # stream before yielding made llama-server et al. appear non-streaming.)
        tool_calls_acc: dict[int, dict] = {}
        full_content = ""
        parser = _ThinkingParser() if self.is_thinking_model else None
        self.last_thinking = ""
        self.last_usage = None

        for chunk in stream:
            if getattr(chunk, "usage", None):
                self.last_usage = usage_from_openai(chunk)
            if not chunk.choices:  # terminal usage chunk (empty choices) or keep-alive
                continue
            delta = chunk.choices[0].delta
            logger.debug("LLM raw chunk: %s", chunk)
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                self.last_thinking += reasoning
                yield StreamChunk(StreamingContentType.THINKING, reasoning)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    acc = tool_calls_acc.setdefault(tc_delta.index, {"name": "", "arguments": ""})
                    if tc_delta.function and tc_delta.function.name:
                        acc["name"] += tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        acc["arguments"] += tc_delta.function.arguments
            elif delta.content is not None:
                if parser:
                    for phase, text in parser.feed(delta.content):
                        if phase == StreamingContentType.THINKING:
                            self.last_thinking += text
                        else:
                            full_content += text
                        yield StreamChunk(phase, text)
                else:
                    full_content += delta.content
                    yield StreamChunk(StreamingContentType.GENERATING, delta.content)

        if not tool_calls_acc:
            self._append_message({"role": "assistant", "content": full_content})
            if self.last_thinking:
                self.messages[-1]["thinking"] = self.last_thinking
            return

        # Tool call path (single turn): prose/thinking already streamed above; now dispatch the
        # tools and return. The model's response to the tool results comes on the next chat() call
        # (the loop lives in Agent). No second stream here.
        tool_calls = [{"name": tc["name"], "arguments": json.loads(tc["arguments"])} for tc in tool_calls_acc.values()]
        tool_turn_thinking = self.last_thinking
        msgs_before = len(self.messages)
        self._record_tool_calls(tool_calls, content=full_content)
        if tool_turn_thinking:
            self.messages[msgs_before]["thinking"] = tool_turn_thinking


# --------------------------------------------------------------------------------------
# Local OpenAI-compatible inference servers. Each subclass just supplies a default
# base_url and a Model enum of server-appropriate ids; all behaviour lives in the base.
# --------------------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaOpenAIModel(Model):
    # Model values are Ollama model tags (as used by `ollama pull`); this shim fronts the same
    # install and the same registry tags as the native OllamaModel catalog above, so the two
    # are kept in parity member-for-member (tests/test_model_catalog_facts.py enforces it).
    #
    # structured_output=True on every member below, matching the native catalog: Ollama's
    # OpenAI-compatible endpoint also supports response_format for JSON-schema-constrained
    # output, per Ollama's own docs ("Structured outputs work through the OpenAI-compatible
    # API via response_format" -- docs/capabilities/structured-outputs.mdx), not just the
    # native API's format= parameter.
    #
    # Alibaba
    # Qwen 3.5/3.6/3.8 are a unified vision-language family (vision is built into the base
    # weights, not a separate -VL variant); the Ollama tag serves image input over the
    # OpenAI-compat endpoint too. (Qwen3 32B/8B/4B below are the older text-only generation and
    # stay vision=False.)
    #
    # Note the name spacing: QWEN_3_8_27B is Qwen *3.8* at 27B, while QWEN_3_8B further down is
    # Qwen *3* at 8B. Both follow the catalog's "version parts, then size" scheme.
    # thinking_levels=True: Qwen 3.8's chat template accepts a reasoning_effort kwarg (see
    # aimu/models/providers/hf/text.py for the verification).
    QWEN_3_8_27B = Wire("qwen3.8:27b", structured_output=True)
    # On Apple Silicon, Ollama 0.19+ runs qwen3.6 tags on its MLX backend automatically; that's
    # transparent to this client (the tag is unchanged), so there is nothing MLX-specific to
    # declare here.
    QWEN_3_6_35B = Wire("qwen3.6:35b", structured_output=True)
    QWEN_3_6_27B = Wire("qwen3.6:27b", structured_output=True)
    QWEN_3_5_9B = Wire("qwen3.5:9b", structured_output=True)
    QWEN_3_32B = Wire("qwen3:32b", structured_output=True)
    QWEN_3_8B = Wire("qwen3:8b", structured_output=True)
    QWEN_3_4B = Wire("qwen3:4b", structured_output=True)
    # Google: these weights support audio; add audio=True once Ollama API exposes audio input
    # (the native OllamaModel catalog omits it for the same reason).
    GEMMA_4_E4B = Wire("gemma4:e4b", structured_output=True)
    GEMMA_4_12B = Wire("gemma4:12b", structured_output=True)
    GEMMA_4_26B = Wire("gemma4:26b", structured_output=True)
    GEMMA_4_31B = Wire("gemma4:31b", structured_output=True)
    GEMMA_3_12B = Wire("gemma3:12b", structured_output=True)
    # NVIDIA
    NEMOTRON_CASCADE_2_30B = Wire("nemotron-cascade-2:30b", structured_output=True)
    NEMOTRON_3_NANO_30B = Wire("nemotron-3-nano:30b", structured_output=True)
    # Zhipu AI: doesn't use tools when expected
    GLM_4_7_FLASH_31B_Q4 = Wire("glm-4.7-flash:q4_K_M", structured_output=True)
    # OpenAI
    GPT_OSS_20B = Wire("gpt-oss:20b", structured_output=True)
    # Mistral
    MAGISTRAL_SMALL_24B = Wire("magistral:24b", structured_output=True)
    MINISTRAL_3_14B = Wire("ministral-3:14b", structured_output=True)
    MISTRAL_7B = Wire("mistral:7b", structured_output=True)
    # Microsoft
    PHI_4_MINI_3_8B = Wire("phi4-mini:3.8b", structured_output=True)
    PHI_4_14B = Wire("phi4:14b", structured_output=True)
    # DeepSeek
    DEEPSEEK_R1_8B = Wire("deepseek-r1:8b", structured_output=True)
    # HuggingFace: tool call responses don't always look correct
    SMOLLM2_1_7B = Wire("smollm2:1.7b", structured_output=True)
    # Meta: Muse Glimmer is a vision-language model whose perception encoder ships in the same
    # weights. It emits channel-scoped reasoning and ATEM-style XML tool calls, which Ollama
    # parses server-side, so reasoning arrives here as reasoning_content and tool calls in the
    # standard OpenAI shape.
    MUSE_GLIMMER_30B = Wire("muse-glimmer:30b", structured_output=True)
    # Llama 3.1/3.2 tool calling verified reliable on current Ollama builds (same backend as
    # the native OllamaModel catalog).
    LLAMA_3_2_3B = Wire("llama3.2:3b", structured_output=True)
    LLAMA_3_1_8B = Wire("llama3.1:8b", structured_output=True)


# Ollama's OpenAI shim maps a fixed OpenAI field set onto its native call and reads none of the three
# extra sampling knobs, so declaring them supported would route them into extra_body to be discarded
# without a word -- the failure the declared table exists to prevent. Its native API does accept two of
# the three, so the remedy is to use the 'ollama' provider.
_OLLAMA_SHIM_IGNORES = "Ollama's OpenAI-compatible endpoint reads only the OpenAI parameter set. "

OLLAMA_OPENAI_GENERATE_KWARGS = {
    **OPENAI_COMPAT_GENERATE_KWARGS,
    "top_k": Unsupported(_OLLAMA_SHIM_IGNORES + "Use the 'ollama' provider, whose native API accepts top_k."),
    # The native provider cannot carry min_p either (its SDK's Options model has no such field), so
    # the Modelfile is the only place this one can be set.
    "min_p": Unsupported(_OLLAMA_SHIM_IGNORES + "Set min_p in the model's Modelfile instead (PARAMETER min_p)."),
    "repetition_penalty": Unsupported(
        _OLLAMA_SHIM_IGNORES + "Use the 'ollama' provider, whose native API accepts it as repeat_penalty."
    ),
    "context_length": Unsupported(
        "Set OLLAMA_CONTEXT_LENGTH on the server, or use the 'ollama' provider, whose native API "
        "accepts context_length per request."
    ),
}


class OllamaOpenAIClient(OpenAICompatClient):
    MODELS = OllamaOpenAIModel

    GENERATE_KWARG_SUPPORT = OLLAMA_OPENAI_GENERATE_KWARGS

    def __init__(self, model: OllamaOpenAIModel, base_url: str = OLLAMA_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)


# Shared by every vision-capable member of both GGUF-serving catalogs below (LM Studio and
# llama-server). Neither server loads an mmproj projector by default -- LM Studio and
# llama-server each need their own explicit flag for it -- so a member whose weights are
# intrinsically vision-capable (per MODEL_FACTS) still overrides vision=False here: the
# catalog describes the default path, and advertising vision would let a caller pass images
# that fail at request time. See tests/test_model_catalog_facts.py::test_gguf_catalogs_do_not_advertise_vision.
_NO_MMPROJ = (
    "the default GGUF path loads no mmproj projector, so advertising vision would let a caller "
    "pass images that fail at request time"
)

# Shared by GEMMA_3_12B's bare (GGUF) entry on both GGUF-serving catalogs below (LM Studio and
# llama-server): combines the tools=True rationale (OpenAI-compat servers parse tool calls
# server-side) with the vision=False rationale above (_NO_MMPROJ), since a GGUF Gemma 3 12B
# member overrides both. Previously duplicated verbatim on each catalog; extracted here so the
# two copies cannot drift.
_GEMMA_3_12B_GGUF_WHY = (
    "tools=True: OpenAI-compat servers parse tool calls server-side; the in-process HF and "
    "native Ollama paths have no tool-parse format assigned for Gemma 3. vision=False: " + _NO_MMPROJ
)

# Shared by every MLX-served GEMMA_3_12B quantization (oMLX and LM Studio's MLX entries further
# down): same tools rationale as above, but MLX serves vision directly (no mmproj limitation),
# so only tools is overridden -- vision stays at MODEL_FACTS' intrinsic True.
_GEMMA_3_12B_TOOLS_WHY = (
    "OpenAI-compat servers parse tool calls server-side; the in-process HF and native Ollama "
    "paths have no tool-parse format assigned for Gemma 3"
)


LMSTUDIO_BASE_URL = "http://localhost:1234/v1"


class LMStudioOpenAIModel(Model):
    # Model values are the model "key" as shown in LM Studio's loaded model list.
    # tools=True consistent with the verified Ollama Llama tool calling and the llama.cpp-based
    # llama-server build (LM Studio runs the same GGUF/llama.cpp engine).
    LLAMA_3_1_8B = Wire("llama-3.1-8b-instruct")
    LLAMA_3_2_3B = Wire("llama-3.2-3b-instruct")
    MISTRAL_7B = Wire("mistral-7b-instruct-v0.3")
    PHI_4_MINI_3_8B = Wire("phi-4-mini-instruct")
    PHI_4_14B = Wire("phi-4")
    QWEN_3_4B = Wire("qwen3-4b")
    QWEN_3_8B = Wire("qwen3-8b")
    QWEN_3_32B = Wire("qwen3-32b")
    # Qwen 3.5/3.6/3.8 are a unified vision-language family in the weights (image input is built
    # into the base model, not a separate -VL variant), but LM Studio's GGUF path loads no mmproj
    # projector by default, so every bare (GGUF) member below overrides vision=False. See
    # _NO_MMPROJ above and the MLX quant siblings further down, which stay vision=True.
    QWEN_3_5_9B = Wire("qwen3.5-9b", why=_NO_MMPROJ, vision=False)
    QWEN_3_6_27B = Wire("qwen3.6-27b", why=_NO_MMPROJ, vision=False)
    # Bare GGUF build of Qwen 3.6 35B-A3B, distinct from the MLX-quantized _4BIT/_8BIT members
    # below: a quant-free LM Studio key resolves to the GGUF build, those quant-suffixed keys to
    # an mlx-community download.
    QWEN_3_6_35B = Wire("qwen3.6-35b-a3b", why=_NO_MMPROJ, vision=False)
    # LM Studio ships an MLX engine alongside llama.cpp and picks it automatically for MLX weights
    # on Apple Silicon. The loaded-model key derives from the downloaded repo, so an mlx-community
    # download keeps its quant suffix -- unlike the GGUF entries above, whose keys are quant-free.
    # Member names match OMLXOpenAIModel's so the cross-provider consistency guard covers the pair.
    # Qwen 3.6 35B-A3B is a unified vision-language MoE (matching OllamaModel.QWEN_3_6_35B); MLX
    # inference is a separate path from the GGUF one this catalog's vision override targets, so
    # these quant-suffixed members are left vision=True (Task 11's business, not this one's). No
    # bf16 -- a 35B unquantized is impractical here.
    QWEN_3_6_35B_4BIT = Wire("qwen3.6-35b-a3b-4bit")
    QWEN_3_6_35B_8BIT = Wire("qwen3.6-35b-a3b-8bit")
    # Bare GGUF build of Qwen 3.8 27B, distinct from the MLX-quantized _4BIT/_8BIT members below.
    # thinking_levels=True on every Qwen 3.8 member: verified against the model's own
    # chat_template.jinja, which validates reasoning_effort against {xhigh, medium, low}. See
    # providers/hf/text.py for the full note.
    QWEN_3_8_27B = Wire("qwen3.8-27b", why=_NO_MMPROJ, vision=False)
    # Qwen 3.8 27B is dense rather than MoE, but the MLX story is identical: quant-suffixed keys
    # from an mlx-community download, no bare member here (the bare GGUF build is catalogued
    # above, separately from these MLX quants). At 27B dense, bf16 is impractical here too, so
    # only the two practical quants are listed.
    QWEN_3_8_27B_4BIT = Wire("qwen3.8-27b-4bit")
    QWEN_3_8_27B_8BIT = Wire("qwen3.8-27b-8bit")
    # No MUSE_GLIMMER_30B here: an mlx-community MLX build does exist (see OMLXOpenAIModel's
    # MUSE_GLIMMER_30B members below), but LM Studio's *GGUF* engine (llama.cpp) is the path this
    # member would represent, and whether that engine parses the model's channel-scoped reasoning
    # and ATEM-style XML tool calls is still undocumented. Adding it here would mean guessing
    # tools/thinking on the GGUF path. See OMLXOpenAIModel for the MLX path that does parse them.
    # (The same reasoning excludes it from LlamaServerOpenAIModel and LlamaCppModel: both run the
    # same llama.cpp-derived engine as LM Studio's GGUF path.)
    DEEPSEEK_R1_7B = Wire("deepseek-r1-distill-qwen-7b")
    DEEPSEEK_R1_8B = Wire("deepseek-r1-8b")
    # Zhipu AI: doesn't use tools reliably (matches OllamaModel's own note); this is an intrinsic
    # weights limitation, not a serving-path one, so no tools=True override here either.
    GLM_4_7_FLASH_31B_Q4 = Wire("glm-4.7-flash-q4_k_m")
    GPT_OSS_20B = Wire("gpt-oss-20b")
    MAGISTRAL_SMALL_24B = Wire("magistral-small-24b")
    MINISTRAL_3_14B = Wire("ministral-3-14b")
    NEMOTRON_CASCADE_2_30B = Wire("nemotron-cascade-2-30b-a3b")
    NEMOTRON_3_NANO_30B = Wire("nemotron-3-nano-30b-a3b")
    SMOLLM2_1_7B = Wire("smollm2-1.7b-instruct")
    # OpenAI-compat servers (LM Studio included) parse tool calls server-side; the in-process HF
    # and native Ollama paths have no tool-parse format assigned for Gemma 3 (same override as
    # HFOpenAIModel/LlamaServerOpenAIModel/SGLangOpenAIModel/VLLMOpenAIModel). vision=False for the
    # same mmproj reason as every other member above.
    GEMMA_3_12B = Wire("gemma-3-12b-it", why=_GEMMA_3_12B_GGUF_WHY, tools=True, vision=False)
    # Gemma 4 E4B/12B support audio natively, but LM Studio has no audio-input path (image-only);
    # leave audio=False. All four also override vision=False: see _NO_MMPROJ above.
    GEMMA_4_E4B = Wire("gemma-4-e4b-it", why=_NO_MMPROJ, vision=False)
    GEMMA_4_12B = Wire("gemma-4-12b-it", why=_NO_MMPROJ, vision=False)
    GEMMA_4_26B = Wire("gemma-4-26b-a4b-it", why=_NO_MMPROJ, vision=False)
    GEMMA_4_31B = Wire("gemma-4-31b-it", why=_NO_MMPROJ, vision=False)

    # Task 11: every other model with a confirmed mlx-community build, mirroring
    # OMLXOpenAIModel's block of the same shape (see its comments for the per-model naming
    # notes and the no-bare-quant-agnostic-form rationale). No bare (quant-free) member is added
    # here either, per this catalog's own rule above: a quant-free LM Studio key would be the
    # GGUF build, not an MLX path. Ids are the lowercase of the matching oMLX directory name --
    # LM Studio derives its loaded-model key from the downloaded repo name, case-folded.
    #
    # Alibaba (Qwen)
    QWEN_3_32B_4BIT = Wire("qwen3-32b-4bit")
    QWEN_3_32B_8BIT = Wire("qwen3-32b-8bit")
    QWEN_3_32B_BF16 = Wire("qwen3-32b-bf16")
    QWEN_3_4B_4BIT = Wire("qwen3-4b-4bit")
    QWEN_3_4B_8BIT = Wire("qwen3-4b-8bit")
    QWEN_3_4B_BF16 = Wire("qwen3-4b-bf16")
    QWEN_3_8B_4BIT = Wire("qwen3-8b-4bit")
    QWEN_3_8B_8BIT = Wire("qwen3-8b-8bit")
    QWEN_3_8B_BF16 = Wire("qwen3-8b-bf16")
    QWEN_3_5_9B_4BIT = Wire("qwen3.5-9b-4bit")
    QWEN_3_5_9B_8BIT = Wire("qwen3.5-9b-8bit")
    QWEN_3_5_9B_BF16 = Wire("qwen3.5-9b-bf16")
    QWEN_3_6_27B_4BIT = Wire("qwen3.6-27b-4bit")
    QWEN_3_6_27B_8BIT = Wire("qwen3.6-27b-8bit")
    QWEN_3_6_27B_BF16 = Wire("qwen3.6-27b-bf16")
    # DeepSeek
    DEEPSEEK_R1_7B_4BIT = Wire("deepseek-r1-distill-qwen-7b-4bit")
    DEEPSEEK_R1_7B_8BIT = Wire("deepseek-r1-distill-qwen-7b-8bit")
    DEEPSEEK_R1_7B_BF16 = Wire("deepseek-r1-distill-qwen-7b-bf16")
    DEEPSEEK_R1_8B_4BIT = Wire("deepseek-r1-distill-llama-8b-4bit")
    DEEPSEEK_R1_8B_8BIT = Wire("deepseek-r1-distill-llama-8b-8bit")
    DEEPSEEK_R1_8B_BF16 = Wire("deepseek-r1-distill-llama-8b-bf16")
    # Google. Vision stays True (no override) for these MLX quants: unlike the bare GGUF members
    # above, LM Studio's MLX engine serves vision with no mmproj projector to load.
    GEMMA_3_12B_4BIT = Wire("gemma-3-12b-it-4bit", why=_GEMMA_3_12B_TOOLS_WHY, tools=True)
    GEMMA_3_12B_8BIT = Wire("gemma-3-12b-it-8bit", why=_GEMMA_3_12B_TOOLS_WHY, tools=True)
    GEMMA_3_12B_BF16 = Wire("gemma-3-12b-it-bf16", why=_GEMMA_3_12B_TOOLS_WHY, tools=True)
    GEMMA_4_E4B_4BIT = Wire("gemma-4-e4b-it-4bit")
    GEMMA_4_E4B_8BIT = Wire("gemma-4-e4b-it-8bit")
    GEMMA_4_E4B_BF16 = Wire("gemma-4-e4b-it-bf16")
    GEMMA_4_12B_4BIT = Wire("gemma-4-12b-it-4bit")
    GEMMA_4_12B_8BIT = Wire("gemma-4-12b-it-8bit")
    GEMMA_4_12B_BF16 = Wire("gemma-4-12b-it-bf16")
    GEMMA_4_26B_4BIT = Wire("gemma-4-26b-a4b-it-4bit")
    GEMMA_4_26B_8BIT = Wire("gemma-4-26b-a4b-it-8bit")
    GEMMA_4_26B_BF16 = Wire("gemma-4-26b-a4b-it-bf16")
    GEMMA_4_31B_4BIT = Wire("gemma-4-31b-it-4bit")
    GEMMA_4_31B_8BIT = Wire("gemma-4-31b-it-8bit")
    GEMMA_4_31B_BF16 = Wire("gemma-4-31b-it-bf16")
    # Zhipu
    GLM_4_7_FLASH_31B_Q4_4BIT = Wire("glm-4.7-flash-4bit")
    GLM_4_7_FLASH_31B_Q4_8BIT = Wire("glm-4.7-flash-8bit")
    GLM_4_7_FLASH_31B_Q4_BF16 = Wire("glm-4.7-flash-bf16")
    # OpenAI (open-weight). See OMLXOpenAIModel's comment: gpt-oss ships natively quantized to
    # mxfp4, so mlx-community's build is a Q4/Q8 requantization of that format, not a plain
    # 4bit/8bit build; no bf16 exists.
    GPT_OSS_20B_MXFP4_Q4 = Wire("gpt-oss-20b-mxfp4-q4")
    GPT_OSS_20B_MXFP4_Q8 = Wire("gpt-oss-20b-mxfp4-q8")
    # Meta
    LLAMA_3_2_3B_4BIT = Wire("llama-3.2-3b-instruct-4bit")
    LLAMA_3_2_3B_8BIT = Wire("llama-3.2-3b-instruct-8bit")
    LLAMA_3_2_3B_BF16 = Wire("llama-3.2-3b-instruct-bf16")
    # See OMLXOpenAIModel's comment: the confirmed matching trio uses mlx-community's
    # "Meta-Llama-3.1-..." naming, not "Llama-3.1-...".
    LLAMA_3_1_8B_4BIT = Wire("meta-llama-3.1-8b-instruct-4bit")
    LLAMA_3_1_8B_8BIT = Wire("meta-llama-3.1-8b-instruct-8bit")
    LLAMA_3_1_8B_BF16 = Wire("meta-llama-3.1-8b-instruct-bf16")
    # Mistral. Only bf16 confirmed for Magistral (see OMLXOpenAIModel's comment); only 4bit/8bit
    # for Mistral 7B (no bf16 build).
    MAGISTRAL_SMALL_24B_BF16 = Wire("magistral-small-2506-bf16")
    MINISTRAL_3_14B_4BIT = Wire("ministral-3-14b-instruct-2512-4bit")
    MINISTRAL_3_14B_8BIT = Wire("ministral-3-14b-instruct-2512-8bit")
    MINISTRAL_3_14B_BF16 = Wire("ministral-3-14b-instruct-2512-bf16")
    MISTRAL_7B_4BIT = Wire("mistral-7b-instruct-v0.3-4bit")
    MISTRAL_7B_8BIT = Wire("mistral-7b-instruct-v0.3-8bit")
    # NVIDIA
    NEMOTRON_CASCADE_2_30B_4BIT = Wire("nemotron-cascade-2-30b-a3b-4bit")
    NEMOTRON_CASCADE_2_30B_8BIT = Wire("nemotron-cascade-2-30b-a3b-8bit")
    NEMOTRON_CASCADE_2_30B_BF16 = Wire("nemotron-cascade-2-30b-a3b-bf16")
    # See OMLXOpenAIModel's comment: the confirmed matching trio carries an extra "-mlx-" infix.
    NEMOTRON_3_NANO_30B_4BIT = Wire("nvidia-nemotron-3-nano-30b-a3b-mlx-4bit")
    NEMOTRON_3_NANO_30B_8BIT = Wire("nvidia-nemotron-3-nano-30b-a3b-mlx-8bit")
    NEMOTRON_3_NANO_30B_BF16 = Wire("nvidia-nemotron-3-nano-30b-a3b-mlx-bf16")
    # Microsoft
    PHI_4_14B_4BIT = Wire("phi-4-4bit")
    PHI_4_14B_8BIT = Wire("phi-4-8bit")
    PHI_4_14B_BF16 = Wire("phi-4-bf16")
    # See OMLXOpenAIModel's comment: the third precision is a distinctly-named "-mlx-fp16"
    # build, not a plain bf16 repo.
    PHI_4_MINI_3_8B_4BIT = Wire("phi-4-mini-instruct-4bit")
    PHI_4_MINI_3_8B_8BIT = Wire("phi-4-mini-instruct-8bit")
    PHI_4_MINI_3_8B_FP16 = Wire("phi-4-mini-instruct-mlx-fp16")
    # No SMOLLM2_1_7B here either: no quantized mlx-community build exists (see
    # OMLXOpenAIModel's comment).


class LMStudioOpenAIClient(OpenAICompatClient):
    MODELS = LMStudioOpenAIModel

    def __init__(self, model: LMStudioOpenAIModel, base_url: str = LMSTUDIO_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)


VLLM_BASE_URL = "http://localhost:8000/v1"


class VLLMOpenAIModel(Model):
    # Model values are HuggingFace repo paths (as used by `vllm serve --model`).
    #
    # tools=True below states a fact about the weights (this model was trained for function
    # calling), not a guarantee that vLLM will surface it. vLLM's OpenAI-compatible server
    # gates tool_calls behind two launch flags -- `--enable-auto-tool-choice` and
    # `--tool-call-parser <name>` -- and without them the model still runs and still emits
    # its tool call, just as unstructured text/JSON inside `content` rather than a
    # structured `tool_calls` entry. That failure mode looks exactly like "the model ignored
    # its tools" to a caller iterating `message.tool_calls`, so if a tools=True model here
    # never calls a tool, check the launch flags before doubting the model. The parser
    # `<name>` is specific to the model family and to the installed vLLM version (values are
    # added and renamed across releases), so this catalog deliberately does not enumerate
    # one per model -- look it up in vLLM's own tool-calling docs
    # (https://docs.vllm.ai/en/latest/features/tool_calling.html) for the model and vLLM
    # version actually running. MUSE_GLIMMER_30B below is the one member whose parser value
    # is pinned in a comment, because that value was independently verified for a specific
    # vLLM release; it is the exception this note explains, not the pattern to copy.
    LLAMA_3_1_8B = Wire("meta-llama/Llama-3.1-8B-Instruct")
    LLAMA_3_2_3B = Wire("meta-llama/Llama-3.2-3B-Instruct")
    # Mistral
    MISTRAL_7B = Wire("mistralai/Mistral-7B-Instruct-v0.3")
    MAGISTRAL_SMALL_24B = Wire("mistralai/Magistral-Small-2506")
    MINISTRAL_3_14B = Wire("mistralai/Ministral-3-14B-Instruct-2512")
    # Microsoft
    PHI_4_MINI_3_8B = Wire("microsoft/Phi-4-mini-instruct")
    PHI_4_14B = Wire("microsoft/phi-4")
    # Alibaba. Qwen 3.5/3.6/3.8 are a unified vision-language family: vision is built into the
    # base weights rather than shipped as a separate -VL variant, so this server serves image
    # input directly from the plain repo. (Qwen3 32B/8B/4B below are the older text-only
    # generation and stay vision=False.)
    QWEN_3_4B = Wire("Qwen/Qwen3-4B")
    QWEN_3_8B = Wire("Qwen/Qwen3-8B")
    QWEN_3_32B = Wire("Qwen/Qwen3-32B")
    QWEN_3_5_9B = Wire("Qwen/Qwen3.5-9B")
    QWEN_3_6_27B = Wire("Qwen/Qwen3.6-27B")
    QWEN_3_6_35B = Wire("Qwen/Qwen3.6-35B-A3B")
    QWEN_3_8_27B = Wire("Qwen/Qwen3.8-27B")
    # DeepSeek
    DEEPSEEK_R1_7B = Wire("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    DEEPSEEK_R1_8B = Wire("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    GEMMA_3_12B = Wire(
        "google/gemma-3-12b-it",
        why="OpenAI-compat servers parse tool calls server-side; the in-process HF and native "
        "Ollama paths have no tool-parse format assigned for Gemma 3",
        tools=True,
    )
    GEMMA_4_E4B = Wire("google/gemma-4-E4B-it")
    GEMMA_4_12B = Wire("google/gemma-4-12b-it")
    GEMMA_4_26B = Wire("google/gemma-4-26B-A4B-it")
    GEMMA_4_31B = Wire("google/gemma-4-31B-it")
    # Gemma 4 E4B/12B support audio natively; vLLM accepts input_audio blocks, but this path is
    # unverified against these weights, so leave audio=False until confirmed. (26B/31B: no native audio.)
    # NVIDIA
    NEMOTRON_CASCADE_2_30B = Wire("nvidia/Nemotron-Cascade-2-30B-A3B")
    # NVIDIA publishes no un-suffixed NVIDIA-Nemotron-3-Nano-30B-A3B repo (confirmed 401 on that
    # path); only precision-suffixed repos exist. Use the -BF16 reference checkpoint id verbatim
    # -- there is no shorter form to fall back to. Distinct from the multimodal
    # Nemotron-3-Nano-Omni-30B-A3B-Reasoning line, which is a different model.
    NEMOTRON_3_NANO_30B = Wire("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    # Zhipu AI: doesn't use tools reliably (matches OllamaModel's own note); this is an
    # intrinsic weights limitation rather than a serving-path one, so no tools=True override
    # here either. The _Q4 in the member name carries over from OllamaModel's quantized
    # glm-4.7-flash:q4_K_M tag for cross-provider name matching; this catalog's resolved repo
    # is the unquantized reference checkpoint.
    GLM_4_7_FLASH_31B_Q4 = Wire("zai-org/GLM-4.7-Flash")
    # OpenAI (open-weight)
    GPT_OSS_20B = Wire("openai/gpt-oss-20b")
    # HuggingFace: tool call responses don't always look correct (matches OllamaModel's note)
    SMOLLM2_1_7B = Wire("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    # Muse Glimmer emits channel-scoped reasoning and ATEM-style XML tool calls instead of
    # <think> tags and JSON, so serving it needs vLLM's dedicated parsers, which must be
    # enabled together (they key off the same framing):
    #   vllm serve meta-models/Muse-Glimmer-30B \
    #     --enable-auto-tool-choice --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer
    # With those set, reasoning arrives as reasoning_content and tool calls in the standard
    # OpenAI shape; without them both channels collapse into content and tools never surface.
    # SGLang and transformers-serve have no equivalent Muse Glimmer parser, so this member stays
    # vLLM-only (tests/test_model_catalog_facts.py's namespace guard carries the exception).
    MUSE_GLIMMER_30B = Wire("meta-models/Muse-Glimmer-30B")


class VLLMOpenAIClient(OpenAICompatClient):
    MODELS = VLLMOpenAIModel

    def __init__(self, model: VLLMOpenAIModel, base_url: str = VLLM_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)


HF_OPENAI_BASE_URL = "http://localhost:8000/v1"


class HFOpenAIModel(Model):
    # Model values are HuggingFace repo paths (as used by `transformers serve <model-id>`).
    #
    # Unlike VLLMOpenAIModel/SGLangOpenAIModel above, transformers serve has no pluggable
    # --tool-call-parser launch flag to gate tool_calls behind. Verified 2026-08-22 against
    # https://huggingface.co/docs/transformers/en/serving ("Tool calling" section), which
    # states verbatim: "Tool calling works with any model whose tokenizer declares tool call
    # tokens. Qwen and Gemma 4 work out of the box." -- i.e. support is detected from the
    # model's own chat template/tokenizer at request time, not opted into via a launch flag.
    # So a tools=True member here needs no launch-time opt-in to surface structured
    # tool_calls; if that ever stops being true, re-check the page above before trusting
    # this comment.
    LLAMA_3_1_8B = Wire("meta-llama/Llama-3.1-8B-Instruct")
    LLAMA_3_2_3B = Wire("meta-llama/Llama-3.2-3B-Instruct")
    # Mistral
    MISTRAL_7B = Wire("mistralai/Mistral-7B-Instruct-v0.3")
    MAGISTRAL_SMALL_24B = Wire("mistralai/Magistral-Small-2506")
    MINISTRAL_3_14B = Wire("mistralai/Ministral-3-14B-Instruct-2512")
    # Microsoft
    PHI_4_MINI_3_8B = Wire("microsoft/Phi-4-mini-instruct")
    PHI_4_14B = Wire("microsoft/phi-4")
    # Alibaba. Qwen 3.5/3.6/3.8 are a unified vision-language family: vision is built into the
    # base weights rather than shipped as a separate -VL variant, so this server serves image
    # input directly from the plain repo. (Qwen3 32B/8B/4B below are the older text-only
    # generation and stay vision=False.)
    QWEN_3_4B = Wire("Qwen/Qwen3-4B")
    QWEN_3_8B = Wire("Qwen/Qwen3-8B")
    QWEN_3_32B = Wire("Qwen/Qwen3-32B")
    QWEN_3_5_9B = Wire("Qwen/Qwen3.5-9B")
    QWEN_3_6_27B = Wire("Qwen/Qwen3.6-27B")
    QWEN_3_6_35B = Wire("Qwen/Qwen3.6-35B-A3B")
    QWEN_3_8_27B = Wire("Qwen/Qwen3.8-27B")
    # DeepSeek
    DEEPSEEK_R1_7B = Wire("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    DEEPSEEK_R1_8B = Wire("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    GEMMA_3_12B = Wire(
        "google/gemma-3-12b-it",
        why="OpenAI-compat servers parse tool calls server-side; the in-process HF and native "
        "Ollama paths have no tool-parse format assigned for Gemma 3",
        tools=True,
    )
    GEMMA_4_E4B = Wire("google/gemma-4-E4B-it")
    GEMMA_4_12B = Wire("google/gemma-4-12b-it")
    GEMMA_4_26B = Wire("google/gemma-4-26B-A4B-it")
    GEMMA_4_31B = Wire("google/gemma-4-31B-it")
    # Gemma 4 E4B/12B support audio natively, but audio input over transformers-serve's OpenAI-compat
    # endpoint is immature, so leave audio=False. (26B/31B: no native audio.)
    # NVIDIA
    NEMOTRON_CASCADE_2_30B = Wire("nvidia/Nemotron-Cascade-2-30B-A3B")
    # NVIDIA publishes no un-suffixed NVIDIA-Nemotron-3-Nano-30B-A3B repo (confirmed 401 on that
    # path); only precision-suffixed repos exist. Use the -BF16 reference checkpoint id verbatim
    # -- there is no shorter form to fall back to. Distinct from the multimodal
    # Nemotron-3-Nano-Omni-30B-A3B-Reasoning line, which is a different model.
    NEMOTRON_3_NANO_30B = Wire("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    # Zhipu AI: doesn't use tools reliably (matches OllamaModel's own note); this is an
    # intrinsic weights limitation rather than a serving-path one, so no tools=True override
    # here either. The _Q4 in the member name carries over from OllamaModel's quantized
    # glm-4.7-flash:q4_K_M tag for cross-provider name matching; this catalog's resolved repo
    # is the unquantized reference checkpoint.
    GLM_4_7_FLASH_31B_Q4 = Wire("zai-org/GLM-4.7-Flash")
    # OpenAI (open-weight)
    GPT_OSS_20B = Wire("openai/gpt-oss-20b")
    # HuggingFace: tool call responses don't always look correct (matches OllamaModel's note)
    SMOLLM2_1_7B = Wire("HuggingFaceTB/SmolLM2-1.7B-Instruct")


class HFOpenAIClient(OpenAICompatClient):
    MODELS = HFOpenAIModel

    def __init__(self, model: HFOpenAIModel, base_url: str = HF_OPENAI_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)


LLAMASERVER_BASE_URL = "http://localhost:8080/v1"


class LlamaServerOpenAIModel(Model):
    # Model values are the GGUF file name (or alias) as loaded by llama-server.
    # llama-server ignores the model field in API requests and always uses the loaded model;
    # these names are used for capability lookup only.
    LLAMA_3_1_8B = Wire("llama-3.1-8b-instruct.gguf")
    LLAMA_3_2_3B = Wire("llama-3.2-3b-instruct.gguf")
    MISTRAL_7B = Wire("mistral-7b-instruct-v0.3.gguf")
    MAGISTRAL_SMALL_24B = Wire("magistral-small-24b.gguf")
    MINISTRAL_3_14B = Wire("ministral-3-14b.gguf")
    PHI_4_MINI_3_8B = Wire("phi-4-mini-instruct.gguf")
    PHI_4_14B = Wire("phi-4.gguf")
    QWEN_3_4B = Wire("qwen3-4b.gguf")
    QWEN_3_8B = Wire("qwen3-8b.gguf")
    QWEN_3_32B = Wire("qwen3-32b.gguf")
    # Qwen 3.5/3.6/3.8 are a unified vision-language family in the weights, but the default GGUF
    # path llama-server loads has no mmproj projector, so every member below overrides
    # vision=False. See _NO_MMPROJ above.
    QWEN_3_5_9B = Wire("qwen3.5-9b.gguf", why=_NO_MMPROJ, vision=False)
    QWEN_3_6_27B = Wire("qwen3.6-27b.gguf", why=_NO_MMPROJ, vision=False)
    QWEN_3_6_35B = Wire("qwen3.6-35b-a3b.gguf", why=_NO_MMPROJ, vision=False)
    # thinking_levels=True on every Qwen 3.8 member: verified against the model's own
    # chat_template.jinja, which validates reasoning_effort against {xhigh, medium, low}. See
    # providers/hf/text.py for the full note.
    QWEN_3_8_27B = Wire("qwen3.8-27b.gguf", why=_NO_MMPROJ, vision=False)
    DEEPSEEK_R1_7B = Wire("deepseek-r1-distill-qwen-7b.gguf")
    DEEPSEEK_R1_8B = Wire("deepseek-r1-8b.gguf")
    GEMMA_3_12B = Wire("gemma-3-12b-it.gguf", why=_GEMMA_3_12B_GGUF_WHY, tools=True, vision=False)
    # Gemma 4 E4B/12B support audio natively, but llama-server (GGUF) has no audio-input path;
    # leave audio=False. All four also override vision=False: see _NO_MMPROJ above.
    GEMMA_4_E4B = Wire("gemma-4-e4b-it.gguf", why=_NO_MMPROJ, vision=False)
    GEMMA_4_12B = Wire("gemma-4-12b-it.gguf", why=_NO_MMPROJ, vision=False)
    GEMMA_4_26B = Wire("gemma-4-26b-a4b-it.gguf", why=_NO_MMPROJ, vision=False)
    GEMMA_4_31B = Wire("gemma-4-31b-it.gguf", why=_NO_MMPROJ, vision=False)
    # NVIDIA
    NEMOTRON_CASCADE_2_30B = Wire("nemotron-cascade-2-30b-a3b.gguf")
    NEMOTRON_3_NANO_30B = Wire("nemotron-3-nano-30b-a3b.gguf")
    # Zhipu AI: doesn't use tools reliably (matches OllamaModel's own note); this is an intrinsic
    # weights limitation, not a serving-path one, so no tools=True override here either.
    GLM_4_7_FLASH_31B_Q4 = Wire("glm-4.7-flash-q4_k_m.gguf")
    # OpenAI (open-weight)
    GPT_OSS_20B = Wire("gpt-oss-20b.gguf")
    # HuggingFace: tool call responses don't always look correct (matches OllamaModel's note)
    SMOLLM2_1_7B = Wire("smollm2-1.7b-instruct.gguf")
    # No MUSE_GLIMMER_30B here: llama-server runs the same llama.cpp engine as LM Studio, whose
    # catalog documents why the model is omitted (undocumented parsing of its channel-scoped
    # reasoning and ATEM-style XML tool calls on this engine -- see LMStudioOpenAIModel).


# llama-server accepts llama.cpp's own /completion sampling parameters on its OpenAI endpoint, where
# the repetition knob is spelled repeat_penalty. vLLM and SGLang use the portable repetition_penalty,
# so this is a one-key override rather than a change to the family.
LLAMASERVER_OPENAI_GENERATE_KWARGS = {
    **OPENAI_COMPAT_GENERATE_KWARGS,
    "repetition_penalty": "repeat_penalty",
}


class LlamaServerOpenAIClient(OpenAICompatClient):
    """Client for llama.cpp's llama-server OpenAI-compatible REST API.

    Start the server with:
        llama-server -m /path/to/model.gguf --port 8080
    """

    MODELS = LlamaServerOpenAIModel

    GENERATE_KWARG_SUPPORT = LLAMASERVER_OPENAI_GENERATE_KWARGS

    def __init__(self, model: LlamaServerOpenAIModel, base_url: str = LLAMASERVER_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)


SGLANG_BASE_URL = "http://localhost:30000/v1"


class SGLangOpenAIModel(Model):
    # Model values are HuggingFace repo paths (as used by `python -m sglang.launch_server --model-path`).
    #
    # tools=True below states a fact about the weights (this model was trained for function
    # calling), not a guarantee that SGLang will surface it. SGLang's OpenAI-compatible
    # server gates tool_calls behind a `--tool-call-parser <name>` launch flag; without it a
    # tool-calling model still runs and still emits its tool call, just as unstructured
    # text/JSON inside `content` rather than a structured `tool_calls` entry -- indistinguishable
    # from "the model ignored its tools" to a caller iterating `message.tool_calls`, so check
    # the launch flag before doubting the model. The parser `<name>` is specific to the model
    # family and to the installed SGLang version (values are added and renamed across
    # releases), so this catalog deliberately does not enumerate one per model -- look it up
    # in SGLang's own tool-parser docs (https://docs.sglang.io/advanced_features/tool_parser.html)
    # for the model and SGLang version actually running.
    LLAMA_3_1_8B = Wire("meta-llama/Llama-3.1-8B-Instruct")
    LLAMA_3_2_3B = Wire("meta-llama/Llama-3.2-3B-Instruct")
    # Mistral
    MISTRAL_7B = Wire("mistralai/Mistral-7B-Instruct-v0.3")
    MAGISTRAL_SMALL_24B = Wire("mistralai/Magistral-Small-2506")
    MINISTRAL_3_14B = Wire("mistralai/Ministral-3-14B-Instruct-2512")
    # Microsoft
    PHI_4_MINI_3_8B = Wire("microsoft/Phi-4-mini-instruct")
    PHI_4_14B = Wire("microsoft/phi-4")
    # Alibaba. Qwen 3.5/3.6/3.8 are a unified vision-language family: vision is built into the
    # base weights rather than shipped as a separate -VL variant, so this server serves image
    # input directly from the plain repo. (Qwen3 32B/8B/4B below are the older text-only
    # generation and stay vision=False.)
    QWEN_3_4B = Wire("Qwen/Qwen3-4B")
    QWEN_3_8B = Wire("Qwen/Qwen3-8B")
    QWEN_3_32B = Wire("Qwen/Qwen3-32B")
    QWEN_3_5_9B = Wire("Qwen/Qwen3.5-9B")
    QWEN_3_6_27B = Wire("Qwen/Qwen3.6-27B")
    QWEN_3_6_35B = Wire("Qwen/Qwen3.6-35B-A3B")
    QWEN_3_8_27B = Wire("Qwen/Qwen3.8-27B")
    # DeepSeek
    DEEPSEEK_R1_7B = Wire("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    DEEPSEEK_R1_8B = Wire("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    GEMMA_3_12B = Wire(
        "google/gemma-3-12b-it",
        why="OpenAI-compat servers parse tool calls server-side; the in-process HF and native "
        "Ollama paths have no tool-parse format assigned for Gemma 3",
        tools=True,
    )
    GEMMA_4_E4B = Wire("google/gemma-4-E4B-it")
    GEMMA_4_12B = Wire("google/gemma-4-12b-it")
    GEMMA_4_26B = Wire("google/gemma-4-26B-A4B-it")
    GEMMA_4_31B = Wire("google/gemma-4-31B-it")
    # Gemma 4 E4B/12B support audio natively; SGLang accepts input_audio blocks, but this path is
    # unverified against these weights, so leave audio=False until confirmed. (26B/31B: no native audio.)
    # NVIDIA
    NEMOTRON_CASCADE_2_30B = Wire("nvidia/Nemotron-Cascade-2-30B-A3B")
    # NVIDIA publishes no un-suffixed NVIDIA-Nemotron-3-Nano-30B-A3B repo (confirmed 401 on that
    # path); only precision-suffixed repos exist. Use the -BF16 reference checkpoint id verbatim
    # -- there is no shorter form to fall back to. Distinct from the multimodal
    # Nemotron-3-Nano-Omni-30B-A3B-Reasoning line, which is a different model.
    NEMOTRON_3_NANO_30B = Wire("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    # Zhipu AI: doesn't use tools reliably (matches OllamaModel's own note); this is an
    # intrinsic weights limitation rather than a serving-path one, so no tools=True override
    # here either. The _Q4 in the member name carries over from OllamaModel's quantized
    # glm-4.7-flash:q4_K_M tag for cross-provider name matching; this catalog's resolved repo
    # is the unquantized reference checkpoint.
    GLM_4_7_FLASH_31B_Q4 = Wire("zai-org/GLM-4.7-Flash")
    # OpenAI (open-weight)
    GPT_OSS_20B = Wire("openai/gpt-oss-20b")
    # HuggingFace: tool call responses don't always look correct (matches OllamaModel's note)
    SMOLLM2_1_7B = Wire("HuggingFaceTB/SmolLM2-1.7B-Instruct")


class SGLangOpenAIClient(OpenAICompatClient):
    """Client for SGLang's OpenAI-compatible REST API.

    Start the server with:
        python -m sglang.launch_server --model-path <model> --port 30000
    """

    MODELS = SGLangOpenAIModel

    def __init__(self, model: SGLangOpenAIModel, base_url: str = SGLANG_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)


OMLX_BASE_URL = "http://localhost:8000/v1"


class OMLXOpenAIModel(Model):
    # oMLX (https://github.com/jundot/omlx) is an MLX inference server for Apple Silicon. Model
    # values are the *subdirectory name* under `--model-dir`: oMLX discovers models from
    # subdirectories automatically, so the id is whatever the user named the folder. These entries
    # follow the convention "directory name == the mlx-community repo's model segment", which is
    # what a copy-pasted download produces:
    #     hf download mlx-community/Qwen3.6-35B-A3B-4bit --local-dir $MODEL_DIR/Qwen3.6-35B-A3B-4bit
    # Like LlamaServerOpenAIModel's GGUF filenames these are conventions, not contracts; a folder
    # named anything else is reachable ad hoc via `omlx:<dir>;tools,thinking,vision` (omlx is in
    # model_client._ADHOC_PROVIDERS). oMLX also accepts a `<model>:<profile>` alias form, e.g.
    # `omlx:Qwen3.6-35B-A3B:fast`; the extra colon survives because only the first ':' splits the
    # provider off, the same way an Ollama `name:tag` does.
    #
    # Qwen 3.6 35B-A3B is a unified vision-language MoE (Qwen3_5MoeForConditionalGeneration, with an
    # image_token_id and an image-text-to-text pipeline tag), so vision lives in the base weights,
    # matching OllamaModel.QWEN_3_6_35B. oMLX emits OpenAI-shaped tool_calls and separates
    # reasoning-model output, so tools and thinking are both live on this path.
    #
    # The bare member is the quant-agnostic layout (one folder holding whichever quant the user
    # downloaded) and carries the cross-provider name that resolve_model_enum("QWEN_3_6_35B") and
    # tests/test_model_catalog_consistency.py match on. The suffixed members are for a machine
    # holding several quants side by side. Every id here is distinct on purpose: two members sharing
    # a ModelSpec.id silently become an enum ALIAS (Model.__init__ assigns _value_ = spec.id before
    # enum's duplicate scan runs), which drops the second member from iteration and discards its
    # flags. tests/test_model_catalog_consistency.py::test_no_silent_enum_aliases guards this.
    QWEN_3_6_35B = Wire("Qwen3.6-35B-A3B")
    QWEN_3_6_35B_4BIT = Wire("Qwen3.6-35B-A3B-4bit")
    QWEN_3_6_35B_8BIT = Wire("Qwen3.6-35B-A3B-8bit")
    QWEN_3_6_35B_BF16 = Wire("Qwen3.6-35B-A3B-bf16")
    # Qwen 3.8 27B is a dense unified vision-language model (Qwen3_5ForConditionalGeneration, with
    # an image_token_id and an image-text-to-text pipeline tag), so the same tools/thinking/vision
    # reasoning as 3.6 above applies. mlx-community publishes 4bit, 8bit, and bf16; at 27B dense the
    # bf16 checkpoint is large but still tractable on a high-memory Mac, so it is listed here (it is
    # omitted from the LM Studio catalog, which lists only the two practical quants).
    QWEN_3_8_27B = Wire("Qwen3.8-27B")
    # thinking_levels=True on the Qwen 3.8 members: verified against the model's own
    # chat_template.jinja, which validates reasoning_effort against {xhigh, medium, low}.
    # See providers/hf/text.py for the full note.
    QWEN_3_8_27B_4BIT = Wire("Qwen3.8-27B-4bit")
    QWEN_3_8_27B_8BIT = Wire("Qwen3.8-27B-8bit")
    QWEN_3_8_27B_BF16 = Wire("Qwen3.8-27B-bf16")
    # Meta's Muse Glimmer emits channel-scoped reasoning and ATEM-style XML tool calls
    # (`<atem:function_calls>`) rather than `<think>` tags and JSON, so a serving path only exposes
    # those capabilities if it parses that framing. oMLX does: 0.5.8.dev3 added Muse Glimmer 30B
    # with "channel-scoped output parsing with ATEM tool calls", so reasoning arrives here as
    # reasoning_content and tool calls in the standard OpenAI shape -- hence tools/thinking are
    # True, matching OllamaModel.MUSE_GLIMMER_30B and VLLMOpenAIModel.MUSE_GLIMMER_30B. Requires
    # oMLX >= 0.5.8.dev3.
    #
    # Use an mlx-community checkpoint. oMLX's own `Jundot/Muse-Glimmer-30B-oQ4e` was quantized by
    # 0.5.8.dev1, before the embedding-normalization fix, and silently breaks tool calling on it:
    # the model emits `<|eot|>` right after the reasoning block and never produces an
    # `<atem:function_calls>` block (jundot/omlx#2589). That is a stale-quantization bug, not a
    # parser one, so it is a checkpoint to avoid rather than a capability to downgrade here.
    #
    # nvfp4/mxfp4 conversions also exist upstream but are omitted: those are NVIDIA / microscaling
    # formats, not the Apple Silicon path this provider serves.
    MUSE_GLIMMER_30B = Wire("Muse-Glimmer-30B")
    MUSE_GLIMMER_30B_4BIT = Wire("Muse-Glimmer-30B-4bit")
    MUSE_GLIMMER_30B_8BIT = Wire("Muse-Glimmer-30B-8bit")
    MUSE_GLIMMER_30B_BF16 = Wire("Muse-Glimmer-30B-bf16")

    # Task 11: every other model with a confirmed mlx-community build. Unlike the three families
    # above, none of these has a genuine quant-agnostic full-precision mlx-community folder to
    # serve as a bare member (checked directly: the bare, unsuffixed repo segment 401s for every
    # one of them), so only quant-suffixed members are catalogued here -- no bare id is invented.
    # Each id below is a directory name confirmed to exist as `mlx-community/<id>` (HTTP 200)
    # before being added; see the Task 11 report for the full per-id verification table. Most
    # follow the plain -4bit/-8bit/-bf16 convention; a few deviate because that is what
    # mlx-community actually published, called out inline below.
    #
    # Alibaba (Qwen)
    QWEN_3_32B_4BIT = Wire("Qwen3-32B-4bit")
    QWEN_3_32B_8BIT = Wire("Qwen3-32B-8bit")
    QWEN_3_32B_BF16 = Wire("Qwen3-32B-bf16")
    QWEN_3_4B_4BIT = Wire("Qwen3-4B-4bit")
    QWEN_3_4B_8BIT = Wire("Qwen3-4B-8bit")
    QWEN_3_4B_BF16 = Wire("Qwen3-4B-bf16")
    QWEN_3_8B_4BIT = Wire("Qwen3-8B-4bit")
    QWEN_3_8B_8BIT = Wire("Qwen3-8B-8bit")
    QWEN_3_8B_BF16 = Wire("Qwen3-8B-bf16")
    QWEN_3_5_9B_4BIT = Wire("Qwen3.5-9B-4bit")
    QWEN_3_5_9B_8BIT = Wire("Qwen3.5-9B-8bit")
    QWEN_3_5_9B_BF16 = Wire("Qwen3.5-9B-bf16")
    QWEN_3_6_27B_4BIT = Wire("Qwen3.6-27B-4bit")
    QWEN_3_6_27B_8BIT = Wire("Qwen3.6-27B-8bit")
    QWEN_3_6_27B_BF16 = Wire("Qwen3.6-27B-bf16")
    # DeepSeek
    DEEPSEEK_R1_7B_4BIT = Wire("DeepSeek-R1-Distill-Qwen-7B-4bit")
    DEEPSEEK_R1_7B_8BIT = Wire("DeepSeek-R1-Distill-Qwen-7B-8bit")
    DEEPSEEK_R1_7B_BF16 = Wire("DeepSeek-R1-Distill-Qwen-7B-bf16")
    DEEPSEEK_R1_8B_4BIT = Wire("DeepSeek-R1-Distill-Llama-8B-4bit")
    DEEPSEEK_R1_8B_8BIT = Wire("DeepSeek-R1-Distill-Llama-8B-8bit")
    DEEPSEEK_R1_8B_BF16 = Wire("DeepSeek-R1-Distill-Llama-8B-bf16")
    # Google. Vision stays at MODEL_FACTS' intrinsic True for all of these -- unlike the GGUF
    # catalogs (LM Studio's bare members, llama-server, llama-cpp), MLX serves vision directly
    # with no mmproj projector to load, so _NO_MMPROJ does not apply here.
    GEMMA_3_12B_4BIT = Wire("gemma-3-12b-it-4bit", why=_GEMMA_3_12B_TOOLS_WHY, tools=True)
    GEMMA_3_12B_8BIT = Wire("gemma-3-12b-it-8bit", why=_GEMMA_3_12B_TOOLS_WHY, tools=True)
    GEMMA_3_12B_BF16 = Wire("gemma-3-12b-it-bf16", why=_GEMMA_3_12B_TOOLS_WHY, tools=True)
    GEMMA_4_E4B_4BIT = Wire("gemma-4-e4b-it-4bit")
    GEMMA_4_E4B_8BIT = Wire("gemma-4-e4b-it-8bit")
    GEMMA_4_E4B_BF16 = Wire("gemma-4-e4b-it-bf16")
    # mlx-community's actual repo casing for this one keeps "12B" capitalized, unlike the other
    # three Gemma 4 sizes (which are fully lowercase) -- confirmed via the Hub API's redirect
    # target, not guessed.
    GEMMA_4_12B_4BIT = Wire("gemma-4-12B-it-4bit")
    GEMMA_4_12B_8BIT = Wire("gemma-4-12B-it-8bit")
    GEMMA_4_12B_BF16 = Wire("gemma-4-12B-it-bf16")
    GEMMA_4_26B_4BIT = Wire("gemma-4-26b-a4b-it-4bit")
    GEMMA_4_26B_8BIT = Wire("gemma-4-26b-a4b-it-8bit")
    GEMMA_4_26B_BF16 = Wire("gemma-4-26b-a4b-it-bf16")
    GEMMA_4_31B_4BIT = Wire("gemma-4-31b-it-4bit")
    GEMMA_4_31B_8BIT = Wire("gemma-4-31b-it-8bit")
    GEMMA_4_31B_BF16 = Wire("gemma-4-31b-it-bf16")
    # Zhipu
    GLM_4_7_FLASH_31B_Q4_4BIT = Wire("GLM-4.7-Flash-4bit")
    GLM_4_7_FLASH_31B_Q4_8BIT = Wire("GLM-4.7-Flash-8bit")
    GLM_4_7_FLASH_31B_Q4_BF16 = Wire("GLM-4.7-Flash-bf16")
    # OpenAI (open-weight). gpt-oss ships natively quantized to mxfp4, so mlx-community
    # re-packages it as Q4/Q8 requantizations of that native format (repo names
    # gpt-oss-20b-MXFP4-Q4/-Q8) rather than plain 4bit/8bit builds -- hence the member names
    # below, and no bf16 (no such build exists).
    GPT_OSS_20B_MXFP4_Q4 = Wire("gpt-oss-20b-MXFP4-Q4")
    GPT_OSS_20B_MXFP4_Q8 = Wire("gpt-oss-20b-MXFP4-Q8")
    # Meta
    LLAMA_3_2_3B_4BIT = Wire("Llama-3.2-3B-Instruct-4bit")
    LLAMA_3_2_3B_8BIT = Wire("Llama-3.2-3B-Instruct-8bit")
    LLAMA_3_2_3B_BF16 = Wire("Llama-3.2-3B-Instruct-bf16")
    # mlx-community's confirmed matching 4bit/8bit/bf16 trio for this model carries Meta's older
    # "Meta-Llama-3.1-..." repo naming rather than the canonical "Llama-3.1-..." (a
    # "Llama-3.1-8B-Instruct-4bit" repo also exists but has no 8bit/bf16 siblings, so the
    # Meta-prefixed family is used for all three quants).
    LLAMA_3_1_8B_4BIT = Wire("Meta-Llama-3.1-8B-Instruct-4bit")
    LLAMA_3_1_8B_8BIT = Wire("Meta-Llama-3.1-8B-Instruct-8bit")
    LLAMA_3_1_8B_BF16 = Wire("Meta-Llama-3.1-8B-Instruct-bf16")
    # Mistral. Only a bf16 build exists under the plain naming convention (a 4bit-DWQ variant
    # also exists, but DWQ is a distinct quantization method, not this catalog's plain
    # 4bit/8bit/bf16 vocabulary, so it is not catalogued here).
    MAGISTRAL_SMALL_24B_BF16 = Wire("Magistral-Small-2506-bf16")
    MINISTRAL_3_14B_4BIT = Wire("Ministral-3-14B-Instruct-2512-4bit")
    MINISTRAL_3_14B_8BIT = Wire("Ministral-3-14B-Instruct-2512-8bit")
    MINISTRAL_3_14B_BF16 = Wire("Ministral-3-14B-Instruct-2512-bf16")
    # Only 4bit/8bit exist for this model (no bf16 build confirmed).
    MISTRAL_7B_4BIT = Wire("Mistral-7B-Instruct-v0.3-4bit")
    MISTRAL_7B_8BIT = Wire("Mistral-7B-Instruct-v0.3-8bit")
    # NVIDIA
    NEMOTRON_CASCADE_2_30B_4BIT = Wire("Nemotron-Cascade-2-30B-A3B-4bit")
    NEMOTRON_CASCADE_2_30B_8BIT = Wire("Nemotron-Cascade-2-30B-A3B-8bit")
    NEMOTRON_CASCADE_2_30B_BF16 = Wire("Nemotron-Cascade-2-30B-A3B-bf16")
    # This model's confirmed matching 4bit/8bit/bf16 trio carries an extra "-MLX-" infix and
    # Titlecase quant token in mlx-community's own repo naming (a differently-named, unmatched
    # "-4bit" repo also exists but has no 8bit/bf16 siblings, so the "-MLX-"-infixed family is
    # used for all three quants, matching the pattern used for LLAMA_3_1_8B above).
    NEMOTRON_3_NANO_30B_4BIT = Wire("NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4Bit")
    NEMOTRON_3_NANO_30B_8BIT = Wire("NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-8Bit")
    NEMOTRON_3_NANO_30B_BF16 = Wire("NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-BF16")
    # Microsoft
    PHI_4_14B_4BIT = Wire("phi-4-4bit")
    PHI_4_14B_8BIT = Wire("phi-4-8bit")
    PHI_4_14B_BF16 = Wire("phi-4-bf16")
    # mlx-community publishes 4bit/8bit plus a distinctly-named "-mlx-fp16" build (no plain bf16
    # repo), hence the FP16 (not BF16) member suffix.
    PHI_4_MINI_3_8B_4BIT = Wire("Phi-4-mini-instruct-4bit")
    PHI_4_MINI_3_8B_8BIT = Wire("Phi-4-mini-instruct-8bit")
    PHI_4_MINI_3_8B_FP16 = Wire("Phi-4-mini-instruct-mlx-fp16")
    # No SMOLLM2_1_7B here: mlx-community publishes only the unquantized bare repo for this
    # model (confirmed present) but no quantized -4bit/-8bit/-bf16 build, so there is nothing to
    # catalogue under this task's quant-suffixed fill.

    # structured_output stays False across every OpenAI-compat local-server catalog (only the native
    # Ollama path grammar-enforces JSON), so `schema=` falls back to prompt-and-parse. audio stays
    # False: these weights are image-text-to-text, with no audio encoder.


class OMLXOpenAIClient(OpenAICompatClient):
    """Client for oMLX's OpenAI-compatible REST API (MLX inference on Apple Silicon).

    Start the server with:
        omlx serve --model-dir ~/models --port 8000

    The default port collides with vLLM and HF Transformers Serve, so pass ``base_url=`` (or
    ``@<base_url>`` in the model string) when running more than one of them.
    """

    MODELS = OMLXOpenAIModel

    def __init__(self, model: OMLXOpenAIModel, base_url: str = OMLX_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)
