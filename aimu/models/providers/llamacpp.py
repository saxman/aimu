import json
import logging
import threading
from typing import Iterator, Optional, Any, Union

# Hard module-load import so HAS_LLAMACPP accurately reflects that llama-cpp-python is
# installed (matches the diffusers/soundfile convention in the HuggingFace clients). The
# Llama model itself is still constructed lazily in __init__ to defer weight loading.
import llama_cpp

from ..base import StreamingContentType, StreamChunk, Model, BaseModelClient, ContextOverflowError, classproperty
from .._catalog import Wire
from .._internal.generate_kwargs import LOCAL_MAX_TOKENS, Unsupported
from .._internal.image_input import _build_user_content_blocks
from .._internal.thinking import pop_thinking
from ._thinking import _reasoning_text, _split_thinking, _ThinkingParser

logger = logging.getLogger(__name__)

_model_registry: dict[tuple, Any] = {}  # cache_key → Llama instance
_registry_lock = threading.Lock()


def _make_cache_key(model_path: str, n_ctx: int, n_gpu_layers: int, chat_format: str | None) -> tuple:
    return (model_path, n_ctx, n_gpu_layers, str(chat_format))


def _stringify_content(content: Any) -> str:
    """Extract plain text from a message's ``content`` for the pre-flight token count below.

    ``content`` is either a plain string or an OpenAI-format content-block list (vision input);
    only the text blocks contribute -- an image block has no token-comparable text, and the
    pre-flight check only needs a reasonable proxy for the prompt's real length, not an exact
    reproduction of what a vision-capable GGUF would render.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content) if content is not None else ""


def _raise_if_prompt_overflows(llm: Any, messages: list[dict]) -> None:
    """Pre-flight guard: raise ``ContextOverflowError`` before an over-long prompt reaches
    ``create_chat_completion()``.

    In-process, so there is no server error to translate. Unlike HuggingFace, llama-cpp-python
    doesn't expose the exact chat-template-rendered token count without reaching into its
    internal chat-format machinery, so this is a deliberate approximation: it tokenizes the plain
    role+text content of *messages* with the model's own loaded tokenizer (real BPE, not a
    chars-per-token estimate) and compares against ``n_ctx()`` -- the model's actual,
    precisely-known context window (an ``__init__`` argument, not a guess). The chat template's
    own formatting tokens (role tags, special tokens) are not counted, so this under-counts the
    real prompt slightly and is biased toward false negatives, not false positives: it will not
    flag a request that's actually still within the window as an overflow.

    A module-level function taking the loaded ``Llama`` instance explicitly, rather than a bound
    method reading ``self._llm``, so it degrades the same way (skipped, not guessed) against a
    minimal test double that doesn't implement ``n_ctx()``/``tokenize()``.
    """
    n_ctx_fn = getattr(llm, "n_ctx", None)
    tokenize_fn = getattr(llm, "tokenize", None)
    if n_ctx_fn is None or tokenize_fn is None:
        return
    n_ctx = n_ctx_fn()
    text = "\n".join(f"{m.get('role', '')}: {_stringify_content(m.get('content'))}" for m in messages)
    token_count = len(tokenize_fn(text.encode("utf-8"), add_bos=True, special=True))
    if token_count <= n_ctx:
        return
    raise ContextOverflowError(
        f"The request no longer fits the model's context window: the conversation is "
        f"approximately {token_count} tokens, but this client was constructed with n_ctx={n_ctx}. "
        "Shorten the conversation, advertise fewer tools, or construct LlamaCppClient(..., "
        "n_ctx=N) with a larger window."
    )


# Shared by every vision-capable member of LlamaCppModel below. llama-cpp vision needs an mmproj
# projector supplied via the chat_handler= constructor kwarg, which the default GGUF path does not
# load, so a member whose weights are intrinsically vision-capable (per MODEL_FACTS) still
# overrides vision=False here: the catalog describes the default path, and advertising vision
# would let a caller pass images that fail at request time. See
# tests/test_model_catalog_facts.py::test_gguf_catalogs_do_not_advertise_vision.
_NO_MMPROJ = (
    "llama-cpp vision needs an mmproj projector supplied via chat_handler=, which the default "
    "GGUF path does not load; advertising vision would let a caller pass images that fail at "
    "request time"
)


class LlamaCppModel(Model):
    # tools=True consistent with the verified Ollama Llama tool calling and the llama.cpp-based
    # llama-server build (llama-cpp-python runs the same engine). Requires a chat_format that
    # supports tool calling (modern GGUFs auto-detect; older ones need chat_format="chatml-function-calling").
    LLAMA_3_1_8B = Wire("llama-3.1-8b")
    LLAMA_3_2_3B = Wire("llama-3.2-3b")
    MISTRAL_7B = Wire("mistral-7b")
    MAGISTRAL_SMALL_24B = Wire("magistral-small-24b")
    MINISTRAL_3_14B = Wire("ministral-3-14b")
    PHI_4_MINI_3_8B = Wire("phi-4-mini")
    PHI_4_14B = Wire("phi-4")
    QWEN_3_4B = Wire("qwen3-4b")
    QWEN_3_8B = Wire("qwen3-8b")
    QWEN_3_32B = Wire("qwen3-32b")
    # Qwen 3.5/3.6/3.8 are a unified vision-language family in the weights, but the default GGUF
    # path here loads no mmproj projector, so every member below overrides vision=False. See
    # _NO_MMPROJ above.
    QWEN_3_5_9B = Wire("qwen3.5-9b", why=_NO_MMPROJ, vision=False)
    QWEN_3_6_27B = Wire("qwen3.6-27b", why=_NO_MMPROJ, vision=False)
    QWEN_3_6_35B = Wire("qwen3.6-35b-a3b", why=_NO_MMPROJ, vision=False)
    # thinking_levels=True on every Qwen 3.8 member: verified against the model's own
    # chat_template.jinja, which validates reasoning_effort against {xhigh, medium, low}. See
    # providers/hf/text.py for the full note.
    QWEN_3_8_27B = Wire("qwen3.8-27b", why=_NO_MMPROJ, vision=False)
    DEEPSEEK_R1_7B = Wire("deepseek-r1-7b")
    DEEPSEEK_R1_8B = Wire("deepseek-r1-8b")
    GEMMA_3_12B = Wire("gemma-3-12b", why=_NO_MMPROJ, vision=False)
    # thinking=True matches every other Gemma 4 catalog entry; llama-cpp surfaces reasoning
    # via the shared <think> parser (same as QWEN_3_8B above). vision is overridden False on all
    # four for the same mmproj reason. Gemma 4 E4B/12B support audio natively, but llama-cpp has
    # no audio-input path; leave audio=False.
    GEMMA_4_E4B = Wire("gemma-4-e4b", why=_NO_MMPROJ, vision=False)
    GEMMA_4_12B = Wire("gemma-4-12b", why=_NO_MMPROJ, vision=False)
    GEMMA_4_26B = Wire("gemma-4-26b-a4b", why=_NO_MMPROJ, vision=False)
    GEMMA_4_31B = Wire("gemma-4-31b", why=_NO_MMPROJ, vision=False)
    NEMOTRON_CASCADE_2_30B = Wire("nemotron-cascade-2-30b-a3b")
    NEMOTRON_3_NANO_30B = Wire("nemotron-3-nano-30b-a3b")
    # Zhipu AI: doesn't use tools reliably (matches OllamaModel's own note); this is an intrinsic
    # weights limitation, not a serving-path one, so no tools=True override here either.
    GLM_4_7_FLASH_31B_Q4 = Wire("glm-4.7-flash-q4")
    GPT_OSS_20B = Wire("gpt-oss-20b")
    SMOLLM2_1_7B = Wire("smollm2-1.7b")
    # No MUSE_GLIMMER_30B here: llama-cpp-python runs the same llama.cpp engine as llama-server
    # and LM Studio, whose catalogs document why the model is omitted (undocumented parsing of
    # its channel-scoped reasoning and ATEM-style XML tool calls on this engine -- see
    # LMStudioOpenAIModel in openai_compat.py).


# llama-cpp-python's create_chat_completion takes every sampling knob, spelling the repetition one
# repeat_penalty. The window is sized when the weights load, so context_length cannot be a request
# parameter here.
LLAMACPP_GENERATE_KWARGS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "presence_penalty": "presence_penalty",
    "repetition_penalty": "repeat_penalty",
    "max_tokens": "max_tokens",
    "context_length": Unsupported("Set it at construction instead: LlamaCppClient(..., n_ctx=N)."),
}


class LlamaCppClient(BaseModelClient):
    MODELS = LlamaCppModel

    GENERATE_KWARG_SUPPORT = LLAMACPP_GENERATE_KWARGS

    DEFAULT_GENERATE_KWARGS = {
        "max_tokens": LOCAL_MAX_TOKENS,
        "temperature": 0.1,
    }

    def __init__(
        self,
        model: LlamaCppModel,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        chat_format: Optional[str] = None,
        chat_handler: Optional[Any] = None,
        verbose: bool = False,
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__(model, model_kwargs, system_message)

        self._cache_key = _make_cache_key(model_path, n_ctx, n_gpu_layers, chat_format)
        with _registry_lock:
            if self._cache_key in _model_registry:
                self._llm = _model_registry[self._cache_key]
                return

        self._llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,
            chat_handler=chat_handler,
            verbose=verbose,
        )
        with _registry_lock:
            _model_registry[self._cache_key] = self._llm

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

    def _rewrite_generate_kwargs(self, kwargs: dict) -> dict:
        # No verified thinking control on this path; the request was resolved and warned about
        # upstream, so drop it rather than let llama_cpp reject an unknown key.
        pop_thinking(kwargs)
        return kwargs

    def _iter_stream(self, stream) -> Iterator[StreamChunk]:
        """Iterate a completion stream, yielding StreamChunks and updating self.last_thinking."""
        self.last_thinking = ""
        parser = _ThinkingParser() if self.is_thinking_model else None

        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            reasoning = _reasoning_text(delta)
            if reasoning:
                self.last_thinking += reasoning
                yield StreamChunk(StreamingContentType.THINKING, reasoning)
            text = delta.get("content") or ""
            if not text:
                continue
            logger.debug("LLM raw chunk: %s", chunk)
            if parser:
                for phase, part in parser.feed(text):
                    if phase == StreamingContentType.THINKING:
                        self.last_thinking += part
                        yield StreamChunk(StreamingContentType.THINKING, part)
                    else:
                        yield StreamChunk(StreamingContentType.GENERATING, part)
            else:
                yield StreamChunk(StreamingContentType.GENERATING, text)

    def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs = self._resolve_generate_kwargs(generate_kwargs)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs, images=images, audio=audio)

        content_in = _build_user_content_blocks(prompt, images) if images else prompt
        # llama.cpp runs in-process (no wire); the literal call kwargs to create_chat_completion
        # are the closest equivalent AIMU has to a wire payload for a local GGUF model.
        # generate_kwargs is splatted first so it cannot silently override "messages".
        payload = {**generate_kwargs, "messages": [{"role": "user", "content": content_in}]}
        self._record_request(payload)
        _raise_if_prompt_overflows(self._llm, payload["messages"])
        response = self._llm.create_chat_completion(**payload)
        logger.debug("LLM raw response: %s", response)
        msg = response["choices"][0]["message"]
        content = msg["content"] or ""

        self.last_thinking = ""
        reasoning = _reasoning_text(msg)
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
        content_in = _build_user_content_blocks(prompt, images) if images else prompt
        # generate_kwargs splatted first: see _chat_streamed for why (stream=True must not be
        # silently overridable).
        payload = {**generate_kwargs, "messages": [{"role": "user", "content": content_in}], "stream": True}
        self._record_request(payload)
        _raise_if_prompt_overflows(self._llm, payload["messages"])
        stream = self._llm.create_chat_completion(**payload)
        yield from self._iter_stream(stream)

    def _chat(
        self,
        user_message: Optional[str] = None,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools, images=images, audio=audio)

        if stream:
            return self._chat_streamed(generate_kwargs, tools)

        # list(...) makes a shallow copy: self.messages is live history that _append_message
        # / _record_tool_calls mutate later in this method, so recording the bare attribute would
        # let the recorded request grow the model's own answer after the fact. generate_kwargs is
        # splatted first so an accidental "messages"/"tools" key in it cannot silently override
        # the real ones.
        payload = {**generate_kwargs, "messages": list(self.messages), "tools": tools if tools else None}
        self._record_request(payload)
        _raise_if_prompt_overflows(self._llm, payload["messages"])
        response = self._llm.create_chat_completion(**payload)
        logger.debug("LLM raw response: %s", response)
        msg = response["choices"][0]["message"]

        self.last_thinking = ""
        # Prefer a server-provided reasoning field over parsing inline <think> tags.
        reasoning = _reasoning_text(msg)

        # Single turn: if the model called tools, execute them and return. The model's response
        # to the tool results comes on the next chat() call (the loop lives in Agent).
        if msg.get("tool_calls"):
            tool_calls = [
                {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"])}
                for tc in msg["tool_calls"]
            ]
            text = msg.get("content") or ""
            if reasoning:
                self.last_thinking = reasoning
            elif self.is_thinking_model:
                self.last_thinking, text = _split_thinking(text)
            msgs_before = len(self.messages)
            self._record_tool_calls(tool_calls, content=text)
            if self.last_thinking:
                self.messages[msgs_before]["thinking"] = self.last_thinking
            return text

        content = msg.get("content") or ""
        if reasoning:
            self.last_thinking = reasoning
        elif self.is_thinking_model:
            self.last_thinking, content = _split_thinking(content)

        self._append_message({"role": "assistant", "content": content})
        if self.last_thinking:
            self.messages[-1]["thinking"] = self.last_thinking
        return content

    def _chat_streamed(self, generate_kwargs: dict[str, Any], tools: list) -> Iterator[StreamChunk]:
        # See _chat: copy the live messages list, and splat generate_kwargs first so it cannot
        # silently override stream=True (a caller-supplied generate_kwargs={"stream": False} would
        # otherwise quietly disable streaming instead of raising).
        payload = {
            **generate_kwargs,
            "messages": list(self.messages),
            "stream": True,
            "tools": tools if tools else None,
        }
        self._record_request(payload)
        _raise_if_prompt_overflows(self._llm, payload["messages"])
        stream = self._llm.create_chat_completion(**payload)

        # Yield content/thinking chunks as they arrive (incremental streaming) while accumulating
        # any tool-call deltas separately; content and tool_call deltas don't require buffering.
        tool_calls_acc: dict[int, dict] = {}
        full_content = ""
        parser = _ThinkingParser() if self.is_thinking_model else None
        self.last_thinking = ""

        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            logger.debug("LLM raw chunk: %s", chunk)
            reasoning = _reasoning_text(delta)
            if reasoning:
                self.last_thinking += reasoning
                yield StreamChunk(StreamingContentType.THINKING, reasoning)
            if delta.get("tool_calls"):
                for tc_delta in delta["tool_calls"]:
                    acc = tool_calls_acc.setdefault(tc_delta["index"], {"name": "", "arguments": ""})
                    fn = tc_delta.get("function") or {}
                    if fn.get("name"):
                        acc["name"] += fn["name"]
                    if fn.get("arguments"):
                        acc["arguments"] += fn["arguments"]
            elif delta.get("content"):
                text = delta["content"]
                if parser:
                    for phase, part in parser.feed(text):
                        if phase == StreamingContentType.THINKING:
                            self.last_thinking += part
                        else:
                            full_content += part
                        yield StreamChunk(phase, part)
                else:
                    full_content += text
                    yield StreamChunk(StreamingContentType.GENERATING, text)

        if not tool_calls_acc:
            self._append_message({"role": "assistant", "content": full_content})
            if self.last_thinking:
                self.messages[-1]["thinking"] = self.last_thinking
            return

        # Single turn: prose/thinking already streamed above; now dispatch the tools (yields
        # TOOL_CALLING chunks via streaming-tool support in the base) and return. The model's
        # response to the tool results comes on the next chat() call (loop lives in Agent).
        tool_calls = [{"name": tc["name"], "arguments": json.loads(tc["arguments"])} for tc in tool_calls_acc.values()]
        tool_turn_thinking = self.last_thinking
        msgs_before = len(self.messages)
        self._record_tool_calls(tool_calls, content=full_content)
        if tool_turn_thinking:
            self.messages[msgs_before]["thinking"] = tool_turn_thinking
