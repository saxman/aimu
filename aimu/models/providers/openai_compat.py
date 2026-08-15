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
    ModelSpec,
    StreamChunk,
    StreamingContentType,
    classproperty,
)
from .._internal.audio_input import _build_audio_content_blocks
from .._internal.image_input import _build_user_content_blocks
from .._internal.message_meta import strip_inert_keys
from .._internal.sdk_config import sdk_client_kwargs
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


class OpenAICompatClient(BaseModelClient):
    MODELS = Model

    DEFAULT_GENERATE_KWARGS = {
        "max_tokens": 1024,
        "temperature": 0.1,
    }

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
        self.default_generate_kwargs = self.DEFAULT_GENERATE_KWARGS.copy()
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

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        if not generate_kwargs:
            return self.default_generate_kwargs.copy()
        return {**self.default_generate_kwargs, **generate_kwargs}

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
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)
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
    # Model values are Ollama model tags (as used by `ollama pull`).
    # Llama 3.1/3.2 tool calling verified reliable on current Ollama builds (same backend as
    # the native OllamaModel catalog).
    LLAMA_3_1_8B = ModelSpec("llama3.1:8b", tools=True)
    LLAMA_3_2_3B = ModelSpec("llama3.2:3b", tools=True)
    MISTRAL_7B = ModelSpec("mistral:7b", tools=True)
    PHI_4_MINI = ModelSpec("phi4-mini:3.8b", tools=True)
    QWEN_3_4B = ModelSpec("qwen3:4b", tools=True, thinking=True)
    QWEN_3_8B = ModelSpec("qwen3:8b", tools=True, thinking=True)
    # Qwen 3.5 is a unified vision-language model (vision built into the base weights); the
    # Ollama tag serves image input over the OpenAI-compat endpoint. Qwen3 8B/4B above are the
    # older text-only generation.
    QWEN_3_5_9B = ModelSpec("qwen3.5:9b", tools=True, thinking=True, vision=True)
    # Qwen 3.6 serves over the OpenAI-compat endpoint exactly like 3.5 above. On Apple Silicon,
    # Ollama 0.19+ runs it on its MLX backend automatically; that's transparent to this client (the
    # tag is unchanged), so there is nothing MLX-specific to declare here.
    QWEN_3_6_35B = ModelSpec("qwen3.6:35b", tools=True, thinking=True, vision=True)
    # Qwen 3.8 27B is the same shape again: a unified vision-language dense model whose Ollama tag
    # carries the vision/tools/thinking capabilities. (QWEN_3_8_27B is Qwen 3.8 at 27B; QWEN_3_8B
    # above is Qwen 3 at 8B.)
    QWEN_3_8_27B = ModelSpec("qwen3.8:27b", tools=True, thinking=True, vision=True)
    DEEPSEEK_R1_8B = ModelSpec("deepseek-r1:8b", thinking=True)
    GEMMA_3_12B = ModelSpec("gemma3:12b", vision=True)
    # Gemma 4 E4B/12B support audio natively, but the Ollama API doesn't expose audio input yet
    # (the native OllamaModel catalog omits audio for the same reason); leave audio=False.
    GEMMA_4_E4B = ModelSpec("gemma4:e4b", tools=True, thinking=True, vision=True)
    GEMMA_4_12B = ModelSpec("gemma4:12b", tools=True, thinking=True, vision=True)
    GEMMA_4_26B = ModelSpec("gemma4:26b", tools=True, thinking=True, vision=True)
    GEMMA_4_31B = ModelSpec("gemma4:31b", tools=True, thinking=True, vision=True)
    # Muse Glimmer is a vision-language model whose perception encoder ships in the same
    # weights. It emits channel-scoped reasoning and ATEM-style XML tool calls, which Ollama
    # parses server-side, so reasoning arrives here as reasoning_content and tool calls in the
    # standard OpenAI shape.
    MUSE_GLIMMER_30B = ModelSpec("muse-glimmer:30b", tools=True, thinking=True, vision=True)


class OllamaOpenAIClient(OpenAICompatClient):
    MODELS = OllamaOpenAIModel

    def __init__(self, model: OllamaOpenAIModel, base_url: str = OLLAMA_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)


LMSTUDIO_BASE_URL = "http://localhost:1234/v1"


class LMStudioOpenAIModel(Model):
    # Model values are the model "key" as shown in LM Studio's loaded model list.
    # tools=True consistent with the verified Ollama Llama tool calling and the llama.cpp-based
    # llama-server build (LM Studio runs the same GGUF/llama.cpp engine).
    LLAMA_3_1_8B = ModelSpec("llama-3.1-8b-instruct", tools=True)
    MISTRAL_7B = ModelSpec("mistral-7b-instruct-v0.3", tools=True)
    PHI_4_MINI = ModelSpec("phi-4-mini-instruct", tools=True)
    QWEN_3_4B = ModelSpec("qwen3-4b", tools=True, thinking=True)
    QWEN_3_8B = ModelSpec("qwen3-8b", tools=True, thinking=True)
    # Qwen 3.5 is a unified vision-language model; load its multimodal GGUF in LM Studio for
    # image input (same convention as the Gemma 4 vision entries below). Qwen3 8B/4B are text-only.
    QWEN_3_5_9B = ModelSpec("qwen3.5-9b", tools=True, thinking=True, vision=True)
    # LM Studio ships an MLX engine alongside llama.cpp and picks it automatically for MLX weights
    # on Apple Silicon. The loaded-model key derives from the downloaded repo, so an mlx-community
    # download keeps its quant suffix -- unlike the GGUF entries above, whose keys are quant-free.
    # Member names match OMLXOpenAIModel's so the cross-provider consistency guard covers the pair.
    # Qwen 3.6 35B-A3B is a unified vision-language MoE, hence vision=True (matching
    # OllamaModel.QWEN_3_6_35B). No bare QWEN_3_6_35B here: a quant-free LM Studio key would be the
    # GGUF build, which is not an MLX path. No bf16 either -- a 35B unquantized is impractical here.
    QWEN_3_6_35B_4BIT = ModelSpec("qwen3.6-35b-a3b-4bit", tools=True, thinking=True, vision=True)
    QWEN_3_6_35B_8BIT = ModelSpec("qwen3.6-35b-a3b-8bit", tools=True, thinking=True, vision=True)
    # Qwen 3.8 27B is dense rather than MoE, but the MLX story is identical: quant-suffixed keys
    # from an mlx-community download, no bare member (a quant-free key would be the GGUF build).
    # At 27B dense, bf16 is impractical here too, so only the two practical quants are listed.
    QWEN_3_8_27B_4BIT = ModelSpec("qwen3.8-27b-4bit", tools=True, thinking=True, vision=True)
    QWEN_3_8_27B_8BIT = ModelSpec("qwen3.8-27b-8bit", tools=True, thinking=True, vision=True)
    # No MUSE_GLIMMER_30B here, for two independent reasons: LM Studio distributes it as GGUF only
    # (no MLX build, so it is not an MLX path at all), and whether its llama.cpp engine parses the
    # model's channel-scoped reasoning and ATEM-style XML tool calls is still undocumented. Adding
    # it would mean guessing tools/thinking. See OMLXOpenAIModel for the path that does parse them.
    DEEPSEEK_R1_7B = ModelSpec("deepseek-r1-distill-qwen-7b", thinking=True)
    # Gemma 4 E4B/12B support audio natively, but LM Studio has no audio-input path (image-only);
    # leave audio=False.
    GEMMA_4_E4B = ModelSpec("gemma-4-e4b-it", tools=True, thinking=True, vision=True)
    GEMMA_4_12B = ModelSpec("gemma-4-12b-it", tools=True, thinking=True, vision=True)
    GEMMA_4_26B = ModelSpec("gemma-4-26b-a4b-it", tools=True, thinking=True, vision=True)
    GEMMA_4_31B = ModelSpec("gemma-4-31b-it", tools=True, thinking=True, vision=True)


class LMStudioOpenAIClient(OpenAICompatClient):
    MODELS = LMStudioOpenAIModel

    def __init__(self, model: LMStudioOpenAIModel, base_url: str = LMSTUDIO_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)


VLLM_BASE_URL = "http://localhost:8000/v1"


class VLLMOpenAIModel(Model):
    # Model values are HuggingFace repo paths (as used by `vllm serve --model`).
    LLAMA_3_1_8B = ModelSpec("meta-llama/Llama-3.1-8B-Instruct", tools=True)
    LLAMA_3_2_3B = ModelSpec("meta-llama/Llama-3.2-3B-Instruct", tools=True)
    MISTRAL_7B = ModelSpec("mistralai/Mistral-7B-Instruct-v0.3", tools=True)
    PHI_4_MINI = ModelSpec("microsoft/Phi-4-mini-instruct", tools=True)
    QWEN_3_4B = ModelSpec("Qwen/Qwen3-4B", tools=True, thinking=True)
    QWEN_3_8B = ModelSpec("Qwen/Qwen3-8B", tools=True, thinking=True)
    DEEPSEEK_R1_7B = ModelSpec("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", thinking=True)
    GEMMA_3_12B = ModelSpec("google/gemma-3-12b-it", tools=True, vision=True)
    GEMMA_4_E4B = ModelSpec("google/gemma-4-E4B-it", tools=True, thinking=True, vision=True)
    GEMMA_4_12B = ModelSpec("google/gemma-4-12b-it", tools=True, thinking=True, vision=True)
    GEMMA_4_26B = ModelSpec("google/gemma-4-26B-A4B-it", tools=True, thinking=True, vision=True)
    GEMMA_4_31B = ModelSpec("google/gemma-4-31B-it", tools=True, thinking=True, vision=True)
    # Gemma 4 E4B/12B support audio natively; vLLM accepts input_audio blocks, but this path is
    # unverified against these weights, so leave audio=False until confirmed. (26B/31B: no native audio.)
    # Muse Glimmer emits channel-scoped reasoning and ATEM-style XML tool calls instead of
    # <think> tags and JSON, so serving it needs vLLM's dedicated parsers, which must be
    # enabled together (they key off the same framing):
    #   vllm serve meta-models/Muse-Glimmer-30B \
    #     --enable-auto-tool-choice --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer
    # With those set, reasoning arrives as reasoning_content and tool calls in the standard
    # OpenAI shape; without them both channels collapse into content and tools never surface.
    MUSE_GLIMMER_30B = ModelSpec("meta-models/Muse-Glimmer-30B", tools=True, thinking=True, vision=True)


class VLLMOpenAIClient(OpenAICompatClient):
    MODELS = VLLMOpenAIModel

    def __init__(self, model: VLLMOpenAIModel, base_url: str = VLLM_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)


HF_OPENAI_BASE_URL = "http://localhost:8000/v1"


class HFOpenAIModel(Model):
    # Model values are HuggingFace repo paths (as used by `transformers serve <model-id>`).
    LLAMA_3_1_8B = ModelSpec("meta-llama/Llama-3.1-8B-Instruct", tools=True)
    LLAMA_3_2_3B = ModelSpec("meta-llama/Llama-3.2-3B-Instruct", tools=True)
    MISTRAL_7B = ModelSpec("mistralai/Mistral-7B-Instruct-v0.3", tools=True)
    PHI_4_MINI = ModelSpec("microsoft/Phi-4-mini-instruct", tools=True)
    QWEN_3_4B = ModelSpec("Qwen/Qwen3-4B", tools=True, thinking=True)
    QWEN_3_8B = ModelSpec("Qwen/Qwen3-8B", tools=True, thinking=True)
    DEEPSEEK_R1_7B = ModelSpec("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", thinking=True)
    GEMMA_3_12B = ModelSpec("google/gemma-3-12b-it", tools=True, vision=True)
    GEMMA_4_E4B = ModelSpec("google/gemma-4-E4B-it", tools=True, thinking=True, vision=True)
    GEMMA_4_12B = ModelSpec("google/gemma-4-12b-it", tools=True, thinking=True, vision=True)
    GEMMA_4_26B = ModelSpec("google/gemma-4-26B-A4B-it", tools=True, thinking=True, vision=True)
    GEMMA_4_31B = ModelSpec("google/gemma-4-31B-it", tools=True, thinking=True, vision=True)
    # Gemma 4 E4B/12B support audio natively, but audio input over transformers-serve's OpenAI-compat
    # endpoint is immature, so leave audio=False. (26B/31B: no native audio.)


class HFOpenAIClient(OpenAICompatClient):
    MODELS = HFOpenAIModel

    def __init__(self, model: HFOpenAIModel, base_url: str = HF_OPENAI_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)


LLAMASERVER_BASE_URL = "http://localhost:8080/v1"


class LlamaServerOpenAIModel(Model):
    # Model values are the GGUF file name (or alias) as loaded by llama-server.
    # llama-server ignores the model field in API requests and always uses the loaded model;
    # these names are used for capability lookup only.
    LLAMA_3_1_8B = ModelSpec("llama-3.1-8b-instruct.gguf", tools=True)
    LLAMA_3_2_3B = ModelSpec("llama-3.2-3b-instruct.gguf", tools=True)
    MISTRAL_7B = ModelSpec("mistral-7b-instruct-v0.3.gguf", tools=True)
    PHI_4_MINI = ModelSpec("phi-4-mini-instruct.gguf", tools=True)
    QWEN_3_4B = ModelSpec("qwen3-4b.gguf", tools=True, thinking=True)
    QWEN_3_8B = ModelSpec("qwen3-8b.gguf", tools=True, thinking=True)
    DEEPSEEK_R1_7B = ModelSpec("deepseek-r1-distill-qwen-7b.gguf", thinking=True)
    GEMMA_3_12B = ModelSpec("gemma-3-12b-it.gguf", tools=True, vision=True)
    # Gemma 4 E4B/12B support audio natively, but llama-server (GGUF) has no audio-input path;
    # leave audio=False.
    GEMMA_4_E4B = ModelSpec("gemma-4-e4b-it.gguf", tools=True, thinking=True, vision=True)
    GEMMA_4_12B = ModelSpec("gemma-4-12b-it.gguf", tools=True, thinking=True, vision=True)
    GEMMA_4_26B = ModelSpec("gemma-4-26b-a4b-it.gguf", tools=True, thinking=True, vision=True)
    GEMMA_4_31B = ModelSpec("gemma-4-31b-it.gguf", tools=True, thinking=True, vision=True)


class LlamaServerOpenAIClient(OpenAICompatClient):
    """Client for llama.cpp's llama-server OpenAI-compatible REST API.

    Start the server with:
        llama-server -m /path/to/model.gguf --port 8080
    """

    MODELS = LlamaServerOpenAIModel

    def __init__(self, model: LlamaServerOpenAIModel, base_url: str = LLAMASERVER_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)


SGLANG_BASE_URL = "http://localhost:30000/v1"


class SGLangOpenAIModel(Model):
    # Model values are HuggingFace repo paths (as used by `python -m sglang.launch_server --model-path`).
    LLAMA_3_1_8B = ModelSpec("meta-llama/Llama-3.1-8B-Instruct", tools=True)
    LLAMA_3_2_3B = ModelSpec("meta-llama/Llama-3.2-3B-Instruct", tools=True)
    MISTRAL_7B = ModelSpec("mistralai/Mistral-7B-Instruct-v0.3", tools=True)
    PHI_4_MINI = ModelSpec("microsoft/Phi-4-mini-instruct", tools=True)
    QWEN_3_4B = ModelSpec("Qwen/Qwen3-4B", tools=True, thinking=True)
    QWEN_3_8B = ModelSpec("Qwen/Qwen3-8B", tools=True, thinking=True)
    DEEPSEEK_R1_7B = ModelSpec("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", thinking=True)
    GEMMA_3_12B = ModelSpec("google/gemma-3-12b-it", tools=True, vision=True)
    GEMMA_4_E4B = ModelSpec("google/gemma-4-E4B-it", tools=True, thinking=True, vision=True)
    GEMMA_4_12B = ModelSpec("google/gemma-4-12b-it", tools=True, thinking=True, vision=True)
    GEMMA_4_26B = ModelSpec("google/gemma-4-26B-A4B-it", tools=True, thinking=True, vision=True)
    GEMMA_4_31B = ModelSpec("google/gemma-4-31B-it", tools=True, thinking=True, vision=True)
    # Gemma 4 E4B/12B support audio natively; SGLang accepts input_audio blocks, but this path is
    # unverified against these weights, so leave audio=False until confirmed. (26B/31B: no native audio.)


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
    # model_client._BASE_URL_PROVIDERS). oMLX also accepts a `<model>:<profile>` alias form, e.g.
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
    QWEN_3_6_35B = ModelSpec("Qwen3.6-35B-A3B", tools=True, thinking=True, vision=True)
    QWEN_3_6_35B_4BIT = ModelSpec("Qwen3.6-35B-A3B-4bit", tools=True, thinking=True, vision=True)
    QWEN_3_6_35B_8BIT = ModelSpec("Qwen3.6-35B-A3B-8bit", tools=True, thinking=True, vision=True)
    QWEN_3_6_35B_BF16 = ModelSpec("Qwen3.6-35B-A3B-bf16", tools=True, thinking=True, vision=True)
    # Qwen 3.8 27B is a dense unified vision-language model (Qwen3_5ForConditionalGeneration, with
    # an image_token_id and an image-text-to-text pipeline tag), so the same tools/thinking/vision
    # reasoning as 3.6 above applies. mlx-community publishes 4bit, 8bit, and bf16; at 27B dense the
    # bf16 checkpoint is large but still tractable on a high-memory Mac, so it is listed here (it is
    # omitted from the LM Studio catalog, which lists only the two practical quants).
    QWEN_3_8_27B = ModelSpec("Qwen3.8-27B", tools=True, thinking=True, vision=True)
    QWEN_3_8_27B_4BIT = ModelSpec("Qwen3.8-27B-4bit", tools=True, thinking=True, vision=True)
    QWEN_3_8_27B_8BIT = ModelSpec("Qwen3.8-27B-8bit", tools=True, thinking=True, vision=True)
    QWEN_3_8_27B_BF16 = ModelSpec("Qwen3.8-27B-bf16", tools=True, thinking=True, vision=True)
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
    MUSE_GLIMMER_30B = ModelSpec("Muse-Glimmer-30B", tools=True, thinking=True, vision=True)
    MUSE_GLIMMER_30B_4BIT = ModelSpec("Muse-Glimmer-30B-4bit", tools=True, thinking=True, vision=True)
    MUSE_GLIMMER_30B_8BIT = ModelSpec("Muse-Glimmer-30B-8bit", tools=True, thinking=True, vision=True)
    MUSE_GLIMMER_30B_BF16 = ModelSpec("Muse-Glimmer-30B-bf16", tools=True, thinking=True, vision=True)
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
