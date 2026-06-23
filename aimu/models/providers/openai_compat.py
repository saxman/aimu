"""Generic OpenAI-compatible client base plus the local-inference-server subclasses.

``OpenAICompatClient`` speaks the OpenAI REST API against any ``base_url``. The
subclasses below are thin: each sets a default ``base_url`` and a ``Model`` enum of
server-appropriate ids. They cover *local* servers that merely expose an OpenAI-compatible
endpoint (Ollama, LM Studio, vLLM, HF-Serve, llama-server, SGLang).

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

from ..base import BaseModelClient, Model, ModelSpec, StreamChunk, StreamingContentType, classproperty
from .._internal.audio_input import _build_audio_content_blocks
from .._internal.image_input import _build_user_content_blocks
from .._internal.sdk_config import sdk_client_kwargs
from .._internal.usage import usage_from_openai
from ._thinking import _ThinkingParser, _split_thinking

logger = logging.getLogger(__name__)


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

        for chunk in stream:
            if getattr(chunk, "usage", None):
                self.last_usage = usage_from_openai(chunk)
            if not chunk.choices:  # terminal usage chunk (empty choices) or keep-alive
                continue
            delta = chunk.choices[0].delta
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
        response = self._client.chat.completions.create(
            model=self.model.value,
            messages=[{"role": "user", "content": content_in}],
            **generate_kwargs,
        )
        logger.debug("LLM raw response: %s", response)
        self.last_usage = usage_from_openai(response)
        content = response.choices[0].message.content or ""

        self.last_thinking = ""
        if self.is_thinking_model:
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
        stream = self._client.chat.completions.create(
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

        response = self._client.chat.completions.create(
            model=self.model.value,
            messages=self.messages,
            tools=tools if tools else openai.NOT_GIVEN,
            **generate_kwargs,
        )
        logger.debug("LLM raw response: %s", response)
        msg = response.choices[0].message

        if msg.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": json.loads(tc.function.arguments)} for tc in msg.tool_calls
            ]
            self._handle_tool_calls(tool_calls)

            response = self._client.chat.completions.create(
                model=self.model.value,
                messages=self.messages,
                tools=tools if tools else openai.NOT_GIVEN,
                **generate_kwargs,
            )
            logger.debug("LLM raw response (after tools): %s", response)
            msg = response.choices[0].message

        self.last_usage = usage_from_openai(response)
        content = msg.content or ""
        self.last_thinking = ""
        if self.is_thinking_model:
            self.last_thinking, content = _split_thinking(content)

        self.messages.append({"role": "assistant", "content": content})
        return content

    def _chat_streamed(self, generate_kwargs: dict[str, Any], tools: list) -> Iterator[StreamChunk]:
        stream = self._client.chat.completions.create(
            model=self.model.value,
            messages=self.messages,
            stream=True,
            stream_options={"include_usage": True},
            tools=tools if tools else openai.NOT_GIVEN,
            **generate_kwargs,
        )

        # Consume the first stream to detect tool calls vs. content (mutually exclusive in practice,
        # but we must buffer content since we can't yield before knowing whether tool calls follow).
        tool_calls_acc: dict[int, dict] = {}
        first_pass_chunks: list[StreamChunk] = []
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
                        first_pass_chunks.append(StreamChunk(phase, text))
                else:
                    first_pass_chunks.append(StreamChunk(StreamingContentType.GENERATING, delta.content))

        if not tool_calls_acc:
            full_content = ""
            for sc in first_pass_chunks:
                if sc.phase == StreamingContentType.GENERATING:
                    full_content += sc.content
                yield sc
            self.messages.append({"role": "assistant", "content": full_content})
            return

        # Tool call path: dispatch calls (yields chunks via streaming-tool support),
        # then stream second response.
        tool_calls = [{"name": tc["name"], "arguments": json.loads(tc["arguments"])} for tc in tool_calls_acc.values()]
        yield from self._handle_tool_calls_streamed(tool_calls)

        stream2 = self._client.chat.completions.create(
            model=self.model.value,
            messages=self.messages,
            stream=True,
            stream_options={"include_usage": True},
            tools=tools if tools else openai.NOT_GIVEN,
            **generate_kwargs,
        )

        full_content = ""
        for sc in self._iter_stream(stream2):
            if sc.phase == StreamingContentType.GENERATING:
                full_content += sc.content
            yield sc
        self.messages.append({"role": "assistant", "content": full_content})


# --------------------------------------------------------------------------------------
# Local OpenAI-compatible inference servers. Each subclass just supplies a default
# base_url and a Model enum of server-appropriate ids; all behaviour lives in the base.
# --------------------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaOpenAIModel(Model):
    # Model values are Ollama model tags (as used by `ollama pull`).
    LLAMA_3_1_8B = ModelSpec("llama3.1:8b")
    LLAMA_3_2_3B = ModelSpec("llama3.2:3b")
    MISTRAL_7B = ModelSpec("mistral:7b", tools=True)
    PHI_4_MINI = ModelSpec("phi4-mini:3.8b", tools=True)
    QWEN_3_4B = ModelSpec("qwen3:4b", tools=True, thinking=True)
    QWEN_3_8B = ModelSpec("qwen3:8b", tools=True, thinking=True)
    QWEN_3_5_9B = ModelSpec("qwen3.5:9b", tools=True, thinking=True)
    DEEPSEEK_R1_8B = ModelSpec("deepseek-r1:8b", thinking=True)
    GEMMA_3_12B = ModelSpec("gemma3:12b")
    GEMMA_4_12B = ModelSpec("gemma4:12b", tools=True)


class OllamaOpenAIClient(OpenAICompatClient):
    MODELS = OllamaOpenAIModel

    def __init__(self, model: OllamaOpenAIModel, base_url: str = OLLAMA_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)


LMSTUDIO_BASE_URL = "http://localhost:1234/v1"


class LMStudioOpenAIModel(Model):
    # Model values are the model "key" as shown in LM Studio's loaded model list.
    LLAMA_3_1_8B = ModelSpec("llama-3.1-8b-instruct")
    MISTRAL_7B = ModelSpec("mistral-7b-instruct-v0.3", tools=True)
    PHI_4_MINI = ModelSpec("phi-4-mini-instruct", tools=True)
    QWEN_3_4B = ModelSpec("qwen3-4b", tools=True, thinking=True)
    QWEN_3_8B = ModelSpec("qwen3-8b", tools=True, thinking=True)
    QWEN_3_5_9B = ModelSpec("qwen3.5-9b", tools=True, thinking=True)
    DEEPSEEK_R1_7B = ModelSpec("deepseek-r1-distill-qwen-7b", thinking=True)
    GEMMA_4_12B = ModelSpec("gemma-4-12b-it", tools=True)


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
    GEMMA_3_12B = ModelSpec("google/gemma-3-12b-it", tools=True)
    GEMMA_4_12B = ModelSpec("google/gemma-4-12b-it", tools=True)


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
    GEMMA_3_12B = ModelSpec("google/gemma-3-12b-it", tools=True)
    GEMMA_4_12B = ModelSpec("google/gemma-4-12b-it", tools=True)


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
    GEMMA_3_12B = ModelSpec("gemma-3-12b-it.gguf", tools=True)
    GEMMA_4_12B = ModelSpec("gemma-4-12b-it.gguf", tools=True)


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
    GEMMA_3_12B = ModelSpec("google/gemma-3-12b-it", tools=True)
    GEMMA_4_12B = ModelSpec("google/gemma-4-12b-it", tools=True)


class SGLangOpenAIClient(OpenAICompatClient):
    """Client for SGLang's OpenAI-compatible REST API.

    Start the server with:
        python -m sglang.launch_server --model-path <model> --port 30000
    """

    MODELS = SGLangOpenAIModel

    def __init__(self, model: SGLangOpenAIModel, base_url: str = SGLANG_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)
