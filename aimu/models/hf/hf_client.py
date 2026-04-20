from ..base import StreamingContentType, StreamChunk, Model, ModelClient, classproperty

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from transformers.utils import logging as log
from transformers import TextIteratorStreamer
import gc
import pprint
import logging
from typing import Iterator, Optional, Any
from enum import Enum
import json
import re
import itertools

logger = logging.getLogger(__name__)
log.set_verbosity_error()

DEFAULT_GENERATE_KWARGS = {
    "max_new_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "num_beams": 1,
}


class ToolCallPrefix(Enum):
    """The prefix string that identifies a tool call in a model's raw response.

    The enum value is the literal prefix used both for detection (via `in`) and
    as the anchor for parsing.
    """

    XML_JSON = "<tool_call>"  # <tool_call>{"name": ..., "arguments": ...}</tool_call>
    XML_QWEN = "<function="  # <tool_call><function=NAME><parameter=KEY>VAL</parameter>...</function></tool_call>
    BRACKETED = "[TOOL_CALLS]"
    JSON_OBJECT = '{"name":'
    JSON_ARRAY = '[{"name":'
    NA = ""

    def detected_in(self, text: str) -> bool:
        return self.value in text

    def parse(self, response: str) -> Optional[list[dict]]:
        """Extract tool calls from a full response. Returns None if the prefix is absent."""
        if not self.detected_in(response):
            return None
        if self == ToolCallPrefix.XML_JSON:
            matches = re.findall(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
            return [json.loads(m.strip()) for m in matches] if matches else None
        elif self == ToolCallPrefix.XML_QWEN:
            matches = re.findall(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
            if not matches:
                return None
            tool_calls = []
            for m in matches:
                content = m.strip()
                name_match = re.search(r"<function=([^>]+)>", content)
                if not name_match:
                    continue
                arguments = {}
                for param_match in re.finditer(
                    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", content, re.DOTALL
                ):
                    key = param_match.group(1).strip()
                    val = param_match.group(2).strip()
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    arguments[key] = val
                tool_calls.append({"name": name_match.group(1).strip(), "arguments": arguments})
            return tool_calls if tool_calls else None
        elif self == ToolCallPrefix.BRACKETED:
            start = response.index("[TOOL_CALLS]") + len("[TOOL_CALLS]")
            return json.loads(response[start:].strip())
        elif self == ToolCallPrefix.JSON_OBJECT:
            return [json.loads(response)]
        elif self == ToolCallPrefix.JSON_ARRAY:
            return json.loads(response)
        return None


class HuggingFaceModel(Model):
    tool_call_prefix: Optional[ToolCallPrefix]
    generate_kwargs: dict[str, Any]
    think_opener_in_prompt: bool

    def __init__(
        self,
        value,
        supports_tools=False,
        supports_thinking=False,
        tool_call_prefix=ToolCallPrefix.XML_JSON,
        generate_kwargs=None,
        think_opener_in_prompt=False,
    ):
        super().__init__(value, supports_tools, supports_thinking)
        self.tool_call_prefix = tool_call_prefix
        self.generate_kwargs = DEFAULT_GENERATE_KWARGS.copy()
        self.generate_kwargs.update(generate_kwargs or {})
        # Qwen 3.5's chat template appends <think>\n to the generation prompt,
        # so the model generates starting inside the thinking block and only
        # emits the closing </think>. Set True for such models so streaming
        # thinking extraction can anchor on </think> instead of a leading
        # <think> token that never appears in the stream.
        self.think_opener_in_prompt = think_opener_in_prompt

    GPT_OSS_20B = (
        "openai/gpt-oss-20b",
        True,
        True,
        ToolCallPrefix.XML_JSON,
        {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
    )
    LLAMA_3_1_8B = ("meta-llama/Meta-Llama-3.1-8B-Instruct", True, False, ToolCallPrefix.JSON_OBJECT)
    LLAMA_3_2_3B = (
        "unsloth/Llama-3.2-3B-Instruct",
        True,
        False,
        ToolCallPrefix.JSON_OBJECT,
    )  # using unsloth's version since gated model
    DEEPSEEK_R1_8B = (
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        False,
        True,
        ToolCallPrefix.XML_JSON,
        {"temperature": 0.6},
    )
    GEMMA_3_12B = "google/gemma-3-12b-it"
    GEMMA_4_E4B = (
        "google/gemma-4-E4B-it",
        True,
        False,
        ToolCallPrefix.NA,
        {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    )
    PHI_4_14B = "microsoft/phi-4"
    PHI_4_MINI_3_8B = "microsoft/Phi-4-mini-instruct"
    MISTRAL_7B = ("mistralai/Mistral-7B-Instruct-v0.3", True, False, ToolCallPrefix.JSON_ARRAY)
    MISTRAL_NEMO_12B = ("mistralai/Mistral-Nemo-Instruct-2407", True, False, ToolCallPrefix.JSON_ARRAY)
    MAGISTRAL_SMALL = ("mistralai/Magistral-Small-2509", True, False, ToolCallPrefix.BRACKETED)
    QWEN_3_5_9B = (
        "Qwen/Qwen3.5-9B",
        True,
        True,
        ToolCallPrefix.XML_QWEN,
        {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        },
        True,  # think_opener_in_prompt
    )
    QWEN_3_8B = (
        "Qwen/Qwen3-8B",
        True,
        True,
        ToolCallPrefix.XML_JSON,
        {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
    )
    SMOLLM3_3B = ("HuggingFaceTB/SmolLM3-3B", True, True)


class HuggingFaceClient(ModelClient):
    MODELS = HuggingFaceModel

    DEFAULT_MODEL_KWARGS = {
        "device_map": "auto",
        "torch_dtype": "auto",
    }

    model: HuggingFaceModel # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        model: HuggingFaceModel,
        model_kwargs: Optional[dict] = None,
        system_message: Optional[str] = None,
    ):
        if model_kwargs is None:
            model_kwargs = self.DEFAULT_MODEL_KWARGS.copy()
        super().__init__(model, model_kwargs, system_message)

        self._hf_processor = None
        self._parsed_tool_calls = None

        if model == self.MODELS.MAGISTRAL_SMALL:
            from transformers import Mistral3ForConditionalGeneration

            self._hf_tokenizer = AutoTokenizer.from_pretrained(model.value, tokenizer_type="mistral")
            
            # device_map="auto" causes OOM on dual 4090 GPUs
            self._hf_model = Mistral3ForConditionalGeneration.from_pretrained(
                model.value, torch_dtype=torch.bfloat16, device_map="sequential"
            )
        elif model == self.MODELS.GPT_OSS_20B and torch.backends.mps.is_available():
            raise ValueError("GPT-OSS-20B is not supported on MPS devices.")
        elif model.value.startswith("google/gemma-4") or model.value.startswith("qwen/qwen3.5"):
            self._hf_processor = AutoProcessor.from_pretrained(model.value)
            self._hf_tokenizer = self._hf_processor.tokenizer
            self._hf_model = AutoModelForCausalLM.from_pretrained(model.value, **model_kwargs)
        else:
            ## TODO ensure we're using attention appropriately for single (flash_attention_2) vs multi GPU setups
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model.value)
            self._hf_model = AutoModelForCausalLM.from_pretrained(model.value, **model_kwargs)

    def __del__(self):
        del self._hf_model
        del self._hf_tokenizer
        if self._hf_processor is not None:
            del self._hf_processor

        gc.collect()

        if torch.cuda.is_available():
            logger.info("Emptying CUDA cache")
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            logger.info("Emptying MPS cache")
            torch.mps.empty_cache()

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_thinking]

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_tools]

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if not generate_kwargs:
            kwargs = self.model.generate_kwargs.copy()
        else:
            kwargs = generate_kwargs.copy()

        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")

        if "repeat_penalty" in kwargs:
            kwargs["repetition_penalty"] = kwargs.pop("repeat_penalty")

        # presence_penalty is an OpenAI API concept; Transformers' generate() does not
        # accept it and raises a ValueError. Drop it silently.
        kwargs.pop("presence_penalty", None)

        return kwargs

    def _apply_chat_template(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Any:
        if self._hf_processor is not None:
            text = self._hf_processor.apply_chat_template(
                messages,
                tools=tools if self.model.supports_tools else None,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.model.supports_thinking,
            )
            return self._hf_processor(text=text, return_tensors="pt").to(self._hf_model.device)
        if self.model == self.MODELS.MAGISTRAL_SMALL:
            # ValueError: Kwargs ['add_generation_prompt', 'enable_thinking', 'xml_tools'] are not supported by `MistralCommonTokenizer.apply_chat_template`.
            text = self._hf_tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                tokenize=False,
                tools=tools,
            )
        elif (tools and len(tools) == 0) or not self.model.supports_tools:
            ## some models (e.g. Llama 3.1) misbehave if 'tools' is present, even if None
            text = self._hf_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
                enable_thinking=True,
            )
        else:
            text = self._hf_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
                enable_thinking=True,
                tools=tools,
                xml_tools=tools,
            )
        return self._hf_tokenizer([text], return_tensors="pt").to(self._hf_model.device)

    def _generate_sync(
        self,
        messages: list[dict],
        generate_kwargs: dict[str, Any],
        tools: Optional[list[dict]] = None,
    ) -> str:
        self.last_thinking = ""
        self._pending_thinking_tokens = []
        self._parsed_tool_calls = None

        model_inputs = self._apply_chat_template(messages, tools)
        generated_ids = self._hf_model.generate(**model_inputs, **generate_kwargs)

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]

        if self._hf_processor is not None:
            raw = self._hf_processor.decode(output_ids, skip_special_tokens=False)
            logger.debug("[raw] processor response: %s", raw)
            parsed = self._hf_processor.parse_response(raw)
            self.last_thinking = (parsed.get("thinking") or "").strip()
            if self.last_thinking:
                logger.debug("[thinking] %s", self.last_thinking)

            tool_calls = parsed.get("tool_calls") or []
            if tool_calls:
                # parse_response returns the OpenAI function-calling envelope;
                # unwrap to the flat {"name": ..., "arguments": ...} form that
                # _handle_tool_calls expects (matches every other client).
                self._parsed_tool_calls = [
                    {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                    for tc in tool_calls
                ]
                logger.debug("[tool_call] parsed: %s", self._parsed_tool_calls)
                response = ""
            else:
                self._parsed_tool_calls = None
                # parse_response leaves EOS in content when decoded with
                # skip_special_tokens=False; use a clean tokenizer decode for
                # the user-facing text.
                response = self._hf_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            logger.debug("[generating] %s", response)
            return response

        response = self._hf_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        logger.debug("[raw] response: %s", response)

        if (
            self.model.supports_thinking
            and (response.startswith("<think>") or self.model.think_opener_in_prompt)
            and "</think>" in response
        ):
            # For models where the <think> opener is injected into the prompt
            # (Qwen 3.5), the response starts with thinking content directly;
            # otherwise skip the explicit opener. Anchor the split on </think>.
            think_end = response.index("</think>")
            think_start = len("<think>") if response.startswith("<think>") else 0
            self.last_thinking = response[think_start:think_end].strip()
            logger.debug("[thinking] %s", self.last_thinking)
            response = response[think_end + len("</think>") :].strip()

        logger.debug("[generating] %s", response)

        return response

    def _generate_streaming(
        self,
        messages: list[dict],
        generate_kwargs: dict[str, Any],
        tools: Optional[list[dict]],
        streamer: TextIteratorStreamer,
    ) -> Iterator[str]:
        self.last_thinking = ""
        self._pending_thinking_tokens = []

        model_inputs = self._apply_chat_template(messages, tools)
        self._hf_model.generate(**model_inputs, **generate_kwargs, streamer=streamer)

        # first part is always empty
        next(streamer)
        response_part = next(streamer)
        logger.debug("[raw] first token: %r", response_part)

        if self.model.supports_thinking and (
            response_part.startswith("<think>") or self.model.think_opener_in_prompt
        ):
            # Advance past the explicit <think> token. For models where the
            # opener is injected into the generation prompt (Qwen 3.5), the
            # first streamed token is already thinking content.
            #
            # Wrap in try/except: a thinking model can hit max_new_tokens
            # before emitting </think>. Treat that as "all thinking, no
            # content" rather than crashing the generator with StopIteration.
            try:
                if response_part.startswith("<think>"):
                    response_part = next(streamer)

                while "</think>" not in response_part:
                    logger.debug("[thinking] token: %r", response_part)
                    self.last_thinking += response_part
                    self._pending_thinking_tokens.append(response_part)
                    response_part = next(streamer)

                next(streamer)  # newline after </think>
                response_part = next(streamer)  # first content token
                logger.debug("[generating] first token (post-thinking): %r", response_part)
            except StopIteration:
                logger.debug("[thinking] stream exhausted before </think>")
                response_part = ""

            self.last_thinking = self.last_thinking.strip()
            logger.debug("[thinking] collected: %s", self.last_thinking)

        return itertools.chain([response_part], streamer)

    def generate(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        messages = [
            {"role": "user", "content": prompt},
        ]

        return self._generate_sync(messages, generate_kwargs)

    def generate_streamed(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        include_thinking: bool = True,
    ) -> Iterator[StreamChunk]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        messages = [
            {"role": "user", "content": prompt},
        ]

        streamer = TextIteratorStreamer(self._hf_tokenizer, skip_prompt=True, skip_special_tokens=True)

        it = self._generate_streaming(messages, generate_kwargs, None, streamer)

        if include_thinking:
            for token in self._pending_thinking_tokens:
                yield StreamChunk(StreamingContentType.THINKING, token)
        self._pending_thinking_tokens = []

        for token in it:
            logger.debug("[generating] token: %r", token)
            yield StreamChunk(StreamingContentType.GENERATING, token)

    def chat(self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True) -> str:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        response = self._generate_sync(self.messages, generate_kwargs, tools)

        if self._hf_processor is not None:
            tool_calls = self._parsed_tool_calls
        else:
            prefix = self.model.tool_call_prefix
            tool_calls = prefix.parse(response) if prefix else None
        if tool_calls:
            logger.debug("[tool_call] parsed: %s", tool_calls)
            msgs_before = len(self.messages)
            self._handle_tool_calls(tool_calls, tools)

            if self.last_thinking:
                self.messages[msgs_before]["thinking"] = self.last_thinking

            response = self._generate_sync(self.messages, generate_kwargs, tools)

        logger.debug("[generating] chat response: %s", response)
        self.messages.append({"role": "assistant", "content": response})

        if self.last_thinking:
            self.messages[-1]["thinking"] = self.last_thinking

        return response

    def chat_streamed(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True
    ) -> Iterator[StreamChunk]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        if self._hf_processor is not None:
            # Gemma 4 uses special tokens (<|channel>, <|tool_call>, ...) and
            # requires processor.parse_response for structured extraction.
            # Token-level streaming with ToolCallPrefix doesn't apply here, so
            # route through _generate_sync and yield the parsed result.
            yield from self._chat_streamed_via_processor(generate_kwargs, tools)
            return

        streamer = TextIteratorStreamer(self._hf_tokenizer, skip_prompt=True, skip_special_tokens=False)
        it = self._generate_streaming(self.messages, generate_kwargs, tools, streamer)

        for token in self._pending_thinking_tokens:
            yield StreamChunk(StreamingContentType.THINKING, token)
        self._pending_thinking_tokens = []

        response_part = next(it)
        content = ""
        msgs_before = len(self.messages)

        prefix = self.model.tool_call_prefix

        if prefix and prefix.detected_in(response_part):
            parts = [response_part]
            parts.extend(it)
            response = "".join(parts)
            logger.debug("[tool_call] raw: %s", response)

            tool_calls = prefix.parse(response)
            if tool_calls:
                logger.debug("[tool_call] parsed: %s", tool_calls)
                self._handle_tool_calls(tool_calls, tools)

                if self.last_thinking:
                    self.messages[msgs_before]["thinking"] = self.last_thinking

                for tc, tr in zip(self.messages[msgs_before]["tool_calls"], self.messages[msgs_before + 1 :]):
                    logger.debug(
                        "[tool_call] response: name=%s, response=%s",
                        tc["function"]["name"],
                        tr["content"],
                    )
                    yield StreamChunk(
                        StreamingContentType.TOOL_CALLING,
                        {"name": tc["function"]["name"], "response": tr["content"]},
                    )

                streamer = TextIteratorStreamer(self._hf_tokenizer, skip_prompt=True, skip_special_tokens=False)
                it = self._generate_streaming(self.messages, generate_kwargs, tools, streamer)
        else:
            content += response_part
            logger.debug("[generating] token: %r", response_part)
            yield StreamChunk(StreamingContentType.GENERATING, response_part)

        for token in self._pending_thinking_tokens:
            logger.debug("[thinking] token: %r", token)
            yield StreamChunk(StreamingContentType.THINKING, token)
        self._pending_thinking_tokens = []

        eos_str = self._hf_tokenizer.decode(self._hf_model.config.eos_token_id)

        for response_part in it:
            if eos_str in response_part:
                response_part = response_part.replace(eos_str, "")

            logger.debug("[generating] token: %r", response_part)
            content += response_part
            yield StreamChunk(StreamingContentType.GENERATING, response_part)

        logger.debug("[generating] collected: %s", content)
        self.messages.append({"role": "assistant", "content": content})

        if self.last_thinking:
            self.messages[-1]["thinking"] = self.last_thinking

    def _chat_streamed_via_processor(
        self, generate_kwargs: dict[str, Any], tools: Optional[list[dict]]
    ) -> Iterator[StreamChunk]:
        """Buffered streaming for models that require processor.parse_response (Gemma 4)."""
        response = self._generate_sync(self.messages, generate_kwargs, tools)

        if self._parsed_tool_calls:
            logger.debug("[tool_call] parsed: %s", self._parsed_tool_calls)
            msgs_before = len(self.messages)
            self._handle_tool_calls(self._parsed_tool_calls, tools)

            if self.last_thinking:
                self.messages[msgs_before]["thinking"] = self.last_thinking

            for tc, tr in zip(self.messages[msgs_before]["tool_calls"], self.messages[msgs_before + 1 :]):
                logger.debug(
                    "[tool_call] response: name=%s, response=%s",
                    tc["function"]["name"],
                    tr["content"],
                )
                yield StreamChunk(
                    StreamingContentType.TOOL_CALLING,
                    {"name": tc["function"]["name"], "response": tr["content"]},
                )

            response = self._generate_sync(self.messages, generate_kwargs, tools)

        if self.last_thinking:
            logger.debug("[thinking] collected: %s", self.last_thinking)
            yield StreamChunk(StreamingContentType.THINKING, self.last_thinking)

        if response:
            logger.debug("[generating] collected: %s", response)
            yield StreamChunk(StreamingContentType.GENERATING, response)

        self.messages.append({"role": "assistant", "content": response})

        if self.last_thinking:
            self.messages[-1]["thinking"] = self.last_thinking

    def print_model_info(self):
        print(f"model : size : {self._hf_model.get_memory_footprint() // 1024**2} MB")

        try:
            print(f"model : is quantized : {self._hf_model.is_quantized}")
            print(f"model : quantization method : {self._hf_model.quantization_method}")
        except AttributeError:
            print("model : is quantized : False")
            pass

        try:
            print(f"model : 8-bit quantized : {self._hf_model.is_loaded_in_8bit}")
        except AttributeError:
            pass

        try:
            print(f"model : 4-bit quantized : {self._hf_model.is_loaded_in_4bit}")
        except AttributeError:
            pass

        param = next(self._hf_model.parameters())
        print(f"model : on GPU (CUDA) : {param.is_cuda}")
        print(f"model : on GPU (MPS) : {param.is_mps}")

    def print_device_map(self):
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(self._hf_model.hf_device_map)
