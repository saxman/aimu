from ..base_client import StreamingContentType, Model, ModelClient, classproperty

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers.utils import logging as log
from transformers import TextIteratorStreamer
import gc
import pprint
import logging
from typing import Iterator, Optional, Any
from enum import Enum
import json
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

    XML = "<tool_call>"
    BRACKETED = "[TOOL_CALLS]"
    JSON_OBJECT = '{"name":'
    JSON_ARRAY = '[{"name":'

    def detected_in(self, text: str) -> bool:
        return self.value in text

    def parse(self, response: str) -> Optional[list[dict]]:
        """Extract tool calls from a full response. Returns None if the prefix is absent."""
        if not self.detected_in(response):
            return None
        if self == ToolCallPrefix.XML:
            start = response.index("<tool_call>") + len("<tool_call>")
            end = response.index("</tool_call>")
            return [json.loads(response[start:end].strip())]
        elif self == ToolCallPrefix.BRACKETED:
            start = response.index("[TOOL_CALLS]") + len("[TOOL_CALLS]")
            return json.loads(response[start:].strip())
        elif self == ToolCallPrefix.JSON_OBJECT:
            return [json.loads(response)]
        elif self == ToolCallPrefix.JSON_ARRAY:
            return [json.loads(response)[0]]
        return None


class HuggingFaceModel(Model):
    tool_call_prefix: Optional[ToolCallPrefix]
    generate_kwargs: dict[str, Any]

    def __init__(
        self,
        value,
        supports_tools=False,
        supports_thinking=False,
        tool_call_prefix=ToolCallPrefix.XML,
        generate_kwargs=None,
    ):
        super().__init__(value, supports_tools, supports_thinking)
        self.tool_call_prefix = tool_call_prefix
        self.generate_kwargs = DEFAULT_GENERATE_KWARGS.copy()
        self.generate_kwargs.update(generate_kwargs or {})

    GPT_OSS_20B = (
        "openai/gpt-oss-20b",
        True,
        True,
        ToolCallPrefix.XML,
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
        ToolCallPrefix.XML,
        {"temperature": 0.6},
    )
    GEMMA_3_12B = "google/gemma-3-12b-it"
    PHI_4_14B = "microsoft/phi-4"
    PHI_4_MINI_3_8B = "microsoft/Phi-4-mini-instruct"
    MISTRAL_7B = ("mistralai/Mistral-7B-Instruct-v0.3", True, False, ToolCallPrefix.JSON_ARRAY)
    MISTRAL_NEMO_12B = ("mistralai/Mistral-Nemo-Instruct-2407", True, False, ToolCallPrefix.JSON_ARRAY)
    MAGISTRAL_SMALL = ("mistralai/Magistral-Small-2509", True, False, ToolCallPrefix.BRACKETED)
    QWEN_3_5_9B = ("Qwen/Qwen3.5-9B", True, True)
    QWEN_3_8B = (
        "Qwen/Qwen3-8B",
        True,
        True,
        ToolCallPrefix.XML,
        {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
    )
    SMOLLM3_3B = ("HuggingFaceTB/SmolLM3-3B", True, True)


class HuggingFaceClient(ModelClient):
    MODELS = HuggingFaceModel

    DEFAULT_MODEL_KWARGS = {
        "device_map": "auto",
        "torch_dtype": "auto",
    }

    model: HuggingFaceModel

    def __init__(
        self,
        model: HuggingFaceModel,
        model_kwargs: Optional[dict] = None,
        system_message: Optional[str] = None,
    ):
        if model_kwargs is None:
            model_kwargs = self.DEFAULT_MODEL_KWARGS.copy()
        super().__init__(model, model_kwargs, system_message)

        if model == self.MODELS.MAGISTRAL_SMALL:
            from transformers import Mistral3ForConditionalGeneration

            self.hf_tokenizer = AutoTokenizer.from_pretrained(model.value, tokenizer_type="mistral")
            # device_map="auto" causes OOM on dual 4090 GPUs
            self.hf_model = Mistral3ForConditionalGeneration.from_pretrained(
                model.value, torch_dtype=torch.bfloat16, device_map="sequential"
            )
        elif model == self.MODELS.GPT_OSS_20B and torch.backends.mps.is_available():
            raise ValueError("GPT-OSS-20B is not supported on MPS devices.")
        else:
            ## TODO ensure we're using attention appropriately for single (flash_attention_2) vs multi GPU setups
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model.value)
            self.hf_model = AutoModelForCausalLM.from_pretrained(model.value, **model_kwargs)

    def __del__(self):
        del self.hf_model
        del self.hf_tokenizer

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

        return kwargs

    def _apply_chat_template(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Any:
        if self.model == self.MODELS.MAGISTRAL_SMALL:
            # ValueError: Kwargs ['add_generation_prompt', 'enable_thinking', 'xml_tools'] are not supported by `MistralCommonTokenizer.apply_chat_template`.
            text = self.hf_tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                tokenize=False,
                tools=tools,
            )
        elif (tools and len(tools) == 0) or not self.model.supports_tools:
            ## some models (e.g. Llama 3.1) misbehave if 'tools' is present, even if None
            text = self.hf_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
                enable_thinking=True,
            )
        else:
            text = self.hf_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
                enable_thinking=True,
                tools=tools,
                xml_tools=tools,
            )
        return self.hf_tokenizer([text], return_tensors="pt").to(self.hf_model.device)

    def _generate_sync(
        self,
        messages: list[dict],
        generate_kwargs: dict[str, Any],
        tools: Optional[list[dict]] = None,
    ) -> str:
        self.last_thinking = ""
        self._pending_thinking_tokens = []

        model_inputs = self._apply_chat_template(messages, tools)
        generated_ids = self.hf_model.generate(**model_inputs, **generate_kwargs)

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        response = self.hf_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        logger.debug("LLM raw response: %s", response)

        if self.model.supports_thinking and response.startswith("<think>"):
            self.last_thinking = response[len("<think>") : response.index("</think>")].strip()
            logger.debug("LLM thinking: %s", self.last_thinking)
            response = response[response.index("</think>") + len("</think>") :].strip()

        logger.debug("LLM final response: %s", response)

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
        self.hf_model.generate(**model_inputs, **generate_kwargs, streamer=streamer)

        # first part is always empty
        next(streamer)
        response_part = next(streamer)

        if self.model.supports_thinking and response_part.startswith("<think>"):
            while True:
                response_part = next(streamer)

                if "</think>" in response_part:
                    next(streamer)  # following </think> there's a newline
                    response_part = next(streamer)  # get the first valid token after thinking
                    break

                self.last_thinking += response_part
                self._pending_thinking_tokens.append(response_part)

            self.last_thinking = self.last_thinking.strip()

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
    ) -> Iterator[str]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        messages = [
            {"role": "user", "content": prompt},
        ]

        streamer = TextIteratorStreamer(self.hf_tokenizer, skip_prompt=True, skip_special_tokens=True)

        it = self._generate_streaming(messages, generate_kwargs, None, streamer)

        if include_thinking:
            for token in self._pending_thinking_tokens:
                self._streaming_content_type = StreamingContentType.THINKING
                yield token
        self._pending_thinking_tokens = []

        for token in it:
            logger.debug("LLM raw token: %r", token)
            self._streaming_content_type = StreamingContentType.GENERATING
            yield token

        self._streaming_content_type = StreamingContentType.DONE

    def chat(self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True) -> str:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        response = self._generate_sync(self.messages, generate_kwargs, tools)

        # TODO: handle multiple tool calls
        prefix = self.model.tool_call_prefix
        tool_calls = prefix.parse(response) if prefix else None
        if tool_calls:
            self._handle_tool_calls(tool_calls, tools)

            # assign thinking to the tool call (the second to last message, before the tool response)
            if self.last_thinking:
                self.messages[-2]["thinking"] = self.last_thinking

            response = self._generate_sync(self.messages, generate_kwargs, tools)

        self.messages.append({"role": "assistant", "content": response})

        if self.last_thinking:
            self.messages[-1]["thinking"] = self.last_thinking

        return response

    def chat_streamed(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True
    ) -> Iterator[str]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        streamer = TextIteratorStreamer(self.hf_tokenizer, skip_prompt=True, skip_special_tokens=False)
        it = self._generate_streaming(self.messages, generate_kwargs, tools, streamer)

        for token in self._pending_thinking_tokens:
            self._streaming_content_type = StreamingContentType.THINKING
            yield token
        self._pending_thinking_tokens = []

        response_part = next(it)
        content = ""
        msgs_before = len(self.messages)

        # TODO: handle multiple tool calls
        prefix = self.model.tool_call_prefix

        if prefix and prefix.detected_in(response_part):
            parts = [response_part]
            parts.extend(it)
            response = "".join(parts)

            tool_calls = prefix.parse(response)
            if tool_calls:
                self._handle_tool_calls(tool_calls, tools)

                if self.last_thinking:
                    self.messages[msgs_before]["thinking"] = self.last_thinking

                for tc, tr in zip(self.messages[msgs_before]["tool_calls"], self.messages[msgs_before + 1 :]):
                    self._streaming_content_type = StreamingContentType.TOOL_CALLING
                    yield json.dumps({"name": tc["function"]["name"], "response": tr["content"]})

                streamer = TextIteratorStreamer(self.hf_tokenizer, skip_prompt=True, skip_special_tokens=False)
                it = self._generate_streaming(self.messages, generate_kwargs, tools, streamer)
        else:
            content += response_part
            self._streaming_content_type = StreamingContentType.GENERATING
            yield response_part

        for token in self._pending_thinking_tokens:
            self._streaming_content_type = StreamingContentType.THINKING
            yield token
        self._pending_thinking_tokens = []

        eos_str = self.hf_tokenizer.decode(self.hf_model.config.eos_token_id)

        for response_part in it:
            if eos_str in response_part:
                response_part = response_part.replace(eos_str, "")

            logger.debug("LLM raw token: %r", response_part)
            content += response_part
            self._streaming_content_type = StreamingContentType.GENERATING
            yield response_part

        self.messages.append({"role": "assistant", "content": content})

        if self.last_thinking:
            self.messages[-1]["thinking"] = self.last_thinking

        self._streaming_content_type = StreamingContentType.DONE

    def print_model_info(self):
        print(f"model : size : {self.hf_model.get_memory_footprint() // 1024**2} MB")

        try:
            print(f"model : is quantized : {self.hf_model.is_quantized}")
            print(f"model : quantization method : {self.hf_model.quantization_method}")
        except AttributeError:
            print("model : is quantized : False")
            pass

        try:
            print(f"model : 8-bit quantized : {self.hf_model.is_loaded_in_8bit}")
        except AttributeError:
            pass

        try:
            print(f"model : 4-bit quantized : {self.hf_model.is_loaded_in_4bit}")
        except AttributeError:
            pass

        param = next(self.hf_model.parameters())
        print(f"model : on GPU (CUDA) : {param.is_cuda}")
        print(f"model : on GPU (MPS) : {param.is_mps}")

    def print_device_map(self):
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(self.hf_model.hf_device_map)
