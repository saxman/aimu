from ..base_client import Model, ModelClient

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers.utils import logging as log
from transformers import TextIteratorStreamer
import gc
import pprint
import logging
from typing import Iterator, Optional, Any
import json
import itertools

logger = logging.getLogger(__name__)
log.set_verbosity_error()


class HuggingFaceModel(Model):
    GPT_OSS_20B = "openai/gpt-oss-20b"

    LLAMA_3_1_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    LLAMA_3_2_3B = "unsloth/Llama-3.2-3B-Instruct"

    DEEPSEEK_R1_8B = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    GEMMA_3_12B = "google/gemma-3-12b-it"

    PHI_4_14B = "microsoft/phi-4"
    PHI_4_MINI_3_8B = "microsoft/Phi-4-mini-instruct"

    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
    MISTRAL_NEMO_12B = "mistralai/Mistral-Nemo-Instruct-2407"
    MISTRAL_SMALL_3_2_24B = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

    QWEN_3_8B = "Qwen/Qwen3-8B"

    SMOLLM3_3B = "HuggingFaceTB/SmolLM3-3B"


class HuggingFaceClient(ModelClient):
    MODELS = HuggingFaceModel

    TOOL_MODELS = [
        # MODELS.GPT_OSS_20B,
        MODELS.QWEN_3_8B,
        MODELS.MISTRAL_7B,
        MODELS.MISTRAL_NEMO_12B,
        MODELS.MISTRAL_SMALL_3_2_24B,
        # MODELS.LLAMA_3_1_8B,
        # MODELS.LLAMA_3_2_3B,
        # MODELS.SMOLLM3_3B,
    ]

    THINKING_MODELS = [
        MODELS.GPT_OSS_20B,
        MODELS.DEEPSEEK_R1_8B,
        MODELS.QWEN_3_8B,
        MODELS.SMOLLM3_3B,
    ]

    DEFAULT_MODEL_KWARGS = {
        "device_map": "auto",
        "torch_dtype": "auto",
    }

    DEFAULT_GENERATE_KWARGS = {
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "num_beams": 1,
    }

    DEFAULT_MODEL_GENERATE_KWARGS = {
        MODELS.GPT_OSS_20B.value: {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
        },
        MODELS.QWEN_3_8B.value: {
            ## thinking
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            ## TODO support non-thinking defaults
            # "temperature": 0.7,
            # "top_p": 0.8,
        },
        MODELS.DEEPSEEK_R1_8B.value: {
            "temperature": 0.6,
        },
    }

    def __init__(
        self,
        model: HuggingFaceModel,
        model_kwargs: dict = DEFAULT_MODEL_KWARGS.copy(),
        system_message: Optional[str] = None,
    ):
        super().__init__(model, model_kwargs, system_message)

        self.hf_tokenizer = AutoTokenizer.from_pretrained(model.value, attn_implementation="eager")
        self.hf_model = AutoModelForCausalLM.from_pretrained(model.value, **model_kwargs)

        self.default_generate_kwargs = self.DEFAULT_GENERATE_KWARGS.copy()

        if self.model in self.DEFAULT_MODEL_GENERATE_KWARGS:
            self.default_generate_kwargs.update(self.DEFAULT_MODEL_GENERATE_KWARGS[self.model.value])

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

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict[str, None]:
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs

        if "max_tokens" in generate_kwargs:
            generate_kwargs["max_new_tokens"] = generate_kwargs.pop("max_tokens")

        if "repeat_penalty" in generate_kwargs:
            generate_kwargs["repetition_penalty"] = generate_kwargs.pop("repeat_penalty")

        return generate_kwargs

    def _generate(
        self,
        messages: list[dict],
        generate_kwargs: dict[str, Any],
        tools: Optional[list[dict]] = None,
        streamer: Optional[TextIteratorStreamer] = None,
    ) -> tuple[str, Iterator[str]]:
        if (tools and len(tools) == 0) or self.model not in self.TOOL_MODELS:
            ## some models (e.g. Llama 3.1) misbehave is tools is present, event if None
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

        self.last_thinking = ""

        model_inputs = self.hf_tokenizer([text], return_tensors="pt").to(self.hf_model.device)
        generated_ids = self.hf_model.generate(**model_inputs, **generate_kwargs, streamer=streamer)
        # TODO: fix the EOS token being returned. Streamer has skip_special_tokens, and have tried eos_token_id=self.hf_model.config.eos_token_id for generator

        if streamer is None:
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
            response = self.hf_tokenizer.decode(output_ids, skip_special_tokens=True)

            if self.model in self.THINKING_MODELS and response.startswith("<think>"):
                self.last_thinking = response[len("<think>") : response.index("</think>")].strip()
                response = response[response.index("</think>") + len("</think>") :].strip()

            return response, iter([])

        # first part is always empty
        next(streamer)
        response_part = next(streamer)

        if self.model in self.THINKING_MODELS and response_part.startswith("<think>"):
            while True:
                response_part = next(streamer)

                if "</think>" in response_part:
                    next(streamer) # following </think> there's a newline
                    response_part = next(streamer) # get the first valid token after thinking
                    break

                self.last_thinking += response_part

            self.last_thinking = self.last_thinking.strip()
                
        return "", itertools.chain([response_part], streamer)

    def generate(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        messages = [
            {"role": "user", "content": prompt},
        ]

        response, _ = self._generate(messages, generate_kwargs)

        return response

    def generate_streamed(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[str]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        messages = [
            {"role": "user", "content": prompt},
        ]

        streamer = TextIteratorStreamer(self.hf_tokenizer, skip_prompt=True, skip_special_tokens=True)

        _, it = self._generate(messages, generate_kwargs, streamer=streamer)

        return it

    def chat(self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True) -> str:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        response, _ = self._generate(self.messages, generate_kwargs, tools)

        # TODO: handle multiple tool calls
        if self.model in [self.MODELS.QWEN_3_8B, self.MODELS.SMOLLM3_3B, self.MODELS.GPT_OSS_20B]:
            if "<tool_call>" in response:
                tool_calls = []
                tool_call_start = response.index("<tool_call>") + len("<tool_call>")
                tool_call_end = response.index("</tool_call>")
                tool_call = response[tool_call_start:tool_call_end].strip()
                tool_calls.append(json.loads(tool_call))

                response = response[tool_call_end + len("</tool_call>") :].strip()

                self._handle_tool_calls(tool_calls, tools)

                # assign thinking to the tool call (the second to last message, before the tool response)
                if self.last_thinking:
                    self.messages[-2]["thinking"] = self.last_thinking

                response, _ = self._generate(self.messages, generate_kwargs, tools)
        elif self.model in [self.MODELS.MISTRAL_SMALL_3_2_24B]:
            if "[TOOL_CALLS]" in response:
                self._handle_tool_calls(json.loads(response), tools)
                response, _ = self._generate(self.messages, generate_kwargs, tools)
        elif self.model in [self.MODELS.LLAMA_3_1_8B, self.MODELS.LLAMA_3_2_3B]:
            if response.startswith('{"name":'):
                self._handle_tool_calls([json.loads(response)], tools)
                response, _ = self._generate(self.messages, generate_kwargs, tools)
        elif self.model in [self.MODELS.MISTRAL_7B, self.MODELS.MISTRAL_NEMO_12B]:
            if response.startswith('[{"name":'):
                self._handle_tool_calls([json.loads(response)[0]], tools)
                response, _ = self._generate(self.messages, generate_kwargs, tools)

        self.messages.append({"role": "assistant", "content": response})

        if self.last_thinking:
            self.messages[-1]["thinking"] = self.last_thinking

        return response

    def chat_streamed(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True
    ) -> Iterator[str]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        streamer = TextIteratorStreamer(self.hf_tokenizer, skip_prompt=True, skip_special_tokens=False)

        _, it = self._generate(self.messages, generate_kwargs, tools, streamer=streamer)

        response_part = next(it)
        content = ""

        # TODO: handle multiple tool calls
        if self.model in [self.MODELS.QWEN_3_8B, self.MODELS.SMOLLM3_3B] and "<tool_call>" in response_part:
            # buffer the entire tool call
            response = response_part
            for response_part in it:
                response += response_part

            tool_calls = []
            tool_call_start = response.index("<tool_call>") + len("<tool_call>")
            tool_call_end = response.index("</tool_call>")
            tool_call = response[tool_call_start:tool_call_end].strip()
            tool_calls.append(json.loads(tool_call))

            response = response[tool_call_end + len("</tool_call>") :].strip()

            self._handle_tool_calls(tool_calls, tools)

            if self.last_thinking:
                self.messages[-2]["thinking"] = self.last_thinking

            streamer = TextIteratorStreamer(self.hf_tokenizer, skip_prompt=True, skip_special_tokens=False)

            _, it = self._generate(self.messages, generate_kwargs, tools, streamer=streamer)
        elif self.model in [self.MODELS.MISTRAL_SMALL_3_2_24B] and "[TOOL_CALLS]" in response_part:
            response = response_part
            for response_part in it:
                response += response_part

            tool_calls_start = response.index("[TOOL_CALLS]") + len("[TOOL_CALLS]")
            tool_calls = response[tool_calls_start:].strip()

            self._handle_tool_calls(json.loads(tool_calls), tools)
            _, it = self._generate(self.messages, generate_kwargs, tools, streamer=streamer)
        elif self.model in [self.MODELS.LLAMA_3_1_8B, self.MODELS.LLAMA_3_2_3B] and '{"name":' in response_part:
            response = response_part
            for response_part in it:
                response += response_part

            self._handle_tool_calls([json.loads(response)], tools)
            _, it = self._generate(self.messages, generate_kwargs, tools)
        elif self.model in [self.MODELS.MISTRAL_7B, self.MODELS.MISTRAL_NEMO_12B] and '[{"name":' in response_part:
            response = response_part
            for response_part in it:
                response += response_part

            self._handle_tool_calls([json.loads(response)[0]], tools)
            _, it = self._generate(self.messages, generate_kwargs, tools)
        else:
            content += response_part
            yield response_part

        eos_str = self.hf_tokenizer.decode(self.hf_model.config.eos_token_id)

        for response_part in it:
            if eos_str in response_part:
                response_part = response_part.replace(eos_str, "")

            content += response_part
            yield response_part

        self.messages.append({"role": "assistant", "content": content})

        if self.last_thinking:
            self.messages[-1]["thinking"] = self.last_thinking

        return

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
