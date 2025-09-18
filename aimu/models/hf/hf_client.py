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

logger = logging.getLogger(__name__)
log.set_verbosity_error()


class HuggingFaceModel(Model):
    GPT_OSS_20B = "openai/gpt-oss-20b"

    LLAMA_3_1_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"

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
        MODELS.GPT_OSS_20B,
        MODELS.QWEN_3_8B,
        MODELS.MISTRAL_7B,
        MODELS.MISTRAL_NEMO_12B,
        MODELS.MISTRAL_SMALL_3_2_24B,
        MODELS.LLAMA_3_1_8B,
        # MODELS.LLAMA_3_2_3B,
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

    # TODO: add more models and enable use by clients
    DEFAULT_MODEL_TEMPERATURES = {
        MODELS.MISTRAL_NEMO_12B: 0.3,
    }

    DEFAULT_GENERATE_KWARGS = {
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 50,
        ## gpt-oss recommended
        # "temperature": 1.0,
        # "top_p": 1.0,
        # "top_k": 0,
        ## qwen3 recommended (thinking)
        # "temperature": 0.6,
        # "top_p": 0.95,
        # "top_k": 20,
        # "min_p": 0,
        ## qwen3 recommended (non-thinking)
        # "temperature": 0.7,
        # "top_p": 0.8,
        # "top_k": 20,
        # "min_p": 0,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "num_beams": 1,
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

    def generate(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        messages = [
            {"role": "user", "content": prompt},
        ]

        text = self.hf_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", tokenize=False
        )

        model_inputs = self.hf_tokenizer([text], return_tensors="pt").to(self.hf_model.device)

        generated_ids = self.hf_model.generate(
            **model_inputs,
            **generate_kwargs,
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        response = self.hf_tokenizer.decode(output_ids, skip_special_tokens=True)

        self.last_thinking = ""
        if self.model in self.THINKING_MODELS and response.startswith("<think>"):
            self.last_thinking = response[len("<think>") : response.index("</think>")].strip()
            response = response[response.index("</think>") + len("</think>") :].strip()

        return response

    def generate_streamed(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[str]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        messages = [
            {"role": "user", "content": prompt},
        ]

        text = self.hf_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", tokenize=False
        )

        model_inputs = self.hf_tokenizer([text], return_tensors="pt").to(self.hf_model.device)

        streamer = TextIteratorStreamer(self.hf_tokenizer, skip_prompt=True, skip_special_tokens=False)

        self.hf_model.generate(**model_inputs, **generate_kwargs, streamer=streamer)

        next(streamer)  # first part is always empty
        response_part = next(streamer)

        self.last_thinking = ""
        if self.model in self.THINKING_MODELS and response_part.startswith("<think>"):
            while True:
                response_part = next(streamer)
                if "</think>" in response_part:
                    response_part = next(streamer)
                    break
                self.last_thinking += response_part

            self.last_thinking = self.last_thinking.strip()

        while True:
            yield response_part
            response_part = next(streamer)
            if response_part == "":
                break

    def __generate_raw(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        model_inputs = self.hf_tokenizer([prompt], return_tensors="pt").to(self.hf_model.device)
        output_tokens = self.hf_model.generate(**model_inputs, **generate_kwargs)

        results = self.hf_tokenizer.decode(output_tokens[0])

        return results

    def _generate_streamed_raw(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[str]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        streamer = TextIteratorStreamer(self.hf_tokenizer, skip_special_tokens=True)

        model_inputs = self.hf_tokenizer([prompt], return_tensors="pt").to(self.hf_model.device)
        _ = self.hf_model.generate(**model_inputs, **generate_kwargs, streamer=streamer)

        for response_part in streamer:
            yield response_part

    def _chat_generate(
        self, generate_kwargs: dict[str, Any], tools: list[dict], streamer: Optional[TextIteratorStreamer] = None
    ) -> Optional[str]:
        input_tokens = self.hf_tokenizer.apply_chat_template(
            self.messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.hf_model.device)

        output_tokens = self.hf_model.generate(**input_tokens, **generate_kwargs, streamer=streamer)

        if streamer is None:
            response = self.hf_tokenizer.decode(
                output_tokens[0][len(input_tokens["input_ids"][0]) :], skip_special_tokens=False
            )

            return response

        return

    def chat(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: Optional[bool] = True
    ) -> str:
        generate_kwargs = self._chat_setup(user_message, generate_kwargs)

        tools = []
        if use_tools and self.mcp_client:
            tools = self.mcp_client.get_tools()

        response = self._chat_generate(generate_kwargs, tools) or ""

        eos_token = self.hf_tokenizer.eos_token if self.hf_tokenizer.eos_token else ""

        thinking = ""
        if self.model in self.THINKING_MODELS and response.startswith("<think>"):
            thinking = response[len("<think>") : response.index("</think>")]
            response = response[response.index("</think>") + len("</think>") :].strip()

        # <tool_call>{"name": "get_current_weather", "arguments": {"location": "Paris"}}</tool_call>
        if self.model in [self.MODELS.QWEN_3_8B, self.MODELS.SMOLLM3_3B]:
            if response.startswith("<tool_call>"):
                tool_calls = []
                while "<tool_call>" in response:
                    tool_call_start = response.index("<tool_call>") + len("<tool_call>")
                    tool_call_end = response.index("</tool_call>")
                    tool_call = response[tool_call_start:tool_call_end].strip()

                    tool_calls.append(json.loads(tool_call))

                    response = response[tool_call_end + len("</tool_call>") :].strip()

                self._handle_tool_calls(tool_calls, tools)

                if thinking:
                    self.messages[-2]["thinking"] = thinking
                    thinking = ""

                response = self._chat_generate(generate_kwargs, tools) or ""

                if self.hf_model in self.THINKING_MODELS and response.startswith("<think>"):
                    thinking = response[len("<think>") : response.index("</think>")]
                    response = response[response.index("</think>") + len("</think>") :].strip()

            response = response[: -len(eos_token)]

        # [TOOL_CALLS] [{"name": "get_current_temperature", "arguments": {"location": "Paris"}}]
        elif self.model in [self.MODELS.MISTRAL_7B, self.MODELS.MISTRAL_NEMO_12B, self.MODELS.MISTRAL_SMALL_3_2_24B]:
            if "[TOOL_CALLS]" in response:
                tool_calls_start = response.index("[TOOL_CALLS]") + len("[TOOL_CALLS]")
                tools_calls_end = response.index(eos_token)
                tool_calls = response[tool_calls_start:tools_calls_end].strip()

                self._handle_tool_calls(json.loads(tool_calls), tools)
                response = self._chat_generate(generate_kwargs, tools) or ""

            response = response[: -len(eos_token)].strip()

        # {"name": "get_current_weather", "arguments": {"location": "Paris"}}
        elif self.model in [self.MODELS.LLAMA_3_1_8B]:  # TODO re-add self.MODELS.LLAMA_3_2_3B
            if response.startswith('{"name":'):
                self._handle_tool_calls([json.loads(response)], tools)
                response = self._chat_generate(generate_kwargs, tools) or ""

            response = response[: -len(eos_token)].strip()

        self.messages.append({"role": "assistant", "content": response})

        if thinking:
            self.messages[-1]["thinking"] = thinking

        return response

    def chat_streamed(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: Optional[bool] = True
    ) -> Iterator[str]:
        generate_kwargs = self._chat_setup(user_message, generate_kwargs)

        if not hasattr(self, "streamer"):
            self.streamer = TextIteratorStreamer(self.hf_tokenizer, skip_prompt=True, skip_special_tokens=False)

        tools = []
        if use_tools and self.mcp_client:
            tools = self.mcp_client.get_tools()

        self._chat_generate(generate_kwargs, tools, streamer=self.streamer)

        content = ""
        eos_token = self.hf_tokenizer.eos_token if self.hf_tokenizer.eos_token else ""

        next(self.streamer)
        response_part = next(self.streamer)

        thinking = ""
        if response_part.startswith("<think>"):
            while True:
                response_part = next(self.streamer)
                if "</think>" in response_part:
                    next(self.streamer)
                    response_part = next(self.streamer)
                    break
                thinking += response_part

        if self.model in [self.MODELS.QWEN_3_8B, self.MODELS.SMOLLM3_3B]:
            if "<tool_call>" in response_part:
                response = response_part
                for response_part in self.streamer:
                    response += response_part

                tool_calls = []
                while "<tool_call>" in response:
                    tool_call_start = response.index("<tool_call>") + len("<tool_call>")
                    tool_call_end = response.index("</tool_call>")
                    tool_call = response[tool_call_start:tool_call_end].strip()

                    tool_calls.append(json.loads(tool_call))

                    response = response[tool_call_end + len("</tool_call>") :].strip()

                self._handle_tool_calls(tool_calls, tools)

                if thinking:
                    self.messages[-2]["thinking"] = thinking
                    thinking = ""

                self._chat_generate(generate_kwargs, tools, streamer=self.streamer)
        elif self.model in [self.MODELS.MISTRAL_7B, self.MODELS.MISTRAL_NEMO_12B, self.MODELS.MISTRAL_SMALL_3_2_24B]:
            next(self.streamer)  # first part is always empty
            response_part = next(self.streamer)

            if "[TOOL_CALLS]" in response_part:
                response = response_part
                for response_part in self.streamer:
                    response += response_part

                tool_calls_start = response.index("[TOOL_CALLS]") + len("[TOOL_CALLS]")
                tool_call_end = response.index(eos_token)
                tool_calls = response[tool_calls_start:tool_call_end].strip()

                self._handle_tool_calls(json.loads(tool_calls), tools)
                self._chat_generate(generate_kwargs, tools, streamer=self.streamer)
            else:
                content = response_part
                yield content
        elif self.model in [self.MODELS.LLAMA_3_1_8B]:  # TODO re-add self.MODELS.LLAMA_3_2_3B
            pass  # TODO: handle tool calls for LLAMA models

        next(self.streamer)
        response_part = next(self.streamer)

        if response_part.startswith("<think>"):
            while True:
                response_part = next(self.streamer)
                if "</think>" in response_part:
                    next(self.streamer)
                    response_part = next(self.streamer)
                    break
                thinking += response_part
        else:
            content += response_part
            yield content

        for response_part in self.streamer:
            if response_part.endswith(eos_token):
                response_part = response_part[: -len(eos_token)]
            content += response_part
            yield response_part

        self.messages.append({"role": "assistant", "content": content})

        if thinking:
            self.messages[-1]["thinking"] = thinking

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
