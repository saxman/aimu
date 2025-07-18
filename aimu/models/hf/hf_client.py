from ..models import ModelClient

import gc
import pprint
import logging
from typing import Iterator
import json

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoTokenizer
    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as log
    from transformers import TextIteratorStreamer

    log.set_verbosity_error()
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    TextIteratorStreamer = None


class HuggingFaceClient(ModelClient):
    MODEL_LLAMA_3_1_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MODEL_LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"

    MODEL_DEEPSEEK_R1_8B = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    MODEL_GEMMA_3_12B = "google/gemma-3-12b-it"

    MODEL_PHI_4_14B = "microsoft/phi-4"
    MODEL_PHI_4_MINI_3_8B = "microsoft/Phi-4-mini-instruct"

    MODEL_MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
    MODEL_MISTRAL_NEMO_12B = "mistralai/Mistral-Nemo-Instruct-2407"
    MODEL_MISTRAL_SMALL_3_2_24B = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

    MODEL_QWEN_3_8B = "Qwen/Qwen3-8B"

    MODEL_SMOLLM3_3B = "HuggingFaceTB/SmolLM3-3B"

    TOOL_MODELS = [
        MODEL_MISTRAL_SMALL_3_2_24B,  ## Potential tokenizer issue with this model
        MODEL_QWEN_3_8B,
        MODEL_LLAMA_3_2_3B,
        MODEL_SMOLLM3_3B,
    ]

    THINKING_MODELS = [MODEL_QWEN_3_8B, MODEL_DEEPSEEK_R1_8B, MODEL_SMOLLM3_3B]

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

    def __init__(self, model_id: str, model_kwargs: dict = DEFAULT_MODEL_KWARGS.copy(), system_message: str = None):
        super().__init__(model_id, model_kwargs, system_message)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, attn_implementation="eager")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    def __del__(self):
        del self.model
        del self.tokenizer

        gc.collect()

        if torch.cuda.is_available():
            logger.info("Emptying CUDA cache")
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            logger.info("Emptying MPS cache")
            torch.mps.empty_cache()

    def _update_generate_kwargs(self, generate_kwargs) -> None:
        if "max_tokens" in generate_kwargs:
            generate_kwargs["max_new_tokens"] = generate_kwargs.pop("max_tokens")

        if "repeat_penalty" in generate_kwargs:
            generate_kwargs["repetition_penalty"] = generate_kwargs.pop("repeat_penalty")

        return generate_kwargs

    def generate(self, prompt: str, generate_kwargs: dict = DEFAULT_GENERATE_KWARGS.copy()) -> str:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, **generate_kwargs)

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def generate_streamed(self, prompt: str, generate_kwargs: dict = DEFAULT_GENERATE_KWARGS.copy()) -> Iterator[str]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        self.model.generate(**model_inputs, **generate_kwargs, streamer=streamer)

        for response_part in streamer:
            yield response_part

    def _chat(self, user_message: str, generate_kwargs, tools: dict = None) -> None:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        if tools and self.model_id not in self.TOOL_MODELS:
            raise ValueError(
                f"Tool usage is not supported for model {self.model_id}. Supported models: {self.TOOL_MODELS}"
            )

        # Add system message if it's the first user message and system_message is set
        if len(self.messages) == 0 and self.system_message:
            self.messages.append({"role": "system", "content": self.system_message}) ## TODO: not all models support system messages

        # Add user message
        self.messages.append({"role": "user", "content": user_message})

        return

    def _handle_tool_calls(self, tools_calls: list, tools: dict) -> None:
        message = {"role": "assistant"}
        self.messages.append(message)

        message["tool_calls"] = []
        for tool_call in tools_calls:
            message["tool_calls"].append({"type": "function", "function": tool_call, "id": "123456789"})

            for tool in tools:
                if tool.__name__ == tool_call["name"]:
                    tool_response = tool(**tool_call["arguments"])
                    self.messages.append(
                        {
                            "role": "tool",
                            "name": tool_call["name"],
                            "content": str(tool_response),
                            "tool_call_id": "123456789",
                        }
                    )

                    logger.debug(f"Tool call response: {self.messages[-1]}")

                    break

        logger.debug(f"Tool calls message: {message}")

        return

    def _chat_generate(self, generate_kwargs: dict, tools: dict, streamer: TextIteratorStreamer = None):
        input_tokens = self.tokenizer.apply_chat_template(
            self.messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        output_tokens = self.model.generate(**input_tokens, **generate_kwargs, streamer=streamer)

        if streamer is None:
            response = self.tokenizer.decode(
                output_tokens[0][len(input_tokens["input_ids"][0]) :], skip_special_tokens=False
            )

            return response

        return

    def chat(self, user_message: str, generate_kwargs: dict = DEFAULT_GENERATE_KWARGS.copy(), tools: dict = None) -> str:
        self._chat(user_message, generate_kwargs, tools)

        response = self._chat_generate(generate_kwargs, tools)
        logger.debug(f"Response: {response}")

        # qwen: <tool_call>{"name": "get_current_weather", "arguments": {"location": "Paris"}}</tool_call>
        if "qwen" in self.model_id:
            think_start = len("<think>")
            think_end = response.index("</think>")
            think = response[think_start:think_end]  # TODO: do something with think

            response = response[think_end + len("</think>") :].strip()

            if "<tool_call>" in response:
                tool_calls = []
                while "<tool_call>" in response:
                    tool_call_start = response.index("<tool_call>") + len("<tool_call>")
                    tool_call_end = response.index("</tool_call>")
                    tool_call = response[tool_call_start:tool_call_end].strip()

                    tool_calls.append(json.loads(tool_call))

                    response = response[tool_call_end + len("</tool_call>") :].strip()

                self._handle_tool_calls(tool_calls, tools)

                response = self._chat_generate(generate_kwargs, tools)

            response = response[: -len("<|im_end|>")]

        # mistral: [TOOL_CALLS] [{"name": "get_current_temperature", "arguments": {"location": "Paris"}}]</s>
        elif "mistral" in self.model_id:
            if "[TOOL_CALLS]" in response:
                tool_calls_start = response.index("[TOOL_CALLS]") + len("[TOOL_CALLS]")
                tools_calls_end = response.index("</s>")
                tool_calls = response[tool_calls_start:tools_calls_end].strip()

                self._handle_tool_calls(json.loads(tool_calls), tools)
                response = self._chat_generate(generate_kwargs, tools)

            response = response[: -len("</s>")].strip()

        # llama: {"name": "get_current_weather", "arguments": {"location": "Paris"}}
        elif "llama" in self.model_id:
            if response.startswith('{"name":'):
                self._handle_tool_calls([json.loads(response)], tools)
                response = self._chat_generate(generate_kwargs, tools)

            response = response[: -len("<|eot_id|>")].strip()

        self.messages.append({"role": "assistant", "content": response})

        return response

    def chat_streamed(self, user_message: str, generate_kwargs: dict = DEFAULT_GENERATE_KWARGS.copy(), tools: dict = None) -> Iterator[str]:
        self._chat(user_message, generate_kwargs, tools)

        if not hasattr(self, "streamer"):
            self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)

        self._chat_generate(generate_kwargs, tools, streamer=self.streamer)

        content = ""
        eos = "" # TODO: set eos based on tokenizer

        # If the model is Qwen, the we can't stream the response since we need to process the entire response to detect tool calls
        # TODO: figure out how to stream non-tool call responses for Qwen models
        # TODO: combine this with other thinking models
        if "qwen" in self.model_id:
            response = ""
            for response_part in self.streamer:
                response += response_part

            think_start = len("<think>")
            think_end = response.index("</think>")
            think = response[think_start:think_end]  # TODO: do something with think

            response = response[think_end + len("</think>") :].strip()

            eos = "<|im_end|>"

            if "<tool_call>" in response:
                tool_calls = []
                while "<tool_call>" in response:
                    tool_call_start = response.index("<tool_call>") + len("<tool_call>")
                    tool_call_end = response.index("</tool_call>")
                    tool_call = response[tool_call_start:tool_call_end].strip()

                    tool_calls.append(json.loads(tool_call))

                    response = response[tool_call_end + len("</tool_call>") :].strip()

                self._handle_tool_calls(tool_calls, tools)
                self._chat_generate(generate_kwargs, tools, streamer=self.streamer)
            else:
                # If there isn't a tool call, we've already processed the response, so we can yield the entire response and return
                response = response[: -len(eos)].strip()
                self.messages.append({"role": "assistant", "content": response})
                yield response
                return
        elif "mistral" in self.model_id:
            next(self.streamer)  # first part is always empty
            response_part = next(self.streamer)

            eos = "</s>"

            if "[TOOL_CALLS]" in response_part:
                response = response_part
                for response_part in self.streamer:
                    response += response_part

                tool_calls_start = response.index("[TOOL_CALLS]") + len("[TOOL_CALLS]")
                tool_call_end = response.index("</s>")
                tool_calls = response[tool_calls_start:tool_call_end].strip()

                self._handle_tool_calls(json.loads(tool_calls), tools)
                self._chat_generate(generate_kwargs, tools, streamer=self.streamer)
            else:
                content = response_part
                yield content
        elif "llama" in self.model_id:
            # TODO: handle tool calls for Llama models
            eos = "<|eot_id|>"
        elif "phi" in self.model_id:
            # TODO: handle tool calls for Phi models
            eos = "<|im_sep|>"  # of "<|im_end|>"

        for response_part in self.streamer:
            if response_part.endswith(eos):
                response_part = response_part[: -len(eos)]
            content += response_part
            yield response_part

        self.messages.append({"role": "assistant", "content": content})

        return

    def print_model_info(self):
        print(f"model : size : {self.model.get_memory_footprint() // 1024**2} MB")

        try:
            print(f"model : is quantized : {self.model.is_quantized}")
            print(f"model : quantization method : {self.model.quantization_method}")
        except AttributeError:
            print("model : is quantized : False")
            pass

        try:
            print(f"model : 8-bit quantized : {self.model.is_loaded_in_8bit}")
        except AttributeError:
            pass

        try:
            print(f"model : 4-bit quantized : {self.model.is_loaded_in_4bit}")
        except AttributeError:
            pass

        param = next(self.model.parameters())
        print(f"model : on GPU (CUDA) : {param.is_cuda}")
        print(f"model : on GPU (MPS) : {param.is_mps}")

    def print_device_map(self):
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(self.model.hf_device_map)
