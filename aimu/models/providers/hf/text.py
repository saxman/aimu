from ...base import StreamingContentType, StreamChunk, Model, BaseModelClient, classproperty
from ..._catalog import Wire
from ..._internal.audio_input import _extract_audio_arrays, _replace_audio_with_placeholder
from ..._internal.generate_kwargs import Unsupported
from ..._internal.image_input import (
    _build_user_content_blocks,
    _extract_pil_images,
    _replace_image_url_with_image_placeholder,
)
from ..._internal.thinking import QWEN_REASONING_EFFORT, ResolvedThinking, pop_thinking

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from transformers.utils import logging as log
from transformers import TextIteratorStreamer
from transformers import Mistral3ForConditionalGeneration
from transformers import BitsAndBytesConfig
import gc
import pprint
import logging
import threading
from typing import Iterator, Optional, Any, Union
from enum import Enum
import json
import re
import itertools

logger = logging.getLogger(__name__)
log.set_verbosity_error()

# Module-level registry so multiple HuggingFaceClient instances with the same
# model share weights rather than loading them independently. Each value is a
# tuple of (model, tokenizer, processor, uses_processor_parse_response).
_model_registry: dict[tuple, tuple] = {}
_registry_lock = threading.Lock()


def _load_profile(model: "HuggingFaceModel") -> str:
    """Tag identifying which __init__ loader branch a model uses.

    Two enum members can share a repo id and model_kwargs yet load via different
    classes (e.g. AutoModelForCausalLM vs AutoModelForImageTextToText). Folding this
    tag into the weight-cache key keeps those from colliding, while members that load
    identically (the text-only and _VL Qwen entries, which share one multimodal
    checkpoint) still share a single cached load. Mirrors how the image/audio/speech
    clients key on pipeline_class / pipeline_type.
    """
    mid = model.value
    if model.name == "MAGISTRAL_SMALL":
        return "mistral3"
    if mid.startswith("google/gemma-4"):
        return "gemma4"
    if mid.startswith("google/gemma-3"):
        return "gemma3"
    if mid.startswith(("Qwen/Qwen3.5", "Qwen/Qwen3.6", "Qwen/Qwen3.8")):
        return "qwen-multimodal"
    return "causal-lm"


def _make_cache_key(spec_id: str, load_profile: str, model_kwargs: dict | None) -> tuple:
    return (spec_id, load_profile, *sorted((k, str(v)) for k, v in (model_kwargs or {}).items()))


DEFAULT_GENERATE_KWARGS = {
    "max_new_tokens": 4096,  # high default to avoid cutting off tool calls or thinking; users can override with generation_kwargs
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "num_beams": 1,
}


class ToolCallFormat(Enum):
    """The prefix string that identifies a tool call in a model's raw response.

    The enum value is the literal prefix used both for detection (via `in`) and
    as the anchor for parsing.
    """

    XML = "<tool_call>"  # <tool_call>{"name": ..., "arguments": ...}</tool_call>
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
        if self == ToolCallFormat.XML:
            matches = re.findall(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
            if not matches:
                return None
            result = []
            for m in matches:
                m = m.strip()
                if "<function=" in m:
                    # Qwen-style: <function=name><parameter=key>value</parameter></function>
                    name_m = re.search(r"<function=(\w+)>", m)
                    args = {k: v.strip() for k, v in re.findall(r"<parameter=(\w+)>(.*?)</parameter>", m, re.DOTALL)}
                    result.append({"name": name_m.group(1), "arguments": args})
                else:
                    result.append(json.loads(m))
            return result
        elif self == ToolCallFormat.BRACKETED:
            start = response.index("[TOOL_CALLS]") + len("[TOOL_CALLS]")
            return json.loads(response[start:].strip())
        elif self == ToolCallFormat.JSON_OBJECT:
            return [json.loads(response)]
        elif self == ToolCallFormat.JSON_ARRAY:
            return json.loads(response)
        return None

    def strip_calls(self, response: str) -> str:
        """Return the natural-language text in ``response`` with the tool-call markup removed.

        Models can emit prose alongside a tool call in one generation; this recovers that prose
        so it can be stored as the assistant message's ``content``. XML keeps the text outside
        ``<tool_call>...</tool_call>``; BRACKETED keeps the text before ``[TOOL_CALLS]``; the
        JSON-only formats are the whole tool call, so there is no accompanying prose.
        """
        if self == ToolCallFormat.XML:
            return re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL).strip()
        if self == ToolCallFormat.BRACKETED:
            return response.split("[TOOL_CALLS]", 1)[0].strip()
        return ""


class HuggingFaceModel(Model):
    """HuggingFace Transformers model catalog.

    Each member's value is a ``(Wire, ToolCallFormat, think_opener_in_prompt)`` tuple (or a
    bare ``Wire`` when the tuple's defaults apply). ``Model.__init__`` resolves the ``Wire``
    against the shared ``MODEL_FACTS`` table (``aimu/models/_catalog.py``) into the
    ``ModelSpec`` that actually carries the capability flags. ``think_opener_in_prompt=True``
    for models like Qwen 3.5 whose chat template appends ``<think>\\n`` to the prompt; the
    model generates *inside* the thinking block and only emits the closing ``</think>``.
    """

    def __init__(
        self,
        spec,
        tool_call_format: ToolCallFormat = ToolCallFormat.NA,
        think_opener_in_prompt: bool = False,
    ):
        super().__init__(spec)
        self.tool_call_format = tool_call_format
        self.think_opener_in_prompt = think_opener_in_prompt

    # Alibaba
    # Qwen 3.8 27B is dense (not MoE) but otherwise the same shape as the 3.6 pair below: one
    # repo holding language model + vision tower, XML tool calls, and a chat template that
    # appends the `<think>` opener to the prompt -- hence think_opener_in_prompt=True, matching
    # Qwen 3.5. FP8 is a separate member for the same hardware-gating reason as 3.6-FP8.
    # thinking_levels=True: Qwen/Qwen3.8-27B's chat_template.jinja reads a `reasoning_effort`
    # kwarg (default "xhigh", validated against {"xhigh", "medium", "low"}), verified 2026-08-17.
    QWEN_3_8_27B = (Wire("Qwen/Qwen3.8-27B"), ToolCallFormat.XML, True)
    QWEN_3_8_27B_FP8 = (Wire("Qwen/Qwen3.8-27B-FP8"), ToolCallFormat.XML, True)
    # Qwen 3.5/3.6/3.8 are natively multimodal (unified vision-language foundation): one HF
    # repo holds both the language model and the vision tower, loaded together via
    # AutoModelForImageTextToText (see __init__ -- the FP8 checkpoint's quant skip-list
    # only matches that module tree). The vision encoder loads regardless, so each model
    # is a single vision=True entry; there is no separate text-only flavor to maintain.
    # The bare member is the unquantized checkpoint, so the enum name describes the model rather
    # than a serving choice, and it stays interchangeable with OllamaModel.QWEN_3_6_27B (which the
    # cross-provider consistency guard links by name). FP8 is a separate member because it is a
    # deliberate, hardware-gated choice, not a default: the checkpoint is e4m3 with dynamic
    # activation scaling, which needs Ada/Hopper-class tensor cores (compute capability >= 8.9), so
    # on Ampere, MPS, or CPU it has no native path. Naming it makes that visible at the call site
    # instead of hiding it in the repo id. Both ids must stay distinct: two members sharing a
    # Wire.id silently alias (Wire's equality/hash are id-only, same as ModelSpec; see
    # tests/test_model_catalog_consistency.py).
    # Both carry think_opener_in_prompt=True: 3.6's chat template appends the bare `<think>` opener
    # exactly as 3.5's and 3.8's do (the tails are byte-identical in that respect).
    QWEN_3_6_27B = (Wire("Qwen/Qwen3.6-27B"), ToolCallFormat.XML, True)
    QWEN_3_6_27B_FP8 = (Wire("Qwen/Qwen3.6-27B-FP8"), ToolCallFormat.XML, True)
    QWEN_3_5_9B = (Wire("Qwen/Qwen3.5-9B"), ToolCallFormat.XML, True)
    QWEN_3_8B = (Wire("Qwen/Qwen3-8B"), ToolCallFormat.XML)

    # Google: Gemma 4 is natively multimodal (vision + audio input) and a thinking model.
    # thinking=True matches every server catalog for these ids; the HF chat-template path
    # emits reasoning the same way as the other thinking models here (no think_opener needed).
    GEMMA_4_E4B = (Wire("google/gemma-4-E4B-it", audio=True), ToolCallFormat.NA)
    GEMMA_4_12B = (Wire("google/gemma-4-12b-it", audio=True), ToolCallFormat.NA)
    # tools=False here is deliberate, not an oversight: the in-process HF client parses
    # tool calls via a ToolCallFormat, and Gemma 3 has no format assigned (ToolCallFormat.NA),
    # so it cannot surface tool calls on this path. OpenAI-compat servers (vLLM/SGLang/etc.)
    # parse tool calls server-side, so they set tools=True for the same model. See
    # tests/test_model_catalog_consistency.py.
    GEMMA_3_12B = Wire("google/gemma-3-12b-it")

    # NVIDIA: Nemotron-H is the multimodal Hybrid series (Mamba + Transformer)
    NEMOTRON_H_8B = (Wire("nvidia/Nemotron-H-8B-Instruct-HF", audio=True), ToolCallFormat.XML)

    # OpenAI
    GPT_OSS_20B = (Wire("openai/gpt-oss-20b"), ToolCallFormat.XML)

    # Mistral
    MAGISTRAL_SMALL = (Wire("mistralai/Magistral-Small-2509"), ToolCallFormat.BRACKETED)
    MISTRAL_NEMO_12B = (Wire("mistralai/Mistral-Nemo-Instruct-2407"), ToolCallFormat.JSON_ARRAY)
    MISTRAL_7B = (Wire("mistralai/Mistral-7B-Instruct-v0.3"), ToolCallFormat.JSON_ARRAY)

    # Microsoft
    PHI_4_MINI_3_8B = Wire("microsoft/Phi-4-mini-instruct")
    PHI_4_14B = Wire("microsoft/phi-4")

    # DeepSeek
    # R1's template appends "<|Assistant|><think>\n" to the generation prompt unconditionally,
    # so like the Qwen thinking models the response starts *inside* the thinking block.
    DEEPSEEK_R1_8B = (Wire("deepseek-ai/DeepSeek-R1-Distill-Llama-8B"), ToolCallFormat.XML, True)

    # HuggingFace
    SMOLLM3_3B = (Wire("HuggingFaceTB/SmolLM3-3B"), ToolCallFormat.XML)

    # Meta: Llama 3.2 uses unsloth's repo because the official one is gated
    LLAMA_3_2_3B = (Wire("unsloth/Llama-3.2-3B-Instruct"), ToolCallFormat.JSON_OBJECT)
    LLAMA_3_1_8B = (Wire("meta-llama/Meta-Llama-3.1-8B-Instruct"), ToolCallFormat.JSON_OBJECT)


# Transformers' generate() accepts min_p and repetition_penalty but has no presence_penalty concept
# (it raises a ValueError on one), and spells the output cap max_new_tokens.
HF_GENERATE_KWARGS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "presence_penalty": Unsupported(
        "Transformers' generate() has no presence_penalty. Use repetition_penalty, its nearest "
        "equivalent, or run the model behind an OpenAI-compatible server that accepts the parameter."
    ),
    "repetition_penalty": "repetition_penalty",
    "max_tokens": "max_new_tokens",
    "context_length": Unsupported("The window is fixed by the model's own max_position_embeddings."),
}


class HuggingFaceClient(BaseModelClient):
    MODELS = HuggingFaceModel

    GENERATE_KWARG_SUPPORT = HF_GENERATE_KWARGS

    # The module constant is the class's kwarg-fallback tier (see _resolve_generate_kwargs). It
    # stays a module global too, since the catalog profiles above are built by merging into it.
    DEFAULT_GENERATE_KWARGS = DEFAULT_GENERATE_KWARGS

    DEFAULT_MODEL_KWARGS = {
        "device_map": "auto",
        "torch_dtype": "auto",
    }

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
        self._parsed_content = ""  # prose emitted alongside a tool call (processor path)
        # Gemma 4 emits structured-token output (channels, tool calls) that requires
        # processor.parse_response() to extract. Other processor-loaded models (Gemma 3,
        # Qwen 3.5/3.6/3.8) emit standard <think>...</think> + tool-call text and use the
        # same parsing path as non-processor models.
        self._uses_processor_parse_response = False

        self._cache_key = _make_cache_key(model.value, _load_profile(model), model_kwargs)
        with _registry_lock:
            if self._cache_key in _model_registry:
                cached = _model_registry[self._cache_key]
                self._hf_model, self._hf_tokenizer, self._hf_processor, self._uses_processor_parse_response = cached
                return

        if model == self.MODELS.MAGISTRAL_SMALL:
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model.value, tokenizer_type="mistral")

            bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
            self._hf_model = Mistral3ForConditionalGeneration.from_pretrained(
                model.value, quantization_config=bnb_config, device_map="balanced"
            )
        elif model == self.MODELS.GPT_OSS_20B and torch.backends.mps.is_available():
            raise ValueError("GPT-OSS-20B is not supported on MPS devices.")
        elif model.value.startswith("google/gemma-4"):
            self._hf_processor = AutoProcessor.from_pretrained(model.value)
            self._hf_tokenizer = self._hf_processor.tokenizer
            self._hf_model = AutoModelForCausalLM.from_pretrained(model.value, **model_kwargs)
            self._uses_processor_parse_response = True
        elif model.value.startswith("google/gemma-3"):
            self._hf_processor = AutoProcessor.from_pretrained(model.value)
            self._hf_tokenizer = self._hf_processor.tokenizer
            self._hf_model = AutoModelForCausalLM.from_pretrained(model.value, **model_kwargs)
        elif model.value.startswith(("Qwen/Qwen3.5", "Qwen/Qwen3.6", "Qwen/Qwen3.8")):
            # Qwen 3.5/3.6/3.8 are unified multimodal checkpoints and must ALWAYS load via
            # AutoModelForImageTextToText -- even for the text-only enum members (which
            # simply leave the vision tower unused; chat(images=...) is gated separately
            # by supports_vision). AutoModelForCausalLM is wrong here even when we don't
            # need vision: it builds a text-only module tree (model.layers.*), but the
            # FP8 checkpoint's quantization_config.modules_to_not_convert is written
            # against the multimodal tree (model.language_model.* / model.visual.*). Under
            # the causal-LM tree those skip-rules match nothing, so layers meant to stay
            # bf16 (router mlp.gate, lm_head, embed_tokens, linear_attn projections)
            # mis-quantize and crash with "mat1 and mat2 to have the same dtype, but got
            # BFloat16 != Float8_e4m3fn". The ImageTextToText tree matches the skip-list.
            self._hf_processor = AutoProcessor.from_pretrained(model.value)
            self._hf_tokenizer = self._hf_processor.tokenizer
            self._hf_model = AutoModelForImageTextToText.from_pretrained(model.value, **model_kwargs)
        else:
            ## TODO ensure we're using attention appropriately for single (flash_attention_2) vs multi GPU setups
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model.value)
            self._hf_model = AutoModelForCausalLM.from_pretrained(model.value, **model_kwargs)

        with _registry_lock:
            _model_registry[self._cache_key] = (
                self._hf_model,
                self._hf_tokenizer,
                self._hf_processor,
                self._uses_processor_parse_response,
            )

    def __del__(self):
        # If the registry still holds a reference, deleting this client's attributes
        # won't free the weights, so skip the gc/cache flush too.
        cache_key = getattr(self, "_cache_key", None)
        if cache_key is not None and cache_key in _model_registry:
            return

        for attr in ("_hf_model", "_hf_tokenizer", "_hf_processor"):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

        gc.collect()

        if torch.cuda.is_available():
            logger.info("Emptying CUDA cache")
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            logger.info("Emptying MPS cache")
            torch.mps.empty_cache()

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:
        return [m for m in cls.MODELS if m.supports_thinking]

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:
        return [m for m in cls.MODELS if m.supports_tools]

    @classproperty
    def VISION_MODELS(cls) -> list[Model]:
        return [m for m in cls.MODELS if m.supports_vision]

    @classproperty
    def AUDIO_MODELS(cls) -> list[Model]:
        return [m for m in cls.MODELS if m.supports_audio]

    def _rewrite_generate_kwargs(self, kwargs: dict) -> dict:
        # A caller who wrote Ollama's spelling still gets the parameter Transformers wants. The reverse
        # direction of the same rename is declared in HF_GENERATE_KWARGS, which runs before this hook.
        if "repeat_penalty" in kwargs:
            kwargs["repetition_penalty"] = kwargs.pop("repeat_penalty")

        return kwargs

    def _apply_chat_template(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        thinking: Optional[ResolvedThinking] = None,
    ) -> Any:
        enable_thinking = self.model.supports_thinking if thinking is None else thinking.enabled
        template_extra: dict[str, Any] = {}
        if thinking is not None and thinking.level is not None and self.model.thinking_levels:
            template_extra["reasoning_effort"] = QWEN_REASONING_EFFORT[thinking.level]

        if self._hf_processor is not None:
            pil_images = _extract_pil_images(messages) if self.model.supports_vision else []
            audio_arrays = _extract_audio_arrays(messages) if self.model.supports_audio else []
            template_messages = _replace_image_url_with_image_placeholder(messages) if pil_images else messages
            template_messages = (
                _replace_audio_with_placeholder(template_messages) if audio_arrays else template_messages
            )
            text = self._hf_processor.apply_chat_template(
                template_messages,
                tools=tools if self.model.supports_tools else None,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **template_extra,
            )
            processor_kwargs = {"text": text, "return_tensors": "pt"}
            if pil_images:
                processor_kwargs["images"] = pil_images
            if audio_arrays:
                processor_kwargs["audio"] = audio_arrays
                processor_kwargs["sampling_rate"] = 16000
            return self._hf_processor(**processor_kwargs).to(self._hf_model.device)
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
                enable_thinking=enable_thinking,
                **template_extra,
            )
        else:
            text = self._hf_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
                enable_thinking=enable_thinking,
                tools=tools,
                xml_tools=tools,
                **template_extra,
            )
        return self._hf_tokenizer([text], return_tensors="pt").to(self._hf_model.device)

    def _generate_sync(
        self,
        messages: list[dict],
        generate_kwargs: dict[str, Any],
        tools: Optional[list[dict]] = None,
    ) -> str:
        self.last_thinking = None
        self._pending_thinking_tokens = []
        self._parsed_tool_calls = None

        # Popped here, not peeked: this dict is splatted into generate() below, which
        # rejects unknown keyword arguments.
        thinking = pop_thinking(generate_kwargs)
        model_inputs = self._apply_chat_template(messages, tools, thinking=thinking)
        generated_ids = self._hf_model.generate(**model_inputs, **generate_kwargs)

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]

        if self._uses_processor_parse_response:
            raw = self._hf_processor.decode(output_ids, skip_special_tokens=False)
            logger.debug("[raw] processor response: %s", raw)
            parsed = self._hf_processor.parse_response(raw)
            thinking = parsed.get("thinking")
            self.last_thinking = thinking.strip() if thinking else None

            tool_calls = parsed.get("tool_calls") or []
            if tool_calls:
                # parse_response returns the OpenAI function-calling envelope;
                # unwrap to the flat {"name": ..., "arguments": ...} form that
                # _record_tool_calls / _prepare_tool_calls expect (matches every other client).
                self._parsed_tool_calls = [
                    {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]} for tc in tool_calls
                ]
                logger.debug("[tool_call] parsed: %s", self._parsed_tool_calls)
                # Preserve any prose the model emitted alongside the tool call (parse_response
                # separates content from tool_calls); _chat stores it on the assistant message.
                parsed_content = parsed.get("content")
                self._parsed_content = parsed_content.strip() if parsed_content else ""
                response = ""
            else:
                self._parsed_tool_calls = None
                self._parsed_content = ""
                # parse_response leaves EOS in content when decoded with
                # skip_special_tokens=False; use a clean tokenizer decode for
                # the user-facing text.
                response = self._hf_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            logger.debug("[generating] %s", response)
            return response

        response = self._hf_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        logger.debug("[raw] response: %s", response)

        if self.model.supports_thinking:
            opened = self.model.think_opener_in_prompt or response.startswith("<think>")
            if "</think>" in response:
                idx = response.index("</think>")
                start = len("<think>") if response.startswith("<think>") else 0
                self.last_thinking = response[start:idx].strip()
                logger.debug("LLM thinking: %s", self.last_thinking)
                response = response[idx + len("</think>") :].strip()
            elif opened:
                # Thinking was opened but truncated before </think> (e.g. token
                # budget exhausted). The whole output is thinking, not an answer.
                start = len("<think>") if response.startswith("<think>") else 0
                self.last_thinking = response[start:].strip() or None
                response = ""

        logger.debug("[generating] %s", response)

        return response

    def _generate_streaming(
        self,
        messages: list[dict],
        generate_kwargs: dict[str, Any],
        tools: Optional[list[dict]],
        streamer: TextIteratorStreamer,
    ) -> Iterator[str]:
        self.last_thinking = None
        self._pending_thinking_tokens = []

        # Popped here, not peeked: this dict is splatted into generate() below, which
        # rejects unknown keyword arguments.
        thinking = pop_thinking(generate_kwargs)
        model_inputs = self._apply_chat_template(messages, tools, thinking=thinking)
        self._hf_model.generate(**model_inputs, **generate_kwargs, streamer=streamer)

        # first part is always empty
        next(streamer)
        response_part = next(streamer)
        logger.debug("[raw] first token: %r", response_part)

        if self.model.supports_thinking:
            # <think> may be pre-filled in the prompt template (e.g. Qwen3.5) and thus
            # consumed by skip_prompt=True before streaming begins. Detect thinking by
            # scanning for </think> rather than requiring <think> at the start.
            opened = self.model.think_opener_in_prompt
            if response_part.startswith("<think>"):
                response_part = response_part[len("<think>") :]
                opened = True

            self.last_thinking = ""
            buffered = []
            cur = response_part
            found_end = False
            while cur is not None:
                if "</think>" in cur:
                    before, _, rest = cur.partition("</think>")
                    if before:
                        self.last_thinking += before
                        self._pending_thinking_tokens.append(before)
                    self.last_thinking = self.last_thinking.strip()
                    rest = rest.lstrip("\n")
                    response_part = rest
                    # Skip any empty tokens between </think> and the first real token
                    while not response_part:
                        response_part = next(streamer, None)
                        if response_part is None:
                            response_part = ""
                            break
                    found_end = True
                    break
                if cur:
                    buffered.append(cur)
                    self.last_thinking += cur
                    self._pending_thinking_tokens.append(cur)
                cur = next(streamer, None)

            if not found_end:
                if opened:
                    # The think block was opened but truncated before </think>
                    # (e.g. token budget exhausted). The buffered content is
                    # thinking, not a final answer, so keep it as last_thinking and
                    # the pending THINKING tokens, and emit nothing as generation.
                    self.last_thinking = self.last_thinking.strip() or None
                    return iter([])
                # No <think> opener and no </think>: genuinely not a thinking turn.
                self.last_thinking = None
                self._pending_thinking_tokens = []
                return iter(buffered)

        return itertools.chain([response_part], streamer)

    def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs = self._resolve_generate_kwargs(generate_kwargs)

        # Vision and audio flow through the content-block message; the processor branch in
        # _apply_chat_template extracts PIL images and audio arrays before templating.
        if images:
            content = _build_user_content_blocks(prompt, images)
        elif audio:
            from ..._internal.audio_input import _build_audio_content_blocks

            content = _build_audio_content_blocks(prompt, audio)
        else:
            content = prompt
        messages = [{"role": "user", "content": content}]

        if stream:
            return self._generate_streamed(messages, generate_kwargs)

        return self._generate_sync(messages, generate_kwargs)

    def _generate_streamed(
        self,
        messages: list,
        generate_kwargs: dict[str, Any],
    ) -> Iterator[StreamChunk]:
        streamer = TextIteratorStreamer(self._hf_tokenizer, skip_prompt=True, skip_special_tokens=True)
        it = self._generate_streaming(messages, generate_kwargs, None, streamer)

        for token in self._pending_thinking_tokens:
            yield StreamChunk(StreamingContentType.THINKING, token)
        self._pending_thinking_tokens = []

        for token in it:
            logger.debug("[generating] token: %r", token)
            yield StreamChunk(StreamingContentType.GENERATING, token)

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

        response = self._generate_sync(self.messages, generate_kwargs, tools)

        if self._uses_processor_parse_response:
            tool_calls = self._parsed_tool_calls
            content = self._parsed_content
        else:
            prefix = self.model.tool_call_format
            tool_calls = prefix.parse(response) if prefix else None
            content = prefix.strip_calls(response) if (prefix and tool_calls) else ""
        # Single turn: if the model called tools, execute them and return. Any prose the model
        # emitted alongside the tool call is preserved as the assistant message's content; the
        # model's response to the tool results comes on the next chat() call (loop lives in Agent).
        if tool_calls:
            logger.debug("[tool_call] parsed: %s", tool_calls)
            msgs_before = len(self.messages)
            self._record_tool_calls(tool_calls, content=content)
            if self.last_thinking is not None:
                self.messages[msgs_before]["thinking"] = self.last_thinking
            return content

        logger.debug("[generating] chat response: %s", response)
        self._append_message({"role": "assistant", "content": response})

        if self.last_thinking is not None:
            self.messages[-1]["thinking"] = self.last_thinking

        return response

    def _chat_streamed(self, generate_kwargs: dict[str, Any], tools: list) -> Iterator[StreamChunk]:
        if self._uses_processor_parse_response:
            # Gemma 4 uses special tokens (<|channel>, <|tool_call>, ...) and
            # requires processor.parse_response for structured extraction.
            # Token-level streaming with ToolCallFormat doesn't apply here, so
            # route through _generate_sync and yield the parsed result.
            yield from self._chat_streamed_via_processor(generate_kwargs, tools)
            return

        streamer = TextIteratorStreamer(self._hf_tokenizer, skip_prompt=True, skip_special_tokens=True)
        it = self._generate_streaming(self.messages, generate_kwargs, tools, streamer)

        for token in self._pending_thinking_tokens:
            yield StreamChunk(StreamingContentType.THINKING, token)
        self._pending_thinking_tokens = []

        response_part = next(it, None)
        content = ""
        msgs_before = len(self.messages)

        prefix = self.model.tool_call_format

        if response_part is None:
            # Thinking-only turn: the model opened a <think> block but generation
            # ended before </think> (e.g. token budget exhausted, common with
            # sampling), so _generate_streaming returned an empty iterator. Mirror
            # _generate_sync: surface the buffered thinking via _pending_thinking_tokens
            # below and finish with empty generated content, rather than letting an
            # unguarded next() raise StopIteration (PEP 479 RuntimeError) on this generator.
            pass
        elif prefix and prefix.detected_in(response_part):
            parts = [response_part]
            parts.extend(it)
            response = "".join(parts)
            logger.debug("[tool_call] raw: %s", response)

            tool_calls = prefix.parse(response)
            if tool_calls:
                logger.debug("[tool_call] parsed: %s", tool_calls)
                # Single turn: preserve any prose emitted alongside the tool call, dispatch, and
                # return. The model's response to the tool results comes on the next chat() call.
                prose = prefix.strip_calls(response)
                if prose:
                    yield StreamChunk(StreamingContentType.GENERATING, prose)
                self._record_tool_calls(tool_calls, content=prose)
                if self.last_thinking is not None:
                    self.messages[msgs_before]["thinking"] = self.last_thinking
                return
            # Tool-call markup detected but not parseable: surface the raw text as content.
            content += response
            yield StreamChunk(StreamingContentType.GENERATING, response)
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
        self._append_message({"role": "assistant", "content": content})

        if self.last_thinking is not None:
            self.messages[-1]["thinking"] = self.last_thinking

    def _chat_streamed_via_processor(
        self, generate_kwargs: dict[str, Any], tools: Optional[list[dict]]
    ) -> Iterator[StreamChunk]:
        """Buffered streaming for models that require processor.parse_response (Gemma 4)."""
        response = self._generate_sync(self.messages, generate_kwargs, tools)

        # Single turn: if the model called tools, execute them and return, preserving any prose
        # emitted alongside. The model's response to the tool results comes on the next chat() call.
        if self._parsed_tool_calls:
            logger.debug("[tool_call] parsed: %s", self._parsed_tool_calls)
            if self._parsed_content:
                yield StreamChunk(StreamingContentType.GENERATING, self._parsed_content)
            msgs_before = len(self.messages)
            self._record_tool_calls(self._parsed_tool_calls, content=self._parsed_content)
            if self.last_thinking is not None:
                self.messages[msgs_before]["thinking"] = self.last_thinking
            return

        if self.last_thinking:
            logger.debug("[thinking] collected: %s", self.last_thinking)
            yield StreamChunk(StreamingContentType.THINKING, self.last_thinking)

        if response:
            logger.debug("[generating] collected: %s", response)
            yield StreamChunk(StreamingContentType.GENERATING, response)

        self._append_message({"role": "assistant", "content": response})

        if self.last_thinking is not None:
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
