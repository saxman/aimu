# Model matrix

Every model enum member shipped with AIMU, with capability flags. The tables are generated from the catalogs by `scripts/generate_model_matrix.py`; run `python scripts/generate_model_matrix.py --write` after any catalog change (`tests/test_docs_model_matrix.py::test_matrix_tables_match_the_generator` fails the suite otherwise). The surrounding prose stays hand-written.

These are declarations, not measurements: the table says what a model can be *asked* to do, so you can rule options in and out before spending a token. What it does when asked is the other half, and you find that out by running it. See [compare models](../how-to/compare-models.md).

Legend: ✅ = supported, ✗ = not supported.

The **Thinking** column says whether a model reasons at all. How far it can be *steered* is a
second question, answered by two `ModelSpec` flags behind the portable
[`thinking=`](../how-to/control-thinking.md) parameter:

- **◆ accepts effort levels** (`thinking_levels=True`), so `thinking="low"/"medium"/"high"` is
  honoured: the Qwen 3.8 27B family and every Anthropic model. On every other reasoning model a
  level is warned about and treated as plain on, which is why swapping models never raises.
  *Where* the level lands is a second axis, shown in the Anthropic table's Thinking column:
  `effort` means it becomes `output_config.effort` (with `high` sent as the vendor's `xhigh`),
  and its absence means it becomes a `budget_tokens` figure instead.
- **◇ always reasons** (`thinking_optional=False`), so `thinking=False` cannot be honoured:
  `GEMINI_2_5_PRO` and `CLAUDE_FABLE_5`. The call proceeds and is billed for reasoning;
  `client.last_usage` shows the cost.

Note that a model declaring a level is not sufficient on its own: the active provider also has to
be able to express it. llama.cpp, the OpenAI cloud client and Gemini emit nothing today, so
`thinking=` warns rather than steering there. The [how-to](../how-to/control-thinking.md) has the
per-provider mechanisms.

## Anthropic (`AnthropicModel`)

<!-- generated:AnthropicModel -->
| Enum member | Model id | Tools | Thinking | Vision |
|---|---|:---:|:---:|:---:|
| `CLAUDE_FABLE_5` ◆ ◇ | `claude-fable-5` | ✅ | ✅ (adaptive, effort) | ✅ |
| `CLAUDE_OPUS_5` ◆ | `claude-opus-5` | ✅ | ✅ (adaptive, effort) | ✅ |
| `CLAUDE_OPUS_4_8` ◆ | `claude-opus-4-8` | ✅ | ✅ (adaptive, effort) | ✅ |
| `CLAUDE_OPUS_4_7` ◆ | `claude-opus-4-7` | ✅ | ✅ (adaptive, effort) | ✅ |
| `CLAUDE_OPUS_4_6` ◆ | `claude-opus-4-6` | ✅ | ✅ (budget) | ✅ |
| `CLAUDE_SONNET_5` ◆ | `claude-sonnet-5` | ✅ | ✅ (adaptive, effort) | ✅ |
| `CLAUDE_SONNET_4_6` ◆ | `claude-sonnet-4-6` | ✅ | ✅ (budget) | ✅ |
| `CLAUDE_HAIKU_4_5` ◆ | `claude-haiku-4-5` | ✅ | ✅ (budget) | ✅ |
<!-- /generated -->

AIMU requests Anthropic reasoning in one of two shapes, fixed per model by a `ThinkingStyle` on each `AnthropicModel` member (an Anthropic-specific enum, analogous to HuggingFace's `ToolCallFormat`):

- **budget**: `thinking={"type": "enabled", "budget_tokens": N}`; the model always thinks up to the budget. Used by Opus 4.6, Sonnet 4.6, and Haiku 4.5.
- **adaptive**: `thinking={"type": "adaptive", "display": "summarized"}`; the model decides per request whether and how much to think (it may not think at all on simple prompts), and `temperature`/`top_p`/`top_k` are not sent. Required by Opus 4.7+ and Fable 5, which reject the budget form with a 400.

Both styles surface reasoning as `THINKING` stream chunks and populate `last_thinking`. The `thinking=` column reflects the universal `supports_thinking` flag; the style only changes how the request is built, handled entirely inside `AnthropicClient`.

## OpenAI (`OpenAIModel`)

<!-- generated:OpenAIModel -->
| Enum member | Model id | Tools | Thinking | Vision |
|---|---|:---:|:---:|:---:|
| `GPT_4O_MINI` | `gpt-4o-mini` | ✅ | ✗ | ✅ |
| `GPT_4O` | `gpt-4o` | ✅ | ✗ | ✅ |
| `GPT_4_1` | `gpt-4.1` | ✅ | ✗ | ✅ |
| `GPT_4_1_MINI` | `gpt-4.1-mini` | ✅ | ✗ | ✅ |
| `GPT_4_1_NANO` | `gpt-4.1-nano` | ✅ | ✗ | ✅ |
| `O4_MINI` | `o4-mini` | ✅ | ✗ | ✅ |
| `O3` | `o3` | ✅ | ✗ | ✅ |
| `O3_MINI` | `o3-mini` | ✅ | ✗ | ✗ |
<!-- /generated -->

o-series models emit reasoning tokens that aren't exposed via the API, so `thinking=False` even though they reason internally. Pass `reasoning_effort` via `generate_kwargs` if needed.

## Google Gemini (`GeminiModel`)

<!-- generated:GeminiModel -->
| Enum member | Model id | Tools | Thinking | Vision |
|---|---|:---:|:---:|:---:|
| `GEMINI_2_0_FLASH` | `gemini-2.0-flash` | ✅ | ✗ | ✅ |
| `GEMINI_2_0_FLASH_LITE` | `gemini-2.0-flash-lite` | ✅ | ✗ | ✅ |
| `GEMINI_1_5_PRO` | `gemini-1.5-pro` | ✅ | ✗ | ✅ |
| `GEMINI_1_5_FLASH` | `gemini-1.5-flash` | ✅ | ✗ | ✅ |
| `GEMINI_2_5_PRO` ◇ | `gemini-2.5-pro` | ✅ | ✅ | ✅ |
| `GEMINI_2_5_FLASH` | `gemini-2.5-flash` | ✅ | ✅ | ✅ |
<!-- /generated -->

Gemini 2.5 thinking models emit `<think>` tags on Google's OpenAI-compatible endpoint.

## Ollama native (`OllamaModel`)

<!-- generated:OllamaModel -->
| Enum member | Model id | Tools | Thinking | Vision |
|---|---|:---:|:---:|:---:|
| `QWEN_3_8_27B` ◆ | `qwen3.8:27b` | ✅ | ✅ | ✅ |
| `QWEN_3_6_35B` | `qwen3.6:35b` | ✅ | ✅ | ✅ |
| `QWEN_3_6_27B` | `qwen3.6:27b` | ✅ | ✅ | ✅ |
| `QWEN_3_5_9B` | `qwen3.5:9b` | ✅ | ✅ | ✅ |
| `QWEN_3_32B` | `qwen3:32b` | ✅ | ✅ | ✗ |
| `QWEN_3_8B` | `qwen3:8b` | ✅ | ✅ | ✗ |
| `QWEN_3_4B` | `qwen3:4b` | ✅ | ✅ | ✗ |
| `GEMMA_4_E4B` | `gemma4:e4b` | ✅ | ✅ | ✅ |
| `GEMMA_4_12B` | `gemma4:12b` | ✅ | ✅ | ✅ |
| `GEMMA_4_26B` | `gemma4:26b` | ✅ | ✅ | ✅ |
| `GEMMA_4_31B` | `gemma4:31b` | ✅ | ✅ | ✅ |
| `GEMMA_3_12B` | `gemma3:12b` | ✗ | ✗ | ✅ |
| `NEMOTRON_CASCADE_2_30B` | `nemotron-cascade-2:30b` | ✅ | ✅ | ✗ |
| `NEMOTRON_3_NANO_30B` | `nemotron-3-nano:30b` | ✅ | ✅ | ✗ |
| `GLM_4_7_FLASH_31B_Q4` | `glm-4.7-flash:q4_K_M` | ✗ | ✅ | ✗ |
| `GPT_OSS_20B` | `gpt-oss:20b` | ✅ | ✅ | ✗ |
| `MAGISTRAL_SMALL_24B` | `magistral:24b` | ✅ | ✅ | ✗ |
| `MINISTRAL_3_14B` | `ministral-3:14b` | ✅ | ✗ | ✗ |
| `MISTRAL_7B` | `mistral:7b` | ✅ | ✗ | ✗ |
| `PHI_4_MINI_3_8B` | `phi4-mini:3.8b` | ✅ | ✗ | ✗ |
| `PHI_4_14B` | `phi4:14b` | ✗ | ✗ | ✗ |
| `DEEPSEEK_R1_8B` | `deepseek-r1:8b` | ✗ | ✅ | ✗ |
| `SMOLLM2_1_7B` | `smollm2:1.7b` | ✗ | ✗ | ✗ |
| `MUSE_GLIMMER_30B` | `muse-glimmer:30b` | ✅ | ✅ | ✅ |
| `LLAMA_3_2_3B` | `llama3.2:3b` | ✅ | ✗ | ✗ |
| `LLAMA_3_1_8B` | `llama3.1:8b` | ✅ | ✗ | ✗ |
<!-- /generated -->

Some Ollama models can technically be asked for tools but produce unreliable tool calls; those are marked `tools=False` and documented in the enum source.

## HuggingFace (`HuggingFaceModel`)

<!-- generated:HuggingFaceModel -->
| Enum member | Repo id | Tools | Thinking | Vision |
|---|---|:---:|:---:|:---:|
| `QWEN_3_8_27B` ◆ | `Qwen/Qwen3.8-27B` | ✅ | ✅ | ✅ |
| `QWEN_3_8_27B_FP8` ◆ § | `Qwen/Qwen3.8-27B-FP8` | ✅ | ✅ | ✅ |
| `QWEN_3_6_27B` | `Qwen/Qwen3.6-27B` | ✅ | ✅ | ✅ |
| `QWEN_3_6_27B_FP8` § | `Qwen/Qwen3.6-27B-FP8` | ✅ | ✅ | ✅ |
| `QWEN_3_5_9B` | `Qwen/Qwen3.5-9B` | ✅ | ✅ | ✅ |
| `QWEN_3_8B` | `Qwen/Qwen3-8B` | ✅ | ✅ | ✗ |
| `GEMMA_4_E4B` | `google/gemma-4-E4B-it` | ✅ | ✅ | ✅ |
| `GEMMA_4_12B` | `google/gemma-4-12b-it` | ✅ | ✅ | ✅ |
| `GEMMA_3_12B` | `google/gemma-3-12b-it` | ✗ | ✗ | ✅ |
| `NEMOTRON_H_8B` | `nvidia/Nemotron-H-8B-Instruct-HF` | ✅ | ✗ | ✗ |
| `GPT_OSS_20B` | `openai/gpt-oss-20b` | ✅ | ✅ | ✗ |
| `MAGISTRAL_SMALL` | `mistralai/Magistral-Small-2509` | ✅ | ✗ | ✗ |
| `MISTRAL_NEMO_12B` | `mistralai/Mistral-Nemo-Instruct-2407` | ✅ | ✗ | ✗ |
| `MISTRAL_7B` | `mistralai/Mistral-7B-Instruct-v0.3` | ✅ | ✗ | ✗ |
| `PHI_4_MINI_3_8B` | `microsoft/Phi-4-mini-instruct` | ✗ | ✗ | ✗ |
| `PHI_4_14B` | `microsoft/phi-4` | ✗ | ✗ | ✗ |
| `DEEPSEEK_R1_8B` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | ✗ | ✅ | ✗ |
| `SMOLLM3_3B` | `HuggingFaceTB/SmolLM3-3B` | ✅ | ✅ | ✗ |
| `LLAMA_3_2_3B` | `unsloth/Llama-3.2-3B-Instruct` | ✅ | ✗ | ✗ |
| `LLAMA_3_1_8B` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | ✅ | ✗ | ✗ |
<!-- /generated -->

§ `QWEN_3_6_27B_FP8` is the e4m3 FP8 checkpoint with dynamic activation scaling. FP8 needs Ada/Hopper-class tensor cores (compute capability ≥ 8.9), so on Ampere, MPS, or CPU there is no native path — pick the bare `QWEN_3_6_27B` there. The quantization is in the member **name** precisely because it is a hardware-gated choice rather than a default; elsewhere in the catalogs a quantization is left out of the name when the provider resolves it itself (Ollama's default tags, LM Studio's keys, llama-cpp's `model_path=`).

`_VL` suffix variants load with `AutoModelForImageTextToText` for the vision encoder.

## llama-cpp (`LlamaCppModel`)

<!-- generated:LlamaCppModel -->
| Enum member | Hint id | Tools | Thinking | Vision |
|---|---|:---:|:---:|:---:|
| `LLAMA_3_1_8B` | `llama-3.1-8b` | ✅ | ✗ | ✗ |
| `LLAMA_3_2_3B` | `llama-3.2-3b` | ✅ | ✗ | ✗ |
| `MISTRAL_7B` | `mistral-7b` | ✅ | ✗ | ✗ |
| `MAGISTRAL_SMALL_24B` | `magistral-small-24b` | ✅ | ✅ | ✗ |
| `MINISTRAL_3_14B` | `ministral-3-14b` | ✅ | ✗ | ✗ |
| `PHI_4_MINI_3_8B` | `phi-4-mini` | ✅ | ✗ | ✗ |
| `PHI_4_14B` | `phi-4` | ✗ | ✗ | ✗ |
| `QWEN_3_4B` | `qwen3-4b` | ✅ | ✅ | ✗ |
| `QWEN_3_8B` | `qwen3-8b` | ✅ | ✅ | ✗ |
| `QWEN_3_32B` | `qwen3-32b` | ✅ | ✅ | ✗ |
| `QWEN_3_5_9B` * | `qwen3.5-9b` | ✅ | ✅ | ✗ |
| `QWEN_3_6_27B` * | `qwen3.6-27b` | ✅ | ✅ | ✗ |
| `QWEN_3_6_35B` * | `qwen3.6-35b-a3b` | ✅ | ✅ | ✗ |
| `QWEN_3_8_27B` ◆ * | `qwen3.8-27b` | ✅ | ✅ | ✗ |
| `DEEPSEEK_R1_7B` | `deepseek-r1-7b` | ✗ | ✅ | ✗ |
| `DEEPSEEK_R1_8B` | `deepseek-r1-8b` | ✗ | ✅ | ✗ |
| `GEMMA_3_12B` * | `gemma-3-12b` | ✗ | ✗ | ✗ |
| `GEMMA_4_E4B` * | `gemma-4-e4b` | ✅ | ✅ | ✗ |
| `GEMMA_4_12B` * | `gemma-4-12b` | ✅ | ✅ | ✗ |
| `GEMMA_4_26B` * | `gemma-4-26b-a4b` | ✅ | ✅ | ✗ |
| `GEMMA_4_31B` * | `gemma-4-31b` | ✅ | ✅ | ✗ |
| `NEMOTRON_CASCADE_2_30B` | `nemotron-cascade-2-30b-a3b` | ✅ | ✅ | ✗ |
| `NEMOTRON_3_NANO_30B` | `nemotron-3-nano-30b-a3b` | ✅ | ✅ | ✗ |
| `GLM_4_7_FLASH_31B_Q4` | `glm-4.7-flash-q4` | ✗ | ✅ | ✗ |
| `GPT_OSS_20B` | `gpt-oss-20b` | ✅ | ✅ | ✗ |
| `SMOLLM2_1_7B` | `smollm2-1.7b` | ✗ | ✗ | ✗ |
<!-- /generated -->

llama-cpp model ids are hints; the actual model is loaded from `model_path=` regardless. Capability flags are honoured by the client.

\* These weights are intrinsically vision-capable (see the Ollama/HuggingFace tables), but llama-cpp's default GGUF path loads no mmproj projector, so each overrides `vision=False` with that rationale (`Wire(..., why=..., vision=False)`; see `tests/test_model_catalog_facts.py::test_gguf_catalogs_do_not_advertise_vision`). Pass `chat_handler=Llava15ChatHandler(clip_model_path=...)` to `LlamaCppClient(...)` to enable vision for these.

No `MUSE_GLIMMER_30B` here: llama-cpp-python runs the same llama.cpp engine as llama-server and LM Studio, both of which omit it for the same reason (see the OpenAI-compatible local servers section below).

## OpenAI-compatible local servers

`OllamaOpenAIModel`, `LMStudioOpenAIModel`, `VLLMOpenAIModel`, `HFOpenAIModel`, `LlamaServerOpenAIModel`, `SGLangOpenAIModel`, and `OMLXOpenAIModel` enumerate a shared set of common open models. Capability flags for a given member are the same across servers (except where footnoted); the **model id format differs per server** — LM Studio uses loaded model keys, Ollama uses `name:tag`, vLLM/SGLang/HF Serve use HuggingFace repo paths, llama-server uses GGUF filenames, and oMLX uses `--model-dir` subdirectory names — so consult the enum source for each server's exact ids.

**"all" in the Servers column means the six non-MLX servers** (Ollama, LM Studio, vLLM, HF Serve, llama-server, SGLang). `OMLXOpenAIModel` ships only MLX conversions, so it carries a quant-suffixed member for every model with a confirmed `mlx-community` build (most of the catalog below) and is named explicitly on the rows it does carry; LM Studio carries the same set under its own MLX-engine ids, alongside its GGUF builds. `tests/test_docs_model_matrix.py` checks this column against the enums.

<!-- generated:servers -->
| Enum member | Tools | Thinking | Vision | Servers |
|---|:---:|:---:|:---:|---|
| `LLAMA_3_1_8B` | ✅ | ✗ | ✗ | all |
| `LLAMA_3_1_8B_4BIT` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `LLAMA_3_1_8B_8BIT` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `LLAMA_3_1_8B_BF16` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `LLAMA_3_2_3B` | ✅ | ✗ | ✗ | all |
| `LLAMA_3_2_3B_4BIT` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `LLAMA_3_2_3B_8BIT` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `LLAMA_3_2_3B_BF16` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `MISTRAL_7B` | ✅ | ✗ | ✗ | all |
| `MISTRAL_7B_4BIT` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `MISTRAL_7B_8BIT` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `MAGISTRAL_SMALL_24B` | ✅ | ✅ | ✗ | all |
| `MAGISTRAL_SMALL_24B_BF16` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `MINISTRAL_3_14B` | ✅ | ✗ | ✗ | all |
| `MINISTRAL_3_14B_4BIT` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `MINISTRAL_3_14B_8BIT` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `MINISTRAL_3_14B_BF16` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `PHI_4_MINI_3_8B` | ✅ | ✗ | ✗ | all |
| `PHI_4_MINI_3_8B_4BIT` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `PHI_4_MINI_3_8B_8BIT` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `PHI_4_MINI_3_8B_FP16` | ✅ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `PHI_4_14B` | ✗ | ✗ | ✗ | all |
| `PHI_4_14B_4BIT` | ✗ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `PHI_4_14B_8BIT` | ✗ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `PHI_4_14B_BF16` | ✗ | ✗ | ✗ | oMLX, LM Studio ¶ |
| `QWEN_3_4B` | ✅ | ✅ | ✗ | all |
| `QWEN_3_4B_4BIT` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `QWEN_3_4B_8BIT` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `QWEN_3_4B_BF16` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `QWEN_3_8B` | ✅ | ✅ | ✗ | all |
| `QWEN_3_8B_4BIT` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `QWEN_3_8B_8BIT` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `QWEN_3_8B_BF16` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `QWEN_3_32B` | ✅ | ✅ | ✗ | all |
| `QWEN_3_32B_4BIT` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `QWEN_3_32B_8BIT` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `QWEN_3_32B_BF16` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `QWEN_3_5_9B` | ✅ | ✅ | ✅ ※ | all |
| `QWEN_3_5_9B_4BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `QWEN_3_5_9B_8BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `QWEN_3_5_9B_BF16` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `QWEN_3_6_27B` | ✅ | ✅ | ✅ ※ | all |
| `QWEN_3_6_27B_4BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `QWEN_3_6_27B_8BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `QWEN_3_6_27B_BF16` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `QWEN_3_6_35B` | ✅ | ✅ | ✅ ※ | Ollama, LM Studio, vLLM, HF Serve, llama-server, SGLang, oMLX ¶ |
| `QWEN_3_6_35B_4BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `QWEN_3_6_35B_8BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `QWEN_3_6_35B_BF16` | ✅ | ✅ | ✅ | oMLX ¶ |
| `QWEN_3_8_27B` ◆ | ✅ | ✅ | ✅ ※ | Ollama, LM Studio, vLLM, HF Serve, llama-server, SGLang, oMLX ¶ |
| `QWEN_3_8_27B_4BIT` ◆ | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `QWEN_3_8_27B_8BIT` ◆ | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `QWEN_3_8_27B_BF16` ◆ | ✅ | ✅ | ✅ | oMLX ¶ |
| `DEEPSEEK_R1_7B` | ✗ | ✅ | ✗ | all except Ollama |
| `DEEPSEEK_R1_7B_4BIT` | ✗ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `DEEPSEEK_R1_7B_8BIT` | ✗ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `DEEPSEEK_R1_7B_BF16` | ✗ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `DEEPSEEK_R1_8B` | ✗ | ✅ | ✗ | all |
| `DEEPSEEK_R1_8B_4BIT` | ✗ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `DEEPSEEK_R1_8B_8BIT` | ✗ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `DEEPSEEK_R1_8B_BF16` | ✗ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `GEMMA_3_12B` | ✅ † | ✗ | ✅ ※ | all |
| `GEMMA_3_12B_4BIT` | ✅ | ✗ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_3_12B_8BIT` | ✅ | ✗ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_3_12B_BF16` | ✅ | ✗ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_4_E4B` | ✅ | ✅ | ✅ ※ | all |
| `GEMMA_4_E4B_4BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_4_E4B_8BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_4_E4B_BF16` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_4_12B` | ✅ | ✅ | ✅ ※ | all |
| `GEMMA_4_12B_4BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_4_12B_8BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_4_12B_BF16` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_4_26B` | ✅ | ✅ | ✅ ※ | all |
| `GEMMA_4_26B_4BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_4_26B_8BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_4_26B_BF16` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_4_31B` | ✅ | ✅ | ✅ ※ | all |
| `GEMMA_4_31B_4BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_4_31B_8BIT` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `GEMMA_4_31B_BF16` | ✅ | ✅ | ✅ | oMLX, LM Studio ¶ |
| `NEMOTRON_CASCADE_2_30B` | ✅ | ✅ | ✗ | all |
| `NEMOTRON_CASCADE_2_30B_4BIT` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `NEMOTRON_CASCADE_2_30B_8BIT` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `NEMOTRON_CASCADE_2_30B_BF16` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `NEMOTRON_3_NANO_30B` | ✅ | ✅ | ✗ | all |
| `NEMOTRON_3_NANO_30B_4BIT` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `NEMOTRON_3_NANO_30B_8BIT` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `NEMOTRON_3_NANO_30B_BF16` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `GLM_4_7_FLASH_31B_Q4` | ✗ | ✅ | ✗ | all |
| `GLM_4_7_FLASH_31B_Q4_4BIT` | ✗ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `GLM_4_7_FLASH_31B_Q4_8BIT` | ✗ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `GLM_4_7_FLASH_31B_Q4_BF16` | ✗ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `GPT_OSS_20B` | ✅ | ✅ | ✗ | all |
| `GPT_OSS_20B_MXFP4_Q4` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `GPT_OSS_20B_MXFP4_Q8` | ✅ | ✅ | ✗ | oMLX, LM Studio ¶ |
| `SMOLLM2_1_7B` | ✗ | ✗ | ✗ | all |
| `MUSE_GLIMMER_30B` | ✅ | ✅ | ✅ | Ollama, vLLM, oMLX ‡ |
| `MUSE_GLIMMER_30B_4BIT` | ✅ | ✅ | ✅ | oMLX ‡ ¶ |
| `MUSE_GLIMMER_30B_8BIT` | ✅ | ✅ | ✅ | oMLX ‡ ¶ |
| `MUSE_GLIMMER_30B_BF16` | ✅ | ✅ | ✅ | oMLX ‡ ¶ |
<!-- /generated -->

¶ Every MLX-served model carries **per-quantization** members — each MLX quantization is a separate `mlx-community` repo (or, occasionally, an official publisher repo; see below), so each is a separate directory/key and therefore a separate id. Three families (Qwen 3.6 35B-A3B, Qwen 3.8 27B, Muse Glimmer 30B) additionally carry a *bare* member (`QWEN_3_6_35B`, `QWEN_3_8_27B`, `MUSE_GLIMMER_30B`): the quant-agnostic layout (one folder holding whichever quant was downloaded), the name shared with the Ollama catalogs. Every other MLX-served model below has **no bare oMLX/LM-Studio-MLX member** — mlx-community publishes no quant-agnostic full-precision folder for them (checked directly: the bare, unsuffixed repo segment 401s), so only `_4BIT` / `_8BIT` / `_BF16` (or the precision-specific suffix a model actually ships under; see below) are catalogued, for a machine holding one or several quants. Every id within an enum must stay distinct: two members sharing a `ModelSpec.id` silently become an enum **alias**, dropping the second from iteration and discarding its flags (guarded by `tests/test_model_catalog_consistency.py::test_no_silent_enum_aliases`). LM Studio's bare `QWEN_3_6_35B`/`QWEN_3_8_27B` members are the *GGUF* build (grouped into the "all" GGUF-server set above, not this MLX footnote) and no bf16 (impractical at 35B). Ollama needs no per-quant members at all: 0.19+ picks its MLX backend automatically on Apple Silicon behind an unchanged tag.

Several models ship under mlx-community naming that deviates from the plain `-4bit`/`-8bit`/`-bf16` convention, so their member suffixes follow the repo rather than the convention: `GPT_OSS_20B_MXFP4_Q4`/`_MXFP4_Q8` (gpt-oss ships natively quantized to mxfp4, so mlx-community's build is a Q4/Q8 requantization of that format, not a plain bit-width quantization; no bf16 exists), `PHI_4_MINI_3_8B_FP16` (the third precision is a distinctly-named `-mlx-fp16` build, not a plain bf16 repo), and `NEMOTRON_3_NANO_30B_4BIT`/`_8BIT`/`_BF16` (the confirmed matching trio carries an extra `-MLX-` infix and Titlecase quant token in the actual repo names, e.g. `NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4Bit`, though the member suffixes themselves stay the plain convention). Two models have fewer than three quantizations confirmed: `MAGISTRAL_SMALL_24B` ships only `_BF16` (a `4bit-DWQ` variant exists but DWQ is a distinct quantization method, out of scope for this plain-quant convention) and `MISTRAL_7B` ships only `_4BIT`/`_8BIT` (no bf16 build). `SMOLLM2_1_7B` has no MLX quant member at all: mlx-community publishes only its unquantized bare repo, no quantized build.

‡ `MUSE_GLIMMER_30B` emits channel-scoped reasoning and ATEM-style XML tool calls instead of `<think>` tags and JSON, so a server needs dedicated parsers to surface either. Ollama parses both; vLLM does with `--tool-call-parser muse_glimmer --reasoning-parser muse_glimmer` (enable them together); and **oMLX** does as of **0.5.8.dev3**, which added Muse Glimmer with channel-scoped output parsing for ATEM tool calls. On oMLX, use an `mlx-community` checkpoint: oMLX's own `Jundot/Muse-Glimmer-30B-oQ4e` was quantized before the embedding-normalization fix and silently breaks tool calling (the model emits `<|eot|>` right after the reasoning block and never produces an `<atem:function_calls>` block, [jundot/omlx#2589](https://github.com/jundot/omlx/issues/2589)) — a stale-quantization bug, not a parser one. Still absent from the remaining server enums: SGLang support is only on a PR branch, and LM Studio's GGUF engine, llama-server, and llama-cpp all run the same llama.cpp engine, whose parsing of this format's channel-scoped reasoning and ATEM-style tool calls is undocumented, so entries there would mean guessing `tools`/`thinking`. (An `mlx-community` MLX build of the model does exist -- see the oMLX members above -- but LM Studio's GGUF path is a separate engine from its own MLX one, and the GGUF engine is what these three catalogs share.)

† `GEMMA_3_12B` is marked `tools=✗` on the Ollama-backed server (`OllamaOpenAIModel`) — the in-process HuggingFace and native-Ollama clients have no tool-call parse format for Gemma 3 — but `tools=✅` everywhere else (vLLM, SGLang, HF Serve, llama-server, LM Studio, and oMLX, including the MLX quant members of all three), which parse Gemma 3 tool calls server-side. (`LLAMA_3_1_8B`/`LLAMA_3_2_3B` were formerly `tools=✗` on the Ollama/LM Studio builds; live testing confirmed reliable tool calling on current runtimes, so they are now `tools=✅` everywhere.)

※ Vision ✗ on the three GGUF-serving catalogs — `LlamaServerOpenAIModel`, `LMStudioOpenAIModel`, and (in its own table above) `LlamaCppModel` — despite ✅ here: none of the three loads an mmproj projector by default, so each overrides `vision=False` with that rationale rather than advertise a capability that fails at request time. The ✅ reflects the servers that do carry vision for this member: Ollama, vLLM, HF Serve, and SGLang (plus oMLX where noted). See `Wire(..., why=..., vision=False)` on the affected members and `tests/test_model_catalog_facts.py::test_gguf_catalogs_do_not_advertise_vision`.

Gemma 4 E4B/12B are natively audio-capable, but `audio` is left off for every OpenAI-compat server because audio input isn't reliably exposed by these local servers (see the inline comments in `openai_compat.py` for the per-server reason). 26B/31B have no native audio.

## Image generation

Image clients use a different spec class than text (`HuggingFaceImageSpec` / `GeminiImageSpec`). The capability flags don't apply, so the matrix shows model-specific defaults instead.

### HuggingFace diffusers (`HuggingFaceImageModel`)

| Enum member | Repo id | Pipeline class | Default steps | Default size | img2img |
|---|---|---|:---:|:---:|:---:|
| `SD_1_5` | `runwayml/stable-diffusion-v1-5` | `StableDiffusionPipeline` | 25 | 512×512 | ✓ (`strength=`) |
| `SDXL_BASE` | `stabilityai/stable-diffusion-xl-base-1.0` | `StableDiffusionXLPipeline` | 30 | 1024×1024 | ✓ (`strength=`) |
| `SD_3_5_MEDIUM` | `stabilityai/stable-diffusion-3.5-medium` | `StableDiffusion3Pipeline` | 28 | 1024×1024 | ✓ (`strength=`) |
| `FLUX_1_DEV` | `black-forest-labs/FLUX.1-dev` | `FluxPipeline` | 28 | 1024×1024 | ✓ (`strength=`) |
| `FLUX_1_SCHNELL` | `black-forest-labs/FLUX.1-schnell` | `FluxPipeline` | 4 | 1024×1024 | ✓ (`strength=`) |
| `FLUX_2_KLEIN_4B` | `black-forest-labs/FLUX.2-klein-4B` | `Flux2KleinPipeline` | 4 | 1024×1024 | ✓ (unified) |
| `FLUX_2_KLEIN_9B` | `black-forest-labs/FLUX.2-klein-9B` | `Flux2KleinPipeline` | 4 | 1024×1024 | ✓ (unified) |

The **img2img** column indicates `reference_image=` support. `strength=` models derive output from a noisy version of the reference (0 = identical, 1 = ignore it; default 0.75). "unified" models (`Flux2KleinPipeline`) condition on the reference directly, with no `strength` parameter; `width`/`height` are derived from the reference.

Spec defaults are starting points: pass `num_inference_steps=`, `guidance_scale=`, `width=`, `height=`, `seed=` to override per call. Power users can bypass the enum with a `"hf:<repo_id>"` string for any HuggingFace diffusers model (defaults to `DiffusionPipeline` auto-detect loader, `img2img_pipeline_class=None`).

### Google Gemini (`GeminiImageModel`)

| Enum member | Model id | Notes |
|---|---|---|
| `NANO_BANANA` | `gemini-2.5-flash-image` | GA channel. Aspect ratio via `aspect_ratio=` (e.g. `"1:1"`, `"16:9"`). |
| `NANO_BANANA_PREVIEW` | `gemini-2.5-flash-image-preview` | Preview channel; kept for users who pinned it. |

Short-name aliases like `"gemini:nano-banana"` resolve to the full model id at construction. Nano Banana's `generate_content` API returns one image per call; `num_images > 1` issues N requests.

## Audio generation

Audio clients use `HuggingFaceAudioSpec`, distinct from the image and text spec classes. The matrix shows generation defaults rather than capability flags.

### HuggingFace (`HuggingFaceAudioModel`)

| Enum member | Repo id | Pipeline type | Default duration | Default steps |
|---|---|---|:---:|:---:|
| `MUSICGEN_SMALL` | `facebook/musicgen-small` | `musicgen` | 10 s | N/A |
| `MUSICGEN_MEDIUM` | `facebook/musicgen-medium` | `musicgen` | 10 s | N/A |
| `MUSICGEN_LARGE` | `facebook/musicgen-large` | `musicgen` | 10 s | N/A |
| `AUDIOLDM2` | `cvssp/audioldm2` | `audioldm2` | 10 s | 200 |
| `STABLE_AUDIO_OPEN` | `stabilityai/stable-audio-open-1.0` | `stable_audio` | 10 s | 200 |

**Pipeline types:**
- `musicgen`: token-autoregressive generation via HuggingFace `transformers`. Duration maps to token count (~50 tokens/s at 32 kHz); `num_inference_steps` does not apply. Single final `AUDIO_GENERATING` chunk when streaming.
- `audioldm2` / `stable_audio`: latent diffusion via HuggingFace `diffusers`. Accepts `num_inference_steps`; emits one progress chunk per step plus a final chunk when streaming.

Override per call with `duration_s=`, `num_inference_steps=`, `seed=`, `num_audio=`. Power users can bypass the enum with `"hf:<repo_id>"` for any compatible model (pipeline type inferred from known repo prefixes, defaulting to `musicgen`).

## Speech generation

Speech clients use `SpeechSpec` subclasses, distinct from image and audio spec classes. Speech is text-to-speech (TTS) only; speech-to-text will use a separate `BaseTranscriptionClient` surface.

### HuggingFace (`HuggingFaceSpeechModel`)

| Enum member | Repo id | Pipeline type | Sample rate | Default voice |
|---|---|---|:---:|---|
| `MMS_TTS_ENG` | `facebook/mms-tts-eng` | `tts_pipeline` | 16 kHz | N/A |
| `SPEECHT5` | `microsoft/speecht5_tts` | `speecht5` | 16 kHz | CMU Arctic xvectors idx 7306 |
| `BARK` | `suno/bark` | `bark` | 24 kHz | `v2/en_speaker_6` |

**Pipeline types:**
- `tts_pipeline`: HuggingFace `pipeline("text-to-speech")`. Any compatible TTS pipeline model.
- `speecht5`: `SpeechT5ForTextToSpeech` + `SpeechT5HifiGan` vocoder + x-vector speaker embeddings. Default embedding loads from `Matthijs/cmu-arctic-xvectors` (index 7306) on first call. Pass `voice="N"` to use a different dataset index (0–1132); the dataset is cached on the client after the first lookup.
- `bark`: zero-shot voice cloning. Pass `voice=` a Bark voice code (`"v2/en_speaker_6"`, `"v2/en_speaker_9"`, etc.).

Power users can bypass the enum with `"hf:<repo_id>"` for any compatible model (pipeline type inferred from known repo prefixes, defaulting to `tts_pipeline`).

### OpenAI (`OpenAISpeechModel`)

Requires `OPENAI_API_KEY`.

| Enum member | Model id | Notes |
|---|---|---|
| `TTS_1` | `tts-1` | Fast, standard quality. Recommended for live narration. |
| `TTS_1_HD` | `tts-1-hd` | Slower, higher quality. |

Available voices: `alloy` (default), `echo`, `fable`, `onyx`, `nova`, `shimmer`. Pass as `voice=` to `generate()`. OpenAI returns raw 24 kHz 16-bit PCM; `encode_audio()` handles WAV conversion.

Override per call with `voice=`, `speed=`, `num_audio=`. Override per-agent with `make_speech_tool(client, voice=..., speed=...)`.

## See also

- [Provider matrix](provider-matrix.md): provider × extra × API key
- [How-to: add a new model](../how-to/add-new-model.md): extending these enums
- [How-to: generate images](../how-to/generate-images.md): using the image surface
- [How-to: generate audio](../how-to/generate-audio.md): using the audio surface
- [How-to: generate speech](../how-to/generate-speech.md): using the speech surface
