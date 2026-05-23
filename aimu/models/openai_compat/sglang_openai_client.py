from ..base import Model, ModelSpec
from .openai_compat_client import OpenAICompatClient

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


class SGLangOpenAIClient(OpenAICompatClient):
    """Client for SGLang's OpenAI-compatible REST API.

    Start the server with:
        python -m sglang.launch_server --model-path <model> --port 30000
    """

    MODELS = SGLangOpenAIModel

    def __init__(self, model: SGLangOpenAIModel, base_url: str = SGLANG_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)
