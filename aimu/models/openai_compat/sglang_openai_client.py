from ..base_client import Model
from .openai_compat_client import OpenAICompatClient

SGLANG_BASE_URL = "http://localhost:30000/v1"


class SGLangOpenAIModel(Model):
    def __init__(self, value, supports_tools=False, supports_thinking=False):
        super().__init__(value, supports_tools, supports_thinking)

    # Model values are HuggingFace repo paths (as used by `python -m sglang.launch_server --model-path`).
    LLAMA_3_1_8B = ("meta-llama/Llama-3.1-8B-Instruct", True)
    LLAMA_3_2_3B = ("meta-llama/Llama-3.2-3B-Instruct", True)
    MISTRAL_7B = ("mistralai/Mistral-7B-Instruct-v0.3", True)
    PHI_4_MINI = ("microsoft/Phi-4-mini-instruct", True)
    QWEN_3_4B = ("Qwen/Qwen3-4B", True, True)
    QWEN_3_8B = ("Qwen/Qwen3-8B", True, True)
    DEEPSEEK_R1_7B = ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", False, True)
    GEMMA_3_12B = ("google/gemma-3-12b-it", True)


class SGLangOpenAIClient(OpenAICompatClient):
    """Client for SGLang's OpenAI-compatible REST API.

    Start the server with:
        python -m sglang.launch_server --model-path <model> --port 30000
    """

    MODELS = SGLangOpenAIModel

    def __init__(self, model: SGLangOpenAIModel, base_url: str = SGLANG_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)
