from ..base import Model
from .openai_compat_client import OpenAICompatClient

HF_OPENAI_BASE_URL = "http://localhost:8000/v1"


class HFOpenAIModel(Model):
    def __init__(self, value, supports_tools=False, supports_thinking=False):
        super().__init__(value, supports_tools, supports_thinking)

    # Model values are HuggingFace repo paths (as used by `transformers serve <model-id>`).
    LLAMA_3_1_8B = ("meta-llama/Llama-3.1-8B-Instruct", True)
    LLAMA_3_2_3B = ("meta-llama/Llama-3.2-3B-Instruct", True)
    MISTRAL_7B = ("mistralai/Mistral-7B-Instruct-v0.3", True)
    PHI_4_MINI = ("microsoft/Phi-4-mini-instruct", True)
    QWEN_3_4B = ("Qwen/Qwen3-4B", True, True)
    QWEN_3_8B = ("Qwen/Qwen3-8B", True, True)
    DEEPSEEK_R1_7B = ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", False, True)
    GEMMA_3_12B = ("google/gemma-3-12b-it", True)


class HFOpenAIClient(OpenAICompatClient):
    MODELS = HFOpenAIModel

    def __init__(self, model: HFOpenAIModel, base_url: str = HF_OPENAI_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)
