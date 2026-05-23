from ..base import Model, ModelSpec
from .openai_compat_client import OpenAICompatClient

HF_OPENAI_BASE_URL = "http://localhost:8000/v1"


class HFOpenAIModel(Model):
    # Model values are HuggingFace repo paths (as used by `transformers serve <model-id>`).
    LLAMA_3_1_8B = ModelSpec("meta-llama/Llama-3.1-8B-Instruct", tools=True)
    LLAMA_3_2_3B = ModelSpec("meta-llama/Llama-3.2-3B-Instruct", tools=True)
    MISTRAL_7B = ModelSpec("mistralai/Mistral-7B-Instruct-v0.3", tools=True)
    PHI_4_MINI = ModelSpec("microsoft/Phi-4-mini-instruct", tools=True)
    QWEN_3_4B = ModelSpec("Qwen/Qwen3-4B", tools=True, thinking=True)
    QWEN_3_8B = ModelSpec("Qwen/Qwen3-8B", tools=True, thinking=True)
    DEEPSEEK_R1_7B = ModelSpec("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", thinking=True)
    GEMMA_3_12B = ModelSpec("google/gemma-3-12b-it", tools=True)


class HFOpenAIClient(OpenAICompatClient):
    MODELS = HFOpenAIModel

    def __init__(self, model: HFOpenAIModel, base_url: str = HF_OPENAI_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)
