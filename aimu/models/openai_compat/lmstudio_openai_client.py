from ..base import Model, ModelSpec
from .openai_compat_client import OpenAICompatClient

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"


class LMStudioOpenAIModel(Model):
    # Model values are the model "key" as shown in LM Studio's loaded model list.
    LLAMA_3_1_8B = ModelSpec("llama-3.1-8b-instruct")
    MISTRAL_7B = ModelSpec("mistral-7b-instruct-v0.3", tools=True)
    PHI_4_MINI = ModelSpec("phi-4-mini-instruct", tools=True)
    QWEN_3_4B = ModelSpec("qwen3-4b", tools=True, thinking=True)
    QWEN_3_8B = ModelSpec("qwen3-8b", tools=True, thinking=True)
    QWEN_3_5_9B = ModelSpec("qwen3.5-9b", tools=True, thinking=True)
    DEEPSEEK_R1_7B = ModelSpec("deepseek-r1-distill-qwen-7b", thinking=True)


class LMStudioOpenAIClient(OpenAICompatClient):
    MODELS = LMStudioOpenAIModel

    def __init__(self, model: LMStudioOpenAIModel, base_url: str = LMSTUDIO_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)
