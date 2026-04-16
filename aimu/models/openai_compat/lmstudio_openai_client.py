from ..base import Model
from .openai_compat_client import OpenAICompatClient

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"


class LMStudioOpenAIModel(Model):
    def __init__(self, value, supports_tools=False, supports_thinking=False):
        super().__init__(value, supports_tools, supports_thinking)

    # Model values are the model "key" as shown in LM Studio's loaded model list.
    LLAMA_3_1_8B = ("llama-3.1-8b-instruct", False)
    MISTRAL_7B = ("mistral-7b-instruct-v0.3", True)
    PHI_4_MINI = ("phi-4-mini-instruct", True)
    QWEN_3_4B = ("qwen3-4b", True, True)
    QWEN_3_8B = ("qwen3-8b", True, True)
    QWEN_3_5_9B = ("qwen3.5-9b", True, True)
    DEEPSEEK_R1_7B = ("deepseek-r1-distill-qwen-7b", False, True)


class LMStudioOpenAIClient(OpenAICompatClient):
    MODELS = LMStudioOpenAIModel

    def __init__(self, model: LMStudioOpenAIModel, base_url: str = LMSTUDIO_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)
