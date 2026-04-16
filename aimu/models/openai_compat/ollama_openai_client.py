from ..base import Model
from .openai_compat_client import OpenAICompatClient

OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaOpenAIModel(Model):
    def __init__(self, value, supports_tools=False, supports_thinking=False):
        super().__init__(value, supports_tools, supports_thinking)

    # Model values are Ollama model tags (as used by `ollama pull`).
    LLAMA_3_1_8B = ("llama3.1:8b", False)
    LLAMA_3_2_3B = ("llama3.2:3b", False)
    MISTRAL_7B = ("mistral:7b", True)
    PHI_4_MINI = ("phi4-mini:3.8b", True)
    QWEN_3_4B = ("qwen3:4b", True, True)
    QWEN_3_8B = ("qwen3:8b", True, True)
    QWEN_3_5_9B = ("qwen3.5:9b", True, True)
    DEEPSEEK_R1_8B = ("deepseek-r1:8b", False, True)
    GEMMA_3_12B = ("gemma3:12b", False)


class OllamaOpenAIClient(OpenAICompatClient):
    MODELS = OllamaOpenAIModel

    def __init__(self, model: OllamaOpenAIModel, base_url: str = OLLAMA_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)
