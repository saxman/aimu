from ..base import Model, ModelSpec
from .openai_compat_client import OpenAICompatClient

OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaOpenAIModel(Model):
    # Model values are Ollama model tags (as used by `ollama pull`).
    LLAMA_3_1_8B = ModelSpec("llama3.1:8b")
    LLAMA_3_2_3B = ModelSpec("llama3.2:3b")
    MISTRAL_7B = ModelSpec("mistral:7b", tools=True)
    PHI_4_MINI = ModelSpec("phi4-mini:3.8b", tools=True)
    QWEN_3_4B = ModelSpec("qwen3:4b", tools=True, thinking=True)
    QWEN_3_8B = ModelSpec("qwen3:8b", tools=True, thinking=True)
    QWEN_3_5_9B = ModelSpec("qwen3.5:9b", tools=True, thinking=True)
    DEEPSEEK_R1_8B = ModelSpec("deepseek-r1:8b", thinking=True)
    GEMMA_3_12B = ModelSpec("gemma3:12b")


class OllamaOpenAIClient(OpenAICompatClient):
    MODELS = OllamaOpenAIModel

    def __init__(self, model: OllamaOpenAIModel, base_url: str = OLLAMA_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)
