from ..base import Model, ModelSpec
from .openai_compat_client import OpenAICompatClient

LLAMASERVER_BASE_URL = "http://localhost:8080/v1"


class LlamaServerOpenAIModel(Model):
    # Model values are the GGUF file name (or alias) as loaded by llama-server.
    # llama-server ignores the model field in API requests and always uses the loaded model;
    # these names are used for capability lookup only.
    LLAMA_3_1_8B = ModelSpec("llama-3.1-8b-instruct.gguf", tools=True)
    LLAMA_3_2_3B = ModelSpec("llama-3.2-3b-instruct.gguf", tools=True)
    MISTRAL_7B = ModelSpec("mistral-7b-instruct-v0.3.gguf", tools=True)
    PHI_4_MINI = ModelSpec("phi-4-mini-instruct.gguf", tools=True)
    QWEN_3_4B = ModelSpec("qwen3-4b.gguf", tools=True, thinking=True)
    QWEN_3_8B = ModelSpec("qwen3-8b.gguf", tools=True, thinking=True)
    DEEPSEEK_R1_7B = ModelSpec("deepseek-r1-distill-qwen-7b.gguf", thinking=True)
    GEMMA_3_12B = ModelSpec("gemma-3-12b-it.gguf", tools=True)


class LlamaServerOpenAIClient(OpenAICompatClient):
    """Client for llama.cpp's llama-server OpenAI-compatible REST API.

    Start the server with:
        llama-server -m /path/to/model.gguf --port 8080
    """

    MODELS = LlamaServerOpenAIModel

    def __init__(self, model: LlamaServerOpenAIModel, base_url: str = LLAMASERVER_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)
