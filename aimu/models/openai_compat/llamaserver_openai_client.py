from ..base_client import Model
from .openai_compat_client import OpenAICompatClient

LLAMASERVER_BASE_URL = "http://localhost:8080/v1"


class LlamaServerOpenAIModel(Model):
    def __init__(self, value, supports_tools=False, supports_thinking=False):
        super().__init__(value, supports_tools, supports_thinking)

    # Model values are the GGUF file name (or alias) as loaded by llama-server.
    # llama-server ignores the model field in API requests and always uses the loaded model;
    # these names are used for capability lookup only.
    LLAMA_3_1_8B = ("llama-3.1-8b-instruct.gguf", True)
    LLAMA_3_2_3B = ("llama-3.2-3b-instruct.gguf", True)
    MISTRAL_7B = ("mistral-7b-instruct-v0.3.gguf", True)
    PHI_4_MINI = ("phi-4-mini-instruct.gguf", True)
    QWEN_3_4B = ("qwen3-4b.gguf", True, True)
    QWEN_3_8B = ("qwen3-8b.gguf", True, True)
    DEEPSEEK_R1_7B = ("deepseek-r1-distill-qwen-7b.gguf", False, True)
    GEMMA_3_12B = ("gemma-3-12b-it.gguf", True)


class LlamaServerOpenAIClient(OpenAICompatClient):
    """Client for llama.cpp's llama-server OpenAI-compatible REST API.

    Start the server with:
        llama-server -m /path/to/model.gguf --port 8080
    """

    MODELS = LlamaServerOpenAIModel

    def __init__(self, model: LlamaServerOpenAIModel, base_url: str = LLAMASERVER_BASE_URL, **kwargs):
        super().__init__(model, base_url=base_url, **kwargs)
