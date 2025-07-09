import logging

logger = logging.getLogger(__name__)


class ModelClient:
    MODEL_LLAMA_3_1_8B = None
    MODEL_LLAMA_3_2_3B = None
    MODEL_LLAMA_3_3_70B = None

    MODEL_GEMMA_3_12B = None

    MODEL_PHI_4_14B = None
    MODEL_PHI_4_MINI_3_8B = None

    MODEL_DEEPSEEK_R1_8B = None

    MODEL_MISTRAL_7B = None
    MODEL_MISTRAL_NEMO_12B = None
    MODEL_MISTRAL_SMALL_3_2_24B = None

    MODEL_QWEN_3_8B = None

    MODEL_GPT_4O_MINI = None
    MODEL_GPT_4O = None

    # TODO: add more models and enable use by clients
    DEFAULT_MODEL_TEMPERATURES = {
        MODEL_MISTRAL_NEMO_12B: 0.3,
    }

    def __init__(self, model_id: str, model_kwargs: dict = None):
        if model_id is None:
            raise ValueError("Model not supported by model client")

        self.model_id = model_id
        self.model_kwargs = model_kwargs

        self.messages = []

        self.mcp_client = None
