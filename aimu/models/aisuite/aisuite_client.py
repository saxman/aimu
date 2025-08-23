from ..base_client import ModelClient

import aisuite
import logging

logger = logging.getLogger(__name__)


class AisuiteClient(ModelClient):
    MODEL_LLAMA_3_1_8B = "ollama:llama3.1:8b"
    MODEL_LLAMA_3_2_3B = "ollama:llama3.2:3b"

    MODEL_GEMMA_2_9B = "ollama:gemma2:9b"

    MODEL_PHI_4_14B = "ollama:phi4:14b"

    MODEL_DEEPSEEK_R1_8B = "ollama:deepseek-r1:8b"

    MODEL_MISTRAL_7B = "ollama:mistral:7b"
    MODEL_MISTRAL_NEMO_12B = "ollama:mistral-nemo"

    MODEL_QWEN_2_5_7B = "ollama:qwen2.5:7b"

    MODEL_GPT_4O_MINI = "openai:gpt-4o-mini"
    MODEL_GPT_4O = "openai:gpt-4o"

    def __init__(self, model_id: str):
        super().__init__(model_id, None)
        self.client = aisuite.Client()

    def generate(self, prompt, generate_kwargs):
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            self.model_id, messages, temperature=generate_kwargs["temperature"]
        )

        return response.choices[0].message.content
