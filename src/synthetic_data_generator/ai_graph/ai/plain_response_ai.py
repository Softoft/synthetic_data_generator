from typing import Optional

from synthetic_data_generator.ai_graph.ai.base_ai_config import OpenAIModelVersion
from synthetic_data_generator.ai_graph.ai.base_ai_model import BaseAIModel
from synthetic_data_generator.ai_graph.ai.model_describer import ModelDescriber
from synthetic_data_generator.ai_graph.ai.open_ai_client import OpenAiClient


class BasicPromptGenerator(ModelDescriber):

    def generate_description(self, prompt: str, **kwargs) -> str:
        return prompt


class PlainResponseAI(BaseAIModel[str]):
    def __init__(self,
                 assistant_name: str,
                 client: OpenAiClient,
                 model: OpenAIModelVersion,
                 temperature: float = 1.0,
                 max_tokens: int = 4000,
                 instructions: Optional[str] = None,
                 retry_wait_min: int = 4,
                 retry_wait_max: int = 128,
                 retry_attempts: int = 20):
        super().__init__(
            assistant_name=assistant_name,
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            instructions=instructions,
            retry_wait_min=retry_wait_min,
            retry_wait_max=retry_wait_max,
            retry_attempts=retry_attempts,
            input_describer=BasicPromptGenerator()
        )
