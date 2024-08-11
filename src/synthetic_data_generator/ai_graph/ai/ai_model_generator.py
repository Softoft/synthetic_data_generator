import pydantic
from openai import AsyncOpenAI

from src.synthetic_data_generator.ai_graph.ai.ai_model import InputModel
from src.synthetic_data_generator.ai_graph.ai.base_ai import BaseAIModel
from src.synthetic_data_generator.ai_graph.ai.base_ai_config import OpenAIModelVersion


class AIModelGenerator(BaseAIModel):
    def __init__(self, assistant_name: str, client: AsyncOpenAI, open_ai_model_version: OpenAIModelVersion,
                 temperature: float,
                 max_tokens: int, instructions: str, output_model: type[pydantic.BaseModel], input_instance: InputModel,
                 retry_wait_min: int = 4, retry_wait_max: int = 128, retry_attempts: int = 20):
        super().__init__(assistant_name, client, open_ai_model_version, temperature, max_tokens, instructions,
                         retry_wait_min,
                         retry_wait_max, retry_attempts)
        self.input_instance = input_instance
        self.output_model = output_model

    def _create_prompt(self) -> str:
        pass

    async def get_parsed_completion(self, output_model: type[pydantic.BaseModel]) -> pydantic.BaseModel:
        prompt = self.input_instance.get_prompt()
        return (await self._client.beta.chat.completions.parse(
            model=self._model.get_model_version(),
            messages=[
                {"role": "system", "content": self._instructions},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            response_format=output_model,
            max_tokens=self._max_tokens
        )).choices[0].message.parsed
