import pydantic
from openai.types.chat import ParsedChatCompletion

from src.synthetic_data_generator.ai_graph.ai.base_ai_config import OpenAIModelVersion
from src.synthetic_data_generator.ai_graph.ai.base_ai_model import BaseAIModel
from synthetic_data_generator.ai_graph.ai.model_describer import ModelDescriber
from synthetic_data_generator.ai_graph.ai.open_ai_client import OpenAiClient


class AIModelGenerator[OM: BaseAIModel](BaseAIModel):
    def __init__(self, assistant_name: str, client: OpenAiClient, open_ai_model_version: OpenAIModelVersion,
                 temperature: float,
                 max_tokens: int, instructions: str,
                 input_describer: ModelDescriber,
                 retry_wait_min: int = 4, retry_wait_max: int = 128, retry_attempts: int = 20, ):
        super().__init__(assistant_name=assistant_name, client=client, model=open_ai_model_version,
                         temperature=temperature, max_tokens=max_tokens, instructions=instructions,
                         retry_wait_min=retry_wait_min, retry_wait_max=retry_wait_max, retry_attempts=retry_attempts,
                         input_describer=input_describer)

    async def _get_chat_completion(self, input_instance, output_type: type[pydantic.BaseModel], *args,
                                   **kwargs) -> ParsedChatCompletion:
        prompt = self._input_describer.generate_description(input_instance)
        return await self._client.get_parsed_chat_completion(prompt=prompt,
                                                             model_version=self._model.get_model_version(),
                                                             temperature=self._temperature,
                                                             max_tokens=self._max_tokens,
                                                             instruction=self._instructions,
                                                             response_format=output_type)

    async def get_parsed_completion(self, input_instance: pydantic.BaseModel, output_type: type[pydantic.BaseModel],
                                    *args, **kwargs) -> OM:
        return (await self._get_chat_completion(input_instance, output_type, *args, **kwargs)).choices[0].message.parsed
