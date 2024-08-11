import logging
from abc import ABC
from typing import Optional

import openai
from openai.types.chat import ChatCompletion
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.synthetic_data_generator.ai_graph.ai.base_ai_config import OpenAIModelVersion
from src.synthetic_data_generator.ai_graph.ai.i_ai_model import IAIModel
from synthetic_data_generator.ai_graph.ai.base_ai_analysis import cost_analyzer
from synthetic_data_generator.ai_graph.ai.model_describer import ModelDescriber
from synthetic_data_generator.ai_graph.ai.open_ai_client import OpenAiClient


@cost_analyzer()
class BaseAIModel[ResultType](IAIModel, ABC):
    def __init__(self,
                 assistant_name: str,
                 client: OpenAiClient,
                 model: OpenAIModelVersion,
                 input_describer: ModelDescriber,
                 temperature: float = 1.0,
                 max_tokens: int = 4000,
                 instructions: Optional[str] = None,
                 retry_wait_min: int = 4,
                 retry_wait_max: int = 128,
                 retry_attempts: int = 20):
        if not isinstance(assistant_name, str):
            raise TypeError("assistant_name must be a string")
        if not isinstance(client, OpenAiClient):
            raise TypeError("client must be of type AsyncOpenAI")
        if not isinstance(model, OpenAIModelVersion):
            raise TypeError("model must be of type OpenAIModelVersion")
        if not isinstance(temperature, int | float):
            raise TypeError("temperature must be a float")
        if not isinstance(max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if instructions is not None and not isinstance(instructions, str):
            raise TypeError("instructions must be a string or None")
        if not isinstance(retry_wait_min, int):
            raise TypeError("retry_wait_min must be an integer")
        if not isinstance(retry_wait_max, int):
            raise TypeError("retry_wait_max must be an integer")
        if not isinstance(retry_attempts, int):
            raise TypeError("retry_attempts must be an integer")

        self._assistant_name = assistant_name
        self._client: OpenAiClient = client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._instructions = instructions
        self._model = model
        self._retry_wait_min = retry_wait_min
        self._retry_wait_max = retry_wait_max
        self._retry_attempts = retry_attempts
        self._input_describer = input_describer

    async def _get_chat_completion(self, *args, **kwargs) -> ChatCompletion:
        prompt = self._input_describer.generate_description(*args, **kwargs)
        return await self._client.get_chat_completion(
            model_version=self._model.get_model_version(),
            prompt=prompt,
            instruction=self._instructions,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    async def _get_string_response(self, *args, **kwargs) -> ResultType:
        response = await self._get_chat_completion(*args, **kwargs)
        logging.info(f"Response: {response}")
        return response.choices[0].message.content

    async def get_response_with_retry(self, *args, **kwargs) -> ResultType:
        response_retry_func = retry(
            wait=wait_random_exponential(min=self._retry_wait_min, max=self._retry_wait_max),
            stop=stop_after_attempt(self._retry_attempts),
            retry=retry_if_exception_type((openai.APITimeoutError, openai.RateLimitError, openai.APIConnectionError))
        )(self._get_string_response)
        return await response_retry_func(*args, **kwargs)

    @property
    def assistant_name(self):
        return self._assistant_name
