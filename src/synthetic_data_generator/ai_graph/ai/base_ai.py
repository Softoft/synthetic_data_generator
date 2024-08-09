from typing import Optional

import openai
import pydantic
from openai import AsyncOpenAI
from pydantic import Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from src.synthetic_data_generator.ai_graph.ai.base_ai_config import AssistantModel


class BaseAI(pydantic.BaseModel):
    """
    Args:
        assistant_name: The name of the assistant.
        client: The OpenAI client.
        temperature: The temperature of the model.
        max_tokens: The maximum number of tokens to generate.
        instructions: The instructions for the assistant.
        model: The model to use.
        retry_wait_min: The minimum wait time for retries.
        retry_wait_max: The maximum wait time for retries.
        retry_attempts: The number of retry attempts.
    """

    class Config:
        arbitrary_types_allowed = True

    assistant_name: str
    client: AsyncOpenAI
    temperature: float = Field(default=1, ge=0, le=2)
    max_tokens: int = Field(default=4_000, ge=0, le=64_000)
    instructions: Optional[str] = None
    model: AssistantModel
    retry_wait_min: int = Field(default=4)
    retry_wait_max: int = Field(default=128)
    retry_attempts: int = Field(default=20)

    async def _get_response(self, prompt: str) -> str:
        return (await self.client.chat.completions.create(
            model=self.model.value,
            messages=[
                { "role": "system", "content": self.instructions },
                { "role": "user", "content": prompt },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )).choices[0].message.content

    # noinspection PyArgumentList
    async def get_response_with_retry(self, prompt: str) -> str:
        response_retry_func = retry(
            wait=wait_random_exponential(min=self.retry_wait_min, max=self.retry_wait_max),
            stop=stop_after_attempt(self.retry_attempts),
            retry=retry_if_exception_type((openai.APITimeoutError, openai.RateLimitError, openai.APIConnectionError))
        )(self._get_response)
        return await response_retry_func(prompt)
