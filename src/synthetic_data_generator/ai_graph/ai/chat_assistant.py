import json
from typing import Any, Coroutine, Optional

import openai
from openai import AsyncOpenAI
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from src.synthetic_data_generator.ai_graph.ai.chat_assistant_analysis import cost_analyzer
from src.synthetic_data_generator.ai_graph.ai.chat_assistant_config import AssistantModel


class ChatAssistant(BaseModel):
    """
    Args:
        assistant_name: The name of the assistant.
        client: The OpenAI client.
        temperature: The temperature of the model.
        max_tokens: The maximum number of tokens to generate.
        response_schema: The response schema.
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

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return {
            "assistant_name": self.assistant_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_schema": self.response_schema,
            "instructions": self.instructions,
            "model": self.model,
            "retry_wait_min": self.retry_wait_min,
            "retry_wait_max": self.retry_wait_max,
            "retry_attempts": self.retry_attempts,
        }

    def generate_instructions(self, instructions: str) -> str:
        return f"Instructions: {instructions}"

    def generate_user_prompt(self, prompt: str) -> str:
        return f"User: {prompt}"

    async def create_run(self, prompt) -> ParsedChatCompletion:
        return await self.client.beta.chat.completions.parse(
            model=self.model.value,
            messages=[
                { "role": "system", "content": self.instructions },
                { "role": "user", "content": prompt },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    async def _get_response(self, prompt: str) -> str:
        return (await self.create_run(prompt)).choices[0].message.content

    async def get_dict_response(self, prompt: str) -> dict:
        return json.loads(await self._get_response(prompt))

    # noinspection PyArgumentList
    async def get_response_with_retry(self, prompt: str) -> str:
        response_retry_func = retry(
            wait=wait_random_exponential(min=self.retry_wait_min, max=self.retry_wait_max),
            stop=stop_after_attempt(self.retry_attempts),
            retry=retry_if_exception_type((openai.APITimeoutError, openai.RateLimitError, openai.APIConnectionError))
        )(self._get_response)
        return await response_retry_func(prompt)
