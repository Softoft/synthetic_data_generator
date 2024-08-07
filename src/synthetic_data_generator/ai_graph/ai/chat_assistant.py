import json
import logging
from typing import Any

import openai
from openai import AsyncOpenAI, BaseModel
from pydantic import Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from src.synthetic_data_generator.ai_graph.ai.chat_assistant_analysis import cost_analyzer


@cost_analyzer()
class ChatAssistant(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    assistant_name: str
    client: AsyncOpenAI
    temperature: float = Field(default=1, ge=0, le=2)
    max_prompt_tokens: int = Field(default=4_000, ge=0, le=64_000)
    max_completion_tokens: int = Field(default=4_000, ge=0, le=64_000)
    retry_wait_min: int = Field(default=4)
    retry_wait_max: int = Field(default=128)
    retry_attempts: int = Field(default=20)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return {
            "assistant_name": self.assistant_name,
            "assistant_id": self.assistant_id,
            "temperature": self.temperature,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_completion_tokens": self.max_completion_tokens
        }

    def generate_instructions(self, instructions: str) -> str:
        return f"Instructions: {instructions}"

    def generate_user_prompt(self, prompt: str) -> str:
        return f"User: {prompt}"

    async def create_run(self, thread_id, assistant_id):
        logging.info(f"Creating run for thread {thread_id} and assistant {assistant_id}")
        return await self.client.beta.threads.runs.create_and_poll(thread_id=thread_id,
                                                                   assistant_id=assistant_id,
                                                                   temperature=self.temperature,
                                                                   max_prompt_tokens=self.max_prompt_tokens,
                                                                   max_completion_tokens=self.max_completion_tokens)

    async def _get_response(self, prompt: str) -> str:
        open_ai_assistant = await self.client.beta.assistants.retrieve(self.assistant_id)
        thread = await self.client.beta.threads.create()
        await self.client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
        await self.create_run(thread.id, open_ai_assistant.id)
        messages = await self.client.beta.threads.messages.list(thread_id=thread.id)
        message = messages.data[0].content[0].text.value
        logging.info(f"Got response: \"{message}\"")
        return message

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
