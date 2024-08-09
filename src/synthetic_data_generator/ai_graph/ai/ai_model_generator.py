import logging

import pydantic

from src.synthetic_data_generator.ai_graph.ai.ai_model import InputModel, OutputModel
from src.synthetic_data_generator.ai_graph.ai.base_ai import BaseAI


class AIModelGenerator(BaseAI):
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
        input_instance: The input instance.
    """
    input_instance: InputModel

    async def get_parsed_completion(self, output_model: type[pydantic.BaseModel]) -> pydantic.BaseModel:
        prompt = self.input_instance.get_prompt()
        return await self.client.beta.chat.completions.parse(
            model=self.model.value,
            messages=[
                { "role": "system", "content": self.instructions },
                { "role": "user", "content": prompt },
            ],
            temperature=self.temperature,
            response_format=output_model,
            max_tokens=self.max_tokens
        )
