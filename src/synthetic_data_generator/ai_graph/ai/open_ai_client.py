from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ParsedChatCompletion


class OpenAiClient:
    def __init__(self, async_open_ai: AsyncOpenAI):
        self.async_open_ai = async_open_ai

    async def get_chat_completion(self, prompt, instruction, model_version, temperature, max_tokens) -> ChatCompletion:
        return await self.async_open_ai.chat.completions.create(
            model=model_version,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

    async def get_parsed_chat_completion(self, prompt, instruction, model_version, temperature, max_tokens,
                                         response_format) -> ParsedChatCompletion:
        return (await self.async_open_ai.beta.chat.completions.parse(
            model=model_version,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            response_format=response_format,
            max_tokens=max_tokens
        ))
