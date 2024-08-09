import pytest
from openai import AsyncOpenAI

from src.synthetic_data_generator.ai_graph.ai.ai_model import InputModel, OutputModel
from src.synthetic_data_generator.ai_graph.ai.ai_model_generator import AIModelGenerator
from src.synthetic_data_generator.ai_graph.ai.base_ai_config import AssistantModel


class City(InputModel):
    name: str

    def get_prompt(self):
        return self.name


class Country(OutputModel):
    name: str

    def get_prompt(self):
        return self.name


@pytest.mark.asyncio
async def test_ai_model_generator():
    city = City(name="Karlsruhe")
    ai_model_generator = AIModelGenerator(
        assistant_name="Test",
        client=AsyncOpenAI(),
        temperature=0.5,
        max_tokens=100,
        instructions="What country is the city in?",
        model=AssistantModel.GPT_4o,
        retry_wait_min=0,
        retry_wait_max=0,
        retry_attempts=1,
        input_instance=city
    )
    country = await ai_model_generator.get_parsed_completion(Country)
