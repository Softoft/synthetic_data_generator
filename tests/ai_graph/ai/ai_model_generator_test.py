import pydantic
import pytest
from openai import AsyncOpenAI

from src.synthetic_data_generator.ai_graph.ai.ai_model_generator import AIModelGenerator
from src.synthetic_data_generator.ai_graph.ai.base_ai_config import AIModelType, OpenAIModelVersion
from synthetic_data_generator.ai_graph.ai.model_describer import ModelDescriber
from synthetic_data_generator.ai_graph.ai.open_ai_client import OpenAiClient


class City(pydantic.BaseModel):
    name: str

    def get_prompt(self):
        return self.name


class CityDescriber(ModelDescriber):
    def generate_description(self, city: City):
        return f"city: {city.name}"


class Country(pydantic.BaseModel):
    name: str

    def get_prompt(self):
        return self.name


@pytest.mark.asyncio
async def test_ai_model_generator():
    city = City(name="Karlsruhe")
    ai_model_generator = AIModelGenerator[Country](
        assistant_name="Test",
        client=OpenAiClient(AsyncOpenAI()),
        temperature=0.5,
        max_tokens=100,
        instructions="country of this city",
        open_ai_model_version=OpenAIModelVersion(model_version=AIModelType.GPT_4o_MINI.value),
        retry_wait_min=0,
        retry_wait_max=0,
        retry_attempts=1,
        input_describer=CityDescriber()
    )
    country = await ai_model_generator.get_parsed_completion(city, Country)
    assert country.name == "Germany"
