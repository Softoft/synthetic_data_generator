import pytest
from pydantic import BaseModel

from src.synthetic_data_generator.ai_graph.ai.chat_assistant_config import AssistantModel


@pytest.mark.asyncio
@pytest.mark.slow
async def test_chat_assistant_correct_response(chat_assistant_gpt4_o_mini):
    response = await chat_assistant_gpt4_o_mini._get_response("What is the capital of Germany?")
    assert "Berlin" in response


@pytest.mark.asyncio
@pytest.mark.slow
async def test_chat_assistant_creation(create_chat_assistant):
    chat_assistant = await create_chat_assistant(AssistantModel.GPT_4o_MINI,
                                                 "You are a Simple Chatbot, Answer in short sentences.")
    response = await chat_assistant._get_response("What is the capital of Germany?")
    assert "Berlin" in response


@pytest.mark.asyncio
@pytest.mark.slow
async def test_chat_assistant_gpt4_o_creation(create_chat_assistant):
    chat_assistant = await create_chat_assistant(AssistantModel.GPT_4o,
                                                 "You are a Simple Chatbot, Answer in short sentences.")
    response = await chat_assistant._get_response("What is the capital of Germany?")
    assert "Berlin" in response


@pytest.mark.asyncio
@pytest.mark.slow
async def test_chat_assistant_json_response(create_chat_assistant):
    class City(BaseModel):
        city: str
        country: str

    chat_assistant = await create_chat_assistant(AssistantModel.GPT_4o,
                                                 "Return a JSON object with attributes: city, country",
                                                 response_format={"type": "json_schema", "schema": City.schema()})
    response = await chat_assistant.get_dict_response("What country is Karlsruhe in?")
    assert response["country"] == "Germany"
    assert response["city"] == "Karlsruhe"
