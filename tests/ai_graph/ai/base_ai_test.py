import pytest
from pydantic import BaseModel

from src.synthetic_data_generator.ai_graph.ai.base_ai_config import AssistantModel


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
