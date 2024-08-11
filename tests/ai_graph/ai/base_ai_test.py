import pytest

from src.synthetic_data_generator.ai_graph.ai.base_ai_config import AIModelType, OpenAIModelVersion


@pytest.mark.asyncio
@pytest.mark.slow
async def test_chat_assistant_creation(create_chat_assistant):
    chat_assistant_gpt4o_mini = create_chat_assistant(OpenAIModelVersion(model_version=AIModelType.GPT_4o_MINI.value),
                                                      instructions="Answer in one word")
    response = await chat_assistant_gpt4o_mini._get_string_response("What is the capital of Germany?")
    assert "Berlin" in response

    chat_assistant = create_chat_assistant(OpenAIModelVersion(model_version=AIModelType.GPT_4o.value),
                                           instructions="1 word answer",
                                           max_tokens=2)
    response = await chat_assistant.get_response_with_retry("capital of Germany?")
    assert "Berlin" in response
