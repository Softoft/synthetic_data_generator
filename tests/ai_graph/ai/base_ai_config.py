import pytest

from src.synthetic_data_generator.ai_graph.ai.base_ai_config import AIModelType, OpenAIModelVersion


def test_base_ai_config():
    model = OpenAIModelVersion(model_version="gpt-4o-2024-08-13")
    assert model.get_model_type() == AIModelType.GPT_4o
    model = OpenAIModelVersion(model_version="gpt-4o-mini-2024-08-13")
    assert model.get_model_type() == AIModelType.GPT_4o_MINI

    model = OpenAIModelVersion(model_version="unknown")
    with pytest.raises(ValueError):
        model.get_model_type()
