import logging

import pytest

from src.synthetic_data_generator.ai_graph.ai.base_ai_analysis import AssistantAnalysisResult, AssistantRun, \
    AssistantRuns, calculate_cost
from src.synthetic_data_generator.ai_graph.ai.base_ai_config import AIModelType, OpenAIModelVersion


@pytest.mark.parametrize("prompt_tokens,completion_tokens,ai_type,expected_cost", [
    (1e6, 2e6, AIModelType.GPT_4o, 35),
    (2e6, 1e6, AIModelType.GPT_4o, 25),
    (2e6, 1e6, AIModelType.GPT_4o_MINI, 0.9),

])
def test_assistant_run(create_assistant_run, prompt_tokens, completion_tokens, ai_type, expected_cost):
    assistant_run: AssistantRun = create_assistant_run(assistant_name="Test", prompt_tokens=prompt_tokens,
                                                       completion_tokens=completion_tokens,
                                                       model=OpenAIModelVersion(model_version=ai_type.value))
    cost = assistant_run.cost
    assert cost == pytest.approx(expected_cost)


def test_assistant_run_composite(create_mocked_assistant_run):
    assistant_run_1 = create_mocked_assistant_run(1e6, 2e6, 100)
    assistant_run_2 = create_mocked_assistant_run(2e6, 1e6, 200)
    assistant_runs = AssistantRuns(_runs=[assistant_run_1, assistant_run_2])
    assert assistant_runs.cost == 300
    assert assistant_runs.prompt_tokens == 3e6
    assert assistant_runs.completion_tokens == 3e6


def test_assistant_analysis_result():
    assistant_analysis_result = AssistantAnalysisResult(name="Test",
                                                        cost=100,
                                                        total_cost=1000,
                                                        prompt_tokens=int(1e6),
                                                        completion_tokens=int(2e6)
                                                        )
    assert assistant_analysis_result.cost == 100
    assert assistant_analysis_result.total_cost == 1000
    assert assistant_analysis_result.percent_total_cost == 10
    assert assistant_analysis_result.prompt_tokens == 1e6
    assert assistant_analysis_result.completion_tokens == 2e6

    assert all([text in str(assistant_analysis_result).lower() for text in ["cost", "100", "10"]])


def test_assistant_analyzer_total_summary(create_mocked_assistant_run, chat_assistant_analyzer):
    assistant_run_1 = create_mocked_assistant_run(1e6, 2e6, 100)
    assistant_run_2 = create_mocked_assistant_run(2e6, 1e6, 200)

    chat_assistant_analyzer.append_assistant_run(assistant_run_1)
    chat_assistant_analyzer.append_assistant_run(assistant_run_2)

    assert chat_assistant_analyzer.total_summary().cost == 300
    assert chat_assistant_analyzer.total_summary().prompt_tokens == 3e6
    assert chat_assistant_analyzer.total_summary().completion_tokens == 3e6


def test_print_assistant_analyzer(create_mocked_assistant_run, chat_assistant_analyzer):
    assistant_run_1 = create_mocked_assistant_run(1e6, 2e6, 100)
    assistant_run_2 = create_mocked_assistant_run(2e6, 1e6, 200)

    chat_assistant_analyzer.append_assistant_run(assistant_run_1)
    chat_assistant_analyzer.append_assistant_run(assistant_run_2)

    assistant_analyzer_text = str(chat_assistant_analyzer)
    assert all([text in assistant_analyzer_text.lower() for text in ["cost", "300"]])


@pytest.mark.asyncio
@pytest.mark.slow
async def test_chat_assistant_analyzer(create_mocked_assistant, chat_assistant_analyzer):
    await create_mocked_assistant("test1", "gpt-4o-mini", 10, 20).get_chat_completion()
    logging.info(chat_assistant_analyzer)
    assert chat_assistant_analyzer.total_summary().cost == calculate_cost(10, 20, AIModelType.GPT_4o_MINI)
    assert chat_assistant_analyzer.total_summary().prompt_tokens == 10
    assert 0 < chat_assistant_analyzer.total_summary().completion_tokens == 20


@pytest.mark.asyncio
@pytest.mark.slow
async def test_chat_assistant_analyzer_get_by_name(create_mocked_assistant, chat_assistant_analyzer):
    await create_mocked_assistant("test1", "gpt-4o-mini", 10, 20).get_chat_completion()
    await create_mocked_assistant("test1", "gpt-4o-mini", 5, 5).get_chat_completion()
    await create_mocked_assistant("test2", "gpt-4o", 10, 10).get_chat_completion()

    summary1 = chat_assistant_analyzer.get_summary_for_assistant("test1")
    summary2 = chat_assistant_analyzer.get_summary_for_assistant("test2")

    assert summary1.prompt_tokens == 15
    assert summary1.completion_tokens == 25
    assert summary1.cost == calculate_cost(15, 25, AIModelType.GPT_4o_MINI)
    assert summary2.prompt_tokens == 10
    assert summary2.completion_tokens == 10
    assert summary2.cost == calculate_cost(10, 10, AIModelType.GPT_4o)
