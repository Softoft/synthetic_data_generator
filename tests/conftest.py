import asyncio
import logging
from enum import auto
from unittest.mock import Mock, create_autospec

import pytest
from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.beta.threads.run import Usage
from openai.types.chat import ChatCompletion

from src.synthetic_data_generator.ai_graph.ai.base_ai import PlainResponseAI
from src.synthetic_data_generator.ai_graph.ai.base_ai_analysis import AssistantAnalyzer, AssistantRun, IAssistantRun, \
    cost_analyzer
from src.synthetic_data_generator.ai_graph.ai.base_ai_config import AIModelType, OpenAIModelVersion
from src.synthetic_data_generator.ai_graph.key_value_store import KeyValueStore
from src.synthetic_data_generator.ai_graph.nodes.executable_node import ExecutableNode
from src.synthetic_data_generator.random_generators.number_interval_generator import NormalizedNumberGenerator, \
    NumberIntervalGenerator
from src.synthetic_data_generator.random_generators.random_collection import RandomCollectionFactory
from src.synthetic_data_generator.random_generators.random_collection_table import RandomTableBuilder
from src.synthetic_data_generator.random_nodes.random_collection_node import RandomCollectionNode
from src.synthetic_data_generator.random_nodes.random_table_node import RandomTableNode
from src.synthetic_data_generator.random_nodes.ticket_field import ComparableEnum


class KeyEnum(ComparableEnum):
    K1 = auto()
    K2 = auto()


class ValueEnum(ComparableEnum):
    V1 = auto()
    V2 = auto()
    V3 = auto()


class BigEnum(ComparableEnum):
    B1 = auto()
    B2 = auto()
    B3 = auto()
    B4 = auto()
    B5 = auto()
    B6 = auto()


class ConfigurableEnumSaveNode(ExecutableNode):
    def __init__(self, key_enum_value: KeyEnum):
        self.key_enum_value = key_enum_value
        super().__init__([])

    async def _execute_node(self, shared_storage: KeyValueStore) -> KeyValueStore:
        if KeyEnum in shared_storage:
            return shared_storage
        shared_storage.save(self.key_enum_value)
        return shared_storage


@pytest.fixture(scope="session")
def create_chat_assistant():
    client = AsyncOpenAI()

    def _create_chat_assistant(model: OpenAIModelVersion, instructions: str, assistant_name="Test Chatbot",
                               temperature: float = 0.5, max_tokens: int = 400, prompt: str = ""):
        if model == AIModelType.GPT_4o:
            logging.error("GPT4o is expensive! Use GPT4o Mini for testing")

        test_chat_assistant = PlainResponseAI(assistant_name=assistant_name, instructions=instructions, client=client,
                                              temperature=temperature, model=model, max_tokens=max_tokens,
                                              retry_wait_min=0, retry_wait_max=0,
                                              retry_attempts=1, prompt=prompt)
        return test_chat_assistant

    return _create_chat_assistant


@pytest.fixture(scope="session")
def create_mocked_assistant(create_chat_assistant):
    @cost_analyzer()
    class TestAssistant:
        def __init__(self, assistant_name, model_version: str, prompt_tokens, completion_tokens):
            self.model_version = model_version
            self.assistant_name = assistant_name
            self.completion_tokens = completion_tokens
            self.prompt_tokens = prompt_tokens

        async def _get_chat_completion(self):
            chat_completion = Mock(spec=ChatCompletion)
            chat_completion.usage = CompletionUsage(completion_tokens=self.completion_tokens,
                                                    prompt_tokens=self.prompt_tokens,
                                                    total_tokens=self.completion_tokens + self.prompt_tokens)
            chat_completion.model = self.model_version
            return chat_completion

        async def get_chat_completion(self):
            return await self._get_chat_completion()

    def _create_mocked_assistant(assistant_name, model_version, prompt_tokens, completion_tokens):
        return TestAssistant(assistant_name, model_version, prompt_tokens, completion_tokens)

    return _create_mocked_assistant


@pytest.fixture
def create_assistant_run():
    def _create_assistant_run(assistant_name, completion_tokens, prompt_tokens, model: OpenAIModelVersion):
        run = Mock(spec=IAssistantRun)
        run.get_usage.return_value = Usage(completion_tokens=completion_tokens, prompt_tokens=prompt_tokens,
                                           total_tokens=completion_tokens + prompt_tokens)
        run.get_model.return_value = model
        return AssistantRun(run=run, assistant_name=assistant_name)

    return _create_assistant_run


@pytest.fixture
def create_mocked_assistant_run():
    def _create_mocked_assistant_run(completion_tokens, prompt_tokens, cost: float):
        run = create_autospec(AssistantRun, instance=True)
        run.assistant_name = "Test"
        run.model = AIModelType.GPT_4o_MINI
        run.prompt_tokens = prompt_tokens
        run.completion_tokens = completion_tokens
        run.cost = cost
        return run

    return _create_mocked_assistant_run


@pytest.fixture()
def chat_assistant_analyzer():
    yield AssistantAnalyzer()
    AssistantAnalyzer().reset()


@pytest.fixture
def create_enum_save_node():
    def _create_enum_save_node(key_enum_value: KeyEnum):
        return ConfigurableEnumSaveNode(key_enum_value)

    return _create_enum_save_node


@pytest.fixture
def create_random_table_node(create_enum_save_node):
    def _create_random_table_node(key_value, table_weight_dict):
        random_table = RandomTableBuilder().build_from_dict(KeyEnum, ValueEnum, table_weight_dict)
        return RandomTableNode(KeyEnum, ValueEnum, [create_enum_save_node(key_value)], random_table)

    return _create_random_table_node


@pytest.fixture
def create_random_collection_node():
    def _create_random_collection_node(value_weight_dict):
        random_collection = RandomCollectionFactory().build_from_value_weight_dict(value_weight_dict)
        return RandomCollectionNode(ValueEnum, [], random_collection)

    return _create_random_collection_node


@pytest.fixture
def normalized_number_generator():
    return NormalizedNumberGenerator()


@pytest.fixture
def get_random_numbers(normalized_number_generator):
    def _get_random_numbers(mean, standard_deviation, number_bounds=None, amount: int = 10_000):
        return [normalized_number_generator.generate_bounded_number(mean, standard_deviation, number_bounds) for _ in
                range(amount)]

    return _get_random_numbers


@pytest.fixture
def get_mocked_random_interval():
    def _get_mocked_random_interval(lower_value, min_upper_bound_difference):
        normalized_number_generator = Mock(spec=NormalizedNumberGenerator)
        normalized_number_generator.generate_bounded_number.return_value = lower_value

        return NumberIntervalGenerator(mean=0, standard_deviation=1,
                                       min_upper_bound_difference=min_upper_bound_difference,
                                       lower_number_generator=normalized_number_generator)

    return _get_mocked_random_interval


@pytest.fixture
def create_key_value_store():
    def _create_key_value_store(*stored_values):
        key_value_store = KeyValueStore()
        for value in stored_values:
            key_value_store.save(value)
        return key_value_store

    return _create_key_value_store


@pytest.fixture(autouse=True)
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
