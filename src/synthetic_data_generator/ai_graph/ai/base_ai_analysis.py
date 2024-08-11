import enum
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import pydantic
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from src.synthetic_data_generator.ai_graph.ai.base_ai_config import AIModelType, OpenAIModelVersion
from src.synthetic_data_generator.ai_graph.ai.i_ai_model import IAIModel


def cost_analyzer(warning_limit: float = 1e-4, error_limit=1e-2):
    def cost_analyzer_decorator(cls: IAIModel):
        original_init = cls.__init__

        def new_init(self: IAIModel, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _get_chat_completion = self._get_chat_completion

            async def wrapped_chat_completion(*_args, **_kwargs):
                chat_completion: ChatCompletion = await _get_chat_completion(*_args, **_kwargs)
                new_assistant_run = AssistantRun(assistant_name=self.assistant_name,
                                                 run=ChatCompletionAssistantRunAdapter(
                                                     chat_completion=chat_completion))
                if new_assistant_run.cost > warning_limit:
                    logging.warning(f"Cost of run is {new_assistant_run.cost}")
                if new_assistant_run.cost > error_limit:
                    logging.error(f"Cost of run is {new_assistant_run.cost}")
                AssistantAnalyzer().append_assistant_run(new_assistant_run)
                return chat_completion

            self._get_chat_completion = wrapped_chat_completion

        cls.__init__ = new_init
        return cls

    return cost_analyzer_decorator


class CostType(Enum):
    INPUT = enum.auto()
    OUTPUT = enum.auto()


class IUsage(ABC):
    @property
    @abstractmethod
    def prompt_tokens(self) -> int:
        pass

    @property
    @abstractmethod
    def completion_tokens(self) -> int:
        pass


class IAssistantRun(ABC):
    @abstractmethod
    def get_model(self) -> str:
        pass

    @abstractmethod
    def get_usage(self) -> IUsage:
        pass


class UsageAdapter(pydantic.BaseModel, IUsage):
    usage: CompletionUsage

    @property
    def prompt_tokens(self):
        return self.usage.prompt_tokens

    @property
    def completion_tokens(self):
        return self.usage.completion_tokens


class ChatCompletionAssistantRunAdapter(pydantic.BaseModel, IAssistantRun):
    chat_completion: ChatCompletion

    def get_model(self) -> OpenAIModelVersion:
        return OpenAIModelVersion(self.chat_completion.model)

    def get_usage(self) -> IUsage:
        assert self.chat_completion.usage is not None, "Usage is None"
        return UsageAdapter(usage=self.chat_completion.usage)


class CostCalculable(ABC):
    @property
    @abstractmethod
    def cost(self) -> float:
        pass

    @property
    @abstractmethod
    def prompt_tokens(self) -> int:
        pass

    @property
    @abstractmethod
    def completion_tokens(self) -> int:
        pass

    def create_summary(self, name, total_cost):
        return AssistantAnalysisResult(
            name=name,
            cost=self.cost,
            total_cost=total_cost,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens
        )


def calculate_cost(prompt_tokens, completion_tokens, model_type):
    cost_map = {
        CostType.INPUT: {
            AIModelType.GPT_4o: 5e-6,
            AIModelType.GPT_4o_MINI: 0.15e-6,
        },
        CostType.OUTPUT: {
            AIModelType.GPT_4o: 15e-6,
            AIModelType.GPT_4o_MINI: 0.6e-6,
        }
    }
    input_cost = cost_map[CostType.INPUT][model_type] * prompt_tokens
    output_cost = cost_map[CostType.OUTPUT][model_type] * completion_tokens
    return input_cost + output_cost


class AssistantRun(CostCalculable):
    def __init__(self, assistant_name: str, run: ChatCompletionAssistantRunAdapter):
        self.assistant_name = assistant_name
        self.run = run

    @property
    def model(self) -> OpenAIModelVersion:
        return self.run.get_model()

    @property
    def prompt_tokens(self) -> int:
        return self.run.get_usage().prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.run.get_usage().completion_tokens

    @property
    def cost(self) -> float:
        return calculate_cost(self.prompt_tokens, self.completion_tokens, self.model.get_model_type())


@dataclass
class AssistantRuns(CostCalculable):
    _runs: List[AssistantRun]

    @property
    def cost(self) -> float:
        return sum(run.cost for run in self._runs)

    @property
    def prompt_tokens(self):
        return sum(run.prompt_tokens for run in self._runs)

    @property
    def completion_tokens(self):
        return sum(run.completion_tokens for run in self._runs)


@dataclass
class AssistantAnalysisResult:
    name: str
    cost: float
    total_cost: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def percent_total_cost(self):
        return (self.cost / self.total_cost) * 100

    def format_cost(self, cost: float) -> str:
        if cost > 100:
            return f"{cost:.0f}$"
        if cost > 0.01:
            return f"{cost:.2f}$"
        return f"{cost * 100:.4f}ct"

    def __repr__(self):
        return (f"{self.name}: Total Cost: {self.format_cost(self.total_cost)}$,"
                f" Cost: {self.format_cost(self.cost)},"
                f" Percentage Total Cost: {self.percent_total_cost:.2f}%,"
                f" Prompt Tokens: {self.prompt_tokens}, Completion Tokens: {self.completion_tokens}")


class AssistantAnalyzer:
    instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(AssistantAnalyzer, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._runs = []
            self._initialized = True

    def __repr__(self):
        assistant_summaries = self.generate_assistant_summaries()
        return f"Assistant Usage Summary: {"\n".join(list(map(str, assistant_summaries)))}"

    def reset(self):
        self._runs.clear()

    def append_assistant_run(self, assistant_run: AssistantRun):
        self._runs.append(assistant_run)

    def total_summary(self):
        return AssistantRuns(_runs=self._runs)

    def get_summary_for_assistant(self, assistant_name):
        assistant_runs = [run for run in self._runs if run.assistant_name == assistant_name]
        return AssistantRuns(_runs=assistant_runs)

    def generate_assistant_summaries(self):
        assistant_name_runs = self._group_runs_by_assistant()
        return [AssistantRuns(_runs=runs).create_summary(assistant_name, self.total_summary().cost) for
                assistant_name, runs in
                assistant_name_runs.items()]

    def _group_runs_by_assistant(self) -> Dict[str, List[AssistantRun]]:
        assistant_names = {run.assistant_name for run in self._runs}
        return {name: [run for run in self._runs if run.assistant_name == name] for name in assistant_names}
