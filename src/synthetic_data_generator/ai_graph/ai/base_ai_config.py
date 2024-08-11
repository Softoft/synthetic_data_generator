import enum
from enum import Enum


class CostType(Enum):
    INPUT = enum.auto()
    OUTPUT = enum.auto()


class AIModelType(Enum):
    GPT_4o = "gpt-4o"
    GPT_4o_MINI = "gpt-4o-mini"


class OpenAIModelVersion:
    def __init__(self, model_version: str):
        self._model_version: str = model_version

    def get_model_version(self) -> str:
        return self._model_version

    def get_model_type(self) -> AIModelType:
        for mode_type in sorted(list(AIModelType), key=lambda x: len(x.value), reverse=True):
            if mode_type.value in self._model_version:
                return mode_type
        raise ValueError(f"Model version {self._model_version} not found in AIModelType")
