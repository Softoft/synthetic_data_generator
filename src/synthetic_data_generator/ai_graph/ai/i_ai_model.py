from abc import ABC, abstractmethod

from openai.types.chat import ChatCompletion


class IAIModel[ResultType](ABC):
    @abstractmethod
    async def _get_chat_completion(self) -> ChatCompletion:
        pass

    @abstractmethod
    async def _get_string_response(self) -> str:
        pass

    @abstractmethod
    def _create_prompt(self) -> str:
        pass

    @property
    def assistant_name(self):
        return ""

    @abstractmethod
    async def get_response_with_retry(self) -> ResultType:
        pass
