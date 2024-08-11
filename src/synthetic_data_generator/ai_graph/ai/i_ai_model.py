from abc import ABC, abstractmethod

from openai.types.chat import ChatCompletion


class IAIModel[ResultType](ABC):
    @abstractmethod
    async def _get_chat_completion(self, *args, **kwargs) -> ChatCompletion:
        pass

    @property
    @abstractmethod
    def assistant_name(self) -> str:
        pass

    @abstractmethod
    async def get_response_with_retry(self, *args, **kwargs) -> ResultType:
        pass
