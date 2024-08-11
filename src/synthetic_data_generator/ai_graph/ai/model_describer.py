from abc import abstractmethod, ABC


class ModelDescriber(ABC):
    @abstractmethod
    def generate_description(self, *args, **kwargs) -> str:
        pass
