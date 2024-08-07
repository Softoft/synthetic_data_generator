import abc


class IRandom[V](abc.ABC):
    @abc.abstractmethod
    def get_random_value(self, *args, **kwargs) -> V:
        pass
