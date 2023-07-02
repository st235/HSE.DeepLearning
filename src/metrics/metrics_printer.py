from abc import ABC, abstractmethod


class MetricsPrinter(ABC):
    @abstractmethod
    def add_sequence(self,
                     sequence: str,
                     metrics: dict[str, float]):
        ...

    @abstractmethod
    def print(self):
        ...
