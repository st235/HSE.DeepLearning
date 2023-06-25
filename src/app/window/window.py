import numpy as np

from abc import ABC, abstractmethod


class Window(ABC):
    def __init__(self,
                 title: str,
                 size: tuple[int, int]):
        assert len(size) == 2

        self.__title = title
        self.__size = size

    @property
    def title(self) -> str:
        return self.__title

    @property
    def size(self) -> tuple[int, int]:
        return self.__size

    @abstractmethod
    def update(self, image: np.ndarray):
        ...

    @abstractmethod
    def destroy(self):
        ...
