import numpy as np

from abc import ABC, abstractmethod
from src.utils.geometry.rect import Rect


class FeaturesExtractor(ABC):
    @abstractmethod
    def extract(self,
                image: np.ndarray,
                boxes: list[Rect]) -> np.ndarray:
        ...
