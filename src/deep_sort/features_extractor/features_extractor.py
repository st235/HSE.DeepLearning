import numpy as np

from abc import ABC, abstractmethod
from src.utils.geometry.rect import Rect


class FeaturesExtractor(ABC):
    """Features extractor is the main abstraction for converting a detection area into a feature vector.
    """

    @abstractmethod
    def extract(self,
                image: np.ndarray,
                boxes: list[Rect]) -> np.ndarray:
        """Returns a feature vectors for the given boxes in the given image.

        Feature vectors go in the same order as detections in the given list.
        """
        ...
