import numpy as np

from src.utils.geometry.rect import Rect


class Segmentation(object):
    """
    This class represents segmentation results in an image.

    Attributes
    ----------
    __bbox: Rect
        Bounding box rect.
    __mask: np.ndarray
        Segmentation mask.
    __confidence : float
        Detector confidence score.
    """

    def __init__(self,
                 bbox: Rect,
                 mask: np.ndarray,
                 confidence: float):
        self.__bbox = bbox
        self.__mask = mask
        self.__confidence = float(confidence)

    @property
    def bbox(self) -> Rect:
        return self.__bbox

    @property
    def mask(self) -> np.ndarray[bool]:
        return self.__mask

    @property
    def confidence(self) -> float:
        return self.__confidence
