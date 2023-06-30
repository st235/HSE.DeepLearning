import numpy as np

from src.utils.geometry.rect import Rect


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Attributes
    ----------
    __origin : Rect
        Bounding box origin rect.
    __confidence : float
        Detector confidence score.
    """

    def __init__(self, origin: Rect, confidence):
        self.__origin = origin
        self.__confidence = float(confidence)

    @property
    def origin(self) -> Rect:
        return self.__origin

    @property
    def confidence(self) -> float:
        return self.__confidence
