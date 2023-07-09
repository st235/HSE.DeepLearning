from src.utils.geometry.rect import Rect


class Detection(object):
    """
    This class represents a single detection in an image.

    Attributes
    ----------
    __origin: Rect
        Bounding box origin rect.
    __confidence: float
        Detector confidence score.
    """

    def __init__(self, origin: Rect, confidence: float):
        self.__origin = origin
        self.__confidence = float(confidence)

    @property
    def origin(self) -> Rect:
        return self.__origin

    @property
    def confidence(self) -> float:
        return self.__confidence
