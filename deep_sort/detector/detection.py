import numpy as np

from utils.geometry.rect import Rect


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Attributes
    ----------
    origin : Rect
        Bounding box origin rect.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.
    """

    def __init__(self, origin: Rect, confidence, feature):
        self.__origin = origin
        self.__confidence = float(confidence)
        self.__feature = np.asarray(feature, dtype=np.float32)

    @property
    def origin(self) -> Rect:
        return self.__origin

    @property
    def confidence(self) -> float:
        return self.__confidence

    @property
    def feature(self):
        return self.__feature
