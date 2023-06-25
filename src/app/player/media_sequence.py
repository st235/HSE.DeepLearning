import numpy as np

from abc import ABC, abstractmethod


class MediaSequence(ABC):
    """Represents media sequence, i.e. ordered set of images.
    """

    FRAME_ID_UNKNOWN = -1

    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def __next__(self) -> (np.ndarray, int):
        ...
