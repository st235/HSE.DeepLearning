import numpy as np

from abc import ABC, abstractmethod
from src.deep_sort.segmentation.segmentation import Segmentation


class SegmentationsProvider(ABC):
    """Segments image and finds human on it.
    """

    @abstractmethod
    def load_detections(self,
                        image: np.ndarray,
                        frame_id: int,
                        min_height: int = 0) -> list[Segmentation]:
        """Creates segmentations for the given frame.

        Parameters
        ----------
        image: np.ndarray
            Current frame with detections
        frame_id : str
            The frame id, should be unique.
        min_height : Optional[int]
            A minimum detection bounding box height. Detections that are smaller
            than this value are disregarded.

        Returns
        -------
        list[Segmentation]
            Returns segmentation results for a given frame.
        """
        ...
