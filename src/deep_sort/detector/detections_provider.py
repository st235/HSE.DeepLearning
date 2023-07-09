import numpy as np

from abc import ABC, abstractmethod
from src.deep_sort.detector.detection import Detection


class DetectionsProvider(ABC):
    """Detects people int the given image.
    """

    @abstractmethod
    def load_detections(self,
                        image: np.ndarray,
                        frame_id: int,
                        min_height: int = 0) -> list[Detection]:
        """Creates detections for the given frame.

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
        list[detector.Detection]
            Returns detection responses at given frame index.
        """
        ...
