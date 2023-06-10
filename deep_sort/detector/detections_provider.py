import numpy as np

from deep_sort.detector.detection import Detection


class DetectionsProvider(object):
    """
    Finds humans on the given image.

    The first 10 columns of the detection matrix are in the standard
    MOTChallenge detection format. In the remaining columns store the
    feature vector associated with each detection.
    """

    def load_detections(self,
                        frame_image: np.ndarray,
                        frame_index: int,
                        min_height: int = 0) -> list[Detection]:
        """Creates detections for given frame index from the file on disk.

        Parameters
        ----------
        frame_image: np.ndarray
            Current frame with detections
        frame_index : int
            The frame index.
        min_height : Optional[int]
            A minimum detection bounding box height. Detections that are smaller
            than this value are disregarded.

        Returns
        -------
        List[detector.Detection]
            Returns detection responses at given frame index.

        """
        return []