import numpy as np

from src.deep_sort.detector.detection import Detection
from src.deep_sort.detector.detections_provider import DetectionsProvider
from src.dataset.mot.mot_ground_truth import MotGroundTruth
from src.utils.geometry.rect import Rect


class GroundTruthDetectionsProvider(DetectionsProvider):
    """
    Provides ground truth as detections.
    """

    def __init__(self, ground_truth: MotGroundTruth):
        self.__ground_truth = ground_truth

    def load_detections(self,
                        image: np.ndarray,
                        frame_id: int,
                        min_height: int = 0) -> list[Detection]:
        frame_detections = self.__ground_truth[frame_id]

        detection_list = []
        for bbox in frame_detections.values():
            detection_list.append(Detection(bbox, 1.0))
        return detection_list
