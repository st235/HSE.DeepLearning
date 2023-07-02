from abc import ABC, abstractmethod
from src.dataset.mot.mot_ground_truth import MotGroundTruth
from src.utils.geometry.rect import Rect


class Metric(ABC):
    def __init__(self,
                 ground_truth: MotGroundTruth):
        assert ground_truth is not None

        self._ground_truth = ground_truth

        self._detections: dict[int, dict[int, Rect]] = dict()
        self._tracks: dict[int, list[(int, Rect)]] = dict()

    def update_frame(self,
                     frame_id: int,
                     detections: dict[int, Rect]):
        assert frame_id not in self._detections

        self._detections[frame_id] = detections

        for track_id, box in detections.items():
            if track_id not in self._tracks:
                self._tracks[track_id] = list()

            self._tracks[track_id].append((frame_id, box))

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        ...
