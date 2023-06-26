from abc import ABC, abstractmethod
from src.dataset.mot.mot_ground_truth import MotGroundTruth
from src.utils.geometry.rect import Rect


class Metric(ABC):
    def __init__(self,
                 ground_truth: MotGroundTruth,
                 supported_metrics: set[str]):
        assert ground_truth is not None

        self._ground_truth = ground_truth
        self.__supported_metrics = [metric.lower() for metric in supported_metrics]

        self._detections: dict[int, dict[int, Rect]] = dict()
        self._tracks: dict[int, list[(int, Rect)]] = dict()

    @property
    def available_metrics(self) -> list[str]:
        return list(iter(self.__supported_metrics))

    def is_metric_supported(self, metric: str) -> bool:
        return metric.lower() in self.__supported_metrics

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
    def evaluate(self) -> dict:
        ...
