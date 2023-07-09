from __future__ import annotations

from src.app.app import App
from src.dataset.mot.mot_ground_truth import MotGroundTruth
from src.deep_sort.track import Track
from src.metrics.metric import Metric
from src.metrics.confusion_matrix_metric import ConfusionMatrixMetric
from src.metrics.fps_metric import FPSMetric
from src.metrics.hota_metric import HotaMetric

_ID_METRIC_HOTA = 0
_ID_METRIC_CONFUSION = 1
_ID_METRIC_FPS = 2


class MetricsMixer(object):

    def __init__(self,
                 metrics: list[Metric]):
        self.__metrics = metrics

    @classmethod
    def get_metric_id(cls,
                      metric: str) -> int:
        lookup = {
            HotaMetric.KEY_METRIC_HOTA: _ID_METRIC_HOTA,
            HotaMetric.KEY_METRIC_ASSA: _ID_METRIC_HOTA,
            HotaMetric.KEY_METRIC_DETA: _ID_METRIC_HOTA,
            ConfusionMatrixMetric.KEY_METRIC_F1: _ID_METRIC_CONFUSION,
            ConfusionMatrixMetric.KEY_METRIC_RECALL: _ID_METRIC_CONFUSION,
            ConfusionMatrixMetric.KEY_METRIC_PRECISION: _ID_METRIC_CONFUSION,
            FPSMetric.KEY_METRIC_FPS: _ID_METRIC_FPS,
        }

        if metric not in lookup:
            raise Exception(f"Unsupported metric for mixing {metric}")

        return lookup[metric]

    @classmethod
    def supported_metrics(cls) -> set[str]:
        return {HotaMetric.KEY_METRIC_HOTA, HotaMetric.KEY_METRIC_ASSA, HotaMetric.KEY_METRIC_DETA,
                ConfusionMatrixMetric.KEY_METRIC_F1, ConfusionMatrixMetric.KEY_METRIC_RECALL,
                ConfusionMatrixMetric.KEY_METRIC_PRECISION, FPSMetric.KEY_METRIC_FPS}

    @classmethod
    def create_for_metrics(cls,
                           app: App,
                           ground_truth: MotGroundTruth,
                           metrics_to_track: set[str]) -> MetricsMixer:
        used_metrics: set[int] = set()
        metrics: list[Metric] = list()

        for metric_name in metrics_to_track:
            metric_id = cls.get_metric_id(metric_name)

            if metric_id in used_metrics:
                continue

            metric: Metric
            if metric_id == _ID_METRIC_HOTA:
                metric = HotaMetric(ground_truth=ground_truth)
            elif metric_id == _ID_METRIC_CONFUSION:
                metric = ConfusionMatrixMetric(ground_truth=ground_truth)
            elif metric_id == _ID_METRIC_FPS:
                metric = FPSMetric(app=app, ground_truth=ground_truth)
            else:
                raise Exception(f"Unknown metric id {metric_id}")

            used_metrics.add(metric_id)
            metrics.append(metric)

        return MetricsMixer(metrics)

    def update_frame(self,
                     frame_id: int,
                     tracks: list[Track]):
        for metric in self.__metrics:
            metric.update_frame(frame_id,
                                {track.track_id: track.bounding_box for track in tracks if
                                 track.is_confirmed() and track.time_since_update <= 1})

    def evaluate(self) -> dict[str, float]:
        results: dict[str, float] = dict()

        for metric in self.__metrics:
            results.update({k: v for k, v in metric.evaluate().items()})

        return results
