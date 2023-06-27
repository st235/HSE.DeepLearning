import numpy as np

from functools import cache
from scipy.optimize import linear_sum_assignment
from src.dataset.mot.mot_ground_truth import MotGroundTruth
from src.metrics.metric import Metric
from src.utils.geometry.rect import Rect
from src.utils.benchmarking import benchmark


class HotaMetric(Metric):
    """Implementation of HOTA metric.

    The metric relies on a research paper (https://arxiv.org/pdf/2009.07736.pdf)
    and an implementation article (https://autonomousvision.github.io/hota-metrics/).

    Lightweight, however, a bit slower than the original implementation of a metric,
    given at https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/hota.py.

    This metric does not implement an alignment score as this score has not been
    described in the original paper.

    Attributes
    ----------
        _ground_truth: MotGroundTruth
            Ground truth object.
        __association_lookup: dict[int, np.ndarray]
            Lookup table for similarity scores between rects.
        _detections: dict[int, dict[int, Rect]]
            Lookup table to find detections per frame per track.
        _tracks: dict[int, list[(int, Rect)]]
            Lookup table to find all detections per track.
    """

    KEY_METRIC_HOTA = "HOTA"
    KEY_METRIC_DETA = "DetA"
    KEY_METRIC_ASSA = "AssA"

    def __init__(self,
                 ground_truth: MotGroundTruth):
        super().__init__(ground_truth=ground_truth,
                         supported_metrics={HotaMetric.KEY_METRIC_HOTA, HotaMetric.KEY_METRIC_DETA,
                                            HotaMetric.KEY_METRIC_ASSA})

    @benchmark
    def evaluate(self) -> dict:
        metrics_dict = dict()

        detection, association = self.__evaluate_sequence()

        metrics_dict[HotaMetric.KEY_METRIC_HOTA] = self.__get_hota(detection, association)
        metrics_dict[HotaMetric.KEY_METRIC_DETA] = np.mean(detection)
        metrics_dict[HotaMetric.KEY_METRIC_ASSA] = np.mean(association)

        return metrics_dict

    @staticmethod
    def __get_hota(detection: np.ndarray[float],
                   association: np.ndarray[float]) -> float:
        return 1 / 19 * np.sum(np.sqrt(detection * association))

    def __evaluate_sequence(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """Evaluates a sequence collected through #update_frame method.

        Computes detection and association scores for alphas from 0.05 up to 0.95.

        Returns
        -------
            A tuple of np.ndarray[float] containing detection and association scores for every alpha.
            In tuple the detection score always comes before the association.
            Index of a corresponding score to some alpha 'a' can be obtained as ('a'/0.05 - 1).
        """

        # Accumulator of metrics per all frames.
        result_metrics = np.zeros(shape=(19, 4))

        for frame_id in self._detections.keys():
            result_metrics += self.__evaluate_frame(frame_id=frame_id)

        detection = result_metrics[:, 0] / np.maximum(1, np.sum(result_metrics[:, 0:3], axis=1))
        # Out parameter is required to utilise where otherwise where is ignored.
        association = np.divide(result_metrics[:, 3], result_metrics[:, 0],
                                out=np.zeros_like(result_metrics[:, 0]),
                                where=result_metrics[:, 0] > 0)

        assert detection.shape == association.shape

        return detection, association

    def __evaluate_frame(self,
                         frame_id: int) -> np.ndarray:
        raw_detections = self._detections[frame_id]
        raw_ground_truth = self._ground_truth[frame_id]

        if len(raw_detections) == 0 and len(raw_ground_truth) == 0:
            return np.repeat([[0, 0, 0, 1]], repeats=19, axis=0)
        elif len(raw_detections) == 0:
            return np.repeat([[0, 0, len(raw_ground_truth), 0]], repeats=19, axis=0)
        elif len(raw_ground_truth) == 0:
            return np.repeat([[0, len(raw_detections), 0, 0]], repeats=19, axis=0)

        detection_ids, detection_boxes = zip(*raw_detections.items())
        ground_truth_ids, ground_truth_boxes = zip(*raw_ground_truth.items())

        scores = np.zeros((len(detection_boxes), len(ground_truth_boxes)), dtype=np.float32)

        for i in range(len(detection_ids)):
            detection_box = detection_boxes[i]
            for j in range(len(ground_truth_ids)):
                ground_truth_box = ground_truth_boxes[j]

                scores[i, j] = detection_box.iou(ground_truth_box)

        # Hungarian or modified Jonker-Volgenant algorithm, subject of scipy version.
        row_indexes, col_indexes = linear_sum_assignment(scores, maximize=True)

        # Metrics (tp, fp, fn, association score) per alpha.
        result_metrics = np.zeros(shape=(19, 4))

        # Start per alpha evaluation to re-use linear sum assignment evaluation.
        for index, alpha in enumerate(np.arange(0.05, 1.00, 0.05)):
            for row, column in zip(row_indexes, col_indexes):
                if scores[row, column] >= alpha:
                    # TP association found: increase TP (index 0).
                    result_metrics[index, 0] += 1
                    # Calculating association score (index 3).
                    result_metrics[index, 3] += self.__evaluate_track(alpha=alpha,
                                                                      detection_track_id=detection_ids[row],
                                                                      ground_truth_track_id=ground_truth_ids[column])

            # FP is index 1, TP is 0.
            result_metrics[index, 1] = len(detection_ids) - result_metrics[index, 0]
            # FN is index 2, TP is 0.
            result_metrics[index, 2] = len(ground_truth_ids) - result_metrics[index, 0]

        return result_metrics

    @cache
    def __evaluate_track(self,
                         alpha: float,
                         detection_track_id: int,
                         ground_truth_track_id: int) -> float:
        assert 0 <= alpha <= 1

        tpa = 0
        fpa = 0
        fna = 0

        detection_track = self._tracks[detection_track_id]
        ground_truth_track = self._ground_truth.get_track(ground_truth_track_id)

        if len(detection_track) == 0 and len(ground_truth_track) == 0:
            # No detections and no ground truth: perhaps this could happen.
            return 1

        dti = 0
        gtti = 0

        while dti < len(detection_track) and gtti < len(ground_truth_track):
            detection_frame, detection_box = detection_track[dti]
            ground_truth_frame, ground_truth_box = ground_truth_track[gtti]

            if detection_frame == ground_truth_frame:
                if detection_box.iou(ground_truth_box) >= alpha:
                    tpa += 1
                else:
                    fpa += 1
                    fna += 1

                dti += 1
                gtti += 1
            elif detection_frame < ground_truth_frame:
                fpa += 1
                dti += 1
            else:
                # detection_frame > ground_truth_frame.
                fna += 1
                gtti += 1

        while dti < len(detection_track):
            fpa += 1
            dti += 1

        while gtti < len(ground_truth_track):
            fna += 1
            gtti += 1

        return tpa / (tpa + fna + fpa)
