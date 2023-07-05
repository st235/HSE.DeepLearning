import numpy as np

from scipy.optimize import linear_sum_assignment
from src.dataset.mot.mot_ground_truth import MotGroundTruth
from src.metrics.metric import Metric
from src.utils.benchmarking import benchmark


class ConfusionMatrixMetric(Metric):

    KEY_METRIC_PRECISION = "Precision"
    KEY_METRIC_RECALL = "Recall"
    KEY_METRIC_F1 = "F1"

    def __init__(self,
                 ground_truth: MotGroundTruth,
                 iou_threshold: float = 0.5):
        assert 0 < iou_threshold <= 1
        super().__init__(ground_truth=ground_truth)

        self.__iou_threshold = iou_threshold

    @benchmark
    def evaluate(self) -> dict[str, float]:
        result_metrics: dict[str, float] = dict()
        overall_tp, overall_fp, overall_fn = 0, 0, 0

        for frame_id in self._detections.keys():
            tp, fp, fn = self.__evaluate_frame(frame_id)

            overall_tp += tp
            overall_fp += fp
            overall_fn += fn

        precision: float = 0.0
        precision_denominator = overall_tp + overall_fp

        recall: float = 0.0
        recall_denominator = overall_tp + overall_fn

        if overall_tp == 0 and precision_denominator == 0:
            precision = 1.0
        else:
            # Precision_denominator == 0 and overall_tp != 0 is
            # impossible as denominator contains overall_tp.
            # The logic holds for if branches below.
            precision = overall_tp / precision_denominator

        if overall_tp == 0 and recall_denominator == 0:
            recall = 1.0
        else:
            recall = overall_tp / recall_denominator

        f1_score: float = 0.0
        f1_denominator = precision + recall

        if 2 * precision * recall == 0 and f1_denominator == 0:
            f1_score = 1.0
        else:
            f1_score = 2 * precision * recall / (precision + recall)

        result_metrics[ConfusionMatrixMetric.KEY_METRIC_PRECISION] = precision
        result_metrics[ConfusionMatrixMetric.KEY_METRIC_RECALL] = recall
        result_metrics[ConfusionMatrixMetric.KEY_METRIC_F1] = f1_score

        return result_metrics

    def __evaluate_frame(self,
                         frame_id: int) -> tuple[int, int, int]:
        raw_detections = self._detections[frame_id]
        raw_ground_truth = self._ground_truth[frame_id]

        if len(raw_detections) == 0 and len(raw_ground_truth) == 0:
            return 0, 0, 0
        elif len(raw_detections) == 0:
            return 0, 0, len(raw_ground_truth)
        elif len(raw_ground_truth) == 0:
            return 0, len(raw_detections), 0

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

        tp, fp, fn = 0, 0, 0

        for row, column in zip(row_indexes, col_indexes):
            if scores[row, column] >= self.__iou_threshold:
                tp += 1

        fp = len(detection_ids) - tp
        fn = len(ground_truth_ids) - tp

        return tp, fp, fn
