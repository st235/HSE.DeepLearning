import numpy as np

from scipy.optimize import linear_sum_assignment
from src.dataset.mot.mot_ground_truth import MotGroundTruth
from src.utils.geometry.rect import Rect


class HotaMetric(object):
    def __init__(self,
                 ground_truth: MotGroundTruth):
        self.__ground_truth = ground_truth

        self.__detections: dict[int, dict[int, Rect]] = dict()
        self.__tracks: dict[int, list[(int, Rect)]] = dict()

    def update_frame(self,
                     frame_id: int,
                     detections: dict[int, Rect]):
        assert frame_id not in self.__detections

        self.__detections[frame_id] = detections

        for track_id, box in detections.items():
            if track_id not in self.__tracks:
                self.__tracks[track_id] = list()

            self.__tracks[track_id].append((frame_id, box))

    def evaluate(self):
        hota_a = 0
        for alpha in range(start=0.05, stop=1.0, step=0.05):
            hota_a += self.__evaluate_sequence_with_alpha(alpha=alpha)
        return 1/19 * hota_a

    def __evaluate_sequence_with_alpha(self,
                                       alpha: float) -> [float, float]:
        tp = 0
        fp = 0
        fn = 0
        accumulated_ass = 0

        for frame_id in self.__detections.keys():
            frame_tp, frame_fp, frame_fn, frame_ass = self.__evaluate_frame(alpha, frame_id)

            tp += frame_tp
            fp += frame_fp
            fn += frame_fn
            accumulated_ass += frame_ass

        detection = tp / (tp + fn + fp)
        association = accumulated_ass / tp

        return detection, association

    def __evaluate_frame(self,
                         alpha: float,
                         frame_id: int) -> tuple[int, int, int, float]:
        assert 0 <= alpha <= 1

        detection_tp = 0
        detection_fp = 0
        detection_fn = 0

        accumulated_ass = 0

        raw_detections = self.__detections[frame_id]
        raw_ground_truth = self.__ground_truth[frame_id]

        detection_ids, detection_boxes = zip(*raw_detections.items())
        ground_truth_ids, ground_truth_boxes = zip(*raw_ground_truth.items())

        assert len(detection_ids) == len(detection_boxes)
        assert len(ground_truth_ids) == len(ground_truth_boxes)

        scores = np.array((len(detection_boxes), len(ground_truth_boxes)), type=np.float)

        for i in range(len(detection_ids)):
            detection_box = detection_boxes[i]
            for j in range(len(ground_truth_ids)):
                ground_truth_box = ground_truth_boxes[j]

                iou_score = detection_box.iou(ground_truth_box)
                scores[i, j] = iou_score if iou_score >= alpha else 0

        row_indexes, col_indexes = linear_sum_assignment(scores, maximize=True)

        for row, column in zip(row_indexes, col_indexes):
            if scores[row, column] >= alpha:
                # TP association found.
                detection_tp += 1
                accumulated_ass += self.__evaluate_track(alpha=alpha, detection_track_id=detection_ids[row],
                                                        ground_truth_track_id=ground_truth_ids[column])

        detection_fp += len(detection_ids) - detection_tp
        detection_fn += len(ground_truth_ids) - detection_tp

        return detection_tp, detection_fp, detection_fn, accumulated_ass

    def __evaluate_track(self,
                         alpha: float,
                         detection_track_id: int,
                         ground_truth_track_id: int) -> float:
        assert 0 <= alpha <= 1

        tpa = 0
        fpa = 0
        fna = 0

        detection_track = self.__tracks[detection_track_id]
        ground_truth_track = self.__ground_truth.get_track(ground_truth_track_id)

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
