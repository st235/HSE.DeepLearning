from __future__ import division, print_function, absolute_import

import argparse
import numpy as np

from src.app.app import App
from src.app.visualization import Visualization
from src.dataset.mot.mot_dataset_descriptor import MotDatasetDescriptor
from src.deep_sort import nn_matching, preprocessing
from src.deep_sort.detector.detections_provider import DetectionsProvider
from src.deep_sort.detector.file_detections_provider import FileDetectionsProvider
from src.deep_sort.tracker import Tracker
from src.utils.geometry.rect import Rect
from src.metrics.hota_metric import HotaMetric
from typing import Optional


def run(sequence_directory: str,
        detections_file: str, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_directory : str
        Path to the MOTChallenge sequence directory.
    detections_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    """
    dataset_descriptor = MotDatasetDescriptor.load(sequence_directory)
    detections_provider: DetectionsProvider = FileDetectionsProvider(detections_file_path=detections_file)

    app = App(dataset_descriptor)

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)

    hota = HotaMetric(ground_truth=dataset_descriptor.ground_truth)

    tracker = Tracker(metric)
    results = []

    def frame_callback(frame_id: int, image: np.ndarray, visualisation: Visualization):
        detections = detections_provider.load_detections(image, frame_id, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([list(detection.origin) for detection in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        visualisation.draw_trackers(tracker.tracks)

        hota.update_frame(frame_id,
                          {track.track_id: Rect.from_tlwh(track.to_tlwh()) for track in tracker.tracks if
                           track.is_confirmed() and track.time_since_update <= 1})

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([frame_id, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run the app.
    app.display_fps()
    app.start(frame_callback)

    print('Started')
    print(hota.evaluate())

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]), file=f)


def _bool_string(input_string):
    if input_string not in {"True", "False"}:
        raise ValueError("Please Enter a valid True/False choice")
    else:
        return input_string == "True"


def _parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
                              " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
                                 "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
                                       "box height. Detections with height smaller than this value are "
                                       "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap", help="Non-maxima suppression threshold: Maximum "
                                  "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
                                      "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
                            "gallery. If None, no budget is enforced.", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget)
