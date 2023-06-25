import argparse
import os
import numpy as np

from src.app.app import App
from src.app.visualization import Visualization
from src.dataset.mot.mot_dataset_descriptor import MotDatasetDescriptor
from src.deep_sort import nn_matching, preprocessing
from src.deep_sort.detector.detections_provider import DetectionsProvider
from src.deep_sort.detector.file_detections_provider import FileDetectionsProvider
from src.deep_sort.tracker import Tracker
from src.metrics.hota_metric import HotaMetric
from src.utils.geometry.rect import Rect
from typing import Optional


def run(sequences_directory: str,
        detections_directory: str):
    """Runs evaluation of all sequences in the directory.

    Default dataset implementation is MotDatasetDescriptor (MOT Challenge).

    Parameters
    ----------
    sequences_directory: str
        Path to the sequence directory.
    detections_directory: str
        Path to the detections directory.
    """

    tracked_metrics = [HotaMetric.KEY_METRIC_HOTA, HotaMetric.KEY_METRIC_DETA, HotaMetric.KEY_METRIC_ASSA]
    sequences_metrics: dict[str, dict] = dict()

    for sequence in os.listdir(sequences_directory):
        sequence_directory = os.path.join(sequences_directory, sequence)
        detection_directory = os.path.join(detections_directory, f"{sequence}.npy")

        print(f"Evaluating sequence {sequence}")

        metrics = __evaluate_single_sequence(sequence_directory, detection_directory)
        assert sequence not in sequences_metrics

        sequences_metrics[sequence] = metrics

    print()

    # Print table header.
    print(f"{'':20}|", end='')
    for metric in tracked_metrics:
        print(f"{metric:10}|", end='')
    print()

    for sequence in sequences_metrics.keys():
        metrics = sequences_metrics[sequence]
        print(f"{sequence:20}|", end='')
        for metric_name in tracked_metrics:
            print(f"{metrics[metric_name]:10}|", end='')
        print()


def __evaluate_single_sequence(sequence_directory: str,
                               detection_file: str) -> dict:
    dataset_descriptor = MotDatasetDescriptor.load(sequence_directory)
    detections_provider: DetectionsProvider = FileDetectionsProvider(detections_file_path=detection_file)

    assert dataset_descriptor.ground_truth is not None, \
        f"Ground truth should not be empty for {dataset_descriptor.name}"

    app = App(dataset_descriptor)
    tracker = Tracker(metric=nn_matching.NearestNeighborDistanceMetric("cosine", 0.2))
    metric = HotaMetric(ground_truth=dataset_descriptor.ground_truth)

    def frame_callback(frame_id: int, image: np.ndarray, visualisation: Visualization):
        detections = detections_provider.load_detections(image, frame_id)
        detections = [d for d in detections if d.confidence >= 0.8]

        # Run non-maxima suppression.
        boxes = np.array([list(detection.origin) for detection in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        visualisation.draw_trackers(tracker.tracks)

        metric.update_frame(frame_id,
                            {track.track_id: Rect.from_tlwh(track.to_tlwh()) for track in tracker.tracks if
                             track.is_confirmed() and track.time_since_update <= 1})

    # Run the app.
    app.display_fps()
    app.start(frame_callback)

    return metric.evaluate_with_sub_metrics()


def __parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Sequence evaluation")
    parser.add_argument(
        "--sequences_dir", help="Path to the sequences directory",
        default=None, required=True)
    parser.add_argument(
        "--detections_dir", help="Path to the detections directory",
        default=None, required=True)
    return parser.parse_args()


def main():
    args = __parse_args()
    run(args.sequences_dir, args.detections_dir)


if __name__ == "__main__":
    main()
