import argparse
import os

import numpy as np

from src.app.app import App
from src.app.visualization import Visualization
from src.commands.utils.cl_arguments_utils import parse_array
from src.dataset.mot.mot_dataset_descriptor import MotDatasetDescriptor
from src.dataset.mot.mot_ground_truth import MotGroundTruth
from src.deep_sort.deep_sort import DeepSort
from src.deep_sort.detector.file_detections_provider import FileDetectionsProvider
from src.deep_sort.features_extractor.tensorflow_v1_features_extractor import TensorflowV1FeaturesExtractor
from src.metrics.confusion_matrix_metric import ConfusionMatrixMetric
from src.metrics.hota_metric import HotaMetric
from src.metrics.metric import Metric
from src.metrics.metrics_printer import MetricsPrinter
from src.utils.geometry.rect import Rect


def run(sequences_directory: str,
        detections_directory: str,
        metrics_to_track: list[str]):
    """Runs evaluation of all sequences in the directory.

    Default dataset implementation is MotDatasetDescriptor (MOT Challenge).

    Parameters
    ----------
    sequences_directory: str
        Path to the sequence directory.
    detections_directory: str
        Path to the directory with detection in npy format.
    metrics_to_track: list[str]
        List of metrics to evaluate.
    """

    metrics_printer = MetricsPrinter(metrics_to_track)

    for sequence in os.listdir(sequences_directory):
        sequence_directory = os.path.join(sequences_directory, sequence)
        detection_directory = os.path.join(detections_directory, f"{sequence}.npy")

        print(f"Evaluating sequence {sequence}")

        metrics = __evaluate_single_sequence(sequence_directory, detection_directory, metrics_to_track)
        metrics_printer.add_sequence(sequence, metrics)

    print()

    metrics_printer.print()


def __evaluate_single_sequence(sequence_directory: str,
                               detection_file: str,
                               metrics_to_track: list[str]) -> dict[str, float]:
    dataset_descriptor = MotDatasetDescriptor.load(sequence_directory)

    deep_sort_builder = DeepSort.Builder(dataset_descriptor=dataset_descriptor)
    deep_sort_builder.detections_provider = FileDetectionsProvider(detections_file_path=detection_file)
    deep_sort_builder.features_extractor = TensorflowV1FeaturesExtractor.create_default()

    assert dataset_descriptor.ground_truth is not None, \
        f"Ground truth should not be empty for {dataset_descriptor.name}"

    last_known_metrics_names: set[str] = set()
    metrics: list[Metric] = list()

    for metric_name in metrics_to_track:
        if metric_name in last_known_metrics_names:
            continue

        metric = __create_metric_by_name(metric_name, dataset_descriptor.ground_truth)
        last_known_metrics_names.update(metric.available_metrics)
        metrics.append(metric)

    app = App(dataset_descriptor)
    deep_sort = deep_sort_builder.build()

    def frame_callback(frame_id: int, image: np.ndarray, visualisation: Visualization):
        tracks = deep_sort.update(frame_id, image)

        visualisation.draw_trackers(tracks)

        for metric in metrics:
            metric.update_frame(frame_id,
                                {track.track_id: track.bounding_box for track in tracks if
                                 track.is_confirmed() and track.time_since_update <= 1})

    # Run the app.
    app.display_fps()
    app.start(frame_callback)

    combined_results: dict[str, float] = dict()

    for metric in metrics:
        combined_results.update({k.lower(): v for k, v in metric.evaluate().items()})

    return combined_results


def __create_metric_by_name(metric: str,
                            ground_truth: MotGroundTruth) -> Metric:
    if metric == HotaMetric.KEY_METRIC_HOTA.lower():
        return HotaMetric(ground_truth)
    elif metric == HotaMetric.KEY_METRIC_DETA.lower():
        return HotaMetric(ground_truth)
    elif metric == HotaMetric.KEY_METRIC_ASSA.lower():
        return HotaMetric(ground_truth)
    elif metric == ConfusionMatrixMetric.KEY_METRIC_PRECISION.lower():
        return ConfusionMatrixMetric(ground_truth)
    elif metric == ConfusionMatrixMetric.KEY_METRIC_PRECISION.lower():
        return ConfusionMatrixMetric(ground_truth)
    elif metric == ConfusionMatrixMetric.KEY_METRIC_PRECISION.lower():
        return ConfusionMatrixMetric(ground_truth)
    else:
        raise Exception(f"Unknown metric {metric}")


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
    parser.add_argument(
        "--metrics", help="List of metrics separated by coma, which will be evaluated on the dataset",
        default="HOTA", required=False)
    return parser.parse_args()


def main():
    args = __parse_args()
    run(args.sequences_dir,
        args.detections_dir,
        [metric.lower() for metric in parse_array(args.metrics)])


if __name__ == "__main__":
    main()
