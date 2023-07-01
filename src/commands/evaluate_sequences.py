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
from src.metrics.metrics_mixer import MetricsMixer
from src.metrics.metrics_printer import MetricsPrinter


def run(sequences_directory: str,
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

        print(f"Evaluating sequence {sequence}")

        metrics = __evaluate_single_sequence(sequence_directory, set(metrics_to_track))
        metrics_printer.add_sequence(sequence, metrics)

    print()

    metrics_printer.print()


def __evaluate_single_sequence(sequence_directory: str,
                               metrics_to_track: set[str]) -> dict[str, float]:
    dataset_descriptor = MotDatasetDescriptor.load(sequence_directory)

    deep_sort_builder = DeepSort.Builder(dataset_descriptor=dataset_descriptor)
    deep_sort_builder.detections_provider = FileDetectionsProvider(detections=dataset_descriptor.detections)
    deep_sort_builder.features_extractor = TensorflowV1FeaturesExtractor.create_default()

    assert dataset_descriptor.ground_truth is not None, \
        f"Ground truth should not be empty for {dataset_descriptor.name}"

    app = App(dataset_descriptor)
    deep_sort = deep_sort_builder.build()

    metrics_mixer = MetricsMixer.create_for_metrics(ground_truth=dataset_descriptor.ground_truth,
                                                    metrics_to_track=metrics_to_track)

    def frame_callback(frame_id: int, image: np.ndarray, visualisation: Visualization):
        tracks = deep_sort.update(frame_id, image)

        visualisation.draw_trackers(tracks)

        metrics_mixer.update_frame(frame_id, tracks)

    # Run the app.
    app.display_fps()
    app.start(frame_callback)

    return metrics_mixer.evaluate()


def __parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Sequence evaluation")
    parser.add_argument(
        "--sequences_dir", help="Path to the sequences directory",
        default=None, required=True)
    parser.add_argument(
        "-m", "--metrics", help=f"List of metrics separated by coma, "
                                f"supported metrics are {', '.join(MetricsMixer.supported_metrics())}",
        default=['HOTA'], required=False, nargs='*')
    return parser.parse_args()


def main():
    args = __parse_args()
    run(args.sequences_dir,
        args.metrics)


if __name__ == "__main__":
    main()
