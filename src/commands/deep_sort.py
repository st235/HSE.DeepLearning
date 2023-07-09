import os

from argparse import ArgumentParser
from src.commands import cmd_show_ground_truth, cmd_run_deep_sort
from src.dataset.mot.mot_utils import is_mot_sequence
from src.metrics.metrics_mixer import MetricsMixer


def __flatmap_mot_directories(directories: list[str]) -> list[str]:
    """List all mot directories flat from the given directories list.
    """

    if len(directories) == 0:
        raise Exception('Cannot flatmap directory')

    result: list[str] = []

    for directory in directories:
        if not os.path.exists(directory) or not os.path.isdir(directory):
            raise Exception(f"File does not exist or not a directory {directory}")

        if is_mot_sequence(directory):
            result.append(directory)
            continue

        content_paths = [os.path.join(directory, content) for content in os.listdir(directory)]
        result.extend(__flatmap_mot_directories(content_paths))

    return result


def __parse_args():
    parser = ArgumentParser(description="Deep SORT")
    subparsers = parser.add_subparsers(help='sub-commands help')

    ground_truth_parser = subparsers.add_parser('ground-truth', help='Shows ground truth visualisation without running'
                                                                     'Deep SORT algorithm.')
    ground_truth_parser.set_defaults(cmd='ground-truth')
    ground_truth_parser.add_argument('sequences', nargs='+')

    deep_sort_parser = subparsers.add_parser('run', help='Runs Deep SORT algorithm.')
    deep_sort_parser.set_defaults(cmd='run')
    deep_sort_parser.add_argument('sequences', nargs='+')
    deep_sort_parser.add_argument('-e', '--eval',
                                  help=f"List of metrics to evaluate, "
                                       f"supported metrics are {', '.join(MetricsMixer.supported_metrics())}",
                                  default=[],
                                  choices=MetricsMixer.supported_metrics(),
                                  required=False,
                                  nargs='*')

    mode_group = deep_sort_parser.add_mutually_exclusive_group()
    mode_group.add_argument('-d', '--detections_provider',
                                  help="Detections provider for finding human.",
                                  default=None,
                                  choices=cmd_run_deep_sort.get_supported_detectors(),
                                  required=False)
    mode_group.add_argument('-s', '--segmentations_provider',
                                  help=f"Segmentations provider for finding human.",
                                  default=None,
                                  choices=cmd_run_deep_sort.get_supported_segmentation_providers(),
                                  required=False)

    deep_sort_parser.add_argument('-fe', '--features_extractor',
                                  help=f"Features extractor for ReID.",
                                  default='tfv1',
                                  choices=cmd_run_deep_sort.get_supported_features_extractor(),
                                  required=False)
    deep_sort_parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
                              " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    deep_sort_parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
                                 "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    deep_sort_parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
                                       "box height. Detections with height smaller than this value are "
                                       "disregarded.", default=0, type=int)
    deep_sort_parser.add_argument(
        "--nms_max_overlap", help="Non-maxima suppression threshold: Maximum "
                                  "detection overlap.", default=1.0, type=float)
    deep_sort_parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
                                      "metric (object appearance).", type=float, default=0.2)
    deep_sort_parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
                            "gallery. If None, no budget is enforced.", type=int, default=None)
    deep_sort_parser.add_argument(
        "--max_iou_distance", help="Gating threshold for iou distance metric.", type=float, default=0.7)
    deep_sort_parser.add_argument(
        "--max_age", help="Maximum number of missed detections before a track is deleted.", type=int, default=30)
    deep_sort_parser.add_argument(
        "--n_init", help="Number of frames that a track remains in initialization phase.", type=int, default=3)
    deep_sort_parser.add_argument(
        "--extra", help="Additional parameter that may be required by a model.", type=str, default=None)
    return parser.parse_args()


def main():
    args = __parse_args()
    print(args)

    sequences = __flatmap_mot_directories(args.sequences)

    if args.cmd == 'ground-truth':
        cmd_show_ground_truth.run(sequences)
    elif args.cmd == 'run':
        cmd_run_deep_sort.run(sequences,
                              args.detections_provider,
                              args.segmentations_provider,
                              args.features_extractor,
                              args.output_file,
                              args.min_confidence,
                              args.nms_max_overlap,
                              args.min_detection_height,
                              args.max_cosine_distance,
                              args.nn_budget,
                              args.max_iou_distance,
                              args.max_age,
                              args.n_init,
                              args.eval,
                              args.extra)


if __name__ == "__main__":
    main()
