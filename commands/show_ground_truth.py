from __future__ import division, print_function, absolute_import

import argparse
import numpy as np

from app.app import App
from app.visualization import Visualization
from dataset.mot.mot_dataset_descriptor import MotDatasetDescriptor


def run(sequence_directory: str):
    """Run ground truth visualisation of MOT Challenge.

    Parameters
    ----------
    sequence_directory : str
        Path to the MOTChallenge sequence directory.
    """
    dataset_descriptor = MotDatasetDescriptor.load(sequence_directory)

    assert dataset_descriptor.ground_truth is not None, \
        f"Ground truth should not be empty for {dataset_descriptor.name}"

    app = App(dataset_descriptor)

    def frame_callback(_: np.ndarray, visualisation: Visualization):
        frame_id = int(visualisation.frame_id)

        ground_truth = dataset_descriptor.ground_truth
        ground_truth_tracks = ground_truth[frame_id]

        visualisation.draw_ground_truth(ground_truth_tracks)

    # Run the app.
    app.display_fps()
    app.start(frame_callback)


def __parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    return parser.parse_args()


def main():
    args = __parse_args()
    run(args.sequence_dir)


if __name__ == "__main__":
    main()
