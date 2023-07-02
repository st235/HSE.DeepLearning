from __future__ import division, print_function, absolute_import

import numpy as np

from src.app.app import App
from src.app.visualization import Visualization
from src.dataset.mot.mot_dataset_descriptor import MotDatasetDescriptor


def run(sequences: list[str]):
    """Run ground truth visualisation of the given dataset.

    Default dataset implementation is MotDatasetDescriptor (MOT Challenge).

    Parameters
    ----------
    sequences : str
        Path to the sequences.
    """

    for sequence in sequences:
        print(f"Running sequence {sequence}")

        dataset_descriptor = MotDatasetDescriptor.load(sequence)

        assert dataset_descriptor.ground_truth is not None, \
            f"Ground truth should not be empty for {dataset_descriptor.name}"

        app = App(dataset_descriptor)

        def frame_callback(frame_id: int, _: np.ndarray, visualisation: Visualization):
            ground_truth = dataset_descriptor.ground_truth
            ground_truth_tracks = ground_truth[frame_id]

            visualisation.draw_ground_truth(ground_truth_tracks)

        # Run the app.
        app.display_fps()
        app.start(frame_callback)
