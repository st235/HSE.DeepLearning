import argparse

import cv2
import numpy as np

from app.app import App
from app.visualization import Visualization
from app.window.virtual_window import VirtualWindow
from challenge.mot_challenge_descriptor import MotChallengeDescriptor
from deep_sort.detector.file_detections_provider import FileDetectionsProvider
from deep_sort.utils.geometry.iou_utils import iou
from deep_sort.utils.geometry.rect import Rect
from typing import Optional

DEFAULT_UPDATE_MS = 20


def run(sequence_directory: str,
        result_file: str,
        show_false_alarms: bool = False,
        detections_file: Optional[str] = None,
        update_ms: Optional[int] = None,
        video_filename: Optional[str] = None):
    """Run tracking result visualization.

    Parameters
    ----------
    sequence_directory : str
        Path to the MOTChallenge sequence directory.
    result_file : str
        Path to the tracking output file in MOTChallenge ground truth format.
    show_false_alarms : Optional[bool]
        If True, false alarms are highlighted as red boxes.
    detections_file : Optional[str]
        Path to the detection file.
    update_ms : Optional[int]
        Number of milliseconds between cosecutive frames. Defaults to (a) the
        frame rate specifid in the seqinfo.ini file or DEFAULT_UDPATE_MS ms if
        seqinfo.ini is not available.
    video_filename : Optional[Str]
        If not None, a video of the tracking results is written to this file.

    """
    challenge_descriptor = MotChallengeDescriptor.load(sequence_directory)
    detections_provider = FileDetectionsProvider(detections_file)
    results = np.loadtxt(result_file, delimiter=',')

    app = App(challenge_descriptor, window=VirtualWindow())

    if show_false_alarms and challenge_descriptor.ground_truth is None:
        raise ValueError("No ground truth available. Cannot show false alarms.")

    def frame_callback(image: np.ndarray, visualisation: Visualization):
        if detections_file is not None:
            detections = detections_provider.load_detections(image, visualisation.frame_id)
            visualisation.draw_detections(detections)

        mask = results[:, 0].astype(np.int32) == int(visualisation.frame_id)
        track_ids = results[mask, 1].astype(np.int32)
        boxes = results[mask, 2:6]
        visualisation.draw_ground_truth(track_ids, boxes)

        if show_false_alarms:
            ground_truth = challenge_descriptor.ground_truth
            mask = ground_truth[:, 0].astype(np.int32) == int(visualisation.frame_id)
            gt_boxes = [Rect.from_tlwh(candidate) for candidate in ground_truth[mask, 2:6]]
            for box in boxes:
                # NOTE(nwojke): This is not strictly correct, because we don't
                # solve the assignment problem here.
                min_iou_overlap = 0.5
                if iou(Rect.from_tlwh(box), gt_boxes).max() < min_iou_overlap:
                    vis.viewer.color = 0, 0, 255
                    vis.viewer.thickness = 4
                    vis.viewer.rectangle(*box.astype(np.int32))

    if update_ms is None:
        update_ms = challenge_descriptor.update_rate
    if update_ms is None:
        update_ms = DEFAULT_UPDATE_MS
    visualizer = visualization.Visualization(challenge_descriptor, update_ms)
    if video_filename is not None:
        visualizer.viewer.enable_videowriter(video_filename)
    visualizer.run(frame_callback)


def _parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Siamese Tracking")
    parser.add_argument(
        "--sequence_dir", help="Path to the MOTChallenge sequence directory.",
        default=None, required=True)
    parser.add_argument(
        "--result_file", help="Tracking output in MOTChallenge file format.",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections (optional).",
        default=None)
    parser.add_argument(
        "--update_ms", help="Time between consecutive frames in milliseconds. "
                            "Defaults to the frame_rate specified in seqinfo.ini, if available.",
        default=None)
    parser.add_argument(
        "--output_file", help="Filename of the (optional) output video.",
        default=None)
    parser.add_argument(
        "--show_false_alarms", help="Show false alarms as red bounding boxes.",
        type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        args.sequence_dir, args.result_file, args.show_false_alarms,
        args.detection_file, args.update_ms, args.output_file)
