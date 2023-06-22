from __future__ import annotations

import os
import numpy as np

from src.utils.geometry.rect import Rect
from typing import Optional


class MotGroundTruth(object):
    """Represents ground truth from MOT challenge.

    Usually, ground truth is located along with the sequence under gt folder in a gt.txt file.

    Every line in MOT ground truth file follow the format below:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

    The conf value contains the detection confidence in the det.txt files.
    For the ground truth, it acts as a flag whether the entry is to be considered.
    A value of 0 means that this particular instance is ignored in the evaluation,
    while any other value can be used to mark it as active.

    For submitted results, all lines in the .txt file are considered.

    The world coordinates x,y,z are ignored for the 2D challenge and can be filled with -1.
    Similarly, the bounding boxes are ignored for the 3D challenge.
    However, each line is still required to contain 10 values.

    All frame numbers, target IDs and bounding boxes are 1-based.
    """

    def __init__(self, raw_data: np.ndarray):
        # Accept only 2-dimensional arrays
        assert len(raw_data.shape) == 2

        self.__frames_lookup: dict[int, dict[int, Rect]] = dict()
        self.__tracks_lookup: dict[int, list[(int, Rect)]] = dict()

        for entry in raw_data:
            frame_id, track_id, bbox, confidence = int(entry[0]), int(entry[1]), entry[2:6], entry[6]

            should_consider = confidence == 1

            if not should_consider:
                continue

            if frame_id not in self.__frames_lookup:
                self.__frames_lookup[frame_id] = dict()

            if track_id not in self.__tracks_lookup:
                self.__tracks_lookup[track_id] = list()

            bbox_rect = Rect.from_tlwh(bbox)
            self.__frames_lookup[frame_id][track_id] = bbox_rect
            self.__tracks_lookup[track_id].append((frame_id, bbox_rect))

    @classmethod
    def load(cls, ground_truth_file: str) -> Optional[MotGroundTruth]:
        if not os.path.exists(ground_truth_file):
            return None

        return MotGroundTruth(raw_data=np.loadtxt(ground_truth_file, delimiter=','))

    def __getitem__(self, frame_id: int) -> dict[int, Rect]:
        """Gets all tracks for the given frame id"""
        assert isinstance(frame_id, int), f"Found {type(frame_id)}: {frame_id}"

        if frame_id not in self.__frames_lookup:
            return dict()

        return self.__frames_lookup[frame_id]

    def get_track(self, track_id: int) -> list[(int, Rect)]:
        assert isinstance(track_id, int), f"Found {type(track_id)}: {track_id}"
        assert track_id in self.__tracks_lookup
        return self.__tracks_lookup[track_id]
