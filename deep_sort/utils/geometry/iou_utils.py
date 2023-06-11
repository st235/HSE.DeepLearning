from __future__ import absolute_import

import numpy as np
from deep_sort import linear_assignment

from deep_sort.utils.geometry.rect import Rect


def iou(bbox: Rect, candidates: list[Rect]) -> np.ndarray[float]:
    """Computer intersection over union.

    Parameters
    ----------
    bbox : Rect
        A bounding box.
    candidates : list[Rect]
        A matrix of candidate bounding boxes (one per row).

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    return np.array([bbox.iou(candidate) for candidate in candidates])


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = Rect.from_tlwh(tracks[track_idx].to_tlwh())
        candidates = [detections[i].origin for i in detection_indices]
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
