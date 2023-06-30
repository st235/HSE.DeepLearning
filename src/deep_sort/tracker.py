from __future__ import absolute_import

import numpy as np

from src.deep_sort import linear_assignment
from src.deep_sort.detector.detection import Detection
from src.deep_sort.kalman_filter import KalmanFilter
from src.deep_sort.track import Track
from src.utils.geometry import iou_utils


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed detections before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    __metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    __max_age : int
        Maximum number of missed detections before a track is deleted.
    __n_init : int
        Number of frames that a track remains in initialization phase.
    __kalman_filter : KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self,
                 metric,
                 max_iou_distance: float = 0.7,
                 max_age: int = 30,
                 n_init: int = 3):
        assert isinstance(max_iou_distance, float) \
            and 0 <= max_iou_distance <= 1
        assert isinstance(max_age, int) \
            and max_age >= 0
        assert isinstance(n_init, int) \
            and n_init >= 0

        self.tracks = []

        self.__metric = metric
        self.__max_iou_distance = max_iou_distance
        self.__max_age = max_age
        self.__n_init = n_init

        self.__kalman_filter = KalmanFilter()
        self.__next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.__kalman_filter)

    def update(self,
               detections: list[Detection],
               features: np.ndarray):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections: List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        features: np.ndarray
            Features associated with the detections.
        """
        assert len(detections) == features.shape[0]

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self.__match(detections, features)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.__kalman_filter, detections[detection_idx], features[detection_idx, :])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self.__initiate_track(detections[detection_idx], features[detection_idx, :])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.__metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def __match(self, detections, features):

        def gated_metric(tracks, dets, fts, track_indices, detection_indices):
            features = np.array([fts[i, :] for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.__metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.__kalman_filter, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.__metric.matching_threshold, self.__max_age,
                self.tracks, detections, features, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_utils.iou_cost, self.__max_iou_distance, self.tracks,
                detections, features, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def __initiate_track(self,
                         detection: Detection,
                         feature: np.ndarray):
        xyah = np.array([detection.origin.center_x, detection.origin.center_y,
                         detection.origin.aspect_ratio, detection.origin.height])
        mean, covariance = self.__kalman_filter.initiate(xyah)
        self.tracks.append(Track(mean, covariance, self.__next_id, self.__n_init, self.__max_age, feature))
        self.__next_id += 1
