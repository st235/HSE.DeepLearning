import numpy as np

from enum import Enum
from src.deep_sort.detector.detection import Detection
from src.utils.geometry.rect import Rect


class TrackState(Enum):
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    __hits : int
        Total number of measurement updates.
    __age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    __state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self,
                 mean,
                 covariance,
                 track_id,
                 n_init,
                 max_age,
                 feature: np.ndarray):
        assert feature is not None

        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.time_since_update = 0

        self.features = []
        self.features.append(feature)

        self.__state = TrackState.Tentative
        self.__hits = 1
        self.__age = 1
        self.__n_init = n_init
        self.__max_age = max_age

    @property
    def bounding_box(self) -> Rect:
        """Get bounding box for current track position.

        Returns
        -------
        Rect
            The bounding box.

        """
        bbox = self.mean[:4].copy()
        bbox[2] *= bbox[3]
        bbox[:2] -= bbox[2:] / 2
        return Rect.from_tlwh(bbox)

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.__age += 1
        self.time_since_update += 1

    def update(self,
               kf,
               detection_bbox: Rect,
               feature: np.ndarray):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf: kalman_filter.KalmanFilter
            The Kalman filter.
        detection_bbox: Rect
            The associated detection.
        feature: np.ndarray
            1 Dimensional array with a feature vector for the given detection.
        """
        xyah = np.array([detection_bbox.center_x, detection_bbox.center_y,
                         detection_bbox.aspect_ratio, detection_bbox.height])

        self.mean, self.covariance = kf.update(self.mean, self.covariance, xyah)
        self.features.append(feature)

        self.__hits += 1
        self.time_since_update = 0
        if self.__state == TrackState.Tentative and self.__hits >= self.__n_init:
            self.__state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        can_be_deleted = self.__state == TrackState.Tentative \
            or self.time_since_update > self.__max_age

        if can_be_deleted:
            self.__state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.__state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.__state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.__state == TrackState.Deleted
