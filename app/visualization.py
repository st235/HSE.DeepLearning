import time
import colorsys
import numpy as np

from app.image_viewer import ImageViewer
from challenge.mot_challenge_descriptor import MotChallengeDescriptor


def _create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def _create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = _create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, challenge_descriptor: MotChallengeDescriptor):
        self.frame_idx = 1
        self.last_idx = challenge_descriptor.sequence_size

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def show_fps(self):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self,
                 challenge_descriptor: MotChallengeDescriptor,
                 update_ms: float):
        image_shape = challenge_descriptor.image_size[::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)

        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % challenge_descriptor.name)
        self.viewer.thickness = 2
        self.__frame_idx = 1
        self.__last_idx = challenge_descriptor.sequence_size

        self.__last_update_time = None

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.__frame_idx > self.__last_idx:
            return False  # Terminate
        frame_callback(self, self.__frame_idx)
        self.__frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = _create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int32), label=str(track_id))

    def draw_detections(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            detection_origin = detection.origin
            self.viewer.rectangle(detection_origin.left, detection_origin.top,
                                  detection_origin.width, detection_origin.height)

    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = _create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int32), label=str(track.track_id))
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)

    def show_fps(self):
        current_time = time.time()

        if self.__last_update_time is not None:
            elapsed_time = current_time - self.__last_update_time

            self.viewer.color = (0, 0, 0)
            self.viewer.thickness = -1
            self.viewer.rectangle(0, 25, 150, 25)
            self.viewer.text_color = (255, 255, 255)
            self.viewer.text(0, 50, f"FPS: {round(1.0 / elapsed_time)}")

        self.__last_update_time = current_time
