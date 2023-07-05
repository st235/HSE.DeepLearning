import cv2
import numpy as np

from src.deep_sort.detector.detection import Detection
from src.deep_sort.detector.detections_provider import DetectionsProvider
from src.utils.geometry.rect import Rect


class HogDetectionsProvider(DetectionsProvider):
    """
    Detects people using Histogram of Oriented Gradients (HOG) approach.
    For classification is using SVM. Detections are happening in real time.
    """

    def __init__(self, svm_people_detector=cv2.HOGDescriptor_getDefaultPeopleDetector()):
        self.__hog = cv2.HOGDescriptor()
        self.__hog.setSVMDetector(svm_people_detector)

    def load_detections(self,
                        image: np.ndarray,
                        frame_id: int,
                        min_height: int = 0) -> list[Detection]:
        """Creates detections for given image.
        """
        detection_list = []

        scale_factor = 0.6
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        dim = (width, height)

        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        (humans, _) = self.__hog.detectMultiScale(resized, winStride=(24, 24))

        # loop over all detected humans
        for (x, y, w, h) in humans:
            real_width = w / scale_factor
            real_height = h / scale_factor

            x /= scale_factor
            y /= scale_factor

            if real_height < min_height:
                continue

            bbox_origin = Rect(left=x, top=y, width=real_width, height=real_height)
            detection_list.append(Detection(bbox_origin, 1.0))

        return detection_list
