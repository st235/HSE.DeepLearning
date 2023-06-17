import cv2
import numpy as np

from deep_sort.detector.detection import Detection
from deep_sort.detector.detections_provider import DetectionsProvider
from deep_sort.utils.geometry.rect import Rect


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
                        frame_id: str,
                        min_height: int = 0) -> list[Detection]:
        """Creates detections for given image.
        """
        detection_list = []

        (humans, _) = self.__hog.detectMultiScale(image, winStride=(15, 15), padding=(32, 32), scale=1.1)

        # loop over all detected humans
        for (x, y, w, h) in humans:
            if h < min_height:
                continue

            bbox_origin = Rect(left=x, top=y, width=w, height=h)
            detection_list.append(Detection(bbox_origin, 1.0, []))

        return detection_list
