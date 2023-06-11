import numpy as np

from depdendencies.definitions import fetch_model

from deep_sort.detector.detection import Detection
from deep_sort.detector.detections_provider import DetectionsProvider
from deep_sort.utils.geometry.rect import Rect

_LABEL_PERSON = 0


class YoloV5DetectionsProvider(DetectionsProvider):
    """
    Reads detections from pre-baked file. No detections happen at real time.

    The first 10 columns of the detection matrix are in the standard
    MOTChallenge detection format. In the remaining columns store the
    feature vector associated with each detection.
    """

    def __init__(self):
        self.__model = fetch_model('yolov5',
                                   model_path='yolov5_binaries/yolov5n.pt')

    def load_detections(self,
                        frame_image: np.ndarray,
                        frame_index: int,
                        min_height: int = 0) -> list[Detection]:
        results = self.__model(frame_image)
        detection_list = []

        for obj in results.pred[0]:
            x0, y0, x1, y1, confidence, label = obj.numpy()

            if label != _LABEL_PERSON:
                continue

            width = x1 - x0
            height = y1 - y0

            if height < min_height:
                continue

            rect = Rect(left=x0, top=y0, width=width, height=height)
            detection_list.append(Detection(rect, confidence, []))

        return detection_list
