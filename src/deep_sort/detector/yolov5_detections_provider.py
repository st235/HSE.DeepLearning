import numpy as np

from dependencies.definitions import fetch_model

from src.deep_sort.detector.detection import Detection
from src.deep_sort.detector.detections_provider import DetectionsProvider
from src.utils.geometry.rect import Rect

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
                        image: np.ndarray,
                        frame_id: int,
                        min_height: int = 0) -> list[Detection]:
        results = self.__model(image)
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
            detection_list.append(Detection(rect, confidence))

        return detection_list
