import numpy as np

from src.deep_sort.detector.detection import Detection
from src.deep_sort.detector.detections_provider import DetectionsProvider
from src.utils.geometry.rect import Rect
from dependencies.definitions import fetch_model
from enum import Enum

_LABEL_PERSON = 0

class YoloV5DetectionsProvider(DetectionsProvider):
    """
    Reads detections from pre-baked file. No detections happen at real time.

    The first 10 columns of the detection matrix are in the standard
    MOTChallenge detection format. In the remaining columns store the
    feature vector associated with each detection.
    """

    class Checkpoint(Enum):
        NANO = 0
        NANO6 = 1
        SMALL = 2
        MEDIUM = 3
        LARGE = 4

    def __init__(self,
                 checkpoint: Checkpoint):
        path = self.__get_model_path_by_checkpoint(checkpoint)
        self.__model = fetch_model('yolov5',
                                   model_path=path)

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

    @staticmethod
    def __get_model_path_by_checkpoint(checkpoint: Checkpoint) -> str:
        if checkpoint == YoloV5DetectionsProvider.Checkpoint.NANO:
            return 'yolov5_binaries/yolov5n.pt'

        if checkpoint == YoloV5DetectionsProvider.Checkpoint.NANO6:
            return 'yolov5_binaries/yolov5n6.pt'

        if checkpoint == YoloV5DetectionsProvider.Checkpoint.SMALL:
            return 'yolov5_binaries/yolov5s.pt'

        if checkpoint == YoloV5DetectionsProvider.Checkpoint.MEDIUM:
            return 'yolov5_binaries/yolov5m.pt'

        if checkpoint == YoloV5DetectionsProvider.Checkpoint.LARGE:
            return 'yolov5_binaries/yolov5l.pt'

        raise Exception(f"Cannot find checkpoint {checkpoint}")
