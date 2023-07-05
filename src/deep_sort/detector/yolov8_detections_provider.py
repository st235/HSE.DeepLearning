import numpy as np

from enum import Enum
from ultralytics import YOLO

from src.deep_sort.detector.detection import Detection
from src.deep_sort.detector.detections_provider import DetectionsProvider
from src.utils.geometry.rect import Rect

_LABEL_PERSON = 0


class YoloV8DetectionsProvider(DetectionsProvider):
    class Checkpoint(Enum):
        NANO = 0
        SMALL = 1
        MEDIUM = 2
        LARGE = 3

    def __init__(self,
                 checkpoint: Checkpoint):
        yolo_checkpoint = self.__get_checkpoint(checkpoint)
        self.__model = YOLO(yolo_checkpoint)

    def load_detections(self,
                        image: np.ndarray,
                        frame_id: int,
                        min_height: int = 0) -> list[Detection]:
        results = self.__model(image)
        detection_list = []

        for result in results:
            for box in result.boxes:

                rect = box.xyxy[0]
                x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]
                confidence = box.conf
                label = box.cls

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
    def __get_checkpoint(checkpoint: Checkpoint) -> str:
        if checkpoint == YoloV8DetectionsProvider.Checkpoint.NANO:
            return 'yolov8n.pt'

        if checkpoint == YoloV8DetectionsProvider.Checkpoint.SMALL:
            return 'yolov8s.pt'

        if checkpoint == YoloV8DetectionsProvider.Checkpoint.MEDIUM:
            return 'yolov8m.pt'

        if checkpoint == YoloV8DetectionsProvider.Checkpoint.LARGE:
            return 'yolov8l.pt'

        raise Exception(f"Cannot find checkpoint {checkpoint}")
