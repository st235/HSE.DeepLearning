import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

from dependencies.definitions import get_file_path
from src.deep_sort.detector.detection import Detection
from src.deep_sort.detector.detections_provider import DetectionsProvider
from src.utils.geometry.rect import Rect

_LABEL_PERSON = 0


class MmdetectionDetectionsProvider(DetectionsProvider):
    """
    Reads detections from pre-baked file. No detections happen at real time.

    The first 10 columns of the detection matrix are in the standard
    MOTChallenge detection format. In the remaining columns store the
    feature vector associated with each detection.
    """

    def __init__(self):
        register_all_modules()
        self.__model = init_detector(
            get_file_path('mmdetection', 'configs', 'dynamic_rcnn', 'dynamic-rcnn_r50_fpn_1x_coco.py'),
            get_file_path('mmdetection_binaries', 'dynamic_rcnn_r50_fpn_1x.pth'), device='cpu')

    def load_detections(self,
                        image: np.ndarray,
                        frame_id: int,
                        min_height: int = 0) -> list[Detection]:
        results = inference_detector(self.__model, image)
        detection_list = []

        predictions = results.pred_instances

        bboxes = predictions.bboxes
        scores = predictions.scores
        labels = predictions.labels

        assert len(bboxes) == len(scores)
        assert len(bboxes) == len(labels)

        for i in range(0, len(bboxes)):
            x0, y0, x1, y1 = bboxes[i]
            confidence = scores[i]
            label = labels[i]

            # if label != _LABEL_PERSON:
            #     continue

            width = x1 - x0
            height = y1 - y0

            if height < min_height:
                continue

            rect = Rect(left=x0, top=y0, width=width, height=height)
            detection_list.append(Detection(rect, confidence))

        return detection_list
