import os
import numpy as np

from enum import Enum
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from typing import Optional

from dependencies.definitions import get_file_path
from src.deep_sort.detector.detection import Detection
from src.deep_sort.detector.detections_provider import DetectionsProvider
from src.utils.geometry.rect import Rect
from src.utils.torch_utils import get_available_device

_LABEL_PERSON = 0


class MmdetectionDetectionsProvider(DetectionsProvider):
    """
    Reads detections from pre-baked file. No detections happen at real time.

    The first 10 columns of the detection matrix are in the standard
    MOTChallenge detection format. In the remaining columns store the
    feature vector associated with each detection.
    """

    class Config(Enum):
        EfficientDet = 0
        DynamicRcnnR50Coco = 1
        CascadeRpnFaster = 2
        MobileNetV2 = 3

    def __init__(self,
                 config: Config,
                 model: str):
        assert model is not None, \
            'Model is required'

        register_all_modules()

        config = get_file_path('mmdetection', 'configs', self.__get_config_and_weights(config))
        self.__model = init_detector(config, model, device=get_available_device())

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

    @staticmethod
    def __get_config_and_weights(config: Config) -> str:
        if config == MmdetectionDetectionsProvider.Config.EfficientDet:
            return os.path.join('efficientnet', 'retinanet_effb3_fpn_8xb4-crop896-1x_coco.py')

        if config == MmdetectionDetectionsProvider.Config.DynamicRcnnR50Coco:
            return os.path.join('dynamic_rcnn', 'dynamic-rcnn_r50_fpn_1x_coco.py')

        if config == MmdetectionDetectionsProvider.Config.CascadeRpnFaster:
            return os.path.join('cascade_rpn', 'cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco.py')

        if config == MmdetectionDetectionsProvider.Config.MobileNetV2:
            return os.path.join('yolo', 'yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py')

        raise Exception(f"Unknown config {config}")
