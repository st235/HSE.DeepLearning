import numpy as np

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from src.deep_sort.segmentation.segmentation import Segmentation
from src.deep_sort.segmentation.segmentations_provider import SegmentationsProvider
from src.utils.torch_utils import get_available_device

_LABEL_PERSON = 0


class Detectron2SegmentationsProvider(SegmentationsProvider):
    def __init__(self):
        config = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        config.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        config.MODEL.DEVICE = get_available_device()

        self.__config = config
        self.__predictor = DefaultPredictor(config)

    def load_detections(self,
                        image: np.ndarray,
                        frame_id: int,
                        min_height: int = 0) -> list[Segmentation]:
        result: list[Segmentation] = list()

        output = self.__predictor(image)

        instances = output["instances"]

        labels = instances.pred_classes.cpu().detach().numpy()
        bboxes = instances.pred_boxes.cpu().detach().numpy()
        masks = instances.pred_masks.cpu().detach().numpy()
        scores = instances.scores.cpu().detach().numpy()

        for label, bbox, mask, score in zip(labels, bboxes, masks, scores):
            if label != _LABEL_PERSON:
                continue

            result.append(Segmentation(bbox=bbox, mask=mask, confidence=score))

        return result
