import torch
import numpy as np

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.model.backbone.repvgg import repvgg_det_model_convert
from nanodet.util import Logger, cfg, load_config, load_model_weight

from src.deep_sort.detector.detection import Detection
from src.deep_sort.detector.detections_provider import DetectionsProvider
from dependencies.nanodet_binaries.nanodet_model import NanoDetPaths
from src.utils.torch_utils import get_available_device
from src.utils.geometry.rect import Rect

_NANODET_DEVICE_DEFAULT = get_available_device()


class NanodetDetectionsProvider(DetectionsProvider):
    def __init__(self,
                 paths: NanoDetPaths = NanoDetPaths.LegacyM,
                 logger: Logger = Logger(-1, use_tensorboard=False)):
        load_config(cfg, paths.config_path)
        self.__config = cfg

        model = build_model(self.__config.model)
        torch_model = torch.load(paths.model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, torch_model, logger)

        if self.__config.model.arch.backbone.name == "RepVGG":
            deploy_config = self.__config.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            model = repvgg_det_model_convert(model, deploy_model)

        self.__model = model.to(_NANODET_DEVICE_DEFAULT).eval()
        self.__pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def load_detections(self,
                        image: np.ndarray,
                        frame_id: str,
                        min_height: int = 0) -> list[Detection]:
        meta, results = self.__inference(image)

        person_index = cfg.class_names.index('person')

        all_box = []
        for label in results[0]:
            if label != person_index:
                continue

            for bbox in results[0][label]:
                score = bbox[-1]
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]

                w = x1 - x0
                h = y1 - y0

                if h < min_height:
                    continue

                rect = Rect(left=x0, top=y0, width=w, height=h)
                all_box.append(Detection(rect, score, []))

        return all_box

    def __inference(self, image):
        img_info = {"id": 0, "file_name": None}

        height, width = image.shape[:2]
        img_info["height"] = height
        img_info["width"] = width

        meta = dict(img_info=img_info, raw_img=image, img=image)
        meta = self.__pipeline(None, meta, self.__config.data.val.input_size)

        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(_NANODET_DEVICE_DEFAULT)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.__model.inference(meta)

        return meta, results
