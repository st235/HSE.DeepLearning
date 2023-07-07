from __future__ import annotations

import numpy as np
from enum import Enum
from torchreid.utils import FeatureExtractor

from src.deep_sort.features_extractor.features_extractor import FeaturesExtractor
from src.deep_sort.features_extractor.utils.images import extract_image_patch
from src.utils.geometry.rect import Rect
from src.utils.torch_utils import get_available_device


class TorchReidFeaturesExtractor(FeaturesExtractor):
    """Feature extractor based on torchreid.
    """

    class Model(Enum):
        Shufflenet = 0
        Mobilenet = 1
        Mobilenet14x = 2
        Mlfn = 3
        Osnet = 4
        Osnet075 = 5
        OsnetIbn = 6
        OsnetAin = 7
        OsnetAin075 = 8

    def __init__(self,
                 model: Model):
        self.__torchreid_extractor = FeatureExtractor(
            model_name=self.__get_model_name(model),
            device=get_available_device()
        )

    def extract(self,
                image: np.ndarray,
                boxes: list[Rect]) -> np.ndarray:
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box)
            if patch is None:
                raise Exception(f"Cannot extract detection {box} from the image.")
            image_patches.append(patch)

        out = self.__torchreid_extractor(image_patches)
        return out.cpu().detach().numpy()

    @staticmethod
    def __get_model_name(model: Model) -> str:
        if model == TorchReidFeaturesExtractor.Model.Shufflenet:
            return 'shufflenet'

        if model == TorchReidFeaturesExtractor.Model.Mobilenet:
            return 'mobilenetv2_x1_0'

        if model == TorchReidFeaturesExtractor.Model.Mobilenet14x:
            return 'mobilenetv2_x1_4'

        if model == TorchReidFeaturesExtractor.Model.Mlfn:
            return 'mlfn'

        if model == TorchReidFeaturesExtractor.Model.Osnet:
            return 'osnet_x1_0'

        if model == TorchReidFeaturesExtractor.Model.Osnet075:
            return 'osnet_x0_75'

        if model == TorchReidFeaturesExtractor.Model.OsnetIbn:
            return 'osnet_ibn_x1_0'

        if model == TorchReidFeaturesExtractor.Model.OsnetAin:
            return 'osnet_ain_x1_0'

        if model == TorchReidFeaturesExtractor.Model.OsnetAin075:
            return 'osnet_ain_x0_75'

        raise Exception(f"Cannot find corresponding model name for {model}")

