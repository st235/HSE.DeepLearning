from __future__ import annotations

import numpy as np
import torchvision.transforms as T
from torchreid.utils import FeatureExtractor

from src.deep_sort.features_extractor.features_extractor import FeaturesExtractor
from src.deep_sort.features_extractor.utils.images import extract_image_patch
from src.utils.geometry.rect import Rect
from src.utils.torch_utils import get_available_device


class TorchReidFeaturesExtractor(FeaturesExtractor):
    """Feature extractor based on torchreid.
    """

    def __init__(self):
        self.__torchreid_extractor = FeatureExtractor(
            model_name='mobilenetv2_x1_4',
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

        out: np.ndarray = None

        for i in range(len(image_patches)):
            patch = image_patches[i]

            features = self.__torchreid_extractor(patch)

            if out is None:
                shape = features.shape
                out = np.zeros((len(image_patches), shape[1]), np.float32)

            out[i, :] = features

        if out is None:
            return np.zeros((0, 0))

        return out
