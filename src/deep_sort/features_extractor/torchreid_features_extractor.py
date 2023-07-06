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

        out = self.__torchreid_extractor(image_patches)
        return out.cpu().detach().numpy()
