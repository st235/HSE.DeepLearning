import cv2
import numpy as np

from typing import Optional
from src.utils.geometry.rect import Rect


def extract_image_patch(image: np.ndarray,
                        bbox: Rect,
                        patch_shape: Optional[tuple[float, float]] = None) -> np.ndarray:
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : Rect
        The bounding box.
    patch_shape : Optional[tuple[float, float]]
        This parameter can be used to enforce a desired patch shape
        (width, height). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    assert isinstance(bbox, Rect), \
        f"Bbox was of different type {type(bbox)}"

    if patch_shape is not None:
        bbox = bbox.resize(target_width=patch_shape[0],
                           target_height=patch_shape[1])

    # Shape is in (h, w) format.
    image_shape = image.shape
    image_width = image_shape[1]
    image_height = image_shape[0]

    image_box = Rect(left=0, top=0,
                     width=image_width, height=image_height)

    # Let's clip the box by the image viewport.
    bbox = image_box.clip(bbox)

    image_patch = image[int(bbox.top):int(bbox.bottom), int(bbox.left):int(bbox.right)]

    target_size: tuple[int, int]
    if patch_shape is not None:
        target_size = int(patch_shape[0]), int(patch_shape[1])
    else:
        target_size = int(bbox.width), int(bbox.height)

    image_patch = cv2.resize(image_patch, target_size)
    return image_patch
