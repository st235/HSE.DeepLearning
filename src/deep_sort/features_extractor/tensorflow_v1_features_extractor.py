import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow.compat.v1 as tf

from src.deep_sort.features_extractor.features_extractor import FeaturesExtractor
from src.utils.geometry.rect import Rect

# Falling back to v1.
tf.disable_v2_behavior()


def _extract_image_patch(image: np.ndarray,
                         bbox: Rect,
                         patch_shape: tuple[float, float]):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : Rect
        The bounding box.
    patch_shape : tuple[float, float]
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

    image_patch = cv2.resize(image_patch, (int(patch_shape[0]), int(patch_shape[1])))
    return image_patch


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)

    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


class TensorflowV1FeaturesExtractor(FeaturesExtractor):
    """Feature extractor based on Tensorflow V1 model.
    """

    def __init__(self,
                 checkpoint_file: str,
                 input_name: str = "images",
                 output_name: str = "features",
                 batch_size: int = 32):
        self.__session = tf.Session()
        with tf.gfile.GFile(checkpoint_file, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())

        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(f"net/{input_name}:0")
        self.output_var = tf.get_default_graph().get_tensor_by_name(f"net/{output_name}:0")

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4

        self.__feature_dimension = self.output_var.get_shape().as_list()[-1]

        raw_image_shape = self.input_var.get_shape().as_list()[1:]
        # Format is h, w.
        self.__image_shape = (float(raw_image_shape[1]), float(raw_image_shape[0]))

        self.__batch_size = batch_size

    def extract(self,
                image: np.ndarray,
                boxes: list[Rect]) -> np.ndarray:
        image_patches = []
        for box in boxes:
            patch = _extract_image_patch(image, box, self.__image_shape)
            if patch is None:
                raise Exception(f"Cannot extract detection {box} from the image.")
            image_patches.append(patch)

        image_patches = np.asarray(image_patches)

        out = np.zeros((len(image_patches), self.__feature_dimension), np.float32)
        _run_in_batches(
            lambda x: self.__session.run(self.output_var, feed_dict=x),
            {self.input_var: image_patches}, out, self.__batch_size)
        return out
