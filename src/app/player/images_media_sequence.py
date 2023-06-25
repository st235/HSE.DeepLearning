import os
import cv2
import numpy as np

from src.app.player.media_sequence import MediaSequence
from typing import List


class ImagesMediaSequence(MediaSequence):
    def __init__(self,
                 image_files: List[str]):
        self.__index = 0
        self.__image_files = image_files

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self) -> (np.ndarray, int):
        if self.__index >= len(self.__image_files):
            raise StopIteration()

        image_file = self.__image_files[self.__index]
        assert os.path.exists(image_file)

        self.__index += 1

        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        image_shape = image.shape
        assert image_shape[0] > 0 and image_shape[1] > 0

        image_file_name = os.path.basename(image_file)
        return image, int(os.path.splitext(image_file_name)[0])
