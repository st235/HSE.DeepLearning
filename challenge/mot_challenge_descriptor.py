from __future__ import annotations

import os
import cv2
import numpy as np

from typing import List, Optional


class MotChallengeDescriptor(object):

    @classmethod
    def load(cls,
             sequence_directory: str) -> MotChallengeDescriptor:
        """Loads sequence information, such as image filenames, detections,
        ground truth and creates MotChallengeDescriptor.

        Parameters
        ----------
        sequence_directory : str
            Path to the MOTChallenge sequence directory.

        Returns
        -------
            MotChallengeDescriptor instance.
        """
        assert os.path.exists(sequence_directory)

        images_directory = os.path.join(sequence_directory, "img1")
        images_files = sorted([os.path.join(images_directory, file) for file in os.listdir(images_directory)])
        ground_truth_file = os.path.join(sequence_directory, "gt", "gt.txt")

        ground_truth = np.loadtxt(ground_truth_file, delimiter=',') if os.path.exists(ground_truth_file) else None

        if len(images_files) > 0:
            image = cv2.imread(next(iter(images_files)), cv2.IMREAD_GRAYSCALE)
            image_size = image.shape
        else:
            image_size = None

        info_file = os.path.join(sequence_directory, "seqinfo.ini")
        if os.path.exists(info_file):
            with open(info_file, "r") as file:
                line_splits = [line.split('=') for line in file.read().splitlines()[1:]]
                info_dict = dict(s for s in line_splits if isinstance(s, list) and len(s) == 2)

            update_ms = 1000 / int(info_dict["frameRate"])
        else:
            update_ms = None

        return MotChallengeDescriptor(name=os.path.basename(sequence_directory),
                                      images_files=images_files,
                                      ground_truth=ground_truth,
                                      image_size=image_size,
                                      update_rate=update_ms)

    def __init__(self,
                 name: str,
                 images_files: List[str],
                 ground_truth: Optional[np.ndarray],
                 image_size: np.ndarray,
                 update_rate: Optional[float]):
        self.__name = name
        self.__images_files = images_files
        self.__ground_truth = ground_truth
        self.__image_size = image_size
        self.__update_rate = update_rate

    @property
    def name(self) -> str:
        return self.__name

    @property
    def images_files(self) -> List[str]:
        return self.__images_files

    @property
    def ground_truth(self) -> np.ndarray:
        return self.__ground_truth

    @property
    def image_size(self) -> np.ndarray:
        return self.__image_size

    @property
    def sequence_size(self) -> int:
        return len(self.__images_files)

    @property
    def update_rate(self) -> Optional[float]:
        return self.__update_rate
