from __future__ import annotations

import cv2
import numpy as np

from src.app.window.window import Window


class OpenCVWindow(Window):
    def __init__(self,
                 title: str,
                 size: tuple[int, int]):
        super().__init__(title, size)
        assert size[0] > 0 and size[1] > 0

    def update(self, image: np.ndarray):
        cv2.imshow(self.title, cv2.resize(image, self.size))

    def destroy(self):
        # Due to a bug in OpenCV we must call imshow after destroying the
        # window. This will make the window appear again as soon as waitKey
        # is called.
        # However, even after calling cv2.waitKey there are still no guarantees
        # that the window will be closed at some GUI.
        #
        # see https://github.com/Itseez/opencv/issues/4535
        cv2.destroyWindow(self.title)
        cv2.waitKey(1)
