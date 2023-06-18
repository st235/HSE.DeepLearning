import cv2
import numpy as np


class Window(object):
    def __init__(self,
                 title: str,
                 size: tuple[int, int]):
        assert size[0] > 0 and size[1] > 0

        self.__title = title
        self.__size = size

    @property
    def size(self) -> tuple[int, int]:
        return self.__size

    def update(self, image: np.ndarray):
        cv2.imshow(self.__title, cv2.resize(image, self.__size))

    def destroy(self):
        # Due to a bug in OpenCV we must call imshow after destroying the
        # window. This will make the window appear again as soon as waitKey
        # is called.
        #
        # see https://github.com/Itseez/opencv/issues/4535
        cv2.destroyWindow(self.__title)
        cv2.waitKey(1)
        cv2.imshow(self.__title, np.zeros(self.__size + (3, ), dtype=np.uint8))
