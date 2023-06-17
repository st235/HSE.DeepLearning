import numpy as np


class MediaSequence(object):
    def __iter__(self):
        return self

    def __next__(self) -> (np.ndarray, str):
        raise StopIteration()
