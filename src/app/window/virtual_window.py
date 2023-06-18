import numpy as np

from src.app.window.window import Window


class VirtualWindow(Window):
    def __init__(self):
        super().__init__('virtual_window', (0, 0))

    def update(self, image: np.ndarray):
        # Empty on purpose.
        pass

    def destroy(self):
        # Empty on purpose.
        pass
