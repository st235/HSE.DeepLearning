import time
import numpy as np

from src.app.visualization import Visualization
from src.app.player.media_player import MediaPlayer
from src.app.player.images_media_sequence import ImagesMediaSequence
from src.app.window.window import Window
from src.app.window.opencv_window import OpenCVWindow
from src.dataset.mot.mot_dataset_descriptor import MotDatasetDescriptor
from src.utils.benchmarking import convert_time_to_fps, measure
from typing import Callable, Optional


class App(object):
    def __init__(self,
                 challenge_descriptor: MotDatasetDescriptor,
                 window: Optional[Window] = None):
        window_shape = challenge_descriptor.image_size[::-1]
        aspect_ratio = float(window_shape[1]) / window_shape[0]
        window_shape = 1024, int(aspect_ratio * 1024)

        images_files = challenge_descriptor.images_files
        assert (images_files is not None) and len(images_files) > 0

        images_media_sequence = ImagesMediaSequence(image_files=images_files)

        update_rate_ms = challenge_descriptor.update_rate
        if update_rate_ms is None:
            update_rate_ms = 5

        self.__media_player = MediaPlayer(media_sequence=images_media_sequence,
                                          update_rate_ms=update_rate_ms)

        self.__display_fps = False

        self.__window = window if window is not None else OpenCVWindow(title=challenge_descriptor.name,
                                                                       size=window_shape)

        self.__fps_records: list[float] = list()

    @property
    def fps_records(self) -> list[float]:
        return self.__fps_records

    def start(self, frame_callback: Callable[[int, np.ndarray, Visualization], None]):
        self.__media_player.play(lambda image, frame_id: self.__on_frame_changed(image, frame_id, frame_callback))

    def __on_frame_changed(self,
                           image: Optional[np.ndarray], frame_id: int,
                           frame_callback: Callable[[int, np.ndarray, Visualization], None]):
        if image is None:
            self.__window.destroy()
            return

        visualisation = Visualization(image=image)

        elapsed_time = measure(lambda: frame_callback(frame_id, image, visualisation))

        frames_per_second = convert_time_to_fps(elapsed_time)
        if self.__display_fps:
            visualisation.draw_info(f"FPS: {frames_per_second}\nFrame: {frame_id}")

        self.__fps_records.append(frames_per_second)
        self.__window.update(visualisation.output_image)

    def display_fps(self):
        self.__display_fps = True
