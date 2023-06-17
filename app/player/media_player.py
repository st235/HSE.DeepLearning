import cv2
import time
import numpy as np

from app.player.media_sequence import MediaSequence
from enum import Enum
from typing import Callable, Optional


def _forward_one_frame(media_sequence_iterator,
                       frame_callback: Callable[[np.ndarray, str], None]) -> (bool, int):
    start_time_s = time.time()

    termination_token = object()
    next_frame = next(media_sequence_iterator, termination_token)

    if next_frame != termination_token:
        image, frame_id = next_frame
        frame_callback(image, frame_id)

    finish_time_s = time.time()
    elapsed_time_ms = int((finish_time_s - start_time_s) * 1e3)

    if next_frame == termination_token:
        return True, elapsed_time_ms
    else:
        return False, elapsed_time_ms


class MediaPlayer(object):
    class KeyEvent(Enum):
        ESCAPE = 27
        SPACE = 32
        S = ord('s')
        A = ord('a')
        D = ord('d')
        F = ord('f')
        B = ord('b')

    def __init__(self,
                 media_sequence: MediaSequence,
                 update_rate_ms: int):
        self.__media_sequence = media_sequence
        self.__update_rate_ms = int(update_rate_ms)

    def play(self, frame_callback: Callable[[Optional[np.ndarray], str], None]):

        media_sequence_iterator = iter(self.__media_sequence)
        is_terminated = False
        is_paused = False

        while not is_terminated:
            elapsed_time_ms = 0

            if not is_paused:
                is_terminated, elapsed_time_ms = _forward_one_frame(media_sequence_iterator, frame_callback)

            remaining_time_ms = max(1, self.__update_rate_ms - elapsed_time_ms)
            key_event = cv2.waitKey(remaining_time_ms) & 0xFF

            if key_event == MediaPlayer.KeyEvent.ESCAPE.value:
                is_terminated = True
            elif key_event == MediaPlayer.KeyEvent.SPACE.value:
                is_paused = not is_paused
            elif key_event == MediaPlayer.KeyEvent.S.value:
                if not is_paused:
                    # Makes no sense to fast-forward when is not paused.
                    continue
                is_terminated, _ = _forward_one_frame(media_sequence_iterator, frame_callback)

        # Signalising the end of the sequence.
        frame_callback(None, '')
