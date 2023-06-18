import numpy as np

from app.drawing.color import Color
from app.drawing.drawing_context import DrawingContext
from app.drawing.paint import Paint
from deep_sort.utils.geometry.rect import Rect


class Visualization(object):
    """Draws tacking output on the given image.
    """

    def __init__(self,
                 frame_id: str,
                 image: np.ndarray):
        self.__paint = Paint()
        self.__frame_id = frame_id
        self.__drawing_context = DrawingContext(image=image)

    @property
    def frame_id(self) -> str:
        return self.__frame_id

    @property
    def image(self) -> np.ndarray:
        return self.__drawing_context.image

    def __draw_label_with_bounding_box(self,
                                       label: str,
                                       x: int, y: int,
                                       padding: int,
                                       text_thickness: int,
                                       text_size: float,
                                       color: Color):
        self.__paint.color = color

        self.__paint.style = Paint.Style.STROKE
        self.__paint.thickness = text_thickness
        self.__paint.text_size = text_size
        label_width, label_height = self.__paint.measure_text(label)
        label_x, label_y = x + padding, y + padding + label_height

        width = label_width + 2 * padding
        height = label_height + 2 * padding

        # Draws bounding box.
        self.__paint.style = Paint.Style.FILL
        self.__paint.thickness = 3
        self.__drawing_context.rectangle(x, y, width, height, self.__paint)

        self.__paint.style = Paint.Style.STROKE
        self.__paint.color = Color(red=255, green=255, blue=255)
        self.__paint.thickness = text_thickness
        self.__paint.text_size = text_size
        self.__drawing_context.text(label_x, label_y, label, self.__paint)

    def __draw_track(self,
                     track_id: int,
                     box: Rect):
        track_color: Color = Color.create_unique(track_id)

        # Draws detection rectangle.
        detection_x, detection_y, detection_width, detection_height = [int(v) for v in list(box)]

        self.__paint.style = Paint.Style.STROKE
        self.__paint.color = track_color
        self.__paint.thickness = 3
        self.__drawing_context.rectangle(detection_x, detection_y, detection_width, detection_height, self.__paint)

        # Draws label rectangle.
        self.__draw_label_with_bounding_box(label=str(track_id),
                                            x=detection_x, y=detection_y,
                                            padding=5,
                                            text_thickness=2,
                                            text_size=1.1,
                                            color=track_color)

    def draw_ground_truth(self, tracks: dict[int, Rect]):
        for track_id, box in tracks.items():
            self.__draw_track(track_id, box)

    def draw_detections(self, detections):
        self.__paint.style = Paint.Style.STROKE
        self.__paint.thickness = 3
        self.__paint.color = Color(red=0, green=0, blue=255)
        for i, detection in enumerate(detections):
            detection_origin = detection.origin
            self.__drawing_context.rectangle(int(detection_origin.left), int(detection_origin.top),
                                             int(detection_origin.width), int(detection_origin.height),
                                             self.__paint)

    def draw_trackers(self, tracks):
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            self.__draw_track(track.track_id, Rect.from_tlwh(track.to_tlwh()))

    def draw_info(self, info: str):
        background_color = Color(red=0, green=0, blue=0)
        # Draws label rectangle.
        self.__draw_label_with_bounding_box(label=info,
                                            x=25, y=25,
                                            padding=5,
                                            text_thickness=2,
                                            text_size=2,
                                            color=background_color)
