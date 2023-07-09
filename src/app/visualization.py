import numpy as np

from src.app.drawing.color import Color
from src.app.drawing.drawing_context import DrawingContext
from src.app.drawing.paint import Paint
from src.deep_sort.detector.detection import Detection
from src.deep_sort.track import Track
from src.deep_sort.segmentation.segmentation import Segmentation
from src.utils.geometry.rect import Rect


class Visualization(object):
    """Draws tacking output on the given image.
    """

    def __init__(self,
                 image: np.ndarray):
        self.__paint = Paint()
        self.__drawing_context = DrawingContext(image=image.copy())

    @property
    def output_image(self) -> np.ndarray:
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

    def draw_detections(self, detections: list[Detection]):
        self.__paint.style = Paint.Style.STROKE
        self.__paint.thickness = 3
        self.__paint.color = Color(red=0, green=0, blue=255)
        for detection in detections:
            detection_bbox = detection.origin
            self.__drawing_context.rectangle(int(detection_bbox.left), int(detection_bbox.top),
                                             int(detection_bbox.width), int(detection_bbox.height),
                                             self.__paint)

    def draw_segmentation(self, segmentations: list[Segmentation]):
        self.__paint.style = Paint.Style.STROKE
        self.__paint.thickness = 3
        self.__paint.color = Color(red=0, green=0, blue=255)

        for segmentation in segmentations:
            segmentation_bbox = segmentation.bbox
            self.__drawing_context.rectangle(int(segmentation_bbox.left), int(segmentation_bbox.top),
                                             int(segmentation_bbox.width), int(segmentation_bbox.height),
                                             self.__paint)

            self.__drawing_context.draw_mask(segmentation.mask)

    def draw_trackers(self, tracks: list[Track]):
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            self.__draw_track(track.track_id, track.bounding_box)

    def draw_info(self, info: str):
        background_color = Color(red=0, green=0, blue=0)
        # Draws label rectangle.
        self.__draw_label_with_bounding_box(label=info,
                                            x=25, y=25,
                                            padding=5,
                                            text_thickness=2,
                                            text_size=2,
                                            color=background_color)
