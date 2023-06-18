import cv2
import numpy as np

from src.app.drawing.paint import Paint


class DrawingContext(object):
    def __init__(self,
                 image: np.ndarray):
        self.__image = image

    @property
    def image(self) -> np.ndarray:
        return self.__image

    def point(self,
               x: int, y: int,
               paint: Paint):
        """Draw a point.

        Parameters
        ----------
        x : int
            Center of the circle (x-axis).
        y : int
            Center of the circle (y-axis).
        paint : Paint
            An object that defines drawing style.
        """
        color = paint.color
        thickness = paint.thickness

        cv2.circle(self.__image, (x, y), 1.0, color.raw, thickness)

    def rectangle(self,
                  x: int, y: int,
                  width: int, height: int,
                  paint: Paint):
        """Draws a rectangle.

        Parameters
        ----------
        x : int
            Top left corner of the rectangle (x-axis).
        y : int
            Top let corner of the rectangle (y-axis).
        width : int
            Width of the rectangle.
        height : int
            Height of the rectangle.
        paint : Paint
            An object that defines drawing style.
        """
        color = paint.color
        thickness = paint.thickness

        cv2.rectangle(self.__image, (x, y), (x + width, y + height), color.raw, thickness)

    def circle(self,
               x: int, y: int,
               radius: int,
               paint: Paint):
        """Draws a circle.

        Parameters
        ----------
        x : int
            Center of the circle (x-axis).
        y : int
            Center of the circle (y-axis).
        radius : int
            Radius of the circle in pixels.
        paint : Paint
            An object that defines drawing style.
        """
        color = paint.color
        thickness = paint.thickness

        cv2.circle(self.__image, (x, y), radius, color.raw, thickness)

    def text(self,
             x: int, y: int,
             text: str,
             paint: Paint):
        """Draws a text string at the given location.

        Parameters
        ----------
        x : int
            Bottom-left corner of the text in the image (x-axis).
        y : int
            Bottom-left corner of the text in the image (y-axis).
        text : str
            The text to be drawn.
        paint : Paint
            An object that defines drawing style.
        """
        # When drawing text paint should always have STROKE style.
        assert paint.style == Paint.Style.STROKE

        color = paint.color
        thickness = paint.thickness
        text_size = paint.text_size

        for i, line in enumerate(reversed(text.split('\n'))):
            _, line_height = paint.measure_text(line)

            cv2.putText(self.__image, line, (x, y), cv2.FONT_HERSHEY_PLAIN, text_size, color.raw, thickness)
            y -= line_height
