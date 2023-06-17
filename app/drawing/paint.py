from __future__ import annotations

import cv2

from app.drawing.color import Color
from enum import Enum


class Paint(object):
    class Style(Enum):
        FILL = 0
        STROKE = 1

    def __init__(self,
                 style: Paint.Style = Style.FILL):
        self.__style = style
        self.__thickness = 0
        self.__text_size = 0
        self.__color = Color(red=0, green=0, blue=0)

    @property
    def style(self) -> Paint.Style:
        return self.__style

    @style.setter
    def style(self, style: Paint.Style):
        self.__style = style

    @property
    def thickness(self):
        if self.__style == Paint.Style.STROKE:
            return self.__thickness
        else:
            return -1

    @thickness.setter
    def thickness(self, thickness: int):
        self.__thickness = thickness

    @property
    def color(self) -> Color:
        return self.__color

    @color.setter
    def color(self, color: Color):
        self.__color = color

    @property
    def text_size(self) -> float:
        return self.__text_size

    @text_size.setter
    def text_size(self, text_size):
        self.__text_size = text_size

    def measure_text(self, text: str) -> tuple[int, int]:
        # When drawing text paint should always have STROKE style.
        assert self.__style == Paint.Style.STROKE

        width = 0
        height = 0

        thickness = self.__thickness
        text_size = self.__text_size

        for i, line in enumerate(text.split('\n')):
            size = cv2.getTextSize(line, cv2.FONT_HERSHEY_PLAIN, text_size, thickness)
            width = max(width, size[0][0])
            height += size[0][1]

        return width, height
