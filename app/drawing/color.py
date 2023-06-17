from __future__ import annotations

import colorsys


class Color(object):
    def __init__(self,
                 red: int,
                 green: int,
                 blue: int):
        assert 0 <= red <= 255
        assert 0 <= green <= 255
        assert 0 <= blue <= 255

        self.__red = red
        self.__green = green
        self.__blue = blue

    @classmethod
    def create_unique(cls,
                      tag: int,
                      hue_step: float = 0.41) -> Color:
        """Creates a unique RGB color code for a given track id (tag).

        The color code is generated in HSV color space by moving along the
        hue angle and gradually changing the saturation.

        Parameters
        ----------
        tag : int
            The unique target identifying tag.
        hue_step : float
            Difference between two neighboring color codes in HSV space (more
            specifically, the distance in hue channel).

        Returns
        -------
        (float, float, float)
            RGB color code in range [0, 1]
        """

        hue, value = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
        return Color.from_hsv(hue, 1.0, value)

    @classmethod
    def from_float(cls,
                   red: float,
                   green: float,
                   blue: float) -> Color:
        assert 0.0 <= red <= 1.0
        assert 0.0 <= green <= 1.0
        assert 0.0 <= blue <= 1.0

        return Color(red=int(red * 255),
                     green=int(green * 255),
                     blue=int(blue * 255))

    @classmethod
    def from_hsv(cls,
                 hue: float,
                 saturation: float,
                 value: float) -> Color:
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        return Color.from_float(red, green, blue)

    @property
    def red(self) -> int:
        return self.__red

    @property
    def green(self) -> int:
        return self.__green

    @property
    def blue(self) -> int:
        return self.__blue

    @property
    def raw(self) -> tuple[int, int, int]:
        return self.__red, self.__green, self.__blue
