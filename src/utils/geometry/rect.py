from __future__ import annotations


class Rect(object):
    """Represents a rectangular area and helps to deal with their geometry.

    Parameters:
    ----------
        __left: float
            Left edge of the rectangle.
        __top: float
            Top edge of the rectangle.
        __width: float
            Width of the rectangle.
        __height: float
            Height of the rectangle.
    """

    def __init__(self,
                 left: float, top: float,
                 width: float, height: float):
        assert width > 0
        assert height > 0

        self.__left = left
        self.__top = top
        self.__width = width
        self.__height = height

    @classmethod
    def from_tlwh(cls, tlwh) -> Rect:
        """Creates a rect from `(top left x, top left y, width, height)` array.
        """
        return Rect(tlwh[0], tlwh[1],
                    tlwh[2], tlwh[3])

    @classmethod
    def from_tlbr(cls, tlbr) -> Rect:
        """Creates a rect from `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)` array.
        """
        return Rect(tlbr[0], tlbr[1],
                    tlbr[2] - tlbr[0] + 1, tlbr[3] - tlbr[1] + 1)

    @classmethod
    def from_xyah(cls, xyah) -> Rect:
        """Creates a rect from `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """

        height = xyah[3]
        width = xyah[2] * height

        left = xyah[0] - width / 2
        top = xyah[1] - height / 2

        return Rect(left, top,
                    width, height)

    @property
    def width(self) -> float:
        """Returns width of the rect.
        """
        return self.__width

    @property
    def height(self) -> float:
        """Returns height of the rect.
        """
        return self.__height

    @property
    def top(self) -> float:
        """Returns first horizontal pixels position (aka top edge) in the original image.
        """
        return self.__top

    @property
    def left(self) -> float:
        """Returns first vertical pixels position (aka left edge) in the original image.
        """
        return self.__left

    @property
    def right(self) -> float:
        """Returns last vertical pixels position (aka right edge) in the original image.
        """
        return self.__left + self.__width - 1

    @property
    def bottom(self) -> float:
        """Returns last horizontal pixels position (aka bottom edge) in the original image.
        """
        return self.__top + self.__height - 1

    @property
    def center_x(self) -> float:
        """Returns horizontal central position.
        """
        return self.__left + self.__width / 2

    @property
    def center_y(self) -> float:
        """Returns vertical central position.
        """
        return self.__top + self.__height / 2

    @property
    def aspect_ratio(self) -> float:
        """Returns ratio of width to height.

         Ration is calculated using the next equation: width / height.
        """
        return self.__width / self.__height

    @property
    def area(self) -> float:
        """Returns area of the rectangle.

         Area is calculated using the next equation: width * height.
        """
        return self.__width * self.__height

    def inset(self,
              left: float, top: float,
              right: float, bottom: float) -> Rect:
        """Adds paddings to the current rect.

        Parameters
        ----------
        left: float
            Left padding.
        top: float
            Top padding.
        right: float
            Right padding.
        bottom: float
            Bottom padding.

        Returns
        -------
            Returns a new rectangle with new coordinates and sizes, reflecting the padding.
        """
        assert left >= 0
        assert top >= 0
        assert right >= 0
        assert bottom >= 0

        return Rect(left=self.__left - left, top=self.__top - top,
                    width=self.__width + left + right, height=self.__height + top + bottom)

    def check_if_intersects(self, that: Rect) -> bool:
        """Checks if the rectangle intersects with the given rect.

        Parameters
        ----------
        that: Rect
            Another Rect we are testing against.

        Returns
        -------
            True if rectangles intersect and False otherwise. If rectangles share a common edge they are not
            considered as intersecting.
        """
        return self.__check_if_lines_intersect(self.left, self.right, that.left, that.right) and \
            self.__check_if_lines_intersect(self.top, self.bottom, that.top, that.bottom)

    def iou(self, that: Rect) -> float:
        """Calculates intersection over union.

        IoU is a ratio of the area of intersection of two rectangles over their combined area (aka union area).

        Returns
        -------
            A ratio. The value is always within [0, 1].
        """

        assert isinstance(that, Rect), f"Expected Rect but found {type(that)}: {that}"

        if not self.check_if_intersects(that):
            return 0

        my_area = self.area
        that_area = that.area

        intersection_top = max(self.top, that.top)
        intersection_left = max(self.left, that.left)
        intersection_bottom = min(self.bottom, that.bottom)
        intersection_right = min(self.right, that.right)

        intersection_width = intersection_right - intersection_left
        intersection_height = intersection_bottom - intersection_top

        intersection_area = intersection_width * intersection_height
        assert intersection_area > 0

        union_area = my_area + that_area - intersection_area
        assert union_area > 0

        return intersection_area / union_area

    def resize(self,
               target_width: float,
               target_height: float) -> Rect:
        """Scales the rectangle to the same aspect ratio as target_width over target_height.

        Parameters
        ----------
        target_width: float
            New width of the rectangle.
        target_height: float
            New height of the rectangle.

        Returns
        -------
            A new rectangle with aspect_ratio equal to target_width / target_height.
        """

        assert isinstance(target_width, float), \
            f"Target width is not float {type(target_width)}"
        assert isinstance(target_height, float), \
            f"Target height is not float {type(target_height)}"

        target_aspect_ratio = target_width / target_height

        new_width = target_aspect_ratio * self.height
        new_left = self.left - (new_width - self.width) / 2

        return Rect(left=new_left, top=self.top, width=new_width, height=self.height)

    def clip(self,
             that: Rect) -> Rect:
        """Clips the other rect by the bounding boxes of this rect.

        Returns
        -------
            Returns a new rect with new bounding boxes or raises exception if the other box is completely
             outside the bounding box of the current rect.
        """
        left = max(that.left, self.left)
        right = min(that.right, self.right)
        top = max(that.top, self.top)
        bottom = min(that.bottom, self.bottom)

        new_width = right - left + 1
        new_height = bottom - top + 1

        return Rect(left=left, top=top, width=new_width, height=new_height)

    @classmethod
    def __check_if_lines_intersect(cls, a_start: float, a_end: float,
                                   b_start: float, b_end: float):
        return max(a_start, b_start) < min(a_end, b_end)

    def __iter__(self):
        return Rect.LeftTopWidthHeightIterator(self)

    def __repr__(self):
        cls = type(self)
        return f"<{cls.__module__}.{cls.__name__}> h: {self.left}-{self.right}, v: {self.top}-{self.bottom}"

    def __str__(self):
        return f"Rect(left={self.__left},top={self.__top},width={self.__width},height={self.height})"

    class LeftTopWidthHeightIterator(object):
        def __init__(self, rect):
            self.__rect = rect
            self.__index = 0

        def __iter__(self):
            self.__index = 0
            return self

        def __next__(self):
            if self.__index == 0:
                self.__index += 1
                return self.__rect.left
            if self.__index == 1:
                self.__index += 1
                return self.__rect.top
            if self.__index == 2:
                self.__index += 1
                return self.__rect.width
            if self.__index == 3:
                self.__index += 1
                return self.__rect.height
            raise StopIteration

    class CenterXCenterYAspectRatioHeightIterator(object):
        def __init__(self, rect):
            self.__rect = rect
            self.__index = 0

        def __iter__(self):
            self.__index = 0
            return self

        def __next__(self):
            if self.__index == 0:
                self.__index += 1
                return self.__rect.center_x
            if self.__index == 1:
                self.__index += 1
                return self.__rect.center_y
            if self.__index == 2:
                self.__index += 1
                return self.__rect.aspect_ratio
            if self.__index == 3:
                self.__index += 1
                return self.__rect.height
            raise StopIteration
