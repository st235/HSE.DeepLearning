class Rect(object):
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
    def from_tlwh(cls, tlwh):
        """Creates a rect from `(top left x, top left y, width, height)` array.
        """
        return Rect(tlwh[0], tlwh[1],
                    tlwh[2], tlwh[3])

    @classmethod
    def from_tlbr(cls, tlbr):
        """Creates a rect from `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)` array.
        """
        return Rect(tlbr[0], tlbr[1],
                    tlbr[2] - tlbr[0], tlbr[3] - tlbr[1])

    @classmethod
    def from_xyah(cls, xyah):
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
        return self.__width

    @property
    def height(self) -> float:
        return self.__height

    @property
    def top(self) -> float:
        return self.__top

    @property
    def left(self) -> float:
        return self.__left

    @property
    def right(self) -> float:
        return self.__left + self.__width

    @property
    def bottom(self) -> float:
        return self.__top + self.__height

    @property
    def center_x(self) -> float:
        return self.__left + self.__width / 2

    @property
    def center_y(self) -> float:
        return self.__top + self.__height / 2

    @property
    def aspect_ratio(self) -> float:
        return self.__width / self.__height

    def __str__(self):
        return f"Rect(left={self.__left},top={self.__top},width={self.__width},height={self.height})"
