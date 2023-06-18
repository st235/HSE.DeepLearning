import pytest

from src.utils.geometry.rect import Rect


@pytest.mark.parametrize("ltwh,expected_lrtbwh", [
    ([0, 0, 10, 100], [0, 10, 0, 100, 10, 100]),
    ([0, 0, 15, 25], [0, 15, 0, 25, 15, 25]),
    ([10, 15, 73, 4], [10, 83, 15, 19, 73, 4]),
])
def test_initWithLeftTopWidthHeight_paramsAreCorrect_isSuccessful(ltwh, expected_lrtbwh):
    rect = Rect(left=ltwh[0], top=ltwh[1],
                width=ltwh[2], height=ltwh[3])

    expected_left, expected_right, \
        expected_top, expected_bottom, \
        expected_width, expected_height = expected_lrtbwh

    assert rect.left == expected_left
    assert rect.right == expected_right
    assert rect.top == expected_top
    assert rect.bottom == expected_bottom
    assert rect.width == expected_width
    assert rect.height == expected_height


@pytest.mark.parametrize("ltwh", [
    ([0, 0, -10, 100]),
    ([0, 0, 0, 100]),
    ([0, 0, 10, 0]),
    ([0, 0, 10, -100]),
    ([0, 0, 0, 0]),
    ([0, 0, -10, -10]),
])
def test_initWithLeftTopWidthHeight_sizeIsNegativeOrNull_initFails(ltwh):
    with pytest.raises(AssertionError):
        rect = Rect(left=ltwh[0], top=ltwh[1],
                    width=ltwh[2], height=ltwh[3])


@pytest.mark.parametrize("ltwh,expected_lrtbwh", [
    ([0, 0, 10, 100], [0, 10, 0, 100, 10, 100]),
    ([0, 0, 15, 25], [0, 15, 0, 25, 15, 25]),
    ([10, 15, 73, 4], [10, 83, 15, 19, 73, 4]),
])
def test_initUsingTlwhFactory_paramsAreCorrect_isSuccessful(ltwh, expected_lrtbwh):
    rect = Rect.from_tlwh(ltwh)

    expected_left, expected_right, \
        expected_top, expected_bottom, \
        expected_width, expected_height = expected_lrtbwh

    assert rect.left == expected_left
    assert rect.right == expected_right
    assert rect.top == expected_top
    assert rect.bottom == expected_bottom
    assert rect.width == expected_width
    assert rect.height == expected_height


@pytest.mark.parametrize("rect,expected_area", [
    ([0, 0, 10, 100], 1000),
    ([0, 0, 15, 25], 375),
    ([10, 15, 73, 4], 292),
    ([11, -5, 61, 43], 2623),
])
def test_area_coordinatesAreCorrect_correctArea(rect, expected_area):
    rect = Rect.from_tlwh(rect)
    assert rect.area == expected_area


@pytest.mark.parametrize("rect_a,rect_b,expected_intersection", [
    ([0, 0, 10, 100], [200, 0, 10, 100], False),
    ([50, 0, 100, 100], [100, 0, 10, 100], True),
    ([0, 0, 10, 100], [0, 160, 10, 100], False),
    ([0, 0, 10, 100], [0, 50, 20, 100], True),
    ([0, 0, 20, 100], [10, 10, 10, 100], True),
    ([0, 0, 10, 100], [-5, 0, 10, 100], True),
    ([0, 0, 50, 120], [20, 20, 10, 10], True),
    ([0, 0, 40, 50], [-10, -10, 200, 200], True),
    ([0, 0, 10, 100], [-50, -50, 10, 10], False),
])
def test_checkIfIntersects_rectsIntersect_methodReturnsCorrectResult(rect_a, rect_b, expected_intersection):
    one = Rect.from_tlwh(rect_a)
    another = Rect.from_tlwh(rect_b)
    assert one.check_if_intersects(another) == expected_intersection

@pytest.mark.parametrize("rect_a,rect_b,expected_intersection", [
    ([0, 0, 10, 100], [200, 0, 10, 100], 0),
    ([0, 0, 10, 100], [0, 160, 10, 100], 0),
    ([0, 0, 10, 100], [-50, -50, 10, 10], 0),
])
def test_iou_rectsDoNotIntersect_methodReturns0(rect_a, rect_b, expected_intersection):
    one = Rect.from_tlwh(rect_a)
    another = Rect.from_tlwh(rect_b)
    assert one.iou(another) == expected_intersection

@pytest.mark.parametrize("rect_a,rect_b,expected_intersection", [
    ([0, 0, 10, 100], [0, 50, 20, 100], 0.1999999992),
    ([0, 0, 20, 100], [10, 10, 10, 100], 0.4285714265306122),
    ([0, 0, 10, 100], [-5, 0, 10, 100], 0.33333333111111113),
    ([0, 0, 50, 120], [20, 20, 10, 10], 0.01666666663888889),
    ([0, 0, 40, 50], [-10, -10, 200, 200], 0.049999999987499995),
])
def test_iou_rectsIntersect_methodReturnsCorrectMetric(rect_a, rect_b, expected_intersection):
    one = Rect.from_tlwh(rect_a)
    another = Rect.from_tlwh(rect_b)
    assert abs(one.iou(another) - expected_intersection) < 1e-5


@pytest.mark.parametrize("rect,expected_order", [
    ([0, 0, 10, 100], [0, 0, 10, 100]),
    ([0, 0, 15, 25], [0, 0, 15, 25]),
    ([10, 15, 73, 4], [10, 15, 73, 4]),
    ([11, -5, 61, 43], [11, -5, 61, 43]),
])
def test_checkDefaultIterator_defaultIteratorIsLtwh_iteratorIsCorrect(rect, expected_order):
    rect = Rect.from_tlwh(rect)
    assert list(rect) == expected_order


@pytest.mark.parametrize("rect,expected_order", [
    ([0, 0, 10, 100], [0, 0, 10, 100]),
    ([0, 0, 15, 25], [0, 0, 15, 25]),
    ([10, 15, 73, 4], [10, 15, 73, 4]),
    ([11, -5, 61, 43], [11, -5, 61, 43]),
])
def test_ltwhIterator_coordinatesAreCorrect_correctLtwhOrder(rect, expected_order):
    rect = Rect.from_tlwh(rect)
    assert list(Rect.LeftTopWidthHeightIterator(rect)) == expected_order


@pytest.mark.parametrize("rect,expected_order", [
    ([0, 0, 10, 100], [5, 50, 1/10, 100]),
    ([0, 0, 15, 25], [15/2, 25/2, 15/25, 25]),
    ([10, 15, 73, 4], [10 + 73/2, 17, 73/4, 4]),
    ([11, -5, 61, 43], [11 + 61/2, -5 + 43/2, 61/43, 43]),
])
def test_xyahIterator_coordinatesAreCorrect_correctXyahOrder(rect, expected_order):
    rect = Rect.from_tlwh(rect)
    assert list(Rect.CenterXCenterYAspectRatioHeightIterator(rect)) == expected_order

