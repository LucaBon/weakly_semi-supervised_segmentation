import numpy as np

from wsss.constants import IGNORE_INDEX
from wsss.labels import calculate_image_labels, convert_from_color, convert_to_color


def test_color_round_trip():
    labels = np.arange(6, dtype=np.uint8).repeat(4).reshape(4, 6)
    np.testing.assert_array_equal(convert_from_color(convert_to_color(labels)), labels)


def test_unknown_color_is_ignored_not_class_zero():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)  # black = eroded boundary
    rgb[0, 0] = (255, 255, 255)
    labels = convert_from_color(rgb)
    assert labels[0, 0] == 0
    assert (labels.ravel()[1:] == IGNORE_INDEX).all()


def test_image_labels_skip_clutter_and_ignore():
    tile = np.array([[0, 0, 5], [4, IGNORE_INDEX, 5]], dtype=np.uint8)
    np.testing.assert_array_equal(calculate_image_labels(tile), [1, 0, 0, 0, 1])


def test_image_labels_min_fraction():
    tile = np.zeros((10, 10), dtype=np.uint8)
    tile[0, 0] = 4  # 1% of the pixels
    np.testing.assert_array_equal(calculate_image_labels(tile, 0.0), [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(calculate_image_labels(tile, 0.05), [1, 0, 0, 0, 0])
