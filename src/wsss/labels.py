import numpy as np

from wsss.constants import COLOR_MAPPING, IGNORE_INDEX, N_TAG_CLASSES


def convert_to_color(arr_2d, palette=COLOR_MAPPING):
    """
    Convert numeric labels to RGB-color encoding. Labels not in the palette
    (e.g. IGNORE_INDEX) are rendered black.
    Args:
        arr_2d (np.ndarray): (H, W) integer labels
        palette (dict): label -> RGB color

    Returns:
        np.ndarray: (H, W, 3) uint8 RGB image
    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for label, color in palette.items():
        arr_3d[arr_2d == label] = color
    return arr_3d


def convert_from_color(arr_3d, palette=COLOR_MAPPING, unknown=IGNORE_INDEX):
    """
    Convert RGB-color encoding to numeric labels. Colors that are not in the
    palette (e.g. black boundaries of the eroded ground truth) become `unknown`
    instead of being silently mapped to class 0.
    Args:
        arr_3d (np.ndarray): (H, W, 3) RGB encoded labels
        palette (dict): label -> RGB color
        unknown (int): value for colors not in the palette

    Returns:
        np.ndarray: (H, W) uint8 labels
    """
    arr_2d = np.full(arr_3d.shape[:2], unknown, dtype=np.uint8)
    for label, color in palette.items():
        mask = np.all(arr_3d == np.array(color, dtype=arr_3d.dtype), axis=-1)
        arr_2d[mask] = label
    return arr_2d


def calculate_image_labels(tile_labels, min_fraction=0.0,
                           n_classes=N_TAG_CLASSES):
    """
    Calculate the multi-label tag vector of a crop. Clutter and ignored pixels
    are neglected.
    Args:
        tile_labels (np.ndarray): (H, W) labels of the crop
        min_fraction (float): a class is present if the fraction of the crop's
            pixels belonging to it is strictly greater than this value
        n_classes (int): number of taggable classes

    Returns:
        np.ndarray: (n_classes,) float32 vector, e.g. [0, 1, 0, 0, 1]
    """
    counts = np.bincount(tile_labels.ravel(), minlength=256)[:n_classes]
    return (counts > min_fraction * tile_labels.size).astype(np.float32)
