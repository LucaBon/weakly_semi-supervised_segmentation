import random

import numpy as np
from sklearn.metrics import confusion_matrix
from skimage import io

from constants import COLOR_MAPPING, INVERSE_COLOR_MAPPING


def extract_random_patch(img, window_shape):
    """
    Extract coordinates of a random patch of size window_shape
    Args:
        img (np.ndarray): image
        window_shape (tuple): (width, height)

    Returns:
        tuple: coordinates of the path extracted

    """

    width, height = window_shape
    image_width, image_height = img.shape[-2:]
    x1 = random.randint(0, image_width - width - 1)
    x2 = x1 + width
    y1 = random.randint(0, image_height - height - 1)
    y2 = y1 + height
    return x1, x2, y1, y2


def split_into_tiles(image_path, tile_size):
    """
    Take a big image and produce non-overlapping tiles out of it.
    Args:
        image_path (str): path to image
        tile_size (tuple): (tile height, tile width)

    Returns:
        list: list of tiles extracted from each image

    """
    tiles = []
    image = io.imread(image_path)
    if len(image.shape) == 2:
        image_height, image_width = image.shape
    elif len(image.shape) == 3:
        image_height, image_width, image_channels = image.shape
    tile_height, tile_width = tile_size
    n_tiles_height = int(image_height / tile_height)
    n_tiles_width = int(image_width / tile_width)
    for i in range(n_tiles_height):
        for j in range(n_tiles_width):
            tile = image[i * tile_height:(i + 1) * tile_height,
                         j * tile_width: (j + 1) * tile_width]
            tiles.append(tile)
    return tiles


def calculate_accuracy(predictions, gts, label_values=None):
    """

    Args:
        predictions:
        gts:
        label_values (None, list): optional, default None. Label names

    Returns:
        float: accuracy
    """
    labels_list = None
    if label_values is not None:
        labels_list = list(range(len(label_values)))
    confusion_matrix_result = confusion_matrix(gts.flatten(),
                                               predictions.flatten(),
                                               labels_list)

    # Compute global accuracy
    total = sum(sum(confusion_matrix_result))
    accuracy = sum([confusion_matrix_result[x][x]
                    for x in range(len(confusion_matrix_result))])
    accuracy *= 100 / float(total)

    return accuracy


def calculate_iou(predictions, gts, label_values=None):
    """

    Args:
        predictions:
        gts:
        label_values (None, list): optional, default None. Label names

    Returns:
        np.ndarray: Intersection over Union for each class
    """
    labels_list = None
    if label_values is not None:
        labels_list = list(range(len(label_values)))
    confusion_matrix_result = confusion_matrix(gts.flatten(),
                                               predictions.flatten(),
                                               labels_list)
    # Compute IoU (Intersection over union)
    iou = np.zeros(len(confusion_matrix_result))
    for i in range(len(confusion_matrix_result)):
        try:
            iou[i] = confusion_matrix_result[i, i] / \
                     (np.sum(confusion_matrix_result[i, :]) +
                      np.sum(confusion_matrix_result[:, i]) -
                      confusion_matrix_result[i, i])
        except:
            # Ignore exception if there is no element in class i for test set
            pass

    return iou


def calculate_f1_scores(predictions, gts, label_values=None):
    """

    Args:
        predictions:
        gts:
        label_values (None, list): optional, default None. Label names

    Returns:
        np.ndarray: f1 scores for each class
    """
    labels_list = None
    if label_values is not None:
        labels_list = list(range(len(label_values)))
    confusion_matrix_result = confusion_matrix(gts.flatten(),
                                               predictions.flatten(),
                                               labels_list)
    f1_score = np.zeros(len(confusion_matrix_result))
    for i in range(len(confusion_matrix_result)):
        try:
            f1_score[i] = 2. * confusion_matrix_result[i, i] / \
                          (np.sum(confusion_matrix_result[i, :]) +
                           np.sum(confusion_matrix_result[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    return f1_score


def calculate_kappa(predictions, gts, label_values=None):
    """

    Args:
        predictions:
        gts:
        label_values (None, list): optional, default None. Label names

    Returns:
        tuple: Cohen's kappa
    """
    labels_list = None
    if label_values is not None:
        labels_list = list(range(len(label_values)))
    confusion_matrix_result = confusion_matrix(gts.flatten(),
                                               predictions.flatten(),
                                               labels_list)

    # Compute kappa coefficient
    total = np.sum(confusion_matrix_result)
    pa = np.trace(confusion_matrix_result) / float(total)
    pe = np.sum(np.sum(confusion_matrix_result, axis=0) *
                np.sum(confusion_matrix_result, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    return kappa


def convert_to_color(arr_2d, palette=COLOR_MAPPING):
    """
    Convert numeric labels to RGB-color encoding
    Args:
        arr_2d (np.ndarray):
        palette (dict): color mapping

    Returns:
        np.ndarray: RGB image
    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color(arr_3d, palette=INVERSE_COLOR_MAPPING):
    """
    Convert RGB-color encoding to numeric labels
    Args:
        arr_3d (np.ndarray):RGB encoded labels
        palette (dict): color to labels mapping

    Returns:
        np.ndarray: array containing numeric labels
    """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def calculate_image_labels(tile_labels,
                           threshold_image_labels):
    """
    Calculate image-level labels. Clutter is neglected.
    Args:
        tile_labels (np.ndarray): labels on a tile of size TILE_SIZE
        threshold_image_labels: Threshold in percentage
            ( # pixels_of_that_class / # all_pixels)
            If the percentage of pixel of a specific class is greater than the
            threshold, that class is considered present (1), otherwise it is
            considered absent (0)

    Returns:
        np.ndarray (5,): label vector ex. [0, 1, 0, 0, 1]
    """

    # only five classes are considered, clutter is neglected
    label_vector = np.zeros(shape=5)

    labels, counts = np.unique(tile_labels, return_counts=True)
    threshold_number_pixel = threshold_image_labels * sum(counts) / 100
    for label, count in zip(labels, counts):
        if label != 5 and count > threshold_number_pixel:
            label_vector[label] = 1

    return label_vector


def per_label_accuracy(predictions, gts):
    """
    It calculates the accuracy for each label.
    For each label: (TP+TN) / N_SAMPLE
    Args:
        predictions (np.ndarray):
        gts (np.ndarray):

    Returns:
        np.ndarray: per label accuracy
    """
    n_samples, n_labels = predictions.shape
    per_label_correct_pred = np.zeros(n_labels)
    for label in range(n_labels):
        for pred_sample, gt_sample in zip(predictions, gts):
            if pred_sample[label] == gt_sample[label]:
                per_label_correct_pred[label] += 1
    return per_label_correct_pred / n_samples


def per_label_precision(predictions, gts):
    """
    For each label it calculates (TP / (TP + FP))
    Args:
        predictions:
        gts:

    Returns:
        np.ndarray: per label precision
    """
    n_samples, n_labels = predictions.shape
    per_label_tp = np.zeros(n_labels)
    per_label_fp = np.zeros(n_labels)
    for label in range(n_labels):
        for pred_sample, gt_sample in zip(predictions, gts):
            if pred_sample[label] == gt_sample[label] and pred_sample[label] == 1:
                per_label_tp[label] += 1
            if pred_sample[label] != gt_sample[label] and pred_sample[label] == 1:
                per_label_fp[label] += 1
    return per_label_tp / (per_label_tp + per_label_fp)


def per_label_recall(predictions, gts):
    """
    For each label it calculates (TP / (TP + FN))
    Args:
        predictions:
        gts:

    Returns:
        np.ndarray: per label recall
    """
    n_samples, n_labels = predictions.shape
    per_label_tp = np.zeros(n_labels)
    per_label_fn = np.zeros(n_labels)
    for label in range(n_labels):
        for pred_sample, gt_sample in zip(predictions, gts):
            if pred_sample[label] == gt_sample[label] and pred_sample[label] == 1:
                per_label_tp[label] += 1
            if pred_sample[label] != gt_sample[label] and pred_sample[label] == 0:
                per_label_fn[label] += 1
    return per_label_tp / (per_label_tp + per_label_fn)