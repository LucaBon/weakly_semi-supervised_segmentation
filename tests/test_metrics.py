import numpy as np
import pytest

from wsss.constants import IGNORE_INDEX
from wsss.metrics import ConfusionMatrix, multilabel_stats


def test_iou_f1_oa_hand_computed():
    gt = np.array([0, 0, 1, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 0, 2])
    cm = ConfusionMatrix(n_classes=3)
    cm.update(pred, gt)
    # class 0: tp 1, fp 1, fn 1 -> IoU 1/3, F1 1/2
    # class 1: tp 2, fp 1, fn 1 -> IoU 2/4, F1 4/6
    np.testing.assert_allclose(cm.iou(), [1 / 3, 0.5, 1.0])
    np.testing.assert_allclose(cm.f1(), [0.5, 2 / 3, 1.0])
    assert cm.overall_accuracy() == pytest.approx(4 / 6)
    assert cm.overall_accuracy(classes=[0, 1]) == pytest.approx(3 / 5)


def test_ignore_index_and_absent_class_is_nan():
    cm = ConfusionMatrix(n_classes=3)
    cm.update(np.array([0, 2]), np.array([0, IGNORE_INDEX]))
    assert cm.matrix.sum() == 1
    assert np.isnan(cm.iou()[1])


def test_accumulated_differs_from_batch_average():
    """Per-batch averaging (the old evaluation) is biased; accumulation is exact."""
    gt_a, pred_a = np.array([1] + [0] * 99), np.array([1] + [0] * 99)
    gt_b, pred_b = np.array([1] * 50 + [0] * 50), np.array([0] * 100)
    total = ConfusionMatrix(n_classes=2)
    total.update(pred_a, gt_a)
    total.update(pred_b, gt_b)
    single = ConfusionMatrix(n_classes=2)
    single.update(np.concatenate([pred_a, pred_b]), np.concatenate([gt_a, gt_b]))
    np.testing.assert_array_equal(total.matrix, single.matrix)
    assert total.iou()[1] == pytest.approx(1 / 51)


def test_kappa_perfect():
    cm = ConfusionMatrix(n_classes=2)
    cm.update(np.array([0, 1, 1]), np.array([0, 1, 1]))
    assert cm.kappa() == pytest.approx(1.0)


def test_multilabel_stats_on_probabilities():
    probabilities = np.array([[0.9, 0.2], [0.6, 0.7]])
    targets = np.array([[1, 0], [0, 1]])
    stats = multilabel_stats(probabilities, targets)
    assert stats["precision"] == [0.5, 1.0]
    assert stats["recall"] == [1.0, 1.0]
