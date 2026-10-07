import torch

from wsss.constants import CLUTTER, IGNORE_INDEX
from wsss.pseudo_label import (allowed_classes, class_thresholds,
                               make_pseudo_labels, tag_masked_probabilities)


def test_clutter_always_allowed():
    allowed = allowed_classes(torch.tensor([[0.0, 1, 0, 0, 0]]))
    assert allowed.tolist() == [[False, True, False, False, False, True]]


def test_absent_classes_get_zero_probability():
    logits = torch.zeros(1, 6, 2, 2)
    logits[0, 0] = 10  # the model says class 0, but the crop is tagged [building]
    probabilities = tag_masked_probabilities(logits, torch.tensor([[0.0, 1, 0, 0, 0]]))
    assert probabilities[0, 0].max() == 0
    torch.testing.assert_close(probabilities.sum(1), torch.ones(1, 2, 2))
    assert probabilities[0, CLUTTER].max() > 0


def test_low_confidence_and_invalid_pixels_ignored():
    probabilities = torch.zeros(1, 6, 1, 3)
    probabilities[0, 1, 0, 0] = 0.99
    probabilities[0, 4, 0, 1] = 0.75  # car, lower threshold
    probabilities[0, 2, 0, 2] = 0.6
    labels, _ = make_pseudo_labels(probabilities, class_thresholds(0.9, car=0.7))
    assert labels.tolist() == [[[1, 4, IGNORE_INDEX]]]
    valid = torch.tensor([[[False, True, True]]])
    labels, _ = make_pseudo_labels(probabilities, class_thresholds(0.9, car=0.7), valid)
    assert labels[0, 0, 0] == IGNORE_INDEX
