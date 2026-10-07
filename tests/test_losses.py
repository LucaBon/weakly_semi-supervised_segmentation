import numpy as np
import pytest
import torch

from wsss.constants import IGNORE_INDEX
from wsss.losses import (class_weights_from_labels, dice_loss, lse_pool,
                         masked_cross_entropy, tag_loss)


def test_lse_pool_between_mean_and_max():
    probabilities = torch.zeros(1, 1, 10, 10)
    probabilities[0, 0, 0, 0] = 1.0
    for r in (1.0, 20.0):
        pooled = lse_pool(probabilities, r=r).item()
        assert 0.01 <= pooled <= 1.0
    assert lse_pool(probabilities, r=1000.0).item() == pytest.approx(1.0, abs=0.01)
    assert lse_pool(torch.zeros(1, 1, 4, 4)).item() == pytest.approx(0.0, abs=1e-6)


def test_lse_pool_respects_valid_mask():
    probabilities = torch.zeros(1, 1, 4, 4)
    probabilities[0, 0, 0, 0] = 1.0
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    valid[0, 0, 0] = False
    assert lse_pool(probabilities, valid).item() == pytest.approx(0.0, abs=1e-6)


def test_tag_loss_gradient_suppresses_absent_class():
    logits = torch.zeros(1, 6, 8, 8, requires_grad=True)
    tags = torch.tensor([[1.0, 0, 0, 0, 0]])
    tag_loss(logits, tags).backward()
    # class 1 is absent: its logits must decrease, class 0 present: increase
    assert (logits.grad[0, 1] > 0).all()
    assert (logits.grad[0, 0] < 0).all()


def test_masked_cross_entropy_all_ignored_is_zero():
    logits = torch.randn(2, 6, 4, 4)
    target = torch.full((2, 4, 4), IGNORE_INDEX)
    assert masked_cross_entropy(logits, target).item() == 0.0


def test_dice_loss_perfect_prediction():
    target = torch.randint(0, 6, (2, 8, 8))
    logits = torch.nn.functional.one_hot(target, 6).permute(0, 3, 1, 2).float() * 100
    assert dice_loss(logits, target).item() == pytest.approx(0.0, abs=1e-3)


def test_class_weights_favour_rare_classes():
    label = np.zeros((10, 10), dtype=np.uint8)
    label[:, 5:] = 1
    label[0, 0] = 4
    weights = class_weights_from_labels([label], max_weight=10.0)
    assert weights[4] == pytest.approx(10.0)
    assert weights[0] <= weights[4]
