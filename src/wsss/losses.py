import numpy as np
import torch
import torch.nn.functional as F

from wsss.constants import IGNORE_INDEX, N_CLASSES, N_TAG_CLASSES


def class_weights_from_labels(labels, n_classes=N_CLASSES, max_weight=10.0):
    """
    Median-frequency balancing weights (Eigen & Fergus), clipped to
    `max_weight`; replaces the hand-tuned weight of 10 for cars.
    """
    counts = np.zeros(n_classes, dtype=np.float64)
    for label in labels:
        counts += np.bincount(label.ravel(), minlength=256)[:n_classes]
    frequencies = counts / counts.sum()
    present = frequencies > 0
    weights = np.ones(n_classes)
    weights[present] = np.median(frequencies[present]) / frequencies[present]
    return torch.tensor(np.clip(weights, 1 / max_weight, max_weight),
                        dtype=torch.float32)


def dice_loss(logits, target, ignore_index=IGNORE_INDEX, eps=1.0):
    """Soft multi-class Dice loss averaged over the classes present in the batch."""
    n_classes = logits.shape[1]
    valid = (target != ignore_index).unsqueeze(1)
    probabilities = logits.float().softmax(1) * valid
    one_hot = F.one_hot(target.clamp(max=n_classes - 1), n_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float() * valid
    intersection = (probabilities * one_hot).sum((0, 2, 3))
    cardinality = probabilities.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    dice = (2 * intersection + eps) / (cardinality + eps)
    present = one_hot.sum((0, 2, 3)) > 0
    if not present.any():
        return logits.sum() * 0
    return 1 - dice[present].mean()


def segmentation_loss(logits, target, class_weights=None, dice_weight=1.0):
    loss = F.cross_entropy(logits.float(), target, weight=class_weights,
                           ignore_index=IGNORE_INDEX)
    if dice_weight > 0:
        loss = loss + dice_weight * dice_loss(logits, target)
    return loss


def lse_pool(probabilities, valid=None, r=20.0):
    """
    Log-Sum-Exp pooling (Pinheiro & Collobert, CVPR 2015) of per-pixel class
    probabilities into crop-level presence probabilities in [0, 1]. It
    interpolates between average (r -> 0) and max pooling (r -> inf): the tag
    loss reaches many pixels, yet a small object such as a car can still raise
    its class score. Pooling softmax probabilities (not raw logits) ties tags
    to the segmentation: an absent class must have low probability everywhere.
    Args:
        probabilities (torch.Tensor): (B, C, H, W) softmax probabilities
        valid (torch.Tensor): optional (B, H, W) bool, pixels to pool over
        r (float): sharpness

    Returns:
        torch.Tensor: (B, C) pooled probabilities
    """
    probabilities = probabilities.float()
    if valid is None:
        valid = torch.ones_like(probabilities[:, 0], dtype=torch.bool)
    valid = valid.unsqueeze(1).expand_as(probabilities)
    scaled = (r * probabilities).masked_fill(~valid, float("-inf")).flatten(2)
    n_valid = valid.flatten(2).sum(-1).clamp(min=1).float()
    return ((torch.logsumexp(scaled, dim=-1) - torch.log(n_valid)) / r).clamp(0, 1)


def tag_scores(logits, valid=None, r=20.0):
    """Crop-level presence probabilities of the taggable classes (no clutter)."""
    return lse_pool(logits.float().softmax(1)[:, :N_TAG_CLASSES], valid, r)


def tag_loss(logits, tags, valid=None, r=20.0, eps=1e-4):
    """Multi-label BCE between pooled segmentation probabilities and crop tags."""
    scores = tag_scores(logits, valid, r).clamp(eps, 1 - eps)
    tags = tags.float()
    # written out: F.binary_cross_entropy is not allowed inside autocast
    return -(tags * scores.log() + (1 - tags) * (1 - scores).log()).mean()


def masked_cross_entropy(logits, target, ignore_index=IGNORE_INDEX):
    """Cross entropy over the non-ignored pixels; 0 (not NaN) if there are none."""
    loss = F.cross_entropy(logits.float(), target, ignore_index=ignore_index,
                           reduction="none")
    valid = (target != ignore_index).float()
    return (loss * valid).sum() / valid.sum().clamp(min=1)
