import torch

from wsss.constants import CAR, CLUTTER, IGNORE_INDEX, N_CLASSES, N_TAG_CLASSES


def allowed_classes(tags):
    """
    (B, N_TAG_CLASSES) tags -> (B, N_CLASSES) bool of the classes a pixel of
    the crop may take. Clutter is never tagged, so it is always allowed.
    """
    allowed = torch.ones(tags.shape[0], N_CLASSES, dtype=torch.bool,
                         device=tags.device)
    allowed[:, :N_TAG_CLASSES] = tags > 0.5
    allowed[:, CLUTTER] = True
    return allowed


def tag_masked_probabilities(logits, tags):
    """
    Softmax restricted to the classes allowed by the crop tags: the prediction
    of a class the annotator said is absent is impossible, and its probability
    mass is redistributed to the present classes.
    """
    allowed = allowed_classes(tags)[:, :, None, None]
    return logits.float().masked_fill(~allowed, float("-inf")).softmax(1)


def make_pseudo_labels(probabilities, thresholds, valid=None):
    """
    Argmax pseudo-labels, ignored where the confidence is below the class
    threshold or outside `valid`.
    Args:
        probabilities (torch.Tensor): (B, C, H, W)
        thresholds (torch.Tensor): (C,) per-class confidence thresholds
        valid (torch.Tensor): optional (B, H, W) bool

    Returns:
        tuple(torch.Tensor, torch.Tensor): (B, H, W) int64 labels with
            IGNORE_INDEX, (B, H, W) confidences
    """
    confidence, labels = probabilities.max(1)
    keep = confidence >= thresholds.to(probabilities.device)[labels]
    if valid is not None:
        keep &= valid
    return labels.masked_fill(~keep, IGNORE_INDEX), confidence


def class_thresholds(default, car=None):
    thresholds = torch.full((N_CLASSES,), float(default))
    if car is not None:
        thresholds[CAR] = float(car)
    return thresholds
