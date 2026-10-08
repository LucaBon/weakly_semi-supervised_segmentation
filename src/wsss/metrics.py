import warnings

import numpy as np

from wsss.constants import EVAL_CLASSES, IGNORE_INDEX, LABEL_NAMES, N_CLASSES


class ConfusionMatrix:
    """
    Confusion matrix accumulated over a whole dataset. Metrics must be computed
    once from the accumulated matrix, not averaged over batches: per-batch
    averaging biases rare classes such as Car.
    Rows are ground truth, columns are predictions.
    """

    def __init__(self, n_classes=N_CLASSES, ignore_index=IGNORE_INDEX):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, predictions, gts):
        predictions = np.asarray(predictions).ravel().astype(np.int64)
        gts = np.asarray(gts).ravel().astype(np.int64)
        valid = (gts != self.ignore_index) & (gts < self.n_classes)
        index = gts[valid] * self.n_classes + predictions[valid]
        self.matrix += np.bincount(
            index, minlength=self.n_classes ** 2).reshape(self.n_classes,
                                                          self.n_classes)

    def iou(self):
        tp = np.diag(self.matrix).astype(np.float64)
        denominator = self.matrix.sum(0) + self.matrix.sum(1) - tp
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(denominator > 0, tp / denominator, np.nan)

    def f1(self):
        tp = np.diag(self.matrix).astype(np.float64)
        denominator = self.matrix.sum(0) + self.matrix.sum(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(denominator > 0, 2 * tp / denominator, np.nan)

    def overall_accuracy(self, classes=None):
        """Accuracy over the pixels whose ground truth is in `classes`."""
        matrix = self.matrix if classes is None else self.matrix[classes]
        total = matrix.sum()
        if classes is None:
            correct = np.trace(matrix)
        else:
            correct = matrix[np.arange(len(classes)), classes].sum()
        return correct / total if total > 0 else np.nan

    def kappa(self):
        total = self.matrix.sum()
        pa = np.trace(self.matrix) / total
        pe = np.sum(self.matrix.sum(0) * self.matrix.sum(1)) / total ** 2
        return (pa - pe) / (1 - pe)

    def summary(self, classes=EVAL_CLASSES, names=LABEL_NAMES):
        """
        ISPRS-style summary: per-class scores and their means over `classes`
        (clutter excluded by default). All scores come from the full matrix,
        so ground-truth clutter predicted as class c is a false positive of c,
        and OA counts every non-ignored pixel.
        Returns:
            dict: per-class IoU/F1, mIoU, mF1, OA
        """
        iou, f1 = self.iou()[classes], self.f1()[classes]
        with warnings.catch_warnings():
            # all-NaN (empty matrix) gives NaN means without a warning
            warnings.simplefilter("ignore", RuntimeWarning)
            miou, mf1 = np.nanmean(iou), np.nanmean(f1)
        return {"per_class_iou": {names[c]: float(v) for c, v in zip(classes, iou)},
                "per_class_f1": {names[c]: float(v) for c, v in zip(classes, f1)},
                "miou": float(miou),
                "mf1": float(mf1),
                "oa": float(self.overall_accuracy())}


def multilabel_stats(probabilities, targets, threshold=0.5):
    """
    Per-label precision and recall for multi-label tag predictions.
    Args:
        probabilities (np.ndarray): (N, C) sigmoid probabilities (not logits)
        targets (np.ndarray): (N, C) binary ground truth
        threshold (float): decision threshold on probabilities

    Returns:
        dict: per-label precision, recall, accuracy as lists
    """
    predictions = probabilities > threshold
    targets = targets.astype(bool)
    tp = (predictions & targets).sum(0)
    fp = (predictions & ~targets).sum(0)
    fn = (~predictions & targets).sum(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
    accuracy = (predictions == targets).mean(0)
    return {"precision": precision.tolist(),
            "recall": recall.tolist(),
            "accuracy": accuracy.tolist()}


def boundary_map(labels):
    """True where a pixel differs from its right or bottom neighbour."""
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return boundary


class BoundaryScore:
    """
    Boundary F-score (Csurka et al., BMVC 2013): a predicted boundary pixel is
    correct if a ground-truth boundary lies within `tolerance` pixels, and
    vice versa. Counts are accumulated over a dataset, for all classes
    together and for each class separately.
    """

    def __init__(self, tolerance=2, classes=(), n_classes=N_CLASSES):
        self.tolerance = tolerance
        self.classes = list(classes)
        self.counts = {key: np.zeros(4) for key in ["all"] + self.classes}

    def _update(self, key, predicted, actual):
        from scipy.ndimage import distance_transform_edt
        if predicted.any():
            to_actual = distance_transform_edt(~actual) if actual.any() else np.full(actual.shape, np.inf)
            self.counts[key][0] += (to_actual[predicted] <= self.tolerance).sum()
        if actual.any():
            to_predicted = distance_transform_edt(~predicted) if predicted.any() else np.full(actual.shape, np.inf)
            self.counts[key][2] += (to_predicted[actual] <= self.tolerance).sum()
        self.counts[key][1] += predicted.sum()
        self.counts[key][3] += actual.sum()

    def update(self, prediction, gt):
        self._update("all", boundary_map(prediction), boundary_map(gt))
        for c in self.classes:
            self._update(c, boundary_map(prediction == c), boundary_map(gt == c))

    def summary(self):
        results = {}
        for key, (tp_p, n_p, tp_r, n_r) in self.counts.items():
            precision = tp_p / n_p if n_p else np.nan
            recall = tp_r / n_r if n_r else np.nan
            if np.isnan(precision) or np.isnan(recall):
                f1 = np.nan
            elif precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            results[key] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
        return results
