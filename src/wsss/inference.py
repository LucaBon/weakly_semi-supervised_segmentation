import numpy as np
import torch

from wsss.constants import CAR, N_CLASSES, N_TAG_CLASSES
from wsss.data import grid_cells, to_tensor
from wsss.losses import tag_scores


def window_origins(length, window, stride):
    """Window start positions covering [0, length) entirely."""
    if length <= window:
        return [0]
    origins = list(range(0, length - window + 1, stride))
    if origins[-1] + window < length:
        origins.append(length - window)
    return origins


@torch.no_grad()
def sliding_window_probabilities(model, image, window=512, stride=256,
                                 batch_size=4, filter_threshold=None,
                                 filter_cell_size=200, tta=False, device="cuda", amp=True):
    """
    Class probabilities for a whole image, averaging overlapping windows, so
    every pixel (borders included) is predicted.
    Args:
        model (nn.Module): network returning logits
        image (np.ndarray): (H, W, 3) uint8
        window (int): window size, divisible by 32
        stride (int): stride between windows
        batch_size (int): windows per forward pass
        filter_threshold (float): if set, prediction filtering (Bae et al.
            2022): the taggable classes whose pooled presence score is below
            the threshold are suppressed
        filter_cell_size (int): scores are pooled over cells of this size,
            as during training (LSE scores depend on the pooled area)
        tta (bool): average over the 8 flips / 90-degree rotations
        device (str): device
        amp (bool): use mixed precision

    Returns:
        np.ndarray: (C, H, W) float32 probabilities
    """
    model.eval()
    height, width = image.shape[:2]
    pad_h, pad_w = max(0, window - height), max(0, window - width)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    padded_h, padded_w = image.shape[:2]
    tensor = to_tensor(image)
    probabilities = torch.zeros(N_CLASSES, padded_h, padded_w)
    counts = torch.zeros(1, padded_h, padded_w)
    origins = [(y, x) for y in window_origins(padded_h, window, stride)
               for x in window_origins(padded_w, window, stride)]
    for start in range(0, len(origins), batch_size):
        batch_origins = origins[start:start + batch_size]
        batch = torch.stack([tensor[:, y:y + window, x:x + window]
                             for y, x in batch_origins]).to(device)
        if tta:
            window_probabilities = dihedral_average(model, batch, device, amp)
            logits = window_probabilities.clamp(min=1e-8).log()
        else:
            with torch.autocast(device_type=device.split(":")[0], enabled=amp):
                logits = model(batch)
            window_probabilities = logits.float().softmax(1)
        if filter_threshold is not None:
            window_probabilities = filter_absent_classes(logits, filter_threshold,
                                                         filter_cell_size)
        window_probabilities = window_probabilities.cpu()
        for (y, x), p in zip(batch_origins, window_probabilities):
            probabilities[:, y:y + window, x:x + window] += p
            counts[:, y:y + window, x:x + window] += 1
    probabilities /= counts
    return probabilities[:, :height, :width].numpy()


def dihedral_average(model, batch, device, amp):
    """Softmax averaged over the 8 dihedral transforms of the input."""
    total = 0
    for flip in (False, True):
        for k in range(4):
            x = torch.rot90(batch, k, (2, 3))
            if flip:
                x = x.flip(3)
            with torch.autocast(device_type=device.split(":")[0], enabled=amp):
                p = model(x).float().softmax(1)
            if flip:
                p = p.flip(3)
            total = total + torch.rot90(p, -k, (2, 3))
    return total / 8


def filter_absent_classes(logits, threshold, cell_size=200):
    """Suppress, in each cell of the window, the taggable classes it is
    predicted not to contain. Returns probabilities."""
    logits = logits.float()
    absent = torch.zeros(logits.shape, dtype=torch.bool, device=logits.device)
    for y, x, h, w in grid_cells(*logits.shape[2:], cell_size):
        present = tag_scores(logits[:, :, y:y + h, x:x + w]) >= threshold
        absent[:, :N_TAG_CLASSES, y:y + h, x:x + w] = ~present[:, :, None, None]
    return logits.masked_fill(absent, float("-inf")).softmax(1)


def decide(probabilities, car_offset=0.0):
    """
    Class decision from (C, H, W) probabilities. `car_offset` is added to the
    car log-probability before the argmax: a negative value lowers the car
    prior, trading car recall for precision (post-hoc calibration).
    """
    if car_offset == 0:
        return probabilities.argmax(0).astype(np.uint8)
    scores = np.log(np.clip(probabilities, 1e-8, None))
    scores[CAR] += car_offset
    return scores.argmax(0).astype(np.uint8)


def predict(model, image, car_offset=0.0, refine=None, **kwargs):
    """
    (H, W) uint8 class prediction for a whole image.
    Args:
        refine (dict): optional PAMR settings (iterations, dilations)
    """
    probabilities = sliding_window_probabilities(model, image, **kwargs)
    if refine:
        from wsss.refine import refine_probabilities
        probabilities = refine_probabilities(probabilities, image,
                                             device=kwargs.get("device", "cuda"), **refine)
    return decide(probabilities, car_offset)
